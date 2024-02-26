import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.math import generate_permute_matrix
from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d
from extensions.reg_att_map_generator import RegionalAttentionMapGenerator


class AOTEngine(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=5,
                 short_term_mem_skip=1):
        super().__init__()

        self.cfg = aot_model.cfg
        self.align_corners = aot_model.cfg.MODEL_ALIGN_CORNERS
        self.AOT = aot_model

        self.max_obj_num = aot_model.max_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.losses = None

        self.ram = RegionalAttentionMapGenerator()

        self.restart_engine()

    def forward(self,
                all_frames,
                all_masks,
                batch_size,
                obj_nums,
                step=0,
                tf_board=False,
                use_prev_pred=False,
                enable_prev_frame=False,
                use_prev_prob=False):  # only used for training
        if self.losses is None:
            self._init_losses()

        self.freeze_id = True if use_prev_pred else False
        aux_weight = self.aux_weight * max(self.aux_step - step,
                                           0.) / self.aux_step

        self.offline_encoder(all_frames, all_masks)

        self.add_reference_frame(frame_step=0, obj_nums=obj_nums)

        grad_state = torch.no_grad if aux_weight == 0 else torch.enable_grad
        with grad_state():
            ref_aux_loss, ref_aux_mask = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step)

        aux_losses = [ref_aux_loss]
        aux_masks = [ref_aux_mask]

        curr_losses, curr_masks = [], []
        if enable_prev_frame:
            self.set_prev_frame(frame_step=1)
            with grad_state():
                prev_aux_loss, prev_aux_mask = self.generate_loss_mask(
                    self.offline_masks[self.frame_step], step)
            aux_losses.append(prev_aux_loss)
            aux_masks.append(prev_aux_mask)
        else:
            self.match_propogate_one_frame()
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step, return_prob=True)
            self.update_short_term_memory(
                curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else self.assign_identity(
                    self.offline_one_hot_masks[self.frame_step]))
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        self.match_propogate_one_frame()
        curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
            self.offline_masks[self.frame_step], step, return_prob=True)
        curr_losses.append(curr_loss)
        curr_masks.append(curr_mask)
        for _ in range(self.total_offline_frame_num - 3):
            self.update_short_term_memory(
                curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else self.assign_identity(
                    self.offline_one_hot_masks[self.frame_step]))
            self.match_propogate_one_frame()
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step, return_prob=True)
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        aux_loss = torch.cat(aux_losses, dim=0).mean(dim=0)
        pred_loss = torch.cat(curr_losses, dim=0).mean(dim=0)

        loss = aux_weight * aux_loss + pred_loss

        all_pred_mask = aux_masks + curr_masks

        all_frame_loss = aux_losses + curr_losses

        boards = {'image': {}, 'scalar': {}}

        return loss, all_pred_mask, all_frame_loss, boards

    def _init_losses(self):
        cfg = self.cfg

        from networks.layers.loss import CrossEntropyLoss, SoftJaccordLoss
        bce_loss = CrossEntropyLoss(
            cfg.TRAIN_TOP_K_PERCENT_PIXELS,
            cfg.TRAIN_HARD_MINING_RATIO * cfg.TRAIN_TOTAL_STEPS)
        iou_loss = SoftJaccordLoss()

        losses = [bce_loss, iou_loss]
        loss_weights = [0.5, 0.5]

        self.losses = nn.ModuleList(losses)
        self.loss_weights = loss_weights
        self.aux_weight = cfg.TRAIN_AUX_LOSS_WEIGHT
        self.aux_step = cfg.TRAIN_TOTAL_STEPS * cfg.TRAIN_AUX_LOSS_RATIO + 1e-5

    def encode_one_img_mask(self, img=None, mask=None, frame_step=-1):
        if frame_step == -1:
            frame_step = self.frame_step

        if self.enable_offline_enc:
            curr_enc_embs = self.offline_enc_embs[frame_step]
        elif img is None:
            curr_enc_embs = None
        else:
            curr_enc_embs = self.AOT.encode_image(img)

        if mask is not None:
            curr_one_hot_mask = one_hot_mask(mask, self.max_obj_num)
        elif self.enable_offline_enc:
            curr_one_hot_mask = self.offline_one_hot_masks[frame_step]
        else:
            curr_one_hot_mask = None

        return curr_enc_embs, curr_one_hot_mask

    def offline_encoder(self, all_frames, all_masks=None):
        self.enable_offline_enc = True
        self.offline_frames = all_frames.size(0) // self.batch_size

        # extract backbone features
        self.offline_enc_embs = self.split_frames(
            self.AOT.encode_image(all_frames), self.batch_size)
        self.total_offline_frame_num = len(self.offline_enc_embs)

        if all_masks is not None:
            # extract mask embeddings
            offline_one_hot_masks = one_hot_mask(all_masks, self.max_obj_num)
            self.offline_masks = list(
                torch.split(all_masks, self.batch_size, dim=0))
            self.offline_one_hot_masks = list(
                torch.split(offline_one_hot_masks, self.batch_size, dim=0))

        if self.input_size_2d is None:
            self.update_size(all_frames.size()[2:],
                             self.offline_enc_embs[0][-1].size()[2:])

    def assign_identity(self, one_hot_mask):
        if self.enable_id_shuffle:
            one_hot_mask = torch.einsum('bohw,bot->bthw', one_hot_mask,
                                        self.id_shuffle_matrix)

        id_emb = self.AOT.get_id_emb(one_hot_mask).view(
            self.batch_size, -1, self.enc_hw).permute(2, 0, 1)

        if self.training and self.freeze_id:
            id_emb = id_emb.detach()

        return id_emb

    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs))

    def add_reference_frame(self,
                            img=None,
                            mask=None,
                            frame_step=-1,
                            obj_nums=None,
                            img_embs=None):
        if self.obj_nums is None and obj_nums is None:
            print('No objects for reference frame!')
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step

        if img_embs is None:
            curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(
                img, mask, frame_step)
        else:
            _, curr_one_hot_mask = self.encode_one_img_mask(
                None, mask, frame_step)
            curr_enc_embs = img_embs

        if curr_enc_embs is None:
            print('No image for reference frame!')
            exit()

        if curr_one_hot_mask is None:
            print('No mask for reference frame!')
            exit()

        if self.input_size_2d is None:
            self.update_size(img.size()[2:], curr_enc_embs[-1].size()[2:])

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        if self.pos_emb is None:
            self.pos_emb = self.AOT.get_pos_emb(curr_enc_embs[-1]).expand(
                self.batch_size, -1, -1,
                -1).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1)

        curr_id_emb = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_emb

        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      None,
                                                      None,
                                                      curr_id_emb,
                                                      pos_emb=self.pos_emb,
                                                      size_2d=self.enc_size_2d)

        self.curr_lstt_output = self.curr_lstt_output[:-1]
        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output

        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
            self.head = self.long_term_memories[0][0].size()[0]
            self.tail = self.long_term_memories[0][0].size()[0]
        else:
            self.update_long_term_memory(lstt_long_memories, mask)

        self.last_mem_step = self.frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories

    def set_prev_frame(self, img=None, mask=None, frame_step=1):
        self.frame_step = frame_step
        curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(
            img, mask, frame_step)

        if curr_enc_embs is None:
            print('No image for previous frame!')
            exit()

        if curr_one_hot_mask is None:
            print('No mask for previous frame!')
            exit()

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        curr_id_emb = self.assign_identity(curr_one_hot_mask)
        self.curr_id_embs = curr_id_emb

        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      None,
                                                      None,
                                                      curr_id_emb,
                                                      pos_emb=self.pos_emb,
                                                      size_2d=self.enc_size_2d)

        self.att_maps = self.curr_lstt_output[-1]
        self.curr_lstt_output = self.curr_lstt_output[:-1]
        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output

        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
        else:
            self.update_long_term_memory(lstt_long_memories)
        self.last_mem_step = frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories

    def compress_attn(self, q_raw, k, v1, v2):
        """
        q [C_A, 1, 128]
        k [C_T, 1, 128]
        v1 [C_T, 1, 512]
        v2 [C_T, 1, 512]
        """
        q = q_raw / q_raw.size()[-1]

        q = q.permute(1, 0, 2) # [1, C_A, 128]
        k = k.permute(1, 2, 0) # [1, 128, C_T]
        v1 = v1.permute(1, 0, 2) # [1, C_T, 512]
        v2 = v2.permute(1, 0, 2) # [1, C_T, 512]

        qk = q @ k # [1, C_A, C_T]
        qk = torch.softmax(qk, dim=-1)

        v1 = qk @ v1
        v2 = qk @ v2
        # q = q.permute(1, 0, 2)
        v1 = v1.permute(1, 0, 2)
        v2 = v2.permute(1, 0, 2)
        after_compress_memory = [q_raw, v1, None, v2]
        return after_compress_memory

    def compress_features(self, att_maps, tobe_compress_memories):
        after_compress_memories = []
        for att_map, tobe_compress_memory in zip(att_maps, tobe_compress_memories):
            values, indices = att_map.topk(self.len_after_compress, dim=-1, largest=True)

            k = tobe_compress_memory[0]
            q = torch.index_select(k, 0, indices[0, 0, :])
            v1 = tobe_compress_memory[1]
            v2 = tobe_compress_memory[3]

            after_compress_memory = self.compress_attn(q, k, v1, v2)
            after_compress_memories.append(after_compress_memory)

        return after_compress_memories

    def to_compress(self):
        att_maps = []
        prev_long_term_memories = []
        last_long_term_memories = []
        tobe_compress_memories = []
        for att_map, memories in zip(self.att_maps, self.long_term_memories):
            slice_att_map = att_map[:, :, self.head:self.head+self.len_tobe_compress]
            # slice_att_map = torch.sum(slice_att_map, dim=-2)
            att_maps.append(slice_att_map)

            prev_long_term_memory = []
            last_long_term_memory = []
            tobe_compress_memory = []
            for e_memory in memories:
                if e_memory is None:
                    prev_long_term_memory.append(None)
                    last_long_term_memory.append(None)
                    tobe_compress_memory.append(None)
                else:
                    prev_long_term_memory.append(e_memory[:self.head, :, :])
                    last_long_term_memory.append(e_memory[self.head+self.len_tobe_compress:, :, :])
                    tobe_compress_memory.append(e_memory[self.head:self.head+self.len_tobe_compress, :, :])

            prev_long_term_memories.append(prev_long_term_memory)
            last_long_term_memories.append(last_long_term_memory)
            tobe_compress_memories.append(tobe_compress_memory)

        after_compress_memories = self.compress_features(att_maps, tobe_compress_memories)

        long_term_memories = []
        for prev_memory, after_memory, last_memory in zip(prev_long_term_memories, after_compress_memories, last_long_term_memories):
            long_term_memory = []
            for i in range(len(prev_memory)):
                if prev_memory[i] is None:
                    long_term_memory.append(None)
                else:
                    e_memory = torch.cat([prev_memory[i], after_memory[i], last_memory[i]], dim=0)
                    long_term_memory.append(e_memory)
            long_term_memories.append(long_term_memory)

        self.long_term_memories = long_term_memories

    def update_long_term_memory(self, new_long_term_memories, curr_mask=None):

        if self.flag_compress and self.tail - self.head >= self.threshold_memory:
            self.head = self.head + self.len_after_compress
            self.to_compress()

        if self.flag_reg_gen and not self.training and curr_mask is not None:
            curr_mask = curr_mask.int()
            n_object = curr_mask.max() + 1
            n_object = n_object.unsqueeze(0)
            att_map, bbox = self.ram(curr_mask, n_object)
            att_map = F.interpolate(att_map.float(), self.enc_size_2d, mode="nearest")
            att_map = att_map.view(1, 1, -1).permute(2, 1, 0)
            indices = torch.nonzero(att_map)
            indices = indices[:, 0]

        if self.long_term_memories is None:
            self.long_term_memories = new_long_term_memories
        updated_long_term_memories = []
        for new_long_term_memory, last_long_term_memory in zip(
                new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(new_long_term_memory,
                                     last_long_term_memory):
                if new_e is None or last_e is None:
                    updated_e.append(None)
                elif self.flag_reg_gen and not self.training and curr_mask is not None:
                    new_e = torch.index_select(new_e, 0, indices)
                    updated_e.append(torch.cat([new_e, last_e], dim=0))
                else:
                    updated_e.append(torch.cat([new_e, last_e], dim=0))
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories
        self.tail = self.long_term_memories[0][0].size()[0]
        self.att_maps = None

    def update_short_term_memory(self, curr_mask, curr_id_emb=None, skip_long_term_update=False):
        if curr_id_emb is None:
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                curr_one_hot_mask = one_hot_mask(curr_mask, self.max_obj_num)
            else:
                curr_one_hot_mask = curr_mask
            curr_id_emb = self.assign_identity(curr_one_hot_mask)

        lstt_curr_memories = self.curr_lstt_output[1]
        lstt_curr_memories_2d = []
        for layer_idx in range(len(lstt_curr_memories)):
            curr_k, curr_v = lstt_curr_memories[layer_idx][
                0], lstt_curr_memories[layer_idx][1]
            curr_k, curr_v = self.AOT.LSTT.layers[layer_idx].fuse_key_value_id(
                curr_k, curr_v, curr_id_emb)
            lstt_curr_memories[layer_idx][0], lstt_curr_memories[layer_idx][
                1] = curr_k, curr_v
            lstt_curr_memories_2d.append([
                seq_to_2d(lstt_curr_memories[layer_idx][0], self.enc_size_2d),
                seq_to_2d(lstt_curr_memories[layer_idx][1], self.enc_size_2d)
            ])

        self.short_term_memories_list.append(lstt_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[
            -self.short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            # skip the update of long-term memory or not
            if not skip_long_term_update: 
                self.update_long_term_memory(lstt_curr_memories, curr_mask)
            self.last_mem_step = self.frame_step

    def match_propogate_one_frame(self, img=None, img_embs=None):
        self.frame_step += 1
        if img_embs is None:
            curr_enc_embs, _ = self.encode_one_img_mask(
                img, None, self.frame_step)
        else:
            curr_enc_embs = img_embs
        self.curr_enc_embs = curr_enc_embs # [[1, 24, 121, 213], [1, 32, 61, 107], [1, 96, 31, 54], [1, 256, 31, 54]]

        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      self.long_term_memories,
                                                      self.short_term_memories,
                                                      None,
                                                      pos_emb=self.pos_emb,
                                                      size_2d=self.enc_size_2d)
        att_maps = []
        for att_map in self.curr_lstt_output[-1]:
            att_maps.append(torch.sum(att_map, dim=-2))
        if self.att_maps is None:
            self.att_maps = att_maps
        else:
            for i in range(len(self.att_maps)):
                self.att_maps[i] = self.att_maps[i] + att_maps[i]

        self.curr_lstt_output = self.curr_lstt_output[:-1]

    def decode_current_logits(self, output_size=None):
        curr_enc_embs = self.curr_enc_embs
        curr_lstt_embs = self.curr_lstt_output[0]

        pred_id_logits = self.AOT.decode_id_logits(curr_lstt_embs,
                                                   curr_enc_embs) # pred_id_logits [1, 11, 121, 213]

        if self.enable_id_shuffle:  # reverse shuffle
            pred_id_logits = torch.einsum('bohw,bto->bthw', pred_id_logits,
                                          self.id_shuffle_matrix)

        # remove unused identities
        for batch_idx, obj_num in enumerate(self.obj_nums):
            pred_id_logits[batch_idx, (obj_num+1):] = - \
                1e+10 if pred_id_logits.dtype == torch.float32 else -1e+4

        self.pred_id_logits = pred_id_logits

        if output_size is not None:
            pred_id_logits = F.interpolate(pred_id_logits,
                                           size=output_size,
                                           mode="bilinear",
                                           align_corners=self.align_corners) # pred_id_logits [1, 11, 480, 854]

        return pred_id_logits

    def predict_current_mask(self, output_size=None, return_prob=False):
        if output_size is None:
            output_size = self.input_size_2d

        pred_id_logits = F.interpolate(self.pred_id_logits,
                                       size=output_size,
                                       mode="bilinear",
                                       align_corners=self.align_corners)
        pred_mask = torch.argmax(pred_id_logits, dim=1)

        if not return_prob:
            return pred_mask
        else:
            pred_prob = torch.softmax(pred_id_logits, dim=1)
            return pred_mask, pred_prob

    def calculate_current_loss(self, gt_mask, step):
        pred_id_logits = self.pred_id_logits

        pred_id_logits = F.interpolate(pred_id_logits,
                                       size=gt_mask.size()[-2:],
                                       mode="bilinear",
                                       align_corners=self.align_corners)

        label_list = []
        logit_list = []
        for batch_idx, obj_num in enumerate(self.obj_nums):
            now_label = gt_mask[batch_idx].long()
            now_logit = pred_id_logits[batch_idx, :(obj_num + 1)].unsqueeze(0)
            label_list.append(now_label.long())
            logit_list.append(now_logit)

        total_loss = 0
        for loss, loss_weight in zip(self.losses, self.loss_weights):
            total_loss = total_loss + loss_weight * \
                loss(logit_list, label_list, step)

        return total_loss

    def generate_loss_mask(self, gt_mask, step, return_prob=False):
        self.decode_current_logits()
        loss = self.calculate_current_loss(gt_mask, step)
        if return_prob:
            mask, prob = self.predict_current_mask(return_prob=True)
            return loss, mask, prob
        else:
            mask = self.predict_current_mask()
            return loss, mask

    def keep_gt_mask(self, pred_mask, keep_prob=0.2):
        pred_mask = pred_mask.float()
        gt_mask = self.offline_masks[self.frame_step].float().squeeze(1)

        shape = [1 for _ in range(pred_mask.ndim)]
        shape[0] = self.batch_size
        random_tensor = keep_prob + torch.rand(
            shape, dtype=pred_mask.dtype, device=pred_mask.device)
        random_tensor.floor_()  # binarize

        pred_mask = pred_mask * (1 - random_tensor) + gt_mask * random_tensor

        return pred_mask

    def restart_engine(self, batch_size=1, enable_id_shuffle=False):

        self.batch_size = batch_size
        self.frame_step = 0
        self.last_mem_step = -1
        self.enable_id_shuffle = enable_id_shuffle
        self.freeze_id = False

        self.obj_nums = None
        self.pos_emb = None
        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None

        self.long_term_memories = None
        self.short_term_memories_list = []
        self.short_term_memories = None

        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0

        self.curr_enc_embs = None
        self.curr_memories = None
        self.curr_id_embs = None

        if enable_id_shuffle:
            self.id_shuffle_matrix = generate_permute_matrix(
                self.max_obj_num + 1, batch_size, gpu_id=self.gpu_id)
        else:
            self.id_shuffle_matrix = None

        self.att_maps = None
        self.head = 0
        self.tail = 0
        self.len_tobe_compress = 4096
        self.len_after_compress = 64
        self.threshold_memory = 10000
        self.flag_reg_gen = True
        self.flag_compress = True

    def update_size(self, input_size, enc_size):
        self.input_size_2d = input_size
        self.enc_size_2d = enc_size
        self.enc_hw = self.enc_size_2d[0] * self.enc_size_2d[1]


class AOTInferEngine(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=5,
                 short_term_mem_skip=1,
                 max_aot_obj_num=None):
        super().__init__()

        self.cfg = aot_model.cfg
        self.AOT = aot_model

        if max_aot_obj_num is None or max_aot_obj_num > aot_model.max_obj_num:
            self.max_aot_obj_num = aot_model.max_obj_num
        else:
            self.max_aot_obj_num = max_aot_obj_num

        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip

        self.aot_engines = []

        self.restart_engine()

    def restart_engine(self):
        del (self.aot_engines)
        self.aot_engines = []
        self.obj_nums = None

    def separate_mask(self, mask, obj_nums):
        if mask is None:
            return [None] * len(self.aot_engines)
        if len(self.aot_engines) == 1:
            return [mask], [obj_nums]

        separated_obj_nums = [
            self.max_aot_obj_num for _ in range(len(self.aot_engines))
        ]
        if obj_nums % self.max_aot_obj_num > 0:
            separated_obj_nums[-1] = obj_nums % self.max_aot_obj_num

        if len(mask.size()) == 3 or mask.size()[0] == 1:
            separated_masks = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_mask = ((mask >= start_id) & (mask <= end_id)).float()
                separated_mask = (fg_mask * mask - start_id + 1) * fg_mask
                separated_masks.append(separated_mask)
            return separated_masks, separated_obj_nums
        else:
            prob = mask
            separated_probs = []
            for idx in range(len(self.aot_engines)):
                start_id = idx * self.max_aot_obj_num + 1
                end_id = (idx + 1) * self.max_aot_obj_num
                fg_prob = prob[start_id:(end_id + 1)]
                bg_prob = 1. - torch.sum(fg_prob, dim=1, keepdim=True)
                separated_probs.append(torch.cat([bg_prob, fg_prob], dim=1))
            return separated_probs, separated_obj_nums

    def min_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_logits = []
        bg_logits = []

        for logit in all_logits:
            bg_logits.append(logit[:, 0:1])
            fg_logits.append(logit[:, 1:1 + self.max_aot_obj_num])

        bg_logit, _ = torch.min(torch.cat(bg_logits, dim=1),
                                dim=1,
                                keepdim=True)
        merged_logit = torch.cat([bg_logit] + fg_logits, dim=1)

        return merged_logit

    def soft_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_probs = []
        bg_probs = []

        for logit in all_logits:
            prob = torch.softmax(logit, dim=1)
            bg_probs.append(prob[:, 0:1])
            fg_probs.append(prob[:, 1:1 + self.max_aot_obj_num])

        bg_prob = torch.prod(torch.cat(bg_probs, dim=1), dim=1, keepdim=True)
        merged_prob = torch.cat([bg_prob] + fg_probs,
                                dim=1).clamp(1e-5, 1 - 1e-5)
        merged_logit = torch.logit(merged_prob)

        return merged_logit

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = AOTEngine(self.AOT, self.gpu_id,
                                   self.long_term_mem_gap,
                                   self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(
            mask, obj_nums)
        img_embs = None
        for aot_engine, separated_mask, separated_obj_num in zip(
                self.aot_engines, separated_masks, separated_obj_nums):
            aot_engine.add_reference_frame(img,
                                           separated_mask,
                                           obj_nums=[separated_obj_num],
                                           frame_step=frame_step,
                                           img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()

    def match_propogate_one_frame(self, img=None):
        img_embs = None
        for aot_engine in self.aot_engines:
            aot_engine.match_propogate_one_frame(img, img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

    def decode_current_logits(self, output_size=None):
        all_logits = []
        for aot_engine in self.aot_engines:
            all_logits.append(aot_engine.decode_current_logits(output_size))
        pred_id_logits = self.soft_logit_aggregation(all_logits)
        return pred_id_logits

    def update_memory(self, curr_mask, skip_long_term_update=False):
        separated_masks, _ = self.separate_mask(curr_mask, self.obj_nums)
        for aot_engine, separated_mask in zip(self.aot_engines,
                                              separated_masks):
            aot_engine.update_short_term_memory(separated_mask, 
                                                skip_long_term_update=skip_long_term_update)

    def update_size(self):
        self.input_size_2d = self.aot_engines[0].input_size_2d
        self.enc_size_2d = self.aot_engines[0].enc_size_2d
        self.enc_hw = self.aot_engines[0].enc_hw
