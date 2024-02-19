import torch

import reg_att_map_generator

def var_or_cuda(x, device=None):
    x = x.contiguous()
    if torch.cuda.is_available() and device != torch.device('cpu'):
        if device is None:
            x = x.cuda(non_blocking=True)
        else:
            x = x.cuda(device=device, non_blocking=True)
    return x

class RegionalAttentionMapGeneratorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, n_objects, n_pts_threshold, n_bbox_loose_pixels):
        att_map, bbox = reg_att_map_generator.forward(mask, n_objects, n_pts_threshold,
                                                      n_bbox_loose_pixels)
        return att_map, bbox

    @staticmethod
    def backward(ctx, grad_att_map, grad_bbox):
        grad_mask = var_or_cuda(torch.ones(grad_att_map.size()).float())
        return grad_mask, None, None, None


class RegionalAttentionMapGenerator(torch.nn.Module):
    def __init__(self):
        super(RegionalAttentionMapGenerator, self).__init__()

    def forward(self, mask, n_objects, n_pts_threshold=10, n_bbox_loose_pixels=32):
        return RegionalAttentionMapGeneratorFunction.apply(mask, n_objects, n_pts_threshold,
                                                           n_bbox_loose_pixels)
