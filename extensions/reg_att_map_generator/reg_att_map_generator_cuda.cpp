#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
reg_att_map_generator_cuda_forward(torch::Tensor mask, torch::Tensor n_objects,
                                   int n_pts_threshold, int n_bbox_loose_pixels,
                                   cudaStream_t stream);

std::vector<torch::Tensor>
reg_att_map_generator_forward(torch::Tensor mask, torch::Tensor n_objects,
                              int n_pts_threshold, int n_bbox_loose_pixels) {
  CHECK_INPUT(mask);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return reg_att_map_generator_cuda_forward(
      mask, n_objects, n_pts_threshold, n_bbox_loose_pixels, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &reg_att_map_generator_forward,
        "Regional Attention Map Generator forward (CUDA)");
}