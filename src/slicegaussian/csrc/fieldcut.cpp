// Python bindings of the fused field-cut kernels (fieldcut_kernel.cu).
#include <torch/extension.h>
#include <vector>

torch::Tensor field_forward(torch::Tensor xyz, torch::Tensor cov6, torch::Tensor amp, torch::Tensor zs,
                            int64_t H, int64_t W, double dx, double dy, double radius, double cull_eps);
std::vector<torch::Tensor> field_backward(torch::Tensor grad_image, torch::Tensor xyz, torch::Tensor cov6,
                                          torch::Tensor amp, torch::Tensor zs, double dx, double dy, double radius,
                                          double cull_eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "SliceGaussian: evaluation of the additive Gaussian field on planes z = z_k (forward and backward)";
  m.def("field_forward", &field_forward, "Field values on the planes zs, (K, H, W)");
  m.def("field_backward", &field_backward, "Gradients w.r.t. centres, covariances (6) and amplitudes");
}
