"""Fused CUDA evaluator of the additive Gaussian field on planes z = z_k (forward and backward in one kernel each).

For a pair (Gaussian i, plane k) with c = Sigma_zz, b = (Sigma_xz, Sigma_yz), dz = z_k - mu_z:
    w      = alpha exp(-dz^2 / (2c))                                   (the 3D Gaussian on the plane)
    m      = (mu_xy + b dz / c) / (dx, dy) - 1/2                       (pixel coordinates; pixel j is centred at j)
    C      = Sigma_xy - b b^T / c   ->   P = C / (dx dx, dx dy, dy dy)  (pixels),  Q = P^-1
    image[k, y, x] += w exp(-1/2 d^T Q d),  d = (x, y) - m,  if d^T Q d <= r^2 and |dz| <= r sqrt(c), |w| >= cull_eps
This evaluates the same operator as field.reference_volume (validated in tests/test_math.py);
here the cut runs inside the kernel instead of as PyTorch tensor operations.

Backward (exact chain rule, per pair; no PyTorch graph over pairs):
    pixel sums (warp reduction): g_w, g_m, g_Q (a, b, c of the conic, b counted once)
    g_P = -Q G_Q Q with G_Q = [[g_a, g_b/2], [g_b/2, g_c]];  g_C = g_P / pixel areas
    C = Sigma_xy - b b^T / c;  m = mu_xy + b dz / c;  w = alpha exp(-dz^2 / 2c)  ->  g_mu, g_Sigma (6), g_alpha
accumulated per Gaussian with atomicAdd (K <= a few dozen planes, low contention). The step Sigma <- (quaternion, scale)
stays in PyTorch autograd (per Gaussian, cheap).
"""
import torch
from torch.utils.cpp_extension import load_inline

_CPP = r"""
#include <torch/extension.h>
#include <vector>
torch::Tensor field_forward(torch::Tensor xyz, torch::Tensor cov6, torch::Tensor amp, torch::Tensor zs,
                            int64_t H, int64_t W, double dx, double dy, double radius, double cull_eps);
std::vector<torch::Tensor> field_backward(torch::Tensor grad_image, torch::Tensor xyz, torch::Tensor cov6,
                                          torch::Tensor amp, torch::Tensor zs, double dx, double dy, double radius,
                                          double cull_eps);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
constexpr int kWarp = 32;
constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

struct Cut {
  bool on;
  float w, E, mx, my, qa, qb, qc, Pxx, Pxy, Pyy, c, bx, by, dz;
  int x0, y0, bw, bh;
};

// cov6 = xx, xy, yy, xz, yz, zz
__device__ __forceinline__ Cut make_cut(const float* xyz, const float* cov, float alpha, float z, int H, int W,
                                        float dx, float dy, float r, float eps) {
  Cut t;
  t.on = false;
  t.c = cov[5]; t.bx = cov[3]; t.by = cov[4];
  t.dz = z - xyz[2];
  if (!(t.c > 0.f) || fabsf(t.dz) > r * sqrtf(t.c)) return t;
  t.E = __expf(-0.5f * t.dz * t.dz / t.c);
  t.w = alpha * t.E;
  if (fabsf(t.w) < eps) return t;
  const float Cxx = cov[0] - t.bx * t.bx / t.c, Cxy = cov[1] - t.bx * t.by / t.c, Cyy = cov[2] - t.by * t.by / t.c;
  t.Pxx = Cxx / (dx * dx); t.Pxy = Cxy / (dx * dy); t.Pyy = Cyy / (dy * dy);
  float det = t.Pxx * t.Pyy - t.Pxy * t.Pxy;
  det = fmaxf(det, 1e-12f);
  t.qa = t.Pyy / det; t.qb = -t.Pxy / det; t.qc = t.Pxx / det;
  t.mx = (xyz[0] + t.bx * t.dz / t.c) / dx - 0.5f;
  t.my = (xyz[1] + t.by * t.dz / t.c) / dy - 0.5f;
  const float rx = r * sqrtf(fmaxf(t.Pxx, 0.f)), ry = r * sqrtf(fmaxf(t.Pyy, 0.f));
  t.x0 = max(0, (int)ceilf(t.mx - rx)); t.y0 = max(0, (int)ceilf(t.my - ry));
  const int x1 = min(W, (int)floorf(t.mx + rx) + 1), y1 = min(H, (int)floorf(t.my + ry) + 1);
  t.bw = x1 - t.x0; t.bh = y1 - t.y0;
  t.on = t.bw > 0 && t.bh > 0;
  return t;
}

__global__ void fwd(const int N, const int K, const float* __restrict__ xyz, const float* __restrict__ cov6,
                    const float* __restrict__ amp, const float* __restrict__ zs, const int H, const int W,
                    const float dx, const float dy, const float r, const float eps, float* __restrict__ image) {
  const long pair = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (pair >= static_cast<long>(N) * K) return;
  const int i = pair / K, k = pair % K;
  const Cut t = make_cut(xyz + 3 * i, cov6 + 6 * i, amp[i], zs[k], H, W, dx, dy, r, eps);
  if (!t.on) return;
  const float cut2 = r * r;
  float* img = image + static_cast<long>(k) * H * W;
  for (int p = lane; p < t.bw * t.bh; p += kWarp) {
    const int x = t.x0 + p % t.bw, y = t.y0 + p / t.bw;
    const float ex = x - t.mx, ey = y - t.my;
    const float q = t.qa * ex * ex + 2.f * t.qb * ex * ey + t.qc * ey * ey;
    if (q <= cut2) atomicAdd(img + static_cast<long>(y) * W + x, t.w * __expf(-0.5f * q));
  }
}

__global__ void bwd(const int N, const int K, const float* __restrict__ gimg_all, const float* __restrict__ xyz,
                    const float* __restrict__ cov6, const float* __restrict__ amp, const float* __restrict__ zs,
                    const int H, const int W, const float dx, const float dy, const float r, const float eps,
                    float* __restrict__ g_xyz, float* __restrict__ g_cov, float* __restrict__ g_amp) {
  const long pair = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (pair >= static_cast<long>(N) * K) return;
  const int i = pair / K, k = pair % K;
  const Cut t = make_cut(xyz + 3 * i, cov6 + 6 * i, amp[i], zs[k], H, W, dx, dy, r, eps);
  if (!t.on) return;
  const float cut2 = r * r;
  const float* gimg = gimg_all + static_cast<long>(k) * H * W;
  float gw = 0.f, gmx = 0.f, gmy = 0.f, ga = 0.f, gb = 0.f, gc = 0.f;
  for (int p = lane; p < t.bw * t.bh; p += kWarp) {
    const int x = t.x0 + p % t.bw, y = t.y0 + p / t.bw;
    const float ex = x - t.mx, ey = y - t.my;
    const float q = t.qa * ex * ex + 2.f * t.qb * ex * ey + t.qc * ey * ey;
    if (q > cut2) continue;
    const float gG = gimg[static_cast<long>(y) * W + x] * __expf(-0.5f * q);
    const float v = t.w * gG;
    gw += gG;
    gmx += v * (t.qa * ex + t.qb * ey);
    gmy += v * (t.qb * ex + t.qc * ey);
    ga -= 0.5f * v * ex * ex;
    gb -= v * ex * ey;
    gc -= 0.5f * v * ey * ey;
  }
  gw = warp_sum(gw); gmx = warp_sum(gmx); gmy = warp_sum(gmy);
  ga = warp_sum(ga); gb = warp_sum(gb); gc = warp_sum(gc);
  if (lane != 0) return;
  // conic Q = P^-1:  G_P = -Q G_Q Q,  G_Q = [[ga, gb/2], [gb/2, gc]]
  const float h = 0.5f * gb;
  const float A11 = ga * t.qa + h * t.qb, A12 = ga * t.qb + h * t.qc;   // G_Q Q
  const float A21 = h * t.qa + gc * t.qb, A22 = h * t.qb + gc * t.qc;
  const float gP11 = -(t.qa * A11 + t.qb * A21);
  const float gP12 = -(t.qa * A12 + t.qb * A22);
  const float gP22 = -(t.qb * A12 + t.qc * A22);
  // pixel -> physical; the off-diagonal P_xy is one parameter occupying both entries
  const float gCxx = gP11 / (dx * dx), gCxy = 2.f * gP12 / (dx * dy), gCyy = gP22 / (dy * dy);
  const float c = t.c, bx = t.bx, by = t.by, dz = t.dz, c2 = c * c;
  float gSxx = gCxx, gSxy = gCxy, gSyy = gCyy;
  float gbx = -2.f * gCxx * bx / c - gCxy * by / c;
  float gby = -2.f * gCyy * by / c - gCxy * bx / c;
  float gcc = (gCxx * bx * bx + gCxy * bx * by + gCyy * by * by) / c2;
  // mean m = (mu_xy + b dz / c) / (dx, dy)
  const float hx = gmx / dx, hy = gmy / dy;
  float gmux = hx, gmuy = hy;
  gbx += hx * dz / c; gby += hy * dz / c;
  float gmuz = -(hx * bx + hy * by) / c;
  gcc += -(hx * bx + hy * by) * dz / c2;
  // weight w = alpha exp(-dz^2 / 2c)
  const float gwE = gw * t.w;
  gmuz += gwE * dz / c;
  gcc += gwE * dz * dz / (2.f * c2);
  atomicAdd(g_amp + i, gw * t.E);
  atomicAdd(g_xyz + 3 * i, gmux); atomicAdd(g_xyz + 3 * i + 1, gmuy); atomicAdd(g_xyz + 3 * i + 2, gmuz);
  atomicAdd(g_cov + 6 * i, gSxx); atomicAdd(g_cov + 6 * i + 1, gSxy); atomicAdd(g_cov + 6 * i + 2, gSyy);
  atomicAdd(g_cov + 6 * i + 3, gbx); atomicAdd(g_cov + 6 * i + 4, gby); atomicAdd(g_cov + 6 * i + 5, gcc);
}

int blocks_for(long pairs) { return static_cast<int>((pairs * kWarp + kThreads - 1) / kThreads); }
}  // namespace

torch::Tensor field_forward(torch::Tensor xyz, torch::Tensor cov6, torch::Tensor amp, torch::Tensor zs,
                            int64_t H, int64_t W, double dx, double dy, double radius, double cull_eps) {
  const int N = xyz.size(0), K = zs.size(0);
  auto image = torch::zeros({K, H, W}, xyz.options());
  if (N == 0 || K == 0) return image;
  fwd<<<blocks_for(static_cast<long>(N) * K), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      N, K, xyz.data_ptr<float>(), cov6.data_ptr<float>(), amp.data_ptr<float>(), zs.data_ptr<float>(), H, W, dx, dy,
      radius, cull_eps, image.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return image;
}

std::vector<torch::Tensor> field_backward(torch::Tensor grad_image, torch::Tensor xyz, torch::Tensor cov6,
                                          torch::Tensor amp, torch::Tensor zs, double dx, double dy, double radius,
                                          double cull_eps) {
  const int N = xyz.size(0), K = zs.size(0);
  auto gx = torch::zeros_like(xyz), gcov = torch::zeros_like(cov6), ga = torch::zeros_like(amp);
  if (N == 0 || K == 0) return {gx, gcov, ga};
  bwd<<<blocks_for(static_cast<long>(N) * K), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      N, K, grad_image.data_ptr<float>(), xyz.data_ptr<float>(), cov6.data_ptr<float>(), amp.data_ptr<float>(),
      zs.data_ptr<float>(), grad_image.size(1), grad_image.size(2), dx, dy, radius, cull_eps, gx.data_ptr<float>(),
      gcov.data_ptr<float>(), ga.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gx, gcov, ga};
}
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(name="slicegaussian_fieldcut", cpp_sources=_CPP, cuda_sources=_CUDA,
                           functions=["field_forward", "field_backward"], extra_cflags=["-O3"],
                           extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    return _EXT


class FieldCut(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, cov6, amp, zs, H, W, dx, dy, radius, cull_eps):
        args = (xyz.float().contiguous(), cov6.float().contiguous(), amp.float().contiguous(), zs.float().contiguous())
        ctx.save_for_backward(*args)
        ctx.cfg = (dx, dy, radius, cull_eps)
        return _ext().field_forward(*args, H, W, dx, dy, radius, cull_eps)

    @staticmethod
    def backward(ctx, grad):
        xyz, cov6, amp, zs = ctx.saved_tensors
        gx, gc, ga = _ext().field_backward(grad.float().contiguous(), xyz, cov6, amp, zs, *ctx.cfg)
        return gx, gc, ga, None, None, None, None, None, None, None


def cov6(R, s):
    """(xx, xy, yy, xz, yz, zz) of Sigma = R diag(s^2) R^T, (N, 6)."""
    M = R * s[:, None, :]
    S = lambda a, b: (M[:, a] * M[:, b]).sum(1)
    return torch.stack([S(0, 0), S(0, 1), S(1, 1), S(0, 2), S(1, 2), S(2, 2)], 1)


def render(xyz, R, s, amp, zs, H, W, dx, dy, radius=3.0, cull_eps=1e-5):
    z = torch.as_tensor(zs, device=xyz.device, dtype=torch.float32).reshape(-1)
    return FieldCut.apply(xyz, cov6(R, s), amp, z, H, W, float(dx), float(dy), float(radius), float(cull_eps))
