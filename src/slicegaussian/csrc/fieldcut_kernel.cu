// Fused CUDA evaluator of the additive Gaussian field on planes z = z_k: forward and analytic backward, one kernel
// each. See slicegaussian/cuda/fieldcut.py for the mathematics.
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


// ---------------------------------------------------------------------------------------------------------------------
// Raw-parameter path: the covariance is built inside the kernels from the quaternion and log-scales of each Gaussian,
// and the backward pass returns gradients with respect to those raw parameters. Only the Gaussians listed in idx (those
// that reach a plane) are visited, so the per-iteration cost no longer grows with the size of the population.
// ---------------------------------------------------------------------------------------------------------------------
struct Frame { float R[3][3]; float s[3]; float qn[4]; float qnorm; };

__device__ __forceinline__ Frame make_frame(const float* q, const float* ls) {
  Frame f;
  float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  n = fmaxf(n, 1e-12f);                                                    // F.normalize
  f.qnorm = n;
  const float w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
  f.qn[0] = w; f.qn[1] = x; f.qn[2] = y; f.qn[3] = z;
  f.R[0][0] = 1.f - 2.f * (y * y + z * z); f.R[0][1] = 2.f * (x * y - w * z); f.R[0][2] = 2.f * (x * z + w * y);
  f.R[1][0] = 2.f * (x * y + w * z); f.R[1][1] = 1.f - 2.f * (x * x + z * z); f.R[1][2] = 2.f * (y * z - w * x);
  f.R[2][0] = 2.f * (x * z - w * y); f.R[2][1] = 2.f * (y * z + w * x); f.R[2][2] = 1.f - 2.f * (x * x + y * y);
  for (int k = 0; k < 3; ++k) f.s[k] = __expf(ls[k]);
  return f;
}

// cov6 = xx, xy, yy, xz, yz, zz of R diag(s^2) R^T
__device__ __forceinline__ void frame_cov(const Frame& f, float* c) {
  const int a[6] = {0, 0, 1, 0, 1, 2}, b[6] = {0, 1, 1, 2, 2, 2};
  for (int e = 0; e < 6; ++e) {
    float v = 0.f;
    for (int k = 0; k < 3; ++k) v += f.R[a[e]][k] * f.R[b[e]][k] * f.s[k] * f.s[k];
    c[e] = v;
  }
}

__global__ void raw_cull(const int N, const int K, const float* __restrict__ xyz, const float* __restrict__ quat,
                         const float* __restrict__ ls, const float* __restrict__ zs, const float r,
                         bool* __restrict__ mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const Frame f = make_frame(quat + 4 * i, ls + 3 * i);
  float czz = 0.f;
  for (int k = 0; k < 3; ++k) czz += f.R[2][k] * f.R[2][k] * f.s[k] * f.s[k];
  const float reach = r * sqrtf(czz), mz = xyz[3 * i + 2];
  bool on = false;
  for (int k = 0; k < K; ++k) on |= fabsf(zs[k] - mz) <= reach;
  mask[i] = on;
}

__global__ void raw_fwd(const int M, const int K, const long* __restrict__ idx, const float* __restrict__ xyz,
                        const float* __restrict__ quat, const float* __restrict__ ls, const float* __restrict__ amp,
                        const float* __restrict__ zs, const int H, const int W, const float dx, const float dy,
                        const float r, const float eps, float* __restrict__ image) {
  const long pair = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (pair >= static_cast<long>(M) * K) return;
  const long i = idx[pair / K];
  const int k = pair % K;
  float cov[6];
  frame_cov(make_frame(quat + 4 * i, ls + 3 * i), cov);
  const Cut t = make_cut(xyz + 3 * i, cov, amp[i], zs[k], H, W, dx, dy, r, eps);
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

__global__ void raw_bwd(const int M, const int K, const long* __restrict__ idx, const float* __restrict__ gimg_all,
                        const float* __restrict__ xyz, const float* __restrict__ quat, const float* __restrict__ ls,
                        const float* __restrict__ amp, const float* __restrict__ zs, const int H, const int W,
                        const float dx, const float dy, const float r, const float eps, float* __restrict__ g_xyz,
                        float* __restrict__ g_quat, float* __restrict__ g_ls, float* __restrict__ g_amp) {
  const long pair = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (pair >= static_cast<long>(M) * K) return;
  const long i = idx[pair / K];
  const int k = pair % K;
  const Frame f = make_frame(quat + 4 * i, ls + 3 * i);
  float cov[6];
  frame_cov(f, cov);
  const Cut t = make_cut(xyz + 3 * i, cov, amp[i], zs[k], H, W, dx, dy, r, eps);
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
  // gradients w.r.t. the six covariance entries and the centre: identical to bwd()
  const float h = 0.5f * gb;
  const float A11 = ga * t.qa + h * t.qb, A12 = ga * t.qb + h * t.qc;
  const float A21 = h * t.qa + gc * t.qb, A22 = h * t.qb + gc * t.qc;
  const float gP11 = -(t.qa * A11 + t.qb * A21);
  const float gP12 = -(t.qa * A12 + t.qb * A22);
  const float gP22 = -(t.qb * A12 + t.qc * A22);
  const float gCxx = gP11 / (dx * dx), gCxy = 2.f * gP12 / (dx * dy), gCyy = gP22 / (dy * dy);
  const float c = t.c, bx = t.bx, by = t.by, dz = t.dz, c2 = c * c;
  float gbx = -2.f * gCxx * bx / c - gCxy * by / c;
  float gby = -2.f * gCyy * by / c - gCxy * bx / c;
  float gcc = (gCxx * bx * bx + gCxy * bx * by + gCyy * by * by) / c2;
  const float hx = gmx / dx, hy = gmy / dy;
  gbx += hx * dz / c; gby += hy * dz / c;
  float gmuz = -(hx * bx + hy * by) / c;
  gcc += -(hx * bx + hy * by) * dz / c2;
  const float gwE = gw * t.w;
  gmuz += gwE * dz / c;
  gcc += gwE * dz * dz / (2.f * c2);
  // covariance entries -> M = R diag(s):  dL/dM = 2 G M, G symmetric with half the off-diagonal entry gradients
  float G[3][3];
  G[0][0] = gCxx; G[1][1] = gCyy; G[2][2] = gcc;
  G[0][1] = G[1][0] = 0.5f * gCxy; G[0][2] = G[2][0] = 0.5f * gbx; G[1][2] = G[2][1] = 0.5f * gby;
  float A[3][3], gs[3] = {0.f, 0.f, 0.f};
  for (int a = 0; a < 3; ++a)
    for (int q = 0; q < 3; ++q) {
      float gM = 0.f;
      for (int b = 0; b < 3; ++b) gM += 2.f * G[a][b] * f.R[b][q] * f.s[q];
      A[a][q] = gM * f.s[q];                                               // dL/dR
      gs[q] += gM * f.R[a][q];                                             // dL/ds
    }
  const float w = f.qn[0], x = f.qn[1], y = f.qn[2], z = f.qn[3];
  float gq[4];
  gq[0] = 2.f * (-z * A[0][1] + y * A[0][2] + z * A[1][0] - x * A[1][2] - y * A[2][0] + x * A[2][1]);
  gq[1] = 2.f * (y * A[0][1] + z * A[0][2] + y * A[1][0] - 2.f * x * A[1][1] - w * A[1][2] + z * A[2][0] + w * A[2][1]
                 - 2.f * x * A[2][2]);
  gq[2] = 2.f * (-2.f * y * A[0][0] + x * A[0][1] + w * A[0][2] + x * A[1][0] + z * A[1][2] - w * A[2][0] + z * A[2][1]
                 - 2.f * y * A[2][2]);
  gq[3] = 2.f * (-2.f * z * A[0][0] - w * A[0][1] + x * A[0][2] + w * A[1][0] - 2.f * z * A[1][1] + y * A[1][2]
                 + x * A[2][0] + y * A[2][1]);
  const float dot = gq[0] * w + gq[1] * x + gq[2] * y + gq[3] * z;         // through the normalisation
  for (int e = 0; e < 4; ++e) atomicAdd(g_quat + 4 * i + e, (gq[e] - f.qn[e] * dot) / f.qnorm);
  for (int q = 0; q < 3; ++q) atomicAdd(g_ls + 3 * i + q, gs[q] * f.s[q]);
  atomicAdd(g_amp + i, gw * t.E);
  atomicAdd(g_xyz + 3 * i, hx); atomicAdd(g_xyz + 3 * i + 1, hy); atomicAdd(g_xyz + 3 * i + 2, gmuz);
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

torch::Tensor raw_mask(torch::Tensor xyz, torch::Tensor quat, torch::Tensor ls, torch::Tensor zs, double radius) {
  const int N = xyz.size(0), K = zs.size(0);
  auto mask = torch::zeros({N}, xyz.options().dtype(torch::kBool));
  if (N == 0 || K == 0) return mask;
  raw_cull<<<(N + kThreads - 1) / kThreads, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      N, K, xyz.data_ptr<float>(), quat.data_ptr<float>(), ls.data_ptr<float>(), zs.data_ptr<float>(), radius,
      mask.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return mask;
}

torch::Tensor raw_forward(torch::Tensor idx, torch::Tensor xyz, torch::Tensor quat, torch::Tensor ls, torch::Tensor amp,
                          torch::Tensor zs, int64_t H, int64_t W, double dx, double dy, double radius,
                          double cull_eps) {
  const int M = idx.size(0), K = zs.size(0);
  auto image = torch::zeros({K, H, W}, xyz.options());
  if (M == 0 || K == 0) return image;
  raw_fwd<<<blocks_for(static_cast<long>(M) * K), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, K, idx.data_ptr<long>(), xyz.data_ptr<float>(), quat.data_ptr<float>(), ls.data_ptr<float>(),
      amp.data_ptr<float>(), zs.data_ptr<float>(), H, W, dx, dy, radius, cull_eps, image.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return image;
}

std::vector<torch::Tensor> raw_backward(torch::Tensor grad_image, torch::Tensor idx, torch::Tensor xyz,
                                        torch::Tensor quat, torch::Tensor ls, torch::Tensor amp, torch::Tensor zs,
                                        double dx, double dy, double radius, double cull_eps) {
  const int M = idx.size(0), K = zs.size(0);
  auto gx = torch::zeros_like(xyz), gq = torch::zeros_like(quat), gl = torch::zeros_like(ls), ga = torch::zeros_like(amp);
  if (M == 0 || K == 0) return {gx, gq, gl, ga};
  raw_bwd<<<blocks_for(static_cast<long>(M) * K), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, K, idx.data_ptr<long>(), grad_image.data_ptr<float>(), xyz.data_ptr<float>(), quat.data_ptr<float>(),
      ls.data_ptr<float>(), amp.data_ptr<float>(), zs.data_ptr<float>(), grad_image.size(1), grad_image.size(2), dx,
      dy, radius, cull_eps, gx.data_ptr<float>(), gq.data_ptr<float>(), gl.data_ptr<float>(), ga.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gx, gq, gl, ga};
}
