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
import os

import torch

_EXT = None
_CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")


def _ext():
    """The compiled extension: the prebuilt slicegaussian._C if the package was installed with
    SLICEGAUSSIAN_BUILD_CUDA=1, otherwise compiled from csrc/ on first use (cached by PyTorch)."""
    global _EXT
    if _EXT is None:
        try:
            from .. import _C as ext
        except ImportError:
            from torch.utils.cpp_extension import load
            ext = load(name="slicegaussian_fieldcut",
                       sources=[os.path.join(_CSRC, "fieldcut.cpp"), os.path.join(_CSRC, "fieldcut_kernel.cu")],
                       extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
        _EXT = ext
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
