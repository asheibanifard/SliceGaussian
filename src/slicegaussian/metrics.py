"""Metrics on the reconstructed scalar field, evaluated at every voxel centre (V_max = 1 after normalization).

PSNR = 10 log10(1 / MSE); 3D SSIM (Gaussian window 11, sigma 1.5 voxels, C1 = 0.01^2, C2 = 0.03^2); MAE;
gradient MAE = mean ||grad V - grad V_hat||_2 (forward differences, physical units x smallest spacing);
band error_s = mean |D_s V - D_s V_hat|, D_s = G_s - G_{s+1}, sigma_s = 1, 2, 4, 8 (x smallest spacing).
The prediction is clipped to [0, 1]. Scores are on the valid planes ("_valid", the primary score) and on all planes.
"""
import torch
import torch.nn.functional as F

from .field import render_volume
from .volume import blur3d


def ssim3d(a, b, data_range=1.0):
    """SSIM of two (Z, H, W) volumes with a 3D Gaussian window 11 / 1.5, as separable 1D convolutions in double
    precision (in single precision the cascade loses about 6e-4 of SSIM to rounding in the local variances)."""
    a, b = a.double(), b.double()
    x = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-0.5 * (x / 1.5) ** 2)
    g = g / g.sum()

    def blur(v):
        for shape, pad in (((11, 1, 1), (5, 0, 0)), ((1, 11, 1), (0, 5, 0)), ((1, 1, 11), (0, 0, 5))):
            v = F.conv3d(v, g.view(1, 1, *shape), padding=pad)
        return v

    a, b = a[None, None], b[None, None]
    mu_a, mu_b = blur(a), blur(b)
    saa, sbb, sab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return float((((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))).mean())


def grad_mae(a, b, vol):
    dz, dy, dx = vol.spacing()
    u = min(dz, dy, dx)
    r = a - b
    gx = (r[:-1, :-1, 1:] - r[:-1, :-1, :-1]) * u / dx
    gy = (r[:-1, 1:, :-1] - r[:-1, :-1, :-1]) * u / dy
    gz = (r[1:, :-1, :-1] - r[:-1, :-1, :-1]) * u / dz
    return float((gx ** 2 + gy ** 2 + gz ** 2).sqrt().mean())


def band_errors(a, b, vol, sigmas=(1, 2, 4, 8)):
    dz, dy, dx = vol.spacing()
    u = min(dz, dy, dx)
    blur = lambda v, s: blur3d(v[None], (s * u / dz, s * u / dy, s * u / dx))[0]
    out, pa, pb = {}, a, b
    for s in sigmas:
        na, nb = blur(a, s), blur(b, s)
        out[f"band{s}"] = float(((pa - na) - (pb - nb)).abs().mean())
        pa, pb = na, nb
    return out


def scores(pred, gt, vol, prefix="", full=True):
    pred = pred.clamp(0, 1)
    mse = float(((pred - gt) ** 2).mean())
    d = {f"{prefix}psnr": float(10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-20)))),
         f"{prefix}mae": float((pred - gt).abs().mean())}
    if full:
        d[f"{prefix}ssim"] = ssim3d(pred, gt)
        d[f"{prefix}grad_mae"] = grad_mae(gt, pred, vol)
        d.update({f"{prefix}{k}": v for k, v in band_errors(gt, pred, vol).items()})
    return d


@torch.no_grad()
def whole(field, vol, full=True, pred=None):
    pred = render_volume(field, vol) if pred is None else pred
    d = scores(pred[vol.valid_ids], vol.gt[vol.valid_ids], vol, "", full)
    if vol.invalid_ids:
        d.update(scores(pred, vol.gt, vol, "all_", full=False))
    return d
