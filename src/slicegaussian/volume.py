"""Slice-based scalar volumes (methodology Stage 1: physical-space preprocessing).

A volume is a (Z, Y, X) stack of slices with physical spacing (dz, dy, dx); pixels must be square (dx == dy).
Intensities are min-max normalised to [0, 1]. The voxel (k, i, j) is centred at ((j + 1/2) dx, (i + 1/2) dy, (k + 1/2) dz).

Invalid slices (a slice holding a single value, e.g. the black slices of CREMI C) are never used for training or scoring
on valid slices; for the Phase I analysis they are filled by linear interpolation between their valid neighbours, so
that they produce no artificial scale-space response.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F


def read_stack(path: str) -> np.ndarray:
    """Reads a (Z, Y, X) stack from .tif/.tiff, .npy or .h5/.hdf (dataset "volumes/raw", the CREMI layout)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        raw = np.load(path)
    elif ext in (".h5", ".hdf", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            raw = f["volumes/raw"][()]
    else:
        import tifffile
        raw = tifffile.imread(path)
    if raw.ndim != 3:
        raise ValueError(f"expected a (Z, Y, X) stack, got shape {raw.shape}")
    return raw


class Volume:
    """A normalised slice stack on the GPU with its spacing and valid-slice mask."""

    def __init__(self, src, spacing, name: str = "", detect_invalid: bool = True, device: str = "cuda"):
        raw = src if isinstance(src, np.ndarray) else read_stack(src)
        vol = raw.astype(np.float32)
        self.lo, self.hi = float(vol.min()), float(vol.max())
        vol = (vol - self.lo) / max(self.hi - self.lo, 1e-12)
        self.Z, self.H, self.W = vol.shape
        self.dz, self.dy, self.dx = (float(s) for s in spacing)
        if abs(self.dx - self.dy) > 1e-9 * self.dx:
            raise ValueError("pixels must be square (dx == dy)")
        self.device = device
        self.volume = vol
        self.gt = torch.from_numpy(vol).to(device)                              # (Z, H, W) in [0, 1]
        self.extent = 1.1 * 0.5 * math.sqrt((self.W * self.dx) ** 2 + (self.H * self.dy) ** 2 + (self.Z * self.dz) ** 2)
        self.invalid_ids = [k for k in range(self.Z) if detect_invalid and float(vol[k].max() - vol[k].min()) == 0.0]
        self.valid_ids = [k for k in range(self.Z) if k not in self.invalid_ids]
        if not self.valid_ids:
            raise ValueError("the volume has no valid slice")
        filled = vol.copy()
        va = np.asarray(self.valid_ids)
        for k in self.invalid_ids:
            hi = np.searchsorted(va, k)
            a, b = va[max(hi - 1, 0)], va[min(hi, len(va) - 1)]
            t = 0.0 if a == b else (k - a) / (b - a)
            filled[k] = (1 - t) * vol[a] + t * vol[b]
        self.analysis_volume = filled
        self.valid_z = torch.ones(self.Z, dtype=torch.bool, device=device)
        self.valid_z[self.invalid_ids] = False
        self.name = name or (os.path.splitext(os.path.basename(src))[0] if isinstance(src, str) else "volume")

    @property
    def shape(self):
        return (self.Z, self.H, self.W)

    def spacing(self):
        return (self.dz, self.dy, self.dx)

    def z(self, k):
        return (k + 0.5) * self.dz

    def centres(self, k_ids=None):
        """Physical coordinates (x, y, z) of all voxel centres of the planes k_ids, (K, H, W, 3)."""
        dev = self.device
        ks = torch.arange(self.Z, device=dev) if k_ids is None else torch.as_tensor(k_ids, device=dev)
        z = (ks.float() + 0.5) * self.dz
        y = (torch.arange(self.H, device=dev).float() + 0.5) * self.dy
        x = (torch.arange(self.W, device=dev).float() + 0.5) * self.dx
        Zg, Yg, Xg = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([Xg, Yg, Zg], -1)

    def describe(self):
        return {"name": self.name, "shape": list(self.shape), "spacing_zyx": [self.dz, self.dy, self.dx],
                "invalid_slices": self.invalid_ids, "intensity_range_raw": [self.lo, self.hi]}


# ---------------------------------------------------------------------------------------------------------------------
# Sampling helpers used by Phase I
# ---------------------------------------------------------------------------------------------------------------------
def to_voxel_grid(pts, vol):
    """World points (..., 3) -> grid_sample coordinates in [-1, 1] (align_corners=True)."""
    v = pts / torch.tensor([vol.dx, vol.dy, vol.dz], device=pts.device) - 0.5
    return torch.stack([v[..., 0] / (vol.W - 1), v[..., 1] / (vol.H - 1), v[..., 2] / (vol.Z - 1)], -1) * 2 - 1


def sample(field, pts, vol):
    """Trilinear samples of field (C, Z, H, W) at world points (N, S, 3) -> (C, N, S)."""
    N, S = pts.shape[:2]
    g = to_voxel_grid(pts, vol).view(1, 1, N, S, 3)
    return F.grid_sample(field[None], g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0]


def inside(pts, vol):
    """Points (N, S, 3) inside the volume and in a valid slice."""
    lim = torch.tensor([vol.W * vol.dx, vol.H * vol.dy, vol.Z * vol.dz], device=pts.device)
    ok = ((pts >= 0) & (pts < lim)).all(-1)
    k = (pts[..., 2] / vol.dz).floor().long().clamp(0, vol.Z - 1)
    return ok & vol.valid_z[k]


def blur3d(v, sigmas_zyx):
    """Separable Gaussian blur of (C, Z, H, W); sigmas in voxels per axis, replicate padding."""
    v = v[None]
    for axis, sigma in zip((2, 3, 4), sigmas_zyx):
        if sigma < 0.01:
            continue
        r = max(1, int(4 * sigma + 0.5))
        x = torch.arange(-r, r + 1, device=v.device, dtype=v.dtype)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        shape = [1, 1, 1, 1, 1]
        shape[axis] = k.numel()
        pad = [0] * 6
        pad[(4 - axis) * 2] = pad[(4 - axis) * 2 + 1] = r
        C = v.shape[1]
        v = F.conv3d(F.pad(v, pad, mode="replicate"), (k / k.sum()).view(shape).repeat(C, 1, 1, 1, 1), groups=C)
    return v[0]
