"""Synthetic EM-like phantom with the CREMI geometry (125 x 256 x 256 voxels at 40 x 4 x 4 nm), noise-free.

    micromamba run -n gaussian-volume python scripts/make_phantom.py [--out data/phantom] [--seed 0]

Structure (all in physical nm, evaluated at the voxel centres):
  * cells: a Voronoi tessellation whose metric stretches z by --elongation, so cells are elongated along z like
    neurites; each cell has its own cytoplasm intensity, 0.62 +- 0.08 (uniform);
  * membranes: the Voronoi boundaries, dark (0.15) with a Gaussian profile of std --membrane-std (6 nm) in the
    distance to the bisector between the two nearest seeds;
  * vesicles: spheres of radius 15-25 nm with a dark rim (0.30, Gaussian profile of std 3 nm) and a 0.55 lumen.
Writes phantom_em.tif (uint16, full range) and provenance.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

SHAPE, SPACING = (125, 256, 256), (40.0, 4.0, 4.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/phantom"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cells", type=int, default=200)
    p.add_argument("--elongation", type=float, default=4.0)
    p.add_argument("--membrane-std", type=float, default=6.0)
    p.add_argument("--vesicles", type=int, default=3000)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    Z, H, W = SHAPE
    dz, dy, dx = SPACING
    ext = torch.tensor([W * dx, H * dy, Z * dz])
    seeds = torch.rand(a.cells, 3, generator=g) * (ext + 400) - 200          # some seeds outside: open cells
    level = 0.62 + 0.16 * (torch.rand(a.cells, generator=g) - 0.5)
    ves_c = torch.rand(a.vesicles, 3, generator=g) * ext
    ves_r = 15 + 10 * torch.rand(a.vesicles, generator=g)
    metric = torch.tensor([1.0, 1.0, 1.0 / a.elongation])
    seeds, level, ves_c, ves_r, metric = (t.to(dev) for t in (seeds, level, ves_c, ves_r, metric))
    out = np.empty(SHAPE, np.float32)
    ys, xs = torch.meshgrid((torch.arange(H, device=dev) + 0.5) * dy, (torch.arange(W, device=dev) + 0.5) * dx,
                            indexing="ij")
    for k in range(Z):
        pts = torch.stack([xs, ys, torch.full_like(xs, (k + 0.5) * dz)], -1).view(-1, 3)
        d = torch.cdist(pts * metric, seeds * metric)                          # stretched metric
        d2, i2 = d.topk(2, largest=False)
        # distance to the bisector between the two nearest seeds, in physical units (metric-scaled normal)
        a1, a2 = seeds[i2[:, 0]] * metric, seeds[i2[:, 1]] * metric
        n = a2 - a1
        mid = (a1 + a2) / 2
        dist = ((pts * metric - mid) * n).sum(1).abs() / (n * metric).norm(dim=1)   # physical distance to the plane
        v = level[i2[:, 0]]
        mem = torch.exp(-0.5 * (dist / a.membrane_std) ** 2)
        v = v * (1 - mem) + 0.15 * mem
        dv = torch.cdist(pts, ves_c)                                           # vesicles
        near = dv - ves_r
        j = near.argmin(1)
        r = near.gather(1, j[:, None])[:, 0]
        rim = torch.exp(-0.5 * (r / 3.0) ** 2)
        inside = (r < 0).float()
        v = v * (1 - inside) + 0.55 * inside
        v = v * (1 - rim) + 0.30 * rim
        out[k] = v.view(H, W).cpu().numpy()
    a.out.mkdir(parents=True, exist_ok=True)
    lo, hi = float(out.min()), float(out.max())
    q = np.round((out - lo) / (hi - lo) * 65535).astype(np.uint16)
    tifffile.imwrite(a.out / "phantom_em.tif", q)
    (a.out / "provenance.json").write_text(json.dumps({**vars(a), "out": str(a.out), "shape": SHAPE,
                                                       "spacing_zyx_nm": SPACING, "range": [lo, hi]}, indent=1))
    print("wrote", a.out / "phantom_em.tif", q.shape, lo, hi)


if __name__ == "__main__":
    main()
