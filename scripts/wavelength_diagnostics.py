"""Diagnostics of Phase I's directional wavelength on synthetic volumes with known local structure.

    python scripts/wavelength_diagnostics.py --out runs/wavelength_diag

Every volume has CREMI's geometry (125 x 256 x 256 voxels at 40 x 4 x 4 nm) and is analysed by the unchanged
phase1.analyse with the paper's settings (configs/paper.json). The estimate is lambda_hat(x; u) = 1 / sqrt(u^T G(x) u),
exactly what Phase I uses for the violation ratio. Statistics are over interior voxels (32 px from the in-plane borders,
10 slices from the z borders) and, for membranes, over voxels on the membrane (|distance| <= w / 2).

  A  plane sinusoids        lambda along the wave vector n (truth: lambda) and the orientation error of G's first
                            eigenvector; in-plane and along z
  B  spherical sinusoids    radial lambda (truth: lambda) against the local radius of curvature
  C  crossing sinusoids     lambda along each normal; truth lambda, and the energy-weighted prediction
                            lambda / sqrt((1 + cos^2 phi) / 2) for two equal-amplitude gratings
  D  Gaussian membranes     the Gaussian size Phase I asks for across the membrane, s* = tau_q lambda_hat(n), relative to
                            the membrane's standard deviation w; planar, curved (spheres), crossing (X); and lambda_hat
                            along the membrane
  E  noise                  D's planar membranes with additive Gaussian noise: s*/w on the membrane and lambda_hat in
                            the empty background (truth: no structure, i.e. long)
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

SHAPE, SPACING = (125, 256, 256), (40.0, 4.0, 4.0)       # z, y, x; nm


def grid():
    Z, H, W = SHAPE
    dz, dy, dx = SPACING
    z = (np.arange(Z) + 0.5) * dz
    y = (np.arange(H) + 0.5) * dy
    x = (np.arange(W) + 0.5) * dx
    Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")
    return Xg, Yg, Zg


X, Y, Z = grid()
CENTRE = np.array([X.mean(), Y.mean(), Z.mean()])
INTERIOR = np.zeros(SHAPE, bool)
INTERIOR[10:-10, 32:-32, 32:-32] = True


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def inplane(deg, tilt_deg=0.0):
    t, a = math.radians(deg), math.radians(tilt_deg)
    return unit([math.cos(t) * math.cos(a), math.sin(t) * math.cos(a), math.sin(a)])


def proj(n):                                              # signed distance along n from the volume centre
    return (X - CENTRE[0]) * n[0] + (Y - CENTRE[1]) * n[1] + (Z - CENTRE[2]) * n[2]


class Analysis:
    def __init__(self, vol_np, cfg):
        from slicegaussian import phase1
        from slicegaussian.volume import Volume
        vol = Volume(vol_np.astype(np.float32), SPACING, name="synthetic", detect_invalid=False)
        f = phase1.analyse(vol, cfg)
        self.G = f["G"]                                   # (6, Z, H, W): xx, xy, yy, xz, yz, zz in physical x, y, z
        self.mat3 = phase1.mat3

    def lam(self, u):
        """lambda_hat along u: u is (3,) or (Z, H, W, 3)."""
        u = torch.as_tensor(np.broadcast_to(u, SHAPE + (3,)).copy(), dtype=torch.float32, device=self.G.device)
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        xx, xy, yy, xz, yz, zz = self.G
        k2 = ux * ux * xx + uy * uy * yy + uz * uz * zz + 2 * (ux * uy * xy + ux * uz * xz + uy * uz * yz)
        return (1.0 / k2.clamp(min=1e-30).sqrt()).cpu().numpy()

    def principal(self, mask, n=50_000, seed=0):
        """First eigenvector (largest k^2: across the structure) at n random voxels of the mask, solved on the CPU."""
        idx = np.flatnonzero(mask)
        idx = np.random.default_rng(seed).choice(idx, min(n, idx.size), replace=False)
        G6 = self.G.reshape(6, -1)[:, torch.as_tensor(idx, device=self.G.device)].double().cpu()
        return torch.linalg.eigh(self.mat3(G6))[1][..., -1].numpy()


def stats(a):
    a = a[np.isfinite(a)]
    return {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "n": int(a.size)}


def angle_deg(v, n):
    c = np.abs(v @ unit(n))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("runs/wavelength_diag"))
    p.add_argument("--config", default="configs/paper.json")
    a = p.parse_args()
    from slicegaussian import phase1
    cfg = phase1.P1Config(**json.load(open(a.config))["p1"])
    tau = cfg.tau_q
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(exp, case, quantity, truth, s, **extra):
        rows.append({"experiment": exp, "case": case, "quantity": quantity, "truth": truth, **s,
                     "median_over_truth": s["median"] / truth if truth else float("nan"), **extra})
        print(f"{exp} {case:38s} {quantity:28s} truth {truth:9.2f}  median {s['median']:9.2f}  "
              f"[{s['p10']:.2f}, {s['p90']:.2f}]  ratio {rows[-1]['median_over_truth']:.3f}"
              + "".join(f"  {k} {v:.2f}" for k, v in extra.items()), flush=True)

    # A: plane sinusoids
    for lam in (16, 32, 64, 128, 256):
        for th in (0, 30, 45):
            n = inplane(th)
            an = Analysis(0.5 + 0.5 * np.cos(2 * np.pi * proj(n) / lam), cfg)
            add("A", f"in-plane lambda={lam} theta={th}", "lambda along n", lam, stats(an.lam(n)[INTERIOR]))
            add("A", f"in-plane lambda={lam} theta={th}", "orientation error (deg)", 0.0,
                stats(angle_deg(an.principal(INTERIOR), n)))
    for lam in (160, 320, 640):
        n = np.array([0.0, 0.0, 1.0])
        an = Analysis(0.5 + 0.5 * np.cos(2 * np.pi * proj(n) / lam), cfg)
        add("A", f"along z lambda={lam}", "lambda along n", lam, stats(an.lam(n)[INTERIOR]))

    # B: curved sinusoids (radial wave) by radius of curvature: cylinders with axis z (curvature in the image plane,
    # as for CREMI membranes), and spheres split into equatorial (radial direction in-plane) and polar (radial direction
    # near z, where a 64 nm wave is below the 80 nm Nyquist limit of the 40 nm slices) points
    lam = 64
    R = np.sqrt((X - CENTRE[0]) ** 2 + (Y - CENTRE[1]) ** 2 + (Z - CENTRE[2]) ** 2)
    radial = np.stack([(X - CENTRE[0]), (Y - CENTRE[1]), (Z - CENTRE[2])], -1) / np.maximum(R, 1e-9)[..., None]
    Rc = np.sqrt((X - CENTRE[0]) ** 2 + (Y - CENTRE[1]) ** 2)
    radial_c = np.stack([(X - CENTRE[0]), (Y - CENTRE[1]), 0 * Z], -1) / np.maximum(Rc, 1e-9)[..., None]
    an = Analysis(0.5 + 0.5 * np.cos(2 * np.pi * Rc / lam), cfg)
    lr = an.lam(radial_c)
    for lo, hi in ((0.5, 1), (1, 2), (2, 4)):
        m = INTERIOR & (Rc >= lo * lam) & (Rc < hi * lam)
        add("B", f"cylinder lambda={lam} R/lambda in [{lo},{hi})", "radial lambda", lam, stats(lr[m]))
    an = Analysis(0.5 + 0.5 * np.cos(2 * np.pi * R / lam), cfg)
    lr = an.lam(radial)
    for name, sel in (("equatorial |u_z|<0.2", np.abs(radial[..., 2]) < 0.2), ("polar |u_z|>0.8", np.abs(radial[..., 2]) > 0.8)):
        m = INTERIOR & sel & (R >= lam) & (R < 8 * lam)
        add("B", f"sphere lambda={lam} {name}", "radial lambda", lam, stats(lr[m]))

    # C: two crossing in-plane sinusoids of equal amplitude
    lam = 64
    for phi in (30, 60, 90):
        n1, n2 = inplane(0), inplane(phi)
        an = Analysis(0.5 + 0.25 * np.cos(2 * np.pi * proj(n1) / lam) + 0.25 * np.cos(2 * np.pi * proj(n2) / lam), cfg)
        pred = lam / math.sqrt((1 + math.cos(math.radians(phi)) ** 2) / 2)
        s1 = stats(an.lam(n1)[INTERIOR])
        add("C", f"crossing phi={phi} lambda={lam}", "lambda along n1", lam, s1, energy_prediction=pred)

    # D: Gaussian-profile membranes (isolated: period 256 nm >> w)
    period = 256.0

    def sheets(d, w):                                     # periodic train of Gaussian sheets, distance d to sheet 0
        r = (d + period / 2) % period - period / 2
        return np.exp(-0.5 * (r / w) ** 2), np.abs(r)

    for w in (4, 8, 16):
        for th in (0, 30):
            n = inplane(th)
            v, dist = sheets(proj(n), w)
            an = Analysis(v, cfg)
            on = INTERIOR & (dist <= w / 2)
            add("D", f"planar w={w} theta={th}", "s*/w across", 1.0, stats(tau * an.lam(n)[on] / w),
                lambda_eq=2 * math.sqrt(2) * math.pi * w)
            t = unit(np.cross(n, [0, 0, 1]))
            add("D", f"planar w={w} theta={th}", "lambda along membrane (nm)", 0.0, stats(an.lam(t)[on]))
    w = 8
    for Rm in (16, 32, 64, 128):                          # curved: single cylindrical membrane (axis z) of radius Rm
        v = np.exp(-0.5 * ((Rc - Rm) / w) ** 2)
        an = Analysis(v, cfg)
        on = INTERIOR & (np.abs(Rc - Rm) <= w / 2)
        add("D", f"cylinder w={w} R={Rm}", "s*/w across", 1.0, stats(tau * an.lam(radial_c)[on] / w))
    for phi in (30, 60, 90):                              # crossing: two planar membranes through the centre
        n1, n2 = inplane(0), inplane(phi)
        d1, d2 = np.abs(proj(n1)), np.abs(proj(n2))
        v = np.maximum(np.exp(-0.5 * (d1 / w) ** 2), np.exp(-0.5 * (d2 / w) ** 2))
        an = Analysis(v, cfg)
        l1 = an.lam(n1)
        near = INTERIOR & (d1 <= w / 2) & (d2 <= 4 * w)
        far = INTERIOR & (d1 <= w / 2) & (d2 > 16 * w)
        add("D", f"X-crossing w={w} phi={phi} near", "s*/w across n1", 1.0, stats(tau * l1[near] / w))
        add("D", f"X-crossing w={w} phi={phi} far", "s*/w across n1", 1.0, stats(tau * l1[far] / w))

    # E: noise on planar membranes (w = 8, theta = 30)
    rng = np.random.default_rng(0)
    n = inplane(30)
    v, dist = sheets(proj(n), 8)
    on = INTERIOR & (dist <= 4)
    bg = INTERIOR & (dist >= 64)
    t = unit(np.cross(n, [0, 0, 1]))
    for sd in (0.0, 0.02, 0.05, 0.1):
        an = Analysis(v + sd * rng.standard_normal(SHAPE), cfg)
        add("E", f"noise sd={sd}", "s*/w across (membrane)", 1.0, stats(tau * an.lam(n)[on] / 8))
        add("E", f"noise sd={sd}", "background lambda along n (nm)", 0.0, stats(an.lam(n)[bg]))
        add("E", f"noise sd={sd}", "background lambda along t (nm)", 0.0, stats(an.lam(t)[bg]))

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out / "wavelength_diagnostics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    json.dump({"config": vars(cfg) if hasattr(cfg, "__dict__") else str(cfg), "shape": SHAPE, "spacing_zyx": SPACING,
               "scales_nm": [cfg.sigma_min * 4, cfg.sigma_max * 4]}, open(a.out / "protocol.json", "w"), indent=1,
              default=str)


if __name__ == "__main__":
    main()
