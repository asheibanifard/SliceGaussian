"""Noise versus structure in the reconstructions of one volume.

    micromamba run -n gaussian-volume python scripts/analyse_noise_fit.py --run-dir runs/neural/epfl \
        --volume data/epfl/epfl_ca1.tif --spacing 5,5,5

For each method's prediction.npy (valid slices, intensities as scored):
  * PSNR against the raw volume and against denoised references (3D Gaussian smoothing, sigma in voxels);
  * noise reproduced: beta = <P - D, V - D> / <V - D, V - D>, with D the sigma = 1 reference: the share of the
    removed component (mostly noise) that the method reproduces (0: none, 1: all);
  * residual energy (P - V) by in-plane spatial frequency (cycles per pixel): low < 0.1, mid 0.1-0.25, high > 0.25;
    the same split of the noise estimate V - D for comparison.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["slicegaussian", "gaussianpile", "3dgs", "sadgs3d", "neurcomp", "inif", "coordnet"]


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def bands(r):
    """Fraction of the energy of r (Z, H, W) in radial in-plane frequency bands."""
    F = torch.fft.rfft2(r) .abs() ** 2
    H, W = r.shape[-2:]
    fy = torch.fft.fftfreq(H, device=r.device)[:, None]
    fx = torch.fft.rfftfreq(W, device=r.device)[None, :]
    f = torch.sqrt(fx ** 2 + fy ** 2)
    w = torch.ones_like(f)
    w[:, 1:-1] = 2                                           # rfft: interior columns count twice
    e = (F * w).sum(0)
    tot = float(e.sum())
    return {k: float(e[m].sum()) / tot for k, m in (("low", f < 0.1), ("mid", (f >= 0.1) & (f < 0.25)),
                                                    ("high", f >= 0.25))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--volume", required=True)
    p.add_argument("--spacing", required=True)
    a = p.parse_args()
    from slicegaussian import load_volume
    from slicegaussian.volume import blur3d
    vol = load_volume(str(ROOT / a.volume), tuple(float(s) for s in a.spacing.split(",")))
    ids = vol.valid_ids
    V = vol.gt
    refs = {s: blur3d(V[None], (s, s, s))[0] for s in (1.0, 1.5)}
    D = refs[1.0]
    noise = (V - D)[ids]
    out = {"reference": {f"sigma{s}": {"psnr_vs_raw": psnr(R[ids], V[ids])} for s, R in refs.items()},
           "noise_bands": bands(noise), "methods": {}}
    print("denoised references vs raw: " + ", ".join(f"sigma {s}: {out['reference'][f'sigma{s}']['psnr_vs_raw']:.2f} dB"
                                                        for s in refs))
    print("noise estimate V - D, energy by band:", {k: round(v, 3) for k, v in out["noise_bands"].items()})
    print(f"{'method':14s} {'raw':>6s} {'ref1':>6s} {'ref1.5':>6s} {'beta':>6s}   residual bands low/mid/high")
    for m in METHODS:
        f = ROOT / a.run_dir / m / "seed_0" / "prediction.npy"
        if not f.exists():
            continue
        P = torch.from_numpy(np.load(f).astype(np.float32).reshape(vol.shape)).to(V.device).clamp(0, 1)
        Pv, Vv = P[ids], V[ids]
        beta = float(((Pv - D[ids]) * noise).sum() / (noise ** 2).sum())
        r = {"psnr_raw": psnr(Pv, Vv), **{f"psnr_ref{s}": psnr(Pv, R[ids]) for s, R in refs.items()},
             "noise_reproduced": beta, "residual_bands": bands(Pv - Vv)}
        out["methods"][m] = r
        b = r["residual_bands"]
        print(f"{m:14s} {r['psnr_raw']:6.2f} {r['psnr_ref1.0']:6.2f} {r['psnr_ref1.5']:6.2f} {beta:6.3f}   "
              f"{b['low']:.3f} / {b['mid']:.3f} / {b['high']:.3f}")
    (ROOT / a.run_dir / "noise_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
