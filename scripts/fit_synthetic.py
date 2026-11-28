"""Fit the complete SliceGaussian method (configs/paper.json: Phase I, three Phase II cycles with re-estimation,
seed 0) to each synthetic volume of the wavelength diagnostic (supplement, Sec. S2), and save the slices shown in the
figure: the central x-y slice and the central x-z cut of the ground truth and of the reconstruction.

    python scripts/fit_synthetic.py --out runs/synthetic_recon

Volumes and their order are those of scripts/plot_synthetic_slices.py. Each run writes <out>/<index>/ with the usual
fit outputs and slices.npz; summary.json collects 3D PSNR, SSIM and the Gaussian count.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "runs/synthetic_recon")
    p.add_argument("--config", default=str(ROOT / "configs/paper.json"))
    a = p.parse_args()
    from slicegaussian.pipeline import Config, fit
    from slicegaussian.volume import Volume
    from slicegaussian.field import render_volume
    ps = _load("plot_synthetic_slices")
    cfg = Config.load(a.config)
    cfg.codec = False
    summary = []
    for i, (fam, title, par, v, view) in enumerate(ps.volumes()):
        out = a.out / f"{i:02d}"
        vol = Volume(v.astype(np.float32), ps.wd.SPACING, name=f"synthetic_{i:02d}", detect_invalid=False)
        if (out / "final.json").exists() and (out / "slices.npz").exists():
            final = json.loads((out / "final.json").read_text())
        else:
            field, final = fit(vol, str(out), cfg, verbose=False)
            with torch.no_grad():
                P = render_volume(field, vol).clamp(0, 1).cpu().numpy()
            G = vol.gt.cpu().numpy()
            z = ps.ZC
            np.savez_compressed(out / "slices.npz", gt_xy=G[z], rec_xy=P[z], gt_xz=G[:, ps.NY // 2, :],
                                rec_xz=P[:, ps.NY // 2, :])
            del field, P
            torch.cuda.empty_cache()
        summary.append({"index": i, "family": fam, "title": title, "setting": par, "view": view,
                        "psnr": final["psnr"], "ssim": final.get("ssim"), "gaussians": final["gaussians"],
                        "seconds": final.get("total_seconds")})
        print(f"{i:02d} {title:28s} {par:28s} PSNR {final['psnr']:.2f}  G {final['gaussians']:,}", flush=True)
        (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
