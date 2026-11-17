"""Table 5 (loss ablation) on EPFL CA1: the complete method with each loss, with the columns of the CREMI table plus the
noise measures of Table 3.

    micromamba run -n gaussian-volume python scripts/loss_table_epfl.py

Columns: MAE, RMSE (sqrt of the MSE, from the PSNR on [0, 1]), PSNR, P99 |alpha|, share of Gaussians at the scale-ratio
limit of 1,000, standard deviation of the PSNR over the last 2,000 iterations, PSNR against the volume smoothed by one
voxel and beta, the share of the noise reproduced (as in Table 3).
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("L = L_vox", "slicegaussian_vox"), ("L = L_vox + b_g L_grad", "slicegaussian_vox_grad"),
        ("L = L_vox + b_m L_multi", "slicegaussian_vox_multi"), ("L = L_vox + b_r L_reg", "slicegaussian_vox_reg"),
        ("L = all four terms (default)", "slicegaussian")]


def main():
    from slicegaussian import load_volume
    from slicegaussian.field import GaussianField, render_volume
    from slicegaussian.volume import blur3d
    vol = load_volume(str(ROOT / "data/epfl/epfl_ca1.tif"), (5.0, 5.0, 5.0))
    V = vol.gt
    D = blur3d(V[None], (1.0, 1.0, 1.0))[0]
    n = V - D
    rows = []
    for label, k in RUNS:
        d = ROOT / f"runs/neural/epfl/{k}/seed_0"
        f = json.loads((d / "final.json").read_text())
        tr = [r for r in csv.DictReader(open(d / "trace.csv")) if int(r["iteration"]) > 28000]
        st = torch.load(d / "model.pt", map_location="cuda")
        s = st["scales"]
        field = GaussianField.load(d / "model.pt")
        with torch.no_grad():
            P = render_volume(field, vol).clamp(0, 1)
        r = {"loss": label, "mae": f["mae"], "rmse": 10 ** (-f["psnr"] / 20), "psnr": f["psnr"],
             "p99_amp": float(st["amp"].abs().quantile(0.99)) if len(st["amp"]) < 16_000_000 else float("nan"),
             "at_limit_pct": 100 * float((s.max(1).values / s.min(1).values >= 999.0).float().mean()),
             "psnr_std_last2k": float(np.std([float(x["psnr"]) for x in tr])),
             "psnr_vs_denoised": float(-10 * torch.log10(((P - D) ** 2).mean())),
             "beta": float(((P - D) * n).sum() / (n ** 2).sum()), "gaussians": f["gaussians"],
             "seconds": f["total_seconds"]}
        rows.append(r)
        print(f"{label:30s} MAE {r['mae']:.4f}  RMSE {r['rmse']:.4f}  PSNR {r['psnr']:.2f}  P99|a| {r['p99_amp']:.3f}  "
              f"limit {r['at_limit_pct']:.2f}%  std {r['psnr_std_last2k']:.3f}  | denoised {r['psnr_vs_denoised']:.2f}  "
              f"beta {r['beta']:.3f}  N {r['gaussians']:,}  {r['seconds']:.0f}s", flush=True)
    (ROOT / "runs/neural/epfl/loss_table.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
