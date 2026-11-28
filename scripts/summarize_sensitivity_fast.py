"""Parameter sensitivity with the raw-parameter CUDA kernels (runs/sensitivity_fast), seed 0, CREMI A/B/C.

    micromamba run -n gaussian-volume python scripts/summarize_sensitivity_fast.py

Default row: runs/remeasure_fast/<volume>/seed_0 (complete method, same implementation). Matched-count controls
(three 10k cycles with optimizer restarts, no re-estimation): m_s7 (sigma_max 16 nm, own count), m_s11at7 (default sigma_max 64 nm, tau_q
bisected to the 16 nm count), m_s11 (default Phase I, own count), m_s7atdef (sigma_max 16 nm, tau_q bisected to the
default Phase I count). Writes runs/sensitivity_fast/summary.json and prints the rows.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs/sensitivity_fast"
VOLS = ["cremi_a", "cremi_b", "cremi_c"]
SWEEP = [("Largest scale", "scales7", "16 nm"), ("Largest scale", "scales9", "32 nm"),
         ("Largest scale", "default", "64 nm (default)"), ("Largest scale", "scales13", "128 nm"),
         ("Percentile", "p75", "P75"), ("Percentile", "default", "P90 (default)"), ("Percentile", "p95", "P95"),
         ("Percentile", "p100", "maximum"),
         ("Threshold", "tau0.8", "0.8"), ("Threshold", "tau0.4", "0.4"), ("Threshold", "tau0.2", "0.2"),
         ("Threshold", "default", "0.1 (default)"), ("Threshold", "tau0.05", "0.05"),
         ("Amplitude weight", "reg0", "0"), ("Amplitude weight", "reg0.01", "0.01"),
         ("Amplitude weight", "default", "0.1 (default)"), ("Amplitude weight", "reg1", "1"),
         ("Scale-ratio limit", "ratio10", "10"), ("Scale-ratio limit", "ratio100", "100"),
         ("Scale-ratio limit", "default", "1,000 (default)")]
MATCHED = ["m_s7", "m_s11at7", "m_s7atdef", "m_s11"]


def load(vol, key):
    d = ROOT / "runs/remeasure_fast" / vol / "seed_0" if key == "default" else R / vol / key / "seed_0"
    f = json.loads((d / "final.json").read_text())
    run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
    held = any(p.get("phase") == "check" and not p.get("fired") and p.get("gaussians", 0) >= 1_000_000
               for p in run.get("phases", []))
    return {"psnr": f["psnr"], "mae": f["mae"], "gaussians": f["gaussians"], "seconds": f["total_seconds"],
            "held": held or f["gaussians"] >= 999_000}


def main():
    out = {"sweep": [], "matched": []}
    for group, key, label in SWEEP:
        rows = {v: load(v, key) for v in VOLS}
        out["sweep"].append({"group": group, "key": key, "label": label, **{v: rows[v] for v in VOLS}})
        print(f"{group:18s} {label:24s} " + " / ".join(f"{rows[v]['psnr']:.2f}" for v in VOLS) + "   "
              + " / ".join(f"{rows[v]['gaussians']:,}{'*' if rows[v]['held'] else ''}" for v in VOLS) + "   "
              + " / ".join(f"{rows[v]['seconds']:.0f}" for v in VOLS))
    for key in MATCHED:
        rows = {v: load(v, key) for v in VOLS}
        out["matched"].append({"key": key, **{v: rows[v] for v in VOLS}})
        print(f"{key:12s} " + " / ".join(f"{rows[v]['psnr']:.2f} ({rows[v]['gaussians']:,})" for v in VOLS))
    (R / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
