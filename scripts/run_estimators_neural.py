"""Table 2 estimators on EPFL CA1 and SNEMI3D with the code that produced the CREMI rows (freqpile/experiments/twophase):
uniform grid, random positions, frequency-only (Phase I with structure = none), structure-only, and Phase I itself (a
consistency check against the new code), each matched to the default Phase I count and followed by one Phase II of
30,000 iterations with the CREMI job settings (copied from final-experiment-2/sec42/CREMI_A/*/seed_0/config.json).

    micromamba run -n gaussian-volume python scripts/run_estimators_neural.py

Results: freqpile/results/paper/final-experiment-2/sec42_neural/<volume>/<method>/seed_0/ (trace.csv, final.json).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT.parent / "freqpile/experiments/twophase"
sys.path.insert(0, str(TP))
import data  # noqa: E402
import run   # noqa: E402

VOLUMES = {"epfl": (ROOT / "data/epfl/epfl_ca1.tif", (5.0, 5.0, 5.0)),
           "snemi3d": (ROOT / "data/snemi3d/snemi3d.tif", (29.0, 6.0, 6.0))}
_registry = data.registry


def registry():
    reg = _registry()
    for name, (path, spacing) in VOLUMES.items():
        reg[name] = ("em", "test", (lambda p=path, s=spacing, n=name: data.Volume(str(p), s, name=n)))
    return reg


data.registry = registry
P2 = {"iterations": 30000, "eval_every": 100, "full_every": 5000, "snapshot_every": 1000,
      "lr_xyz": 1.6e-05, "lr_log_scale": 0.001, "lr_quat": 0.0002, "lr_amp": 0.00025}       # as the CREMI jobs
JOBS = [("init_uniform", "uniform", {}), ("init_random", "random", {}), ("one_no_structure", "proposed", {"structure": "none"}),
        ("init_structure", "structure", {}), ("proposed", "proposed", {})]

if __name__ == "__main__":
    for v in VOLUMES:
        for folder, method, p1 in JOBS:
            job = {"out": f"../final-experiment-2/sec42_neural/{v}/{folder}/seed_0", "volume": v, "method": method,
                   "M": "proposed@default", "seed": 0, "p1": p1, "p2": P2, "save_model": False, "codec": False}
            print(v, folder, run.run_job(job), flush=True)
