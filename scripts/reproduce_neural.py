"""Main comparison on two further neural EM volumes (EPFL CA1 FIB-SEM, SNEMI3D ATUM-SEM): every method with its
CREMI settings, one run each (seed 0). Finished runs are skipped; cheap methods run first, CoordNet last.

    python scripts/prepare_neural.py
    python scripts/reproduce_neural.py [--volumes epfl snemi3d] [--methods ...]

Results: runs/neural/<volume>/<method>/seed_0/. Every method is scored by slicegaussian.metrics.whole on the same
min-max normalised ground truth (valid slices); the INRs through their saved prediction (score.json).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "freqpile"
VOLUMES = {"epfl": ("data/epfl/epfl_ca1.tif", "5,5,5"), "snemi3d": ("data/snemi3d/snemi3d.tif", "29,6,6")}
METHODS = ["slicegaussian", "gaussianpile", "3dgs", "sadgs3d", "neurcomp", "inif", "coordnet"]


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(map(str, cmd)), flush=True)
    with log.open("w") as f:
        subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def score(volume, spacing, out):
    import numpy as np
    import torch
    from slicegaussian import load_volume, metrics
    vol = load_volume(volume, tuple(float(s) for s in spacing.split(",")))
    pred = torch.from_numpy(np.load(out / "prediction.npy").astype(np.float32).reshape(vol.shape)).cuda()
    (out / "score.json").write_text(json.dumps(metrics.whole(None, vol, full=True, pred=pred), indent=1))


def done_file(m):
    return {"slicegaussian": "final.json", "gaussianpile": "metrics.json", "3dgs": "metrics.json",
            "sadgs3d": "metrics.json"}.get(m, "score.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--volumes", nargs="+", default=list(VOLUMES), choices=list(VOLUMES))
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    a = p.parse_args()
    py = sys.executable
    for m in a.methods:                                   # method-major: CoordNet runs last on both volumes
        for v in a.volumes:
            volume, spacing = str(ROOT / VOLUMES[v][0]), VOLUMES[v][1]
            out = ROOT / "runs/neural" / v / m / "seed_0"
            if (out / done_file(m)).exists():
                print(f"skip {v}/{m}", flush=True)
                continue
            log = out.parent / "seed_0.log"
            if m == "slicegaussian":
                run([Path(py).parent / "slicegaussian", "fit", volume, "--spacing", spacing,
                     "--config", "configs/paper.json", "--seed", 0, "--out", out, "--quiet"], log)
            elif m in ("gaussianpile", "3dgs"):
                run([py, "scripts/gaussian_baseline.py", volume, "--spacing", spacing, "--method", m,
                     "--valid-only", "--seed", 0, "--out", out], log)
            elif m == "sadgs3d":
                run([py, "baselines/sadgs3d/sadgs3d.py", volume, "--spacing", spacing, "--cap", 1_000_000,
                     "--out", out], log)
            else:
                # INIF sizes its network from the file's bytes: ratio 16 on uint8, as on CREMI (same parameters
                # per voxel). NeurComp ratio 40 and CoordNet 50,000 steps, as on CREMI.
                flags = {"neurcomp": ["--ratio", "40"], "coordnet": ["--steps", "50000"], "inif": ["--ratio", "16"]}[m]
                if not (out / "metrics.json").exists():
                    if m == "inif":
                        run(["bash", REF / "baselines/inif/run.sh", volume, out, *flags], log)
                    else:
                        run([py, REF / "baselines/inr/inr.py", volume, out, "--model", m, *flags], log)
                if not (out / "prediction.npy").exists():
                    if m == "inif":
                        run(["micromamba", "run", "-n", "inif", "python", REF / "baselines/inif/inif.py",
                             volume, out, *flags, "--predict_only"], out / "predict.log")
                    else:
                        run([py, REF / "baselines/inr/inr.py", volume, out, "--model", m, "--predict_only",
                             *(flags if m == "neurcomp" else [])], out / "predict.log")
                score(volume, spacing, out)
            print(f"done {v}/{m}", flush=True)


if __name__ == "__main__":
    main()
