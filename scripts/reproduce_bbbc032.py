"""Main comparison on the BBBC032 confocal crops (nuclei, membrane): SliceGaussian, GaussianPile, 3DGS,
NeurComp, INIF and CoordNet, each with the settings used on CREMI, one run each (seed 0; the seed-to-seed spread is
reported on CREMI). Finished runs are skipped.

    python scripts/prepare_bbbc032.py --out data/bbbc032
    python scripts/reproduce_bbbc032.py [--volumes nuclei membrane] [--methods ...]

Results: runs/bbbc032/<volume>/<method>/seed_<s>/. Every method is scored by slicegaussian.metrics on the same
min-max normalised ground truth; the INRs through their saved prediction (score.json).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "freqpile"
SPACING = "500,101,101"                      # nm, from the file metadata (0.5 x 0.101 x 0.101 um)
METHODS = ["slicegaussian", "gaussianpile", "3dgs", "neurcomp", "inif", "coordnet"]   # CoordNet last: ~1 h per volume


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(map(str, cmd)), flush=True)
    with log.open("w") as f:
        subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def score(volume, out):
    import numpy as np
    import torch
    from slicegaussian import load_volume, metrics
    vol = load_volume(volume, tuple(float(s) for s in SPACING.split(",")))
    pred = torch.from_numpy(np.load(out / "prediction.npy").astype(np.float32)).cuda()
    (out / "score.json").write_text(json.dumps(metrics.whole(None, vol, full=True, pred=pred), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--volumes", nargs="+", default=["nuclei", "membrane"])
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    a = p.parse_args()
    py = sys.executable
    for v in a.volumes:
        volume = ROOT / "data" / "bbbc032" / f"bbbc032_{v}.tif"
        base = ROOT / "runs" / "bbbc032" / v
        for m in a.methods:
            for s in [0]:                    # single seed here; seed variance is reported on CREMI
                out = base / m / f"seed_{s}"
                done = out / ("final.json" if m == "slicegaussian" else
                              "metrics.json" if m in ("gaussianpile", "3dgs") else "score.json")
                if done.exists():
                    print(f"skip {v}/{m}/seed_{s}", flush=True)
                    continue
                log = base / m / f"seed_{s}.log"
                if m == "slicegaussian":
                    run([Path(py).parent / "slicegaussian", "fit", volume, "--spacing", SPACING,
                         "--config", "configs/paper.json", "--seed", s, "--out", out, "--quiet"], log)
                elif m in ("gaussianpile", "3dgs"):
                    run([py, "scripts/gaussian_baseline.py", volume, "--spacing", SPACING, "--method", m,
                         "--seed", s, "--out", out], log)
                else:
                    if not (out / "metrics.json").exists():
                        if m == "inif":
                            # INIF sizes its network from the file's bytes; the CREMI runs used ratio 16 on uint8.
                            # These crops are uint16, so ratio 32 gives the same parameters per voxel (~128k).
                            run(["bash", REF / "baselines/inif/run.sh", volume, out, "--ratio", "32"], log)
                        else:
                            flags = ["--ratio", "40"] if m == "neurcomp" else ["--steps", "50000"]
                            run([py, REF / "baselines/inr/inr.py", volume, out, "--model", m, *flags], log)
                    if not (out / "prediction.npy").exists():
                        if m == "inif":
                            run(["micromamba", "run", "-n", "inif", "python", REF / "baselines/inif/inif.py",
                                 volume, out, "--ratio", "32", "--predict_only"], out / "predict.log")
                        else:
                            run([py, REF / "baselines/inr/inr.py", volume, out, "--model", m, "--predict_only",
                                 *(["--ratio", "40"] if m == "neurcomp" else [])], out / "predict.log")
                    score(volume, out)
                print(f"done {v}/{m}/seed_{s}", flush=True)


if __name__ == "__main__":
    main()
