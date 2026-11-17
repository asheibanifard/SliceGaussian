"""Two new populations plus the saved 1M baseline form each Gaussian baseline's count curve."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for sample in "abc":
        for method in ("gaussianpile", "3dgs"):
            for count in (100000, 300000):
                out = root / "runs/rate_distortion/models" / f"cremi_{sample}" / method / f"initial_{count}"
                out.mkdir(parents=True, exist_ok=True)
                if (out / "model.pt").exists() and (out / "metrics.json").exists():
                    print(f"Skipping {sample}/{method}/{count}", flush=True)
                    continue
                print(f"Starting {sample.upper()}/{method}/{count}", flush=True)
                with (out / "console.log").open("w") as log:
                    subprocess.run([sys.executable, str(root / "scripts/rd_gaussian_sweep.py"),
                        str(root / f"data/cremi/cremi_{sample}.tif"), "--method", method,
                        "--initial-n", str(count), "--out", str(out)], cwd=root,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
                print(f"Finished {sample.upper()}/{method}/{count}", flush=True)


if __name__ == "__main__":
    main()
