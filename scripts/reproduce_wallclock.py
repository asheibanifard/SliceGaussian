"""Run six fresh processes sequentially: full SliceGaussian and GaussianPile on CREMI A/B/C."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    # Dependency compilation is installation work, outside each measured fresh process.
    # CUDA context creation and runtime extension loading still occur inside every fit clock.
    preflight = ("import sys; sys.path.insert(0, " + repr(str(root.parent / "gaussianpile")) + "); "
                 "from slicegaussian.cuda.fieldcut import _ext; _ext(); "
                 "from gaussianpile import _backend; _backend.get()")
    subprocess.run([sys.executable, "-c", preflight], cwd=root, check=True)
    for sample in "abc":
        for method in ("slicegaussian", "gaussianpile"):
            out = root / "runs" / "wallclock" / f"cremi_{sample}" / method
            out.mkdir(parents=True, exist_ok=True)
            if (out / "result.json").exists():
                print(f"Skipping completed {sample}/{method}", flush=True)
                continue
            print(f"Starting {sample.upper()}/{method}", flush=True)
            with (out / "console.log").open("w") as log:
                subprocess.run([sys.executable, str(root / "scripts/wallclock_benchmark.py"),
                                str(root / f"data/cremi/cremi_{sample}.tif"), "--method", method,
                                "--out", str(out)], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"Finished {sample.upper()}/{method}", flush=True)


if __name__ == "__main__":
    main()
