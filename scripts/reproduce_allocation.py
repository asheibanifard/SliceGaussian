"""Run the five allocation arms sequentially on CREMI A/B/C (one seed)."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for sample in "abc":
        out = root / "runs" / "allocation" / f"cremi_{sample}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "allocation.json").exists():
            print(f"Skipping completed {sample.upper()}", flush=True)
            continue
        print(f"Starting CREMI {sample.upper()}", flush=True)
        with (out / "console.log").open("w") as log:
            subprocess.run([sys.executable, str(root / "scripts" / "allocation_ablation.py"),
                            str(root / "data" / "cremi" / f"cremi_{sample}.tif"),
                            "--spacing", "40,4,4", "--out", str(out)],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Finished CREMI {sample.upper()}", flush=True)


if __name__ == "__main__":
    main()
