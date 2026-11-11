"""Prints mean +- std of the final metrics over seeds for every runs/<volume>/seed_<s>/final.json."""
import glob
import json
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = {}
for f in sorted(glob.glob(os.path.join(root, "*", "seed_*", "final.json"))):
    rows.setdefault(f.split(os.sep)[-3], []).append(json.load(open(f)))
keys = ["psnr", "all_psnr", "ssim", "mae", "gaussians", "total_seconds", "encoded_bytes"]
print("volume".ljust(12) + "".join(k.rjust(18) for k in keys))
for v, rs in rows.items():
    cells = []
    for k in keys:
        x = [r[k] for r in rs if r.get(k) is not None]
        cells.append(f"{np.mean(x):.4g} ± {np.std(x):.2g}".rjust(18) if x else "-".rjust(18))
    print(f"{v} (n={len(rs)})".ljust(12) + "".join(cells))
