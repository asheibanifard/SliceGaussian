#!/usr/bin/env bash
# Reproduces the SliceGaussian runs of the paper: CREMI A, B and C, seeds 0, 1 and 2, 3 cycles x 10,000 iterations.
# About 5-6 minutes per run on an RTX A4000. Results: runs/cremi_<v>/seed_<s>/final.json
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_cremi.py --out data/cremi
for v in a b c; do
  for s in 0 1 2; do
    slicegaussian fit "data/cremi/cremi_${v}.tif" --spacing 40,4,4 --config configs/paper.json --seed "$s" \
      --out "runs/cremi_${v}/seed_${s}" --quiet
  done
done
python scripts/summarize.py runs
