#!/bin/bash
# Dense (every 100 it) time-to-PSNR traces for GaussianPile and 3DGS on BBBC032, seed 0.
set -e
cd "$(dirname "$0")/.."
for v in nuclei membrane; do for m in 3dgs gaussianpile; do
  out=runs/bbbc032_time/$v/$m/seed_0
  [ -f $out/metrics.json ] && { echo "skip $v/$m"; continue; }
  mkdir -p $out; echo "start $v/$m $(date +%T)"
  micromamba run -n gaussian-volume python -u scripts/gaussian_baseline.py data/bbbc032/bbbc032_$v.tif \
    --spacing 500,101,101 --method $m --seed 0 --log-every 100 --out $out > $out.log 2>&1
  echo "done $v/$m $(date +%T)"
done; done
