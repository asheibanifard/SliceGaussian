#!/usr/bin/env bash
# SAD-GS adapted to slice volumes (baselines/sadgs3d), seed 0, 30,000 iterations: (1) budget 10^6 (the operating point
# of the GaussianPile and 3DGS rows) on CREMI A/B/C and BBBC032 nuclei/membrane; (2) own count (6M memory limit) on CREMI.
cd "$(dirname "$0")/../.."
run() {  # volume spacing out cap
  [ -f $3/metrics.json ] && return
  mkdir -p $3; echo "[$(date +%H:%M)] start $3"
  micromamba run -n gaussian-volume python baselines/sadgs3d/sadgs3d.py $1 --spacing $2 --out $3 --cap $4 > $3/train.log 2>&1
  echo "[$(date +%H:%M)] exit $? $3"
}
for v in a b c; do run data/cremi/cremi_$v.tif 40,4,4 runs/baselines_cremi/cremi_$v/sadgs3d_1m/seed_0 1000000; done
for v in nuclei membrane; do run data/bbbc032/bbbc032_$v.tif 500,101,101 runs/bbbc032/$v/sadgs3d_1m/seed_0 1000000; done
for v in a b c; do run data/cremi/cremi_$v.tif 40,4,4 runs/baselines_cremi/cremi_$v/sadgs3d_own/seed_0 6000000; done
echo "[$(date +%H:%M)] sadgs done"
