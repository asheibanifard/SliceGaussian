#!/usr/bin/env bash
# Reviewer round 2, weakness 3 (noise). (1) BBBC032: fixed in-plane Gaussian denoising (1 or 2 px) of the analysed
# volume only; training and scoring on the raw data. (2) CREMI A with added Gaussian noise (std 0.02, 0.05):
# SliceGaussian at tau_q 0.2 / 0.1 / 0.05, with 1 px analysis denoising at tau_q 0.1, and 3DGS. Seed 0.
cd "$(dirname "$0")/../.."
R=runs/noise
sg() {  # volume spacing config out
  [ -f $4/final.json ] && return
  mkdir -p $(dirname $4); echo "[$(date +%H:%M)] start $4"
  micromamba run -n gaussian-volume slicegaussian fit $1 --spacing $2 --config $R/configs/$3.json --seed 0 --out $4 --quiet > $4.log 2>&1
  echo "[$(date +%H:%M)] exit $? $4"
}
for v in nuclei membrane; do for ps in 1 2; do
  sg data/bbbc032/bbbc032_$v.tif 500,101,101 presmooth$ps $R/bbbc032_$v/presmooth$ps
done; done
for n in 0.02 0.05; do
  for t in 0.2 0.1 0.05; do sg data/cremi_noise/cremi_a_n$n.tif 40,4,4 tau$t $R/cremi_a_n$n/tau$t; done
  sg data/cremi_noise/cremi_a_n$n.tif 40,4,4 presmooth1 $R/cremi_a_n$n/presmooth1
  out=$R/cremi_a_n$n/3dgs
  if [ ! -f $out/metrics.json ]; then
    echo "[$(date +%H:%M)] start $out"
    micromamba run -n gaussian-volume python scripts/gaussian_baseline.py data/cremi_noise/cremi_a_n$n.tif --spacing 40,4,4 \
      --method 3dgs --out $out > $out.log 2>&1
    echo "[$(date +%H:%M)] exit $? $out"
  fi
done
echo "[$(date +%H:%M)] noise done"
