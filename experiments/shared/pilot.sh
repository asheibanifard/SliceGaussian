#!/bin/bash
# Pilot: SliceGaussian under the shared protocol (L1 + 0.2 D-SSIM, GaussianPile learning rates, one decay over 30k).
cd "$(dirname "$0")/../.."
for spec in "cremi_a data/cremi/cremi_a.tif 40,4,4" "bbbc032_nuclei data/bbbc032/bbbc032_nuclei.tif 500,101,101"; do
  set -- $spec; out=runs/shared/$1/slicegaussian/seed_0
  [ -f $out/final.json ] && continue
  mkdir -p $(dirname $out); echo "[$(date +%H:%M)] start $out"
  micromamba run -n gaussian-volume slicegaussian fit $2 --spacing $3 --config configs/shared.json --seed 0 --out $out --quiet > $out.log 2>&1
  echo "[$(date +%H:%M)] exit $? $out"
done
echo "[$(date +%H:%M)] pilot done"
