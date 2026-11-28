#!/usr/bin/env bash
# Ours with a fixed 1 px in-plane smoothing of the analysed volume (Phase I only; training and scoring on raw data).
cd "$(dirname "$0")/../.."
export SLICEGAUSSIAN_JIT=1
for v in "epfl data/epfl/epfl_ca1.tif 5,5,5" "snemi3d data/snemi3d/snemi3d.tif 29,6,6"; do
  set -- $v
  out=runs/neural/$1/slicegaussian_presmooth1/seed_0
  [ -f $out/final.json ] && continue
  mkdir -p $(dirname $out); echo "[$(date +%H:%M)] start $out"
  micromamba run -n gaussian-volume slicegaussian fit $2 --spacing $3 --config runs/noise/configs/presmooth1.json \
    --seed 0 --out $out --quiet > $(dirname $out)/seed_0.log 2>&1
  echo "[$(date +%H:%M)] exit $? $out"
done
echo "[$(date +%H:%M)] presmooth done"
