#!/usr/bin/env bash
# Table 5 (loss ablation) on EPFL CA1: the complete method (3 cycles with re-estimation, seed 0) with each loss.
cd "$(dirname "$0")/../.."
export SLICEGAUSSIAN_JIT=1
for v in vox vox_grad vox_multi; do
  out=runs/neural/epfl/slicegaussian_$v/seed_0
  [ -f $out/final.json ] && continue
  mkdir -p $(dirname $out); echo "[$(date +%H:%M)] start $out"
  micromamba run -n gaussian-volume slicegaussian fit data/epfl/epfl_ca1.tif --spacing 5,5,5 --config runs/neural/loss_$v.json \
    --seed 0 --out $out --quiet > $(dirname $out)/seed_0.log 2>&1
  echo "[$(date +%H:%M)] exit $? $out"
done
echo "[$(date +%H:%M)] loss epfl done"
