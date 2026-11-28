#!/usr/bin/env bash
# Stronger loss weights: L_vox + beta L_term, beta in {0.3, 1} for L_grad and L_multi, and beta_r = 0.3;
# complete method (3 cycles with re-estimation, seed 0) on CREMI A and EPFL CA1.
cd "$(dirname "$0")/../.."
export SLICEGAUSSIAN_JIT=1
for vol in "cremi_a data/cremi/cremi_a.tif 40,4,4" "epfl data/epfl/epfl_ca1.tif 5,5,5"; do
  set -- $vol
  for k in grad0.3 grad1 multi0.3 multi1 reg0.3; do
    out=runs/loss_sweep/$1/$k/seed_0
    [ -f $out/final.json ] && continue
    mkdir -p $(dirname $out); echo "[$(date +%H:%M)] start $out"
    micromamba run -n gaussian-volume slicegaussian fit $2 --spacing $3 --config runs/neural/loss_$k.json \
      --seed 0 --out $out --quiet > $(dirname $out)/seed_0.log 2>&1
    echo "[$(date +%H:%M)] exit $? $out"
  done
done
echo "[$(date +%H:%M)] loss sweep done"
