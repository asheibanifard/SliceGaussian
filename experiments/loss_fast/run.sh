#!/usr/bin/env bash
# Loss ablation and coefficient sweep with the current implementation (raw-parameter CUDA kernels), seeds 0-2.
# Complete loss: runs/remeasure_fast (CREMI A/B/C seeds 0-2, EPFL seed 0) plus EPFL seeds 1-2 here.
cd "$(dirname "$0")/../.."
R=runs/loss_fast
SG=~/micromamba/envs/gaussian-volume/bin/slicegaussian
fit() {  # name volume spacing variant seed
  out=$R/$1/$4/seed_$5
  [ -f $out/final.json ] && return
  mkdir -p $(dirname $out); echo "[$(date +%H:%M)] $1 $4 seed $5"
  $SG fit $2 --spacing $3 --config $R/configs/$4.json --seed $5 --out $out --quiet > $out.log 2>&1 || echo "FAILED $out"
}
declare -A VOL=([cremi_a]="data/cremi/cremi_a.tif 40,4,4" [cremi_b]="data/cremi/cremi_b.tif 40,4,4"
                [cremi_c]="data/cremi/cremi_c.tif 40,4,4" [epfl]="data/epfl/epfl_ca1.tif 5,5,5")
for s in 0 1 2; do
  for v in cremi_a cremi_b cremi_c epfl; do
    for k in vox vox_grad vox_multi vox_reg; do fit $v ${VOL[$v]} $k $s; done
  done
  [ $s -gt 0 ] && { mkdir -p $R/epfl/full; [ -f $R/epfl/full/seed_$s/final.json ] || { echo "[$(date +%H:%M)] epfl full seed $s"; \
     $SG fit data/epfl/epfl_ca1.tif --spacing 5,5,5 --config configs/paper.json --seed $s --out $R/epfl/full/seed_$s --quiet > $R/epfl/full/seed_$s.log 2>&1 || echo "FAILED epfl full $s"; }; }
  for v in cremi_a epfl; do
    for k in grad0.3 grad1 multi0.3 multi1 reg0.3 reg1; do fit $v ${VOL[$v]} $k $s; done
  done
done
echo "[$(date +%H:%M)] all done"
