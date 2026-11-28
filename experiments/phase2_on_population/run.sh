#!/usr/bin/env bash
# Our Phase II (30,000 iterations, no re-estimation) on the final Gaussians of GaussianPile, 3DGS and SAD-GS
# (CREMI A/B/C, the models behind Table 1; C: the valid-slice retrains). See scripts/phase2_on_population.py.
cd "$(dirname "$0")/../.."
export SLICEGAUSSIAN_JIT=1
F=../freqpile/results/paper
for v in a b c; do
  V=$(echo $v | tr a-z A-Z)
  if [ $v = c ]; then gp=runs/baselines_cremi/cremi_c/gaussianpile_validonly/seed_0/model.pt; g3=runs/baselines_cremi/cremi_c/3dgs_validonly/seed_0/model.pt
  else gp=$F/CREMI_$V/final_experiments_gaussianpile/seed_0/model.pt; g3=$F/CREMI_$V/final_experiments_3dgs/seed_0/model.pt; fi
  for m in "gaussianpile $gp" "3dgs $g3" "sadgs3d runs/baselines_cremi/cremi_$v/sadgs3d_1m/seed_0/model.pt"; do
    set -- $m
    out=runs/phase2_on_population/cremi_$v/$1
    [ -f $out/result.json ] && continue
    mkdir -p $out; echo "[$(date +%H:%M)] start $out"
    micromamba run -n gaussian-volume python scripts/phase2_on_population.py --method $1 --model $2 \
      --volume data/cremi/cremi_$v.tif --spacing 40,4,4 --out $out > $out/run.log 2>&1
    echo "[$(date +%H:%M)] exit $? $out"
  done
done
echo "[$(date +%H:%M)] p2pop done"
