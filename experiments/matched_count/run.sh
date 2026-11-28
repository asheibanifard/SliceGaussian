#!/usr/bin/env bash
# Baselines at SliceGaussian's count (CREMI, seed 0): initial population = cap = the mean final count of the complete
# SliceGaussian (three seeds): A 519,148 / B 473,200 / C 673,912. Everything else as in the main comparison
# (GaussianPile with sigma_z = 0; slice-adapted 3DGS; SAD-GS with the budget filled by the largest violations first).
cd "$(dirname "$0")/../.."
R=runs/matched_count
declare -A N=([a]=519148 [b]=473200 [c]=673912)
run() {  # name command...
  out=$R/$1/seed_0; shift
  [ -f $out/metrics.json ] && return
  mkdir -p $out; echo "[$(date +%H:%M)] start $out"
  "$@" --out $out > $out.log 2>&1
  echo "[$(date +%H:%M)] exit $? $out"
}
for v in a b c; do
  n=${N[$v]}; vol=data/cremi/cremi_$v.tif
  for m in gaussianpile 3dgs; do
    run cremi_$v/$m micromamba run -n gaussian-volume python scripts/gaussian_baseline.py $vol --spacing 40,4,4 \
      --method $m --valid-only --seed 0 --initial-n $n --cap $n
  done
  run cremi_$v/sadgs3d micromamba run -n gaussian-volume python baselines/sadgs3d/sadgs3d.py $vol --spacing 40,4,4 \
    --seed 0 --initial-n $n --cap $n
done
echo "[$(date +%H:%M)] all done"
