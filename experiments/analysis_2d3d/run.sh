#!/usr/bin/env bash
# Native 3D vs image-space (2D per slice) analysis in Phase I, CREMI A/B/C, seed 0: both matched by bisection on
# tau_q to the default Phase I count, one Phase II of 30,000 iterations, no re-estimation (Table 1 protocol).
cd "$(dirname "$0")/../.."
for v in a b c; do
  for an in slice volume; do
    out=runs/analysis_2d3d/cremi_$v/$an
    [ -f $out/final.json ] && continue
    mkdir -p runs/analysis_2d3d/cremi_$v
    echo "[$(date +%H:%M)] start $out"
    micromamba run -n gaussian-volume slicegaussian fit data/cremi/cremi_$v.tif --spacing 40,4,4 \
      --config runs/analysis_2d3d/${an}_$v.json --seed 0 --out $out --quiet > $out.log 2>&1
    echo "[$(date +%H:%M)] exit $? $out"
  done
done
echo "[$(date +%H:%M)] 2d3d done"
