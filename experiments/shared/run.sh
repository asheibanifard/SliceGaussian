#!/bin/bash
# Shared protocol: every method keeps its own loss and learning rates; shared are 30k iterations, valid training
# slices, the Gaussian count (initial = SliceGaussian's Phase I count, cap = its final count), evaluation every 100
# iterations with slicegaussian.metrics, seed 0. SliceGaussian itself: the default runs (runs/wallclock, runs/bbbc032,
# runs/neural).
cd "$(dirname "$0")/../.."
R=runs/shared
while read name vol sp n0 cap; do
  for m in gaussianpile 3dgs sadgs3d; do
    out=$R/$name/$m/seed_0
    [ -f $out/metrics.json ] && { echo "skip $out"; continue; }
    mkdir -p $out; echo "[$(date +%H:%M)] start $out"
    if [ $m = sadgs3d ]; then
      micromamba run -n gaussian-volume python baselines/sadgs3d/sadgs3d.py $vol --spacing $sp --out $out \
        --initial-n $n0 --cap $cap --log-every 100 > $out.log 2>&1
    else
      micromamba run -n gaussian-volume python scripts/gaussian_baseline.py $vol --spacing $sp --method $m --out $out \
        --initial-n $n0 --cap $cap --valid-only --log-every 100 > $out.log 2>&1
    fi
    echo "[$(date +%H:%M)] exit $? $out"
  done
done <<'L'
cremi_a data/cremi/cremi_a.tif 40,4,4 419707 516786
cremi_b data/cremi/cremi_b.tif 40,4,4 358757 472136
cremi_c data/cremi/cremi_c.tif 40,4,4 496167 674309
bbbc032_nuclei data/bbbc032/bbbc032_nuclei.tif 500,101,101 588413 999993
bbbc032_membrane data/bbbc032/bbbc032_membrane.tif 500,101,101 558302 999995
epfl data/epfl/epfl_ca1.tif 5,5,5 213428 607351
snemi3d data/snemi3d/snemi3d.tif 29,6,6 603833 1000000
L
echo "[$(date +%H:%M)] shared done"
