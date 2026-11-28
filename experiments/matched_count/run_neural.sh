#!/usr/bin/env bash
# Baselines at SliceGaussian's count on EPFL CA1 and SNEMI3D (seed 0): initial population = cap = the final count of
# SliceGaussian on that volume (runs/neural/<v>/slicegaussian/seed_0): EPFL 607,351; SNEMI3D 1,000,000 (the count at
# which its re-estimation stopped at the safety limit). Settings otherwise as in the main comparison.
cd "$(dirname "$0")/../.."
R=runs/matched_count
run() {  # name command...
  out=$R/$1/seed_0; shift
  [ -f $out/metrics.json ] && return
  mkdir -p $out; echo "[$(date +%H:%M)] start $out"
  "$@" --out $out > $out.log 2>&1
  echo "[$(date +%H:%M)] exit $? $out"
}
for spec in "epfl data/epfl/epfl_ca1.tif 5,5,5 607351" "snemi3d data/snemi3d/snemi3d.tif 29,6,6 1000000"; do
  set -- $spec; v=$1; vol=$2; sp=$3; n=$4
  for m in gaussianpile 3dgs; do
    run $v/$m micromamba run -n gaussian-volume python scripts/gaussian_baseline.py $vol --spacing $sp \
      --method $m --valid-only --seed 0 --initial-n $n --cap $n
  done
  run $v/sadgs3d micromamba run -n gaussian-volume python baselines/sadgs3d/sadgs3d.py $vol --spacing $sp \
    --seed 0 --initial-n $n --cap $n
done
echo "[$(date +%H:%M)] all done"
