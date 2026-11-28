#!/usr/bin/env bash
# Framework experiment (Table 2) on EPFL CA1 and SNEMI3D, seed 0: our Phase I + one Phase II (30,000 iterations, no
# re-estimation), and our Phase II on the final Gaussians of GaussianPile, 3DGS and SAD-GS (scripts/phase2_on_population.py).
cd "$(dirname "$0")/../.."
export SLICEGAUSSIAN_JIT=1
for v in "epfl data/epfl/epfl_ca1.tif 5,5,5" "snemi3d data/snemi3d/snemi3d.tif 29,6,6"; do
  set -- $v
  out=runs/phase2_on_population/$1/phase1
  if [ ! -f $out/final.json ]; then
    mkdir -p $out; echo "[$(date +%H:%M)] start $out"
    micromamba run -n gaussian-volume slicegaussian fit $2 --spacing $3 --config runs/phase2_on_population/phase1_single_cycle.json \
      --seed 0 --out $out --quiet > $out.log 2>&1
    echo "[$(date +%H:%M)] exit $? $out"
  fi
  for m in gaussianpile 3dgs sadgs3d; do
    out=runs/phase2_on_population/$1/$m
    [ -f $out/result.json ] && continue
    mkdir -p $out; echo "[$(date +%H:%M)] start $out"
    micromamba run -n gaussian-volume python scripts/phase2_on_population.py --method $m \
      --model runs/neural/$1/$m/seed_0/model.pt --volume $2 --spacing $3 --out $out > $out/run.log 2>&1
    echo "[$(date +%H:%M)] exit $? $out"
  done
done
echo "[$(date +%H:%M)] p2pop neural done"
