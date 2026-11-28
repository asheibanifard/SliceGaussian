#!/usr/bin/env bash
# EPFL CA1 and SNEMI3D: (1) Gaussian baselines at SliceGaussian's count and (2) rate-distortion of every representation.
# Starts after the CREMI rate-distortion run (runs/rate_distortion/finish.sh) has left the GPU. Finished steps are
# skipped, so the script can be restarted. Seed 0 throughout.
cd "$(dirname "$0")/.."
PY="micromamba run -n gaussian-volume python"
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
while pgrep -f "rate_distortion/finish.sh" > /dev/null; do sleep 60; done
log "GPU free, starting"

declare -A VOL=([epfl]=data/epfl/epfl_ca1.tif [snemi3d]=data/snemi3d/snemi3d.tif)
declare -A SP=([epfl]=5,5,5 [snemi3d]=29,6,6)
# SliceGaussian's final count (complete method, seed 0; runs/neural/<volume>/slicegaussian/seed_0/final.json)
declare -A N=([epfl]=607351 [snemi3d]=1000000)

# 1. Baselines at equal count: initial population = cap = SliceGaussian's final count, as on CREMI (runs/matched_count).
for v in epfl snemi3d; do
  for m in gaussianpile 3dgs sadgs3d; do
    out=runs/matched_count/$v/$m/seed_0
    [ -f $out/metrics.json ] && { log "skip $out"; continue; }
    mkdir -p $out; log "start $out"
    if [ $m = sadgs3d ]; then
      $PY baselines/sadgs3d/sadgs3d.py ${VOL[$v]} --spacing ${SP[$v]} --seed 0 --initial-n ${N[$v]} --cap ${N[$v]} \
        --out $out > $out.log 2>&1
    else
      $PY scripts/gaussian_baseline.py ${VOL[$v]} --spacing ${SP[$v]} --method $m --valid-only --seed 0 \
        --initial-n ${N[$v]} --cap ${N[$v]} --out $out > $out.log 2>&1
    fi
    log "exit $? $out"
  done
done

# 2a. Extra operating points: SliceGaussian at tau_q 0.8 / 0.4 / 0.2 / 0.05 (0.1 is the main-comparison run) and
#     GaussianPile / 3DGS from 1e5 and 3e5 initial Gaussians (1e6 is the main-comparison run), as on CREMI.
for v in epfl snemi3d; do
  for t in 0.8 0.4 0.2 0.05; do
    out=runs/rate_distortion/models/$v/slicegaussian/tau_$t
    [ -f $out/model.pt ] && [ -f $out/final.json ] && { log "skip $out"; continue; }
    mkdir -p $(dirname $out); log "start $out"
    micromamba run -n gaussian-volume slicegaussian fit ${VOL[$v]} --spacing ${SP[$v]} \
      --config runs/rate_distortion/configs/tau_$t.json --seed 0 --out $out --quiet > $out.log 2>&1
    log "exit $? $out"
  done
  for m in gaussianpile 3dgs; do
    for n in 100000 300000; do
      out=runs/rate_distortion/models/$v/$m/initial_$n
      [ -f $out/model.pt ] && [ -f $out/metrics.json ] && { log "skip $out"; continue; }
      mkdir -p $out; log "start $out"
      $PY scripts/rd_gaussian_sweep.py ${VOL[$v]} --spacing ${SP[$v]} --method $m --initial-n $n --out $out \
        > $out/console.log 2>&1
      log "exit $? $out"
    done
  done
done

# 2b. Encode, decode and score every operating point (CoordNet follows in step 3).
log "start HEVC"; $PY scripts/rd_measure_hevc.py --volumes epfl,snemi3d >> runs/rate_distortion/hevc.log 2>&1; log "exit $? HEVC"
log "start torch RD"
$PY scripts/rd_measure_torch.py --volumes epfl,snemi3d --methods slicegaussian,gaussianpile,3dgs,neurcomp \
  >> runs/rate_distortion/torch.log 2>&1
log "exit $? torch RD"
log "start INIF RD"
micromamba run -n inif python scripts/rd_measure_inif.py --volumes epfl,snemi3d >> runs/rate_distortion/inif.log 2>&1
log "exit $? INIF RD"

# 3. CoordNet was never trained on these volumes: train it with its CREMI settings (about 1 h per volume), then score.
log "start CoordNet training"
$PY scripts/reproduce_neural.py --volumes epfl snemi3d --methods coordnet >> runs/neural/driver.log 2>&1
log "exit $? CoordNet training"
$PY scripts/rd_measure_torch.py --volumes epfl,snemi3d --methods coordnet >> runs/rate_distortion/torch.log 2>&1
log "exit $? CoordNet RD"
log "all done"
