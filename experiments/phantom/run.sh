#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export SLICEGAUSSIAN_JIT=1   # JIT-compiled CUDA extension from src/slicegaussian/csrc for both methods
V=data/phantom/phantom_em.tif
echo "[$(date +%H:%M)] start ours"
micromamba run -n gaussian-volume slicegaussian fit $V --spacing 40,4,4 --config configs/paper.json --seed 0 --out runs/phantom/ours --quiet > runs/phantom/ours.log 2>&1
echo "[$(date +%H:%M)] exit $? ours"
echo "[$(date +%H:%M)] start gs3d"
micromamba run -n gaussian-volume python baselines/gs3d/gs3d.py $V --spacing 40,4,4 --out runs/phantom/gs3d > runs/phantom/gs3d.log 2>&1
echo "[$(date +%H:%M)] exit $? gs3d"
echo "[$(date +%H:%M)] phantom done"
