#!/bin/bash
# Finish the rate-distortion measurements: Torch representations on CREMI A/B/C (finished operating points are
# skipped via points.json), then INIF with its official decoder.
cd "$(dirname "$0")/../.."
echo "[$(date +%H:%M)] torch start"
micromamba run -n gaussian-volume python scripts/rd_measure_torch.py --samples abc >> runs/rate_distortion/torch.log 2>&1
echo "[$(date +%H:%M)] torch exit $?"
micromamba run -n inif python scripts/rd_measure_inif.py >> runs/rate_distortion/inif.log 2>&1
echo "[$(date +%H:%M)] inif exit $?"
