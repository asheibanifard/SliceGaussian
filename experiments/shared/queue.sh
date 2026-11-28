#!/bin/bash
# Wait until the EPFL/SNEMI3D driver has left the GPU, then start the shared-protocol runs.
cd "$(dirname "$0")/../.."
while pgrep -f "scripts/reproduce_neural.py" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] GPU free, starting"
exec runs/shared/run.sh
