# Experiments

The run scripts and configurations of the experiments reported in the paper, one folder per experiment, copied
unchanged from the `runs/` folder in which they were run. Results, models and logs are not included.

Each script changes to the repository root and reads its configurations from `runs/<experiment>/`, so copy the folder
there before running it:

    cp -r experiments/sensitivity_fast runs/
    bash runs/sensitivity_fast/run.sh

The scripts call the `slicegaussian` command of the `gaussian-volume` environment (`environment.yml`) by its path,
`~/micromamba/envs/gaussian-volume/bin/slicegaussian`; change `SG=` at the top of a script if your environment is
elsewhere. The volumes are prepared with `scripts/prepare_cremi.py` and `scripts/prepare_neural.py` (see
`docs/running.md`). Finished runs are skipped, so a script can be restarted. The tables of the paper are produced from
the run outputs by the `scripts/summarize_*.py` scripts.

| Folder | Experiment |
|---|---|
| `remeasure_fast` | Complete method, three seeds, with the CUDA kernels (main results and timing) |
| `shared`, `matched_count`, `sadgs3d`, `gaussianpile_sigmaz` | Baselines, at their own and at equal count; GaussianPile axial width |
| `neural`, `neural_queue.sh` | EPFL CA1 and SNEMI3D; loss and smoothed-analysis runs on EPFL |
| `rate_distortion` | Rate--distortion of the Gaussian, neural and HEVC representations |
| `phase2_on_population` | The same Phase II on populations from eight estimators |
| `allocation_early` | Allocation arms with early schedules |
| `reest_ablation` | Re-estimation controls |
| `analysis_2d3d` | Volume versus per-slice analysis |
| `noise` | CREMI A with added noise |
| `loss_fast` | Loss ablation and weight sweep |
| `sensitivity_fast` | Parameter sensitivity and equal-count controls for the largest scale |
| `phantom`, `bbbc032`, `bbbc032_time.sh` | Synthetic phantom and BBBC032 runs |
