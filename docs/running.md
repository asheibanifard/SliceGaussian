# Running SliceGaussian

SliceGaussian is run from the command line. Command-line arguments say what to run (volume, spacing, output folder);
a `config.yml` file holds the method's settings. This guide assumes the package is installed as described in the
[README](../README.md#installation). Fitting, rendering and scoring need an NVIDIA GPU.

## 1. Run it

From the root of the repository:

```bash
# download the CREMI volumes and write the crops used in the paper
python scripts/prepare_cremi.py --out data/cremi

# fit
slicegaussian fit data/cremi/cremi_a.tif --spacing 40,4,4 --config configs/config.yml --out runs/cremi_a
```

A complete fit takes about 5–6 minutes on an RTX A4000. To check the installation first, run one short cycle:

```bash
slicegaussian fit data/cremi/cremi_a.tif --spacing 40,4,4 --config configs/fast.json --out runs/check
```

The first run also compiles the CUDA kernel (about a minute) if it was not built at installation.

## 2. Use your own volume

- **Format:** a stack in (Z, Y, X) order, as `.tif`, `.npy` or CREMI-style `.hdf`.
- **Spacing:** `--spacing DZ,DY,DX`, separated by commas without spaces. Any physical unit works. Pixels must be
  square (`DY` equal to `DX`).
- **Intensities** are rescaled to [0, 1] from the minimum and maximum of the volume.
- **Constant slices** (for example black ones) are detected and left out of training and of the main scores.

## 3. Change the settings

Copy the configuration, edit it, and pass the copy:

```bash
cp configs/config.yml my_config.yml
slicegaussian fit my_volume.tif --spacing 40,4,4 --config my_config.yml --out runs/mine
```

[`configs/config.yml`](../configs/config.yml) holds the settings of the paper, with a comment on each group. The ones
you are most likely to change are:

| Setting | Paper | Effect |
|---|---|---|
| `p1.tau_q` | 0.1 | lower gives more Gaussians and higher fidelity |
| `reestimation.cycles` | 3 | number of Phase II cycles |
| `reestimation.cycle_iters` | 10000 | iterations per cycle |
| `p1.max_gaussians`, `reestimation.budget` | 2000000, 1000000 | safety limits on the count; lower them if the GPU runs out of memory |
| `reestimation.target` | `null` | stop at the end of a cycle once this 3D PSNR (dB) is reached |
| `p2.beta_g`, `p2.beta_m`, `p2.beta_r` | 0.1 each | loss weights |
| `p2.eval_every` | 100 | iterations between whole-volume evaluations |
| `seed` | 0 | random seed |

A file may contain only the settings you change; the others keep the code's defaults:

```yaml
p1:
  tau_q: 0.2
reestimation:
  cycles: 1
```

Two things to know:

- A misspelled setting inside `p1`, `p2` or `reestimation` stops the run with an error. A misspelled name at the top
  level is ignored.
- Without `--config`, the code's defaults are used. They match the paper's settings except for `p2.eval_every` (500).

JSON files work in the same way (`configs/paper.json` has the same settings as `config.yml`).

## 4. Override a setting from the command line

Arguments take priority over the file, and the file over the defaults:

```bash
slicegaussian fit data/cremi/cremi_a.tif --spacing 40,4,4 --config configs/config.yml \
  --tau-q 0.2 --seed 1 --out runs/cremi_a_tau02_seed1
```

| Argument of `fit` | Meaning |
|---|---|
| `volume` | the volume to fit (required) |
| `--spacing DZ,DY,DX` | voxel spacing (required) |
| `--out DIR` | output folder (required); files of an earlier run there are overwritten |
| `--config FILE` | settings: `.yml`, `.yaml` or `.json` |
| `--seed N` | overrides `seed` |
| `--tau-q X` | overrides `p1.tau_q` |
| `--cycles N` | overrides `reestimation.cycles` |
| `--cycle-iters N` | overrides `reestimation.cycle_iters` |
| `--target-psnr X` | overrides `reestimation.target` |
| `--eval-every N` | overrides `p2.eval_every` |
| `--snapshot-every N` | saves ground truth / prediction / difference images every N iterations; set it here, not in the file |
| `--no-codec` | skips the encoded-size measurement and does not write `model.sgz` |
| `--quiet` | prints only the volume description and the final scores |

`slicegaussian fit --help` prints this list.

## 5. Read the results

The final scores are printed and saved in the output folder:

| File | Content |
|---|---|
| `final.json` | final scores, Gaussian count and times |
| `trace.csv`, `progress.log` | scores during training |
| `run.json` | Phase I statistics and re-estimation events |
| `model.pt`, `model.sgz` | the fitted Gaussians, and their encoded version |
| `config.json` | every setting of the run, including command-line overrides |
| `slices/` | images, with `--snapshot-every` |

To repeat a run, pass its saved settings: `--config runs/cremi_a/config.json`. The volume and spacing must be given
again.

## 6. Render or score a fitted model

```bash
slicegaussian render runs/cremi_a/model.pt --shape 125,256,256 --spacing 40,4,4 --out runs/cremi_a/recon.tif
slicegaussian eval   runs/cremi_a/model.pt data/cremi/cremi_a.tif --spacing 40,4,4
```

Use the spacing of the fit. `--shape` is Z,Y,X. The rendered volume has values in [0, 1] as float32; add `--uint8` for
8-bit.

## 7. Several volumes or seeds

```bash
for v in a b c; do
  for s in 0 1 2; do
    slicegaussian fit data/cremi/cremi_$v.tif --spacing 40,4,4 --config configs/config.yml --seed $s \
      --out runs/cremi_$v/seed_$s --quiet
  done
done
python scripts/summarize.py runs
```

This is what `scripts/reproduce_cremi.sh` does, with `configs/paper.json`.

## Common problems

| Message | Fix |
|---|---|
| `expected three comma-separated values (z,y,x)` | write `--spacing 40,4,4`, without spaces |
| `pixels must be square (dx == dy)` | resample the slices to square pixels |
| `unknown setting p1.…` | the name is misspelled or belongs to another group |
| CUDA out of memory | raise `tau_q`, or lower `max_gaussians` and `budget` |
