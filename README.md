<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo-light.svg" alt="SliceGaussian" width="560">
  </picture>
</p>

# SliceGaussian

**Frequency-aware Gaussian representation of slice-based volumes.**

SliceGaussian represents a 3D image stack (for example serial-section electron microscopy) as a sum of anisotropic 3D
Gaussians. Unlike Gaussian splatting methods that start from a fixed population and densify during training, it
**estimates the Gaussian population from the volume before training**:

1. **Phase I: population estimation.** A multiscale band-pass analysis and a multiscale structure tensor give, at every
   location, the local wavelength along any direction and the principal directions of the structure. A grid of
   candidate Gaussians is compared with them through a per-axis *violation ratio* (Gaussian scale ÷ local wavelength,
   aggregated over the Gaussian's support), and each candidate is split only along the axes that need it. The number
   of Gaussians follows from the data; it is not a user setting.
2. **Phase II: optimisation with a fixed population.** Centres, scales, rotations and amplitudes are optimised slice by
   slice (one slice per iteration, a 2D loss), with no splitting or pruning inside the loop.
3. **Conditional re-estimation.** Between Phase II cycles, Gaussians that still violate the wavelength condition are
   subdivided again; the children's amplitudes are fitted by least squares so that the fitted field is disturbed as
   little as possible.

The field is evaluated by a fused CUDA kernel (forward and analytic backward), compiled on first use.

## Results on CREMI

3D PSNR / SSIM on the 125 × 256 × 256 crops of CREMI A, B and C (40 × 4 × 4 nm), mean ± std over 3 seeds, 3 cycles of
10,000 iterations (CREMI C scored on its 123 valid slices; slices 14 and 74 are black). One NVIDIA RTX A4000.

| Volume | 3D PSNR (dB) | 3D SSIM | Gaussians | Total time (Phase I included) |
|---|---|---|---|---|
| CREMI A | 39.05 ± 0.003 | 0.987 | 519k | 324 s |
| CREMI B | 37.29 ± 0.006 | 0.988 | 473k | 303 s |
| CREMI C | 39.64 ± 0.012 | 0.992 | 674k | 340 s |

For comparison, GaussianPile reaches 32.3 / 31.1 / 31.5 dB with about 1.0M Gaussians in about 208 s; SliceGaussian
reaches that fidelity after 1,500–2,100 iterations (18–23 s including Phase I). See the paper for the full comparison,
the ablations and the limitations.

## Installation

Requirements: Linux, an NVIDIA GPU, Python ≥ 3.9 and PyTorch ≥ 2.1 with a CUDA toolkit (`nvcc`) of the same CUDA
version, plus a host compiler that `nvcc` supports. The reference environment, in which the paper's results were
produced, is Python 3.11 + PyTorch 2.7.1 + CUDA 11.8 + gcc 11:

```bash
git clone https://github.com/asheibanifard/SliceGaussian.git
cd SliceGaussian
micromamba env create -f environment.yml          # or: conda env create -f environment.yml
micromamba activate slicegaussian
SLICEGAUSSIAN_BUILD_CUDA=1 pip install -e . --no-build-isolation
```

`SLICEGAUSSIAN_BUILD_CUDA=1` compiles the CUDA extension at install time (`slicegaussian._C`). Without it,
`pip install -e .` installs the Python package only, and the kernel is compiled from `src/slicegaussian/csrc/` on first
use (about a minute, then cached by PyTorch). `--no-build-isolation` lets the build see the installed PyTorch.

## Quick start

```bash
# 1. data: downloads CREMI A, B, C and writes the 125 x 256 x 256 crops used in the paper
python scripts/prepare_cremi.py --out data/cremi

# 2. fit (settings of the paper)
slicegaussian fit data/cremi/cremi_a.tif --spacing 40,4,4 --config configs/paper.json --out runs/cremi_a

# 3. render the fitted model back to a volume, or score it
slicegaussian render runs/cremi_a/model.pt --shape 125,256,256 --spacing 40,4,4 --out runs/cremi_a/recon.tif
slicegaussian eval   runs/cremi_a/model.pt data/cremi/cremi_a.tif --spacing 40,4,4
```

Any (Z, Y, X) stack in `.tif`, `.npy` or CREMI-style `.hdf` works; pixels must be square (`dy == dx`). Spacing is in
physical units (any unit, used consistently). `configs/fast.json` runs a single short cycle for a quick check.

Useful options of `slicegaussian fit`:

| Option | Meaning |
|---|---|
| `--seed N` | random seed (Phase I support sampling, slice order) |
| `--cycles`, `--cycle-iters` | Phase II cycles and iterations per cycle (default 3 × 10,000) |
| `--tau-q` | subdivision threshold: lower gives more Gaussians and higher fidelity (default 0.1) |
| `--target-psnr` | stop at the end of a cycle once this 3D PSNR is reached |
| `--snapshot-every N` | save ground truth / prediction / difference images of three slices every N iterations |
| `--eval-every N` | whole-volume evaluation interval (excluded from the reported time) |

### Python API

```python
from slicegaussian import Config, fit, load_volume

vol = load_volume("data/cremi/cremi_a.tif", spacing=(40, 4, 4))
cfg = Config.load("configs/paper.json")
field, final = fit(vol, "runs/cremi_a", cfg)
print(final["psnr"], len(field))
```

## Outputs

Each run folder contains:

| File | Content |
|---|---|
| `final.json` | final metrics: 3D PSNR, SSIM, MAE, gradient and band errors (valid slices; `all_*` on all slices), Gaussian count, times, encoded size and decoded fidelity |
| `trace.csv` | the same metrics at every evaluation, with the loss terms and peak GPU memory |
| `progress.log` | a readable log, including each cycle's adequacy check and re-estimation |
| `run.json` | Phase I statistics and stage timings, and every re-estimation event |
| `model.pt` | the Gaussians (centres, scales, unit quaternions, amplitudes; float32) |
| `model.sgz` | the encoded model (16-bit positions, float16 attributes, lzma) |
| `config.json` | all settings, the volume description and the software/GPU environment |
| `slices/` | snapshot images, if requested |

## Reproducing the paper

```bash
bash scripts/reproduce_cremi.sh      # CREMI A, B, C x seeds 0, 1, 2; about 5-6 min per run on an RTX A4000
```

`scripts/summarize.py runs` prints mean ± std over seeds. Timings depend on the GPU. The baselines (GaussianPile,
slice-adapted 3DGS, NeurComp, CoordNet, INIF, HEVC) were run with their own code and are not part of this repository.

## Repository layout

```
src/slicegaussian/
  volume.py       loading, normalisation, valid-slice detection, sampling helpers (Stage 1)
  field.py        the Gaussian field, rotations, evaluators, amplitude calibration
  csrc/           C++/CUDA sources: fieldcut_kernel.cu (planar cut + rasterisation + analytic backward), fieldcut.cpp (bindings)
  cuda/fieldcut.py  autograd wrapper; loads the prebuilt extension or compiles csrc/ on first use
  phase1.py       Stages 2-6: frequency and structure analysis, violation ratio, anisotropic subdivision
  phase2.py       slice-based optimisation with a fixed population and the 2D loss
  reestimate.py   cycles, adequacy check, re-estimation with least-squares children
  metrics.py      3D PSNR, 3D SSIM, MAE, gradient and band errors
  codec.py        encoded size and decoding
  snapshots.py    ground truth / prediction / difference images
  pipeline.py     end-to-end fit() and the configuration
  cli.py          command-line interface
configs/          paper.json (settings of the paper), fast.json
environment.yml   reference conda/micromamba environment (Python 3.11, PyTorch 2.7.1, CUDA 11.8, gcc 11)
setup.py          optional ahead-of-time build of the CUDA extension
scripts/          CREMI download and crop, reproduction script, summary
tests/            checks of the mathematics and the kernel (GPU tests are skipped without CUDA)
docs/method.md    the method in more detail, with every implementation choice
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The GPU tests compare the CUDA kernel with the direct field formula (values and gradients) and run a short end-to-end
fit on a synthetic volume.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`; the entry will be completed on publication).

## Acknowledgements

The CREMI data are from the MICCAI 2016 challenge on circuit reconstruction from electron microscopy images
(https://cremi.org). The violation ratio and anisotropic splitting adapt ideas from SAD-GS (Lyu et al., SIGGRAPH 2026)
to scalar volumes; the slice-based training follows GaussianPile (Kong et al., CVPR 2026).

## License

See [LICENSE](LICENSE).
