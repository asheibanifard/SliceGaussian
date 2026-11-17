# SAD-GS adapted to slice-based volumes (3D structure tensor)

Baseline for the paper's comparison: the densification of SAD-GS (Lyu et al., *Faster 3D Gaussian Splatting Convergence
via Structure-Aware Densification*, SIGGRAPH 2026) on the slice-adapted 3DGS of Table 1, with the structure tensor
computed on the 3D volume instead of on rendered images.

```bash
micromamba run -n gaussian-volume python baselines/sadgs3d/sadgs3d.py data/cremi/cremi_a.tif --spacing 40,4,4 \
    --out runs/baselines_cremi/cremi_a/sadgs3d/seed_0
```

Source: the official code, SADGS repository commit `029814f` (before any local volume changes):
`train.py` (densification block), `utils/freq_utils.py::update_freq_stats_online`,
`utils/loss_utils.py::get_multiscale_structure_tensor_v1`, `scene/gaussian_model.py::densify_and_prune_structgs`,
`densify_and_split_structgs`, `densify_and_clone_structgs`.

## Base model (unchanged from the 3DGS row)

10^6 random initial Gaussians with a grey value each (GaussianPile initialiser, same seed), slice rendering by masking
Gaussians with |z - z_k| > 3 sigma_z and orthographic alpha compositing (gsplat), L1 + 0.2 D-SSIM, Adam with GaussianPile's
learning rates decayed to 0.1x, 30,000 iterations, one slice per iteration. Scored with `slicegaussian.metrics.whole`
on the valid slices, as every method in the paper.

## Kept from SAD-GS

| Component | SAD-GS | Here |
| --- | --- | --- |
| Structure tensor | multiscale v1: base tensor at sigma 1 px, 4 levels, octave 1.5, band weights band^3, trace-normalised, scaled by band frequency^2 | the same algorithm in 3D on the volume (6 components); scales in in-plane pixels, converted to physical spacing per axis |
| Violation | eta_k = projected 1-sigma axis length / shortest local wavelength, wavelength = 1 / sqrt(lambda_1) ("wavelength" mode, default) | eta_k = 3D 1-sigma axis length (in-plane pixels) / 1 / sqrt(lambda_1 of the 3x3 tensor) |
| Sampling | random point in the projected 1-sigma ellipse | random point in the 1-sigma ellipsoid |
| Per-view votes | high if max_k eta > 1, low if <= 0.1; active if opacity > 0.05 and screen gradient > freq threshold | the same; a view is a slice |
| Split rule | split if high in > 80 % of views and gradient >= gate, or abs-gradient test; clone small (max scale <= 0.001 extent); k_k = ceil(sqrt(max eta_k)) children per axis on a grid of spacing sqrt(12) s / k, scales s / k; importance: seen in >= 1 view | the same |
| Pruning | low in > 80 % of views; opacity < 0.1; screen radius > 20 px or scale > 0.1 extent after iteration 3000; opacity < 0.1 at iterations 4000, 8000; opacities capped at 0.8 after each densification | the same |
| Schedule | densify from 500 to 15,000 | the same; interval 500, as every released SAD-GS configuration |

## Translated

* **Gradient units.** SAD-GS's thresholds are set on 3DGS screen-space gradients (2e-4 = 3DGS's standard, 2e-5 for the
  frequency votes, 1e-5 as the split gate). The base model densifies at 5e-5 on the 2D-mean gradient in pixels, so the
  same ratios are used in pixels: 5e-5, 5e-6, 2.5e-6. With the absolute values the population exceeded 5 million
  Gaussians within 3,000 iterations.
* **Views.** A view is a slice. A Gaussian is visible only on its own few slices, so the per-view statistics are
  accumulated at every iteration instead of every 10th.

## Removed

* **Opacity reset** (3DGS, x0.1 every 3,000 iterations). With one slice per iteration a Gaussian receives a gradient only
  when its slice is drawn; its opacity does not recover before the next densification and the min-opacity prune removes
  the whole population (observed twice: 3.6 M -> 2 k Gaussians). The base model has no reset.
* RGB and spherical harmonics, the 3D filter, the hybrid optimiser and its feature learning rates, bounding-box points,
  camera sampling: no counterpart for a grey scalar volume, or not used by the default configuration.
* A safety limit of 6 million Gaussians (GPU memory) that SAD-GS does not have; a run that reaches it is reported.
