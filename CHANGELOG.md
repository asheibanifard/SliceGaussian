# Changelog

## 0.2.0 — 2026-11-28
Release for the journal version of the paper.
- Slices are rendered in CUDA directly from the raw parameters, visiting only the Gaussians that reach the slice.
- GaussianPile focus-width option, noise-share scoring and figure scripts.
- Rate--distortion (Gaussian codec, neural representations, HEVC) and time-to-quality scripts, extended to EPFL CA1
  and SNEMI3D.
- Scripts for the loss study, parameter sensitivity with equal-count controls, and the synthetic wavelength diagnostic.
- `experiments/`: the run scripts and configurations of every experiment in the paper.

## 0.1.0 — 2026-11-11
- First release: Phase I population estimation, slice-based Phase II, three-cycle re-estimation with least-squares
  children, fused CUDA field kernel, metrics, codec, command-line interface, CREMI scripts and tests.
