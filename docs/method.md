# Method and implementation notes

This page maps the methodology to the code and lists every implementation choice the methodology leaves open.

## Field (Section 3.1) — `field.py`, `cuda/fieldcut.py`

V̂(x) = Σᵢ αᵢ exp(−½ (x − μᵢ)ᵀ Σᵢ⁻¹ (x − μᵢ)), Σᵢ = Rᵢ diag(sᵢ²) Rᵢᵀ, with αᵢ a peak amplitude (no normalisation, no
opacity, no compositing). Parameters: centre, log-scales, unit quaternion (normalised on use), amplitude.

On a plane z = z_k a 3D Gaussian factorises exactly, so rendering every plane evaluates V̂ at every voxel centre. The
CUDA kernel truncates each Gaussian at 3 standard deviations and skips plane contributions below 1e-5; the tests compare
it with the direct formula. The anisotropy ratio max(s)/min(s) is clamped at 1,000 during training (float32 accuracy of
the covariance).

## Phase I (Section 3.2) — `phase1.py`

| Stage | Implementation |
|---|---|
| 1 Preprocessing | min-max normalisation to [0, 1]; spacing-aware filters and differences; slices holding a single value are invalid, are filled by interpolation for the analysis only, and are never trained on |
| 2 Frequency | scale space from 0.5 to 16 × the smallest spacing, ratio √2; band D_s = V_s − V_{s+1}; local energy E_s = G_ρ * D_s² (ρ = 2σ_{s+1}); weights E_s² normalised; k_s = exact peak frequency of the DoG band; λ = 1.078 / k̄ (calibration on synthetic sinusoids) |
| 3 Structure | J_s = G_ρ * ∇V_s∇V_sᵀ, trace-normalised, aggregated with the band weights |
| 4 Matching | candidate parents on a grid of spacing 10 × the smallest spacing (40 nm on CREMI), isotropic std h/2, axes = eigenvectors of J̄; directional wavelength from k(x; u)² = Σ_s w_s k_s² uᵀ T̂_s u; violation ratio q_ik = s_ik / λ_k |
| 5 Aggregation | 64 samples uniform in the Mahalanobis ball of radius 2 (centre included), 90th percentile, at least 8 valid samples |
| 6 Subdivision | n_ik = clamp(⌈q̂_ik / τ_q⌉, 1, 4), τ_q = 0.1; at most 64 children per parent; children on the grid ((j + ½)/n − ½)·2b·s, std s/n, b = 1; amplitude V(centre)·(2b/√(2π))³, then one global least-squares amplitude scale |

The Gaussian count is the output of Stage 6. `max_gaussians` (2M) and the re-estimation `budget` (1M) are only
GPU-memory safety limits.

## Phase II (Section 3.3) — `phase2.py`

One slice per iteration, drawn uniformly from the valid slices. All loss terms are 2D on that slice:

L = L_vox + 0.1 L_grad + 0.1 L_multi + 0.1 L_reg

- L_vox: mean absolute error.
- L_grad: mean Euclidean norm of the in-plane gradient difference (forward differences in physical units).
- L_multi: mean absolute error after 2D Gaussian smoothing at 2 and 4 in-plane pixels.
- L_reg: mean |αᵢ|.

Adam; learning rates 1.6e-5 × scene extent (centres), 1e-3 (log-scales), 2e-4 (quaternions), 2.5e-4 (amplitudes), each
decayed exponentially to 0.1× over a cycle. The methodology evaluates the loss on voxel samples without fixing the
batch; one slice per iteration and the 2D loss are our protocol for slice-based data.

## Re-estimation — `reestimate.py`

Three cycles of 10,000 iterations, each with a fresh optimiser. After cycles 1 and 2 the violation ratio is recomputed
on the optimised Gaussians (with the cycle-0 analysis fields), and every Gaussian with max_k q̂ > τ_q is subdivided.
Child amplitudes are set by a closed-form per-axis least-squares fit of the parent's profile (`refit = "ls"`); the
alternative `"grid"` rule is kept for comparison. The methodology specifies where to subdivide (the violation ratio) but
not when the representation is adequate; here an optional PSNR target stops the run, and otherwise the fixed number of
cycles is used.
