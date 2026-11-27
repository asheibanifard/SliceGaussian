"""Train GaussianPile or 3DGS with its native recipe on any slice stack and score it with SliceGaussian's metrics.

    python scripts/gaussian_baseline.py VOLUME --spacing DZ,DY,DX --method gaussianpile|3dgs --out DIR [--seed N]

The reference modules (freqpile/experiments/final_experiments_{gaussianpile,3dgs}.py) are imported unchanged; only
their fixed CREMI spacing and their evaluation are replaced. Initialisation (10^6 random Gaussians), optimiser,
density control, supervision and the 30,000 iterations are the paper settings. Evaluation (every 5,000 iterations and
at the end, excluded from train_seconds) is slicegaussian.metrics.whole on the valid slices, as for SliceGaussian.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def naive_render(mod, torch):
    """Naive all-in-focus 3DGS (the "3DGS" of the GaussianPile paper, Sec. 2.2 / Fig. 1b; its code is unpublished).

    Slice k is rendered by the unchanged 3DGS rasterizer from an orthographic camera at z = (k - 1/2) dz looking along
    +z: every Gaussian in front of the camera contributes with its full projected footprint, whatever its depth, and
    is alpha-composited front to back. There is no axial mask; only the camera position differs between slices.
    Everything else (model, colour, initialisation, optimiser, loss, densification) is the slice-adapted 3DGS's.
    """
    def render(model, k, dz, dx, H, W, means2d_full=None, radii_full=None):
        viewmat = torch.eye(4, device="cuda")[None]
        viewmat[0, 2, 3] = -(k - 0.5) * dz                         # camera half a slice behind slice k
        K = torch.tensor([[1.0 / dx, 0.0, 0.5], [0.0, 1.0 / dx, 0.5], [0.0, 0.0, 1.0]], device="cuda")[None]
        img, _, info = mod.rasterization(model.xyz, model.quats, model.scales, model.alphas, model.color[:, None],
                                         viewmat, K, W, H, camera_model="ortho", packed=False,
                                         near_plane=1e-6 * dz, far_plane=1e10, render_mode="RGB")
        idx = torch.arange(model.num, device="cuda")
        return img[0, ..., 0], idx, info["means2d"], info["radii"]
    return render


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volume")
    p.add_argument("--spacing", required=True, help="dz,dy,dx in physical units")
    p.add_argument("--method", choices=["gaussianpile", "3dgs", "3dgs_naive"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial-n", type=int, default=1_000_000)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--log-every", type=int, default=5000,
                   help="evaluation interval; 100 matches SliceGaussian's trace for time-to-PSNR")
    p.add_argument("--valid-only", action="store_true",
                   help="train on valid slices only (as SliceGaussian); the reference code draws from all Z slices")
    p.add_argument("--reference-root", type=Path, default=Path(__file__).resolve().parents[2] / "freqpile")
    p.add_argument("--sigma-z", default="0",
                   help="GaussianPile only: axial width of the focus model, in physical units, or 'dz' for the slice "
                        "spacing (the GaussianPile paper's recommended setting); 0 = zero-thickness planes (default)")
    p.add_argument("--cap", type=int, default=0,
                   help="shared protocol: final Gaussian budget; a densification step that would exceed it keeps the "
                        "candidates with the largest accumulated gradient (0: no cap, the native recipe)")
    a = p.parse_args()
    spacing = tuple(float(s) for s in a.spacing.split(","))
    sys.path[:0] = [str(a.reference_root / "experiments"), str(a.reference_root / "standalone"),
                    str(a.reference_root.parent / "gaussianpile")]
    import torch
    from slicegaussian import load_volume, metrics
    vol = load_volume(a.volume, spacing)
    mod = importlib.import_module("final_experiments_gaussianpile" if a.method == "gaussianpile"
                                  else "final_experiments_3dgs")              # 3dgs_naive reuses the 3DGS module
    gp = mod if a.method == "gaussianpile" else mod.fg     # the 3DGS module reads SPACING and LOG_EVERY from GaussianPile's
    gp.SPACING = spacing
    gp.LOG_EVERY = a.log_every
    sigma_z = spacing[0] if a.sigma_z == "dz" else float(a.sigma_z)
    if a.method == "gaussianpile":
        mod.SETTINGS = mod.RenderSettings(sigma_z=sigma_z, norm_mode="paper")
    elif sigma_z:
        raise SystemExit("--sigma-z applies to GaussianPile only")
    mod.fe.N0 = a.initial_n

    if a.method == "gaussianpile":
        @torch.no_grad()
        def evaluate(model, cams, stack):
            pred = mod.render_stack(model.means, model.scales, model.quats, model.alphas, cams, mod.SETTINGS)
            return metrics.whole(None, vol, full=True, pred=pred)
    else:
        @torch.no_grad()
        def evaluate(model, stack):
            pred = mod.render_stack(model, vol.Z, vol.dz, vol.dx, vol.H, vol.W)
            return metrics.whole(None, vol, full=True, pred=pred)
    mod.evaluate = evaluate
    if a.valid_only and vol.invalid_ids:
        valid = torch.tensor(vol.valid_ids)

        class _Torch:
            """The reference module's torch, except that randint(Z, ...) draws a valid slice index."""
            def __getattr__(self, name):
                return getattr(torch, name)

            @staticmethod
            def randint(high, size, *args, **kw):
                if high == vol.Z:
                    return valid[torch.randint(len(valid), size, *args, **kw)]
                return torch.randint(high, size, *args, **kw)
        mod.torch = _Torch()
    if a.method == "3dgs_naive":
        mod.render = naive_render(mod, torch)
    if a.cap:
        # Clone and split each add one Gaussian per candidate, so a step adds at most #candidates; pruning in the same
        # call only frees room, so the cap is never exceeded.
        model_cls = mod.GaussianModel if a.method == "gaussianpile" else mod.Model3DGS
        native = model_cls.densify_and_prune

        def capped(self, grad_thr, *args, **kw):
            with torch.no_grad():
                grads = self.grad_accum / self.denom.clamp_min(1)
                cand = torch.nonzero(grads >= grad_thr).squeeze(1)
                room = max(a.cap - self.num, 0)
                if cand.numel() > room:
                    drop = cand[torch.argsort(grads[cand], descending=True)[room:]]
                    self.grad_accum[drop] = 0.0
            return native(self, grad_thr, *args, **kw)
        model_cls.densify_and_prune = capped
    mod.train(a.volume, str(a.out), a.seed, SimpleNamespace(iterations=a.iterations, save_every=0))
    (a.out / "protocol.json").write_text(json.dumps({
        "method": a.method, "volume": vol.describe() if hasattr(vol, "describe") else str(a.volume),
        "spacing_zyx": spacing, "initial_n": a.initial_n, "iterations": a.iterations, "seed": a.seed,
        "cap": a.cap or None,
        "sigma_z": sigma_z if a.method == "gaussianpile" else None,
        "evaluation": f"slicegaussian.metrics.whole on valid slices, every {a.log_every} iterations and at the end",
        "training_slices": "valid only" if a.valid_only else "all Z (reference code)",
        "training": "unchanged native initialisation, optimiser, density control and supervision",
        "reference_source": mod.__file__}, indent=2))


if __name__ == "__main__":
    main()
