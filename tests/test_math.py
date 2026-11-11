"""Checks of the mathematics. CPU tests always run; GPU tests (CUDA kernel, end to end) are skipped without CUDA."""
import math

import numpy as np
import pytest
import torch

from slicegaussian import phase1
from slicegaussian.field import covariance, quat_to_rot, rot_to_quat
from slicegaussian.reestimate import ls_coefficients

_trapz = getattr(np, "trapezoid", None) or np.trapz

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_rotation_roundtrip_and_covariance():
    g = torch.Generator().manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(500, 4, generator=g), dim=1)
    R = quat_to_rot(q)
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(500), atol=1e-5)
    assert torch.allclose(quat_to_rot(rot_to_quat(R)), R, atol=1e-5)
    s = torch.rand(500, 3, generator=g) + 0.1
    S = covariance(R, s)
    assert torch.allclose(S, S.transpose(1, 2), atol=1e-6)
    ev = torch.linalg.eigvalsh(S.double())
    assert torch.allclose(ev, (s ** 2).double().sort(1).values, rtol=1e-4)


def test_band_frequency_is_the_peak_of_the_dog_response():
    sa, sb = 1.0, math.sqrt(2.0)
    f = np.linspace(1e-4, 1.0, 200001)
    resp = np.exp(-2 * np.pi ** 2 * sa ** 2 * f ** 2) - np.exp(-2 * np.pi ** 2 * sb ** 2 * f ** 2)
    assert abs(f[resp.argmax()] - phase1.band_frequency(sa, sb)) < 1e-4


def test_children_grid():
    mu = torch.tensor([[1.0, 2.0, 3.0]])
    R = torch.eye(3)[None]
    s = torch.tensor([[2.0, 1.0, 1.0]])
    n = torch.tensor([[3, 2, 1]])
    parent, c_mu, c_s, off = phase1.children(mu, R, s, n, b=1.0)
    assert len(c_mu) == 6 and (parent == 0).all()
    assert torch.allclose(c_mu.mean(0), mu[0])                          # symmetric about the parent centre
    assert torch.allclose(c_s, s / n.float())
    xs = sorted(set(round(v, 6) for v in c_mu[:, 0].tolist()))
    assert np.allclose(np.diff(xs), 2 * 1.0 * 2.0 / 3)                  # spacing 2 b s / n along the split axis


def test_least_squares_children():
    tab = ls_coefficients(1.0, 4, device="cpu").numpy()
    assert abs(tab[1, 0] - 1.0) < 1e-6                                   # one child = the parent itself
    x = np.linspace(-6, 6, 4001)
    parent = np.exp(-x ** 2 / 2)
    for n in range(2, 5):
        t = ((np.arange(n) + 0.5) / n - 0.5) * 2
        approx = sum(tab[n, j] * np.exp(-(x - t[j]) ** 2 / (2 / n ** 2)) for j in range(n))
        fit = np.sqrt(_trapz((approx - parent) ** 2, x) / _trapz(parent ** 2, x))
        # least squares can only beat any other choice, e.g. the grid rule
        grid = sum(np.exp(-t[j] ** 2 / 2) * 2 / math.sqrt(2 * math.pi) * np.exp(-(x - t[j]) ** 2 / (2 / n ** 2))
                   for j in range(n))
        assert fit <= np.sqrt(_trapz((grid - parent) ** 2, x) / _trapz(parent ** 2, x)) + 1e-9


@cuda
def test_kernel_matches_direct_formula():
    from slicegaussian.field import GaussianField, reference_volume, render_volume
    from slicegaussian.volume import Volume
    g = torch.Generator().manual_seed(1)
    vol = Volume(np.random.default_rng(0).random((12, 40, 48)).astype(np.float32), (4.0, 1.0, 1.0))
    M = 300
    xyz = torch.rand(M, 3, generator=g) * torch.tensor([48.0, 40.0, 48.0])
    s = torch.rand(M, 3, generator=g) * 3 + 0.7
    q = torch.nn.functional.normalize(torch.randn(M, 4, generator=g), dim=1)
    amp = torch.randn(M, generator=g)
    field = GaussianField(xyz.cuda(), s.cuda(), q.cuda(), amp.cuda())
    fast = render_volume(field, vol, radius_sigma=6.0, cull_eps=0.0).double()
    ref = reference_volume(field.xyz.detach(), field.scales.detach(), field.rotation.detach(), field.amp.detach(), vol)
    assert (fast - ref).abs().max() < 1e-4 * max(1.0, ref.abs().max().item())


@cuda
def test_kernel_gradients_match_autograd_of_direct_formula():
    from slicegaussian.field import GaussianField, covariance as cov, render_planes
    from slicegaussian.volume import Volume
    g = torch.Generator().manual_seed(2)
    vol = Volume(np.random.default_rng(1).random((6, 24, 24)).astype(np.float32), (2.0, 1.0, 1.0))
    M = 40
    field = GaussianField((torch.rand(M, 3, generator=g) * torch.tensor([24.0, 24.0, 12.0])).cuda(),
                          (torch.rand(M, 3, generator=g) * 2 + 0.8).cuda(),
                          torch.nn.functional.normalize(torch.randn(M, 4, generator=g), dim=1).cuda(),
                          torch.randn(M, generator=g).cuda())
    ks = [1, 3]
    w = torch.randn(len(ks), vol.H, vol.W, device="cuda")
    (render_planes(field, vol, ks, radius_sigma=8.0, cull_eps=0.0) * w).sum().backward()
    g_fast = {k: field.p[k].grad.clone() for k in field.PARAMS}
    for p in field.p.values():
        p.grad = None
    X = vol.centres(ks).reshape(-1, 3)
    P = torch.linalg.inv(cov(field.rotation, field.scales))
    d = X[:, None] - field.xyz[None]
    val = (field.amp[None] * torch.exp(-0.5 * torch.einsum("vgi,gij,vgj->vg", d, P, d))).sum(1)
    (val.view(len(ks), vol.H, vol.W) * w).sum().backward()
    for k in field.PARAMS:
        ref = field.p[k].grad
        assert torch.allclose(g_fast[k], ref, rtol=2e-3, atol=2e-3 * ref.abs().max().item()), k


@cuda
def test_end_to_end_small(tmp_path):
    from slicegaussian import Config, Volume, fit
    Z, H, W = 16, 64, 64
    z, y, x = np.meshgrid(np.arange(Z) * 4.0, np.arange(H), np.arange(W), indexing="ij")
    vol = Volume((0.5 + 0.4 * np.sin(2 * np.pi * x / 12) * np.cos(2 * np.pi * z / 40)).astype(np.float32),
                 (4.0, 1.0, 1.0), name="synthetic")
    cfg = Config()
    cfg.reestimation.cycles, cfg.reestimation.cycle_iters = 2, 300
    cfg.p2.eval_every, cfg.p2.full_every = 300, 600
    field, final = fit(vol, str(tmp_path), cfg, verbose=False)
    assert final["psnr"] > final["initial_psnr"]
    for f in ("model.pt", "model.sgz", "final.json", "trace.csv", "progress.log", "config.json"):
        assert (tmp_path / f).exists(), f
