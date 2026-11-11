"""Conditional re-estimation (methodology: Algorithm, steps 10-11): Phase II -> Phase I -> Phase II.

The run is a sequence of `cycles` Phase II runs of `cycle_iters` iterations, each with a fresh optimiser and its own
full learning-rate decay. After each cycle the adequacy measures are recorded: whole-volume 3D PSNR, its gain over the
cycle, and Q_remaining (the fraction of Gaussians whose largest violation ratio exceeds tau_q). If a target PSNR is given
and reached, the run stops. Otherwise, before the next cycle, Phase I is applied to the optimised Gaussians (Stages 4-6
with the analysis fields of cycle 0): every Gaussian with max_k q_hat > tau_q is subdivided along its violating axes,
largest violation first, while the count stays within `budget` (a GPU-memory safety limit). The population never
changes inside Phase II.

Children keep the parent's frame, sit on the Stage 6 grid and have std s / n. Their amplitudes are set by

  refit = "ls"    (default) least squares: per axis, the n child amplitudes minimise the L2 error between the parent's
                  1D profile exp(-x^2 / 2 s^2) and the sum of the children, in closed form (Gaussian overlap integrals);
                  the 3D factor is the product over the axes. This keeps the fitted field as unchanged as the child
                  geometry allows, so the PSNR drop at a re-estimation is much smaller.
  refit = "grid"  the parent field at the child centre x (2b / sqrt(2 pi))^d over the d subdivided axes.

Surviving Gaussians keep their parameters; the optimiser is rebuilt at the start of every cycle.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics, phase1, phase2
from .field import GaussianField, Timer, calibrate_amplitude

_LS = {}


def ls_coefficients(b, n_max, device="cuda"):
    """Least-squares child amplitude factors per axis, (n_max + 1, n_max): row n holds the n factors a_j minimising
    || exp(-x^2 / 2) - sum_j a_j exp(-(x - t_j)^2 / 2 (1/n)^2) ||_2, t_j = ((j + 1/2) / n - 1/2) 2b (unit parent std;
    the factors do not depend on the parent's scale)."""
    key = (b, n_max, str(device))
    if key not in _LS:
        tab = np.zeros((n_max + 1, n_max))
        for n in range(1, n_max + 1):
            sc = 1.0 / n
            t = np.array([((j + 0.5) / n - 0.5) * 2 * b for j in range(n)])
            G = math.sqrt(math.pi) * sc * np.exp(-(t[:, None] - t[None]) ** 2 / (4 * sc ** 2))
            r = math.sqrt(2 * math.pi) * sc / math.sqrt(1 + sc * sc) * np.exp(-t ** 2 / (2 * (1 + sc * sc)))
            tab[n, :n] = np.linalg.solve(G, r)
        _LS[key] = torch.tensor(tab, dtype=torch.float32, device=device)
    return _LS[key]


def _replace(field, keep, new):
    """Rebuilds the parameters: rows `keep` (bool) followed by `new` (dict of raw parameters)."""
    opt = field.opt
    for g in opt.param_groups:
        old, name = g["params"][0], g["name"]
        st = opt.state.pop(old, None)
        p = torch.nn.Parameter(torch.cat([old.detach()[keep], new[name].to(old.dtype)]).contiguous())
        if st is not None:
            for k in ("exp_avg", "exp_avg_sq"):
                st[k] = torch.cat([st[k][keep], torch.zeros_like(new[name], dtype=st[k].dtype)])
            opt.state[p] = st
        g["params"][0] = field.p[name] = p


@torch.no_grad()
def subdivide(field, vol, fields, p1, budget, seed, refit="ls"):
    """Phase I on the optimised Gaussians. Returns the number of subdivided Gaussians."""
    mu, R, s = field.xyz.detach(), field.rotation.detach(), field.scales.detach()
    q = phase1.violation(mu, R, s, fields, vol, p1, seed)
    n = torch.clamp(torch.ceil(q / p1.tau_q), min=1, max=p1.n_max).long()
    n = phase1._per_parent_budget(n, p1.max_children)
    n = phase1._global_budget(n, q.max(1).values, budget, len(mu))
    sub = n.prod(1) > 1
    if not sub.any():
        return 0
    idx = torch.nonzero(sub).squeeze(1)
    parent, c_mu, c_s, off = phase1.children(mu[idx], R[idx], s[idx], n[idx], p1.b)
    gp = idx[parent]
    if refit == "ls":
        tab = ls_coefficients(p1.b, p1.n_max, mu.device)
        nk = n[gp]
        j = torch.round((off / (2 * p1.b) + 0.5) * nk - 0.5).long().clamp(min=0)
        amp = field.amp.detach()[gp] * tab[nk, torch.minimum(j, nk - 1)].prod(1)
    elif refit == "grid":
        d = (n[gp] > 1).float().sum(1)
        amp = field.amp.detach()[gp] * torch.exp(-0.5 * (off ** 2).sum(1)) * (2 * p1.b / math.sqrt(2 * math.pi)) ** d
    else:
        raise ValueError(f"unknown refit {refit!r}")
    new = {"xyz": c_mu, "log_scale": c_s.log(), "quat": F.normalize(field.p["quat"].detach()[gp], dim=1), "amp": amp}
    _replace(field, ~sub, new)
    return int(sub.sum())


@torch.no_grad()
def q_remaining(field, vol, fields, p1, seed):
    q = phase1.violation(field.xyz.detach(), field.rotation.detach(), field.scales.detach(), fields, vol, p1, seed)
    return float((q.max(1).values > p1.tau_q).float().mean())


def run_cycles(vol, p1, p2, cycles=3, cycle_iters=10000, budget=1_000_000, target=None, refit="ls",
               log=None, snapshot_dir=None):
    """Phase I, then `cycles` x (Phase II of cycle_iters iterations, check, re-estimation if needed).
    Returns (field, info, trace)."""
    timer = Timer()
    t0 = time.perf_counter()
    init, st, fields = phase1.run(vol, p1, timer=timer)
    field = GaussianField(init["xyz"], init["scales"], init["R"], init["amp"], device=vol.device)
    scale = calibrate_amplitude(field, vol)
    p1_seconds = time.perf_counter() - t0
    phases = [{"phase": "I", "cycle": 0, "seconds": p1_seconds, "gaussians": len(field)}]
    trace, used, sec = [], 0, 0.0
    for c in range(cycles):
        cfg = phase2.P2Config(**{**p2.__dict__, "iterations": cycle_iters, "seed": p2.seed + 101 * c})
        tr = phase2.train(field, vol, cfg, log=log, snapshot_dir=snapshot_dir, it_offset=used, sec_offset=sec)
        trace += [{**r, "cycle": c} for r in tr]
        start_psnr = trace[0]["psnr"] if c == 0 else phases[-1].get("psnr_after", phases[-1]["psnr"])
        used, sec = tr[-1]["iteration"], tr[-1]["seconds"]
        t1 = time.perf_counter()
        qr = q_remaining(field, vol, fields, p1, seed=7 + c)
        ev = {"cycle": c, "iteration": used, "psnr": tr[-1]["psnr"], "ssim": tr[-1].get("ssim"),
              "psnr_gain_in_cycle": tr[-1]["psnr"] - start_psnr, "Q_remaining": qr, "gaussians": len(field),
              "adequate": bool(target is not None and tr[-1]["psnr"] >= target)}
        last = c == cycles - 1
        ev["fired"] = bool(not last and not ev["adequate"] and qr > 0 and len(field) < budget)
        if ev["fired"]:
            ev["subdivided"] = subdivide(field, vol, fields, p1, budget, seed=11 + c, refit=refit)
            ev["gaussians_after"] = len(field)
            ev["psnr_after"] = metrics.whole(field, vol, full=False)["psnr"]
        dt = time.perf_counter() - t1
        sec += dt
        phases.append({"phase": "II", "cycle": c, "end_iteration": used, "psnr": tr[-1]["psnr"]})
        phases.append({"phase": "check", "seconds": dt, **ev})
        if log:
            log({"event": ev})
        if ev["adequate"]:
            break
    info = {"phases": phases, "phase1": st, "amplitude_scale": scale, "init_seconds": p1_seconds,
            "gaussians": len(field), "total_seconds": p1_seconds + sec,
            "reestimation": {"cycles": cycles, "cycle_iters": cycle_iters, "budget": budget, "target": target,
                             "refit": refit}}
    return field, info, trace
