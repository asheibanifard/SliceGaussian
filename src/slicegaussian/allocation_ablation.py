"""Capacity-matched structure densification and a frozen-plan allocation timing control.

This is a volume-native baseline, not an implementation of image-based SAD-GS.
"""
from dataclasses import asdict, replace
import time

import torch

from . import metrics, phase1, phase2
from .field import GaussianField, calibrate_amplitude
from .pipeline import environment, seed_everything
from .reestimate import ls_coefficients
from .volume import sample


@torch.no_grad()
def structure_plan(field, vol, fields, count):
    """Binary splits ranked by scale times directional structure energy; exactly count additions.

    The multiscale tensor J is shared with Phase I (including its band weighting).
    No wavelength or violation threshold enters the split score.
    """
    if not 0 <= count <= len(field):
        raise ValueError("one round must add between zero and the current population")
    J = phase1.mat3(sample(fields["J"], field.xyz[:, None], vol)[:, :, 0])
    R = field.rotation
    energy = torch.einsum("nik,nij,njk->nk", R, J, R).clamp(min=0)
    scores = field.scales * energy.sqrt()
    priority, axes = scores.max(1)
    indices = torch.argsort(priority, descending=True, stable=True)[:count]
    return indices, axes[indices]


@torch.no_grad()
def violation_plan(field, vol, fields, p1, count):
    """Volume-adapted SAD-GS-style support violation, with exact-capacity binary splits."""
    if not 0 <= count <= len(field):
        raise ValueError("one round must add between zero and the current population")
    q = phase1.violation(field.xyz, field.rotation, field.scales, fields, vol, p1, p1.seed)
    priority, axes = q.max(1)
    indices = torch.argsort(priority, descending=True, stable=True)[:count]
    return indices, axes[indices]


@torch.no_grad()
def apply_plan(field, plan, b):
    """Keep row order deterministic; fit children to the current parent using the same LS rule."""
    indices, axes = plan
    if not len(indices):
        return
    n = torch.ones((len(indices), 3), dtype=torch.long, device=field.xyz.device)
    n[torch.arange(len(indices), device=n.device), axes] = 2
    parent, xyz, scales, off = phase1.children(
        field.xyz[indices], field.rotation[indices], field.scales[indices], n, b)
    nk = n[parent]
    j = torch.round((off / (2 * b) + 0.5) * nk - 0.5).long().clamp(min=0)
    tab = ls_coefficients(b, 2, field.xyz.device)
    amp = field.amp[indices[parent]] * tab[nk, torch.minimum(j, nk - 1)].prod(1)
    keep = torch.ones(len(field), dtype=torch.bool, device=n.device)
    keep[indices] = False
    new = {"xyz": xyz, "log_scale": scales.log(),
           "quat": field.p["quat"][indices[parent]], "amp": amp}
    # All arms restart Adam at identical cycle boundaries, so no state is carried here.
    field.p = {k: torch.nn.Parameter(torch.cat([v.detach()[keep], new[k].detach()]).contiguous())
               for k, v in field.p.items()}
    field.opt = None


def additions(initial, target, rounds):
    """Exact, approximately geometric growth; reject schedules impossible with binary splits."""
    if initial < 1 or target < initial or rounds < 1 or target > initial * 2 ** rounds:
        raise ValueError("target must be reachable from the parent grid with the requested split rounds")
    counts, current = [], initial
    for r in range(1, rounds + 1):
        desired = target if r == rounds else round(initial * (target / initial) ** (r / rounds))
        desired = min(2 * current, max(current, desired))
        counts.append(desired - current)
        current = desired
    if current != target:
        raise ValueError("split schedule did not reach the requested count")
    return counts


ARMS = ("phase1_upfront", "violation_upfront", "violation_replay", "violation_online", "structure_online")


def run(vol, cfg, rounds=3, cycle_iters=7500, log=None, on_arm=None, segments=None, arms=ARMS):
    """Five arms, common (rounds+1) cycles; returns metrics, traces, and split audit records.

    segments: optional rounds+1 optimisation lengths (default cycle_iters each). Split round r follows segment r;
    a zero-length segment applies the next round immediately (e.g. 1000,0,0,29000: all splits after 1,000 updates).
    Every arm restarts Adam at the same non-empty segment boundaries.

    Violation upfront and replay share the exact split tree. Online recomputes its tree on
    optimised parents. Reference uses ordinary Phase I at its natural count, no re-estimation.
    """
    segments = [cycle_iters] * (rounds + 1) if segments is None else [int(x) for x in segments]
    if len(segments) != rounds + 1 or min(segments) < 0 or segments[-1] < 1:
        raise ValueError("segments needs rounds+1 non-negative lengths with a positive last one")
    seed_everything(cfg.seed)
    p1 = replace(cfg.p1, seed=cfg.seed, target_count=0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    init, stats, fields = phase1.run(vol, p1)
    mu, R, s, _ = phase1.parents(vol, fields, p1)
    torch.cuda.synchronize()
    reference_seconds = time.perf_counter() - start
    target = len(init["xyz"])
    growth = additions(len(mu), target, rounds)

    def parent_field():
        V = torch.from_numpy(vol.analysis_volume).to(vol.device)
        amp = sample(V[None], mu[:, None], vol)[0, :, 0] * (2 * p1.b / (2 * torch.pi) ** 0.5) ** 3
        f = GaussianField(mu.clone(), s.clone(), R.clone(), amp, device=vol.device)
        calibrate_amplitude(f, vol)
        return f

    torch.cuda.synchronize()
    start = time.perf_counter()
    planner = parent_field()
    plans = []
    for count in growth:
        plan = violation_plan(planner, vol, fields, p1, count)
        plans.append(tuple(x.clone() for x in plan))
        apply_plan(planner, plan, p1.b)
    torch.cuda.synchronize()
    planning_seconds = time.perf_counter() - start
    del planner
    results = {}
    for arm in arms:
        if log:
            log({"arm": arm, "event": "start", "target_count": target})
        seed_everything(cfg.seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        if arm == "phase1_upfront":
            field = GaussianField(init["xyz"], init["scales"], init["R"], init["amp"], device=vol.device)
            calibrate_amplitude(field, vol)
        else:
            field = parent_field()
        events = []

        def split(r):
            psnr_before = metrics.whole(field, vol, full=False)["psnr"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if arm == "structure_online":
                plan = structure_plan(field, vol, fields, growth[r])
            elif arm == "violation_online":
                plan = violation_plan(field, vol, fields, p1, growth[r])
            else:
                plan = plans[r]
            before = len(field)
            apply_plan(field, plan, p1.b)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            events.append({"round": r, "before": before, "after": len(field),
                           "indices": plan[0].cpu().tolist(), "axes": plan[1].cpu().tolist(), "seconds": dt,
                           "psnr_before": psnr_before, "psnr_after": metrics.whole(field, vol, full=False)["psnr"]})
            return dt

        torch.cuda.synchronize()
        overhead = time.perf_counter() - start
        if arm == "violation_upfront":
            for r in range(rounds):
                overhead += split(r)
        # Shared analysis and frozen-tree construction are charged to both timing-control arms.
        overhead += reference_seconds + (planning_seconds if arm in ("violation_upfront", "violation_replay") else 0)
        trace, opt_seconds = [], 0.0
        for c in range(rounds + 1):
            if segments[c] > 0:
                p2 = replace(cfg.p2, iterations=segments[c], seed=cfg.seed + 101 * c)
                def report(row):
                    if log:
                        log({"arm": arm, "cycle": c, **row})

                tr = phase2.train(field, vol, p2, log=report, it_offset=sum(segments[:c]), sec_offset=opt_seconds)
                opt_seconds = tr[-1]["seconds"]
                trace.extend({**row, "cycle": c, "total_seconds": row["seconds"] + overhead} for row in tr)
            if c < rounds and arm in ("violation_replay", "violation_online", "structure_online"):
                overhead += split(c)
        assert len(field) == target
        results[arm] = {"final": {**metrics.whole(field, vol, full=True), "gaussians": len(field),
                                  "total_seconds": overhead + opt_seconds}, "trace": trace, "splits": events}
        if on_arm:
            on_arm(arm, results[arm], field)
        del field
    paired = ({key: results["violation_upfront"]["final"][key] - results["violation_replay"]["final"][key]
               for key in ("psnr", "ssim", "mae", "total_seconds")}
              if {"violation_upfront", "violation_replay"} <= set(results) else None)
    return {"paired_upfront_minus_replay": paired, "protocol": {"seed": cfg.seed, "target_count": target, "parent_count": len(mu),
                         "growth": growth, "rounds": rounds, "cycle_iters": cycle_iters, "segments": segments,
                         "config": asdict(cfg), "volume": vol.describe(), "environment": environment(), "phase1": stats,
                         "shared_analysis_seconds": reference_seconds, "planning_seconds": planning_seconds},
            "arms": results}
