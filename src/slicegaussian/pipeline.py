"""End-to-end SliceGaussian: Phase I, then Phase II with conditional re-estimation, then scoring and saving."""
from __future__ import annotations

import csv
import json
import os
import platform
import random
from dataclasses import asdict, dataclass, field, fields as dc_fields

import numpy as np
import torch

from . import codec, phase1, phase2, reestimate
from .field import CULL_EPS, RADIUS_SIGMA
from .volume import Volume


@dataclass
class ReestimationConfig:
    cycles: int = 3                   # Phase II cycles; re-estimation runs between them when needed
    cycle_iters: int = 10000          # iterations per cycle (paper: 3 x 10,000 = 30,000)
    budget: int = 1_000_000           # safety limit on the count for re-estimation (GPU memory), not a target
    target: float | None = None       # optional adequacy target (3D PSNR on valid slices, dB): stop when reached
    refit: str = "ls"                 # ls | grid: child amplitudes at a re-estimation


@dataclass
class Config:
    p1: phase1.P1Config = field(default_factory=phase1.P1Config)
    p2: phase2.P2Config = field(default_factory=phase2.P2Config)
    reestimation: ReestimationConfig = field(default_factory=ReestimationConfig)
    seed: int = 0
    codec: bool = True                # also measure the encoded size and the decoded fidelity

    @staticmethod
    def from_dict(d):
        c = Config()
        for name, sub in (("p1", c.p1), ("p2", c.p2), ("reestimation", c.reestimation)):
            for k, v in (d.get(name) or {}).items():
                if k not in {f.name for f in dc_fields(sub)}:
                    raise KeyError(f"unknown setting {name}.{k}")
                setattr(sub, k, v)
        for k in ("seed", "codec"):
            if k in d:
                setattr(c, k, d[k])
        return c

    @staticmethod
    def load(path):
        with open(path) as f:
            return Config.from_dict(json.load(f))


def environment():
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "precision": "float32", "radius_sigma": RADIUS_SIGMA, "cull_eps": CULL_EPS}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "gpu_memory_gb": round(p.total_memory / 2 ** 30, 1)})
    return info


def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(volume, out, cfg: Config | None = None, snapshots=False, verbose=True):
    """Fits SliceGaussian to `volume` (a Volume) and writes the results to `out`. Returns (field, final metrics)."""
    cfg = cfg or Config()
    os.makedirs(out, exist_ok=True)
    seed_everything(cfg.seed)
    p1 = phase1.P1Config(**{**asdict(cfg.p1), "seed": cfg.seed})
    p2 = phase2.P2Config(**{**asdict(cfg.p2), "seed": cfg.seed})
    _save_json(os.path.join(out, "config.json"), {**asdict(cfg), "volume": volume.describe(), "environment": environment()})
    prog = open(os.path.join(out, "progress.log"), "w")

    def log(r):
        if "event" in r:
            e = r["event"]
            msg = (f"=== end of cycle {e['cycle'] + 1} at it {e['iteration']}: PSNR {e['psnr']:.2f} "
                   f"(gain in cycle {e['psnr_gain_in_cycle']:+.2f} dB), Q_remaining {e['Q_remaining']:.3f}, "
                   f"G {e['gaussians']} -> ")
            if e["fired"]:
                msg += (f"re-estimated: {e['subdivided']} subdivided, G {e['gaussians_after']}, "
                        f"PSNR right after {e['psnr_after']:.2f}")
            else:
                msg += "adequate, stop" if e["adequate"] else "no re-estimation"
        else:
            msg = (f"it {r['iteration']:>6}  t {r['seconds']:8.1f}s  PSNR {r['psnr']:6.2f}  "
                   f"SSIM {r.get('ssim', float('nan')):.4f}  G {r['gaussians']}")
        prog.write(msg + "\n")
        prog.flush()
        if verbose:
            print(msg, flush=True)

    re = cfg.reestimation
    field, info, trace = reestimate.run_cycles(volume, p1, p2, cycles=re.cycles, cycle_iters=re.cycle_iters,
                                               budget=re.budget, target=re.target, refit=re.refit, log=log,
                                               snapshot_dir=os.path.join(out, "slices") if snapshots else None)
    prog.close()
    keys = sorted({k for r in trace for k in r}, key=lambda k: (k not in ("iteration", "seconds", "gaussians"), k))
    with open(os.path.join(out, "trace.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(trace)
    final = dict(trace[-1])
    final.update({"init_seconds": info["init_seconds"], "total_seconds": info["total_seconds"],
                  "initial_psnr": trace[0]["psnr"], "gaussians": len(field)})
    field.save(os.path.join(out, "model.pt"))
    if cfg.codec:
        final.update(codec.measure(field, volume))
        with open(os.path.join(out, "model.sgz"), "wb") as f:
            f.write(codec.encode(field, volume))
    _save_json(os.path.join(out, "run.json"), info)
    _save_json(os.path.join(out, "final.json"), final)
    return field, final


def load_volume(path, spacing, name=""):
    return Volume(path, tuple(float(s) for s in spacing), name=name)
