"""Capacity and frozen-tree invariants for the allocation experiment (CPU)."""
import pytest
import torch

from slicegaussian.allocation_ablation import additions, apply_plan, structure_plan
from slicegaussian.field import GaussianField, reference_volume
from slicegaussian.volume import Volume


def field():
    return GaussianField(torch.tensor([[2., 2., 2.], [6., 2., 2.]]), torch.ones(2, 3),
                         torch.eye(3).repeat(2, 1, 1), torch.ones(2), device="cpu")


def test_exact_capacity_schedule():
    for initial, target, rounds in [(2, 13, 3), (84500, 419707, 3), (84500, 496167, 3), (3, 3, 2)]:
        count = initial
        for delta in additions(initial, target, rounds):
            assert 0 <= delta <= count
            count += delta
        assert count == target
    with pytest.raises(ValueError):
        additions(2, 17, 3)


def test_structure_score_respects_rotated_axes():
    import numpy as np
    vol = Volume(np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8), (1, 1, 1), device="cpu")
    f = field()
    # J = diag(1, 0, 0); the second parent's local y is the world x axis.
    from slicegaussian.field import rot_to_quat
    f.p["quat"].data[1] = rot_to_quat(torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]]))[0]
    J = torch.zeros(6, 8, 8, 8)
    J[0] = 1
    indices, axes = structure_plan(f, vol, {"J": J}, 2)
    assert dict(zip(indices.tolist(), axes.tolist())) == {0: 0, 1: 1}


def test_frozen_tree_replay_identical_without_optimisation():
    import numpy as np
    vol = Volume(np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8), (1, 1, 1), device="cpu")
    upfront, replay = field(), field()
    J = torch.zeros(6, 8, 8, 8)
    J[0], J[2], J[5] = 1., .5, .1
    plans = []
    for delta in additions(2, 7, 2):
        plan = structure_plan(upfront, vol, {"J": J}, delta)
        plans.append(plan)
        apply_plan(upfront, plan, 1.)
    for plan in plans:
        apply_plan(replay, plan, 1.)
    assert len(upfront) == len(replay) == 7
    for k in upfront.p:
        assert torch.equal(upfront.p[k], replay.p[k])
    a = reference_volume(upfront.xyz, upfront.scales, upfront.rotation, upfront.amp, vol)
    b = reference_volume(replay.xyz, replay.scales, replay.rotation, replay.amp, vol)
    assert torch.equal(a, b)


def test_violation_plan_uses_shared_support_score(monkeypatch):
    from slicegaussian import phase1
    from slicegaussian.allocation_ablation import violation_plan
    f = field()
    cfg = phase1.P1Config(seed=17)
    marker_vol, marker_fields = object(), object()

    def violation(mu, R, scales, fields, vol, p1, seed):
        assert fields is marker_fields and vol is marker_vol
        assert p1 is cfg and seed == 17
        return torch.tensor([[0.2, 0.9, 0.1], [1.1, 0.3, 0.1]])

    monkeypatch.setattr(phase1, "violation", violation)
    indices, axes = violation_plan(f, marker_vol, marker_fields, cfg, 1)
    assert indices.tolist() == [1] and axes.tolist() == [0]
