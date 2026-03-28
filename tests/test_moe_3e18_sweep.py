# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MoE 3e18 FLOP hparam sweep (issue #4018)."""

from experiments.moe_3e18_hparam_sweep import (
    BASELINE,
    TARGET_FLOPS,
    all_sweep_points,
    _steps_for_budget,
    _build_step,
)


def test_baseline_steps_within_budget():
    """Steps computed for baseline config should consume ~3e18 FLOPs."""
    from levanter.utils.flop_utils import lm_flops_per_token

    batch_size = 256
    steps = _steps_for_budget(BASELINE, batch_size)
    fpt = lm_flops_per_token(
        hidden_dim=BASELINE.hidden_dim,
        intermediate_dim=BASELINE.intermediate_dim,
        shared_intermediate_dim=BASELINE.shared_expert_intermediate_dim,
        num_layers=BASELINE.num_layers,
        num_kv_heads=BASELINE.num_kv_heads,
        num_heads=BASELINE.num_heads,
        seq_len=BASELINE.max_seq_len,
        vocab_size=BASELINE.vocab_size,
        glu=True,
        num_experts=BASELINE.num_experts,
        num_shared_experts=1 if BASELINE.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=BASELINE.num_experts_per_token,
    )
    actual_flops = 3 * fpt * BASELINE.max_seq_len * batch_size * steps
    ratio = actual_flops / TARGET_FLOPS
    assert 0.9 <= ratio <= 1.0, f"Budget ratio {ratio:.3f} outside [0.9, 1.0]"


def test_all_sweep_points_no_duplicates():
    points = all_sweep_points()
    names = [p.name for p in points]
    assert len(names) == len(set(names)), f"Duplicate sweep point names: {names}"


def test_all_sweep_points_have_valid_models():
    """Every sweep point should have a valid GrugModelConfig (post_init passes)."""
    for pt in all_sweep_points():
        assert pt.model.num_experts_per_token <= pt.model.num_experts
        assert pt.model.hidden_dim % pt.model.num_heads == 0
        assert pt.model.vocab_size > 0
        assert _steps_for_budget(pt.model, pt.batch_size) > 0


def test_sweep_point_count():
    """Sweep should produce a reasonable number of arms."""
    points = all_sweep_points()
    assert 8 <= len(points) <= 30, f"Expected 8-30 sweep points, got {len(points)}"


def test_build_step_produces_executor_step():
    """_build_step should return a valid ExecutorStep for each sweep point."""
    points = all_sweep_points()
    for pt in points[:3]:
        step = _build_step(pt)
        assert step.name.startswith("moe-3e18/")
        assert step.fn is not None
