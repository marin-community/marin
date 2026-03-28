# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-scale AdamH vs Adam isoflop experiment config generation."""

from experiments.grug.moe.adamh_vs_adam_isoflop import (
    BUDGETS,
    SCALE_POINTS,
    all_steps,
    compute_train_steps,
    make_adam_optimizer,
    make_adamh_optimizer,
    make_model,
    make_scale_point,
)
from levanter.optim import AdamConfig, GrugAdamHConfig


def test_scale_points_cover_all_budgets():
    assert len(SCALE_POINTS) == len(BUDGETS)
    for sp, budget in zip(SCALE_POINTS, BUDGETS):
        assert sp.budget == budget


def test_hidden_dim_increases_with_budget():
    dims = [sp.hidden_dim for sp in SCALE_POINTS]
    for i in range(1, len(dims)):
        assert dims[i] >= dims[i - 1], f"hidden_dim should not decrease: {dims}"


def test_model_configs_are_valid():
    for sp in SCALE_POINTS:
        model = make_model(sp)
        assert model.hidden_dim == sp.hidden_dim
        assert model.num_layers == sp.num_layers
        assert model.num_experts == 8
        assert model.num_experts_per_token == 2
        assert model.hidden_dim % 128 == 0, "hidden_dim must be divisible by head_dim=128"


def test_train_steps_positive():
    for sp in SCALE_POINTS:
        steps = compute_train_steps(sp)
        assert steps > 0, f"train_steps must be positive for budget={sp.budget}"


def test_adam_optimizer_types():
    for sp in SCALE_POINTS:
        adam = make_adam_optimizer(sp)
        adamh = make_adamh_optimizer(sp)
        assert isinstance(adam, AdamConfig)
        assert isinstance(adamh, GrugAdamHConfig)
        assert adam.learning_rate > 0
        assert adamh.learning_rate > 0
        assert adamh.adam_lr > 0


def test_adamh_lr_follows_heuristic():
    """AdamH scale-invariant LR = sqrt(adam_lr * 0.1)."""
    import math

    for sp in SCALE_POINTS:
        adamh = make_adamh_optimizer(sp)
        expected = math.sqrt(adamh.adam_lr * 0.1)
        assert abs(adamh.learning_rate - expected) < 1e-12, (
            f"AdamH LR {adamh.learning_rate} != sqrt({adamh.adam_lr} * 0.1) = {expected}"
        )


def test_all_steps_generated():
    # 2 optimizers x 4 budgets = 8 steps
    assert len(all_steps) == 8
    names = [s.name for s in all_steps]
    # Each budget should have an adam and adamh step
    for budget in BUDGETS:
        budget_str = f"{budget:.0e}"
        adam_names = [n for n in names if "adam-" in n and budget_str in n]
        adamh_names = [n for n in names if "adamh-" in n and budget_str in n]
        assert len(adam_names) == 1, f"Expected 1 adam step for {budget_str}, got {adam_names}"
        assert len(adamh_names) == 1, f"Expected 1 adamh step for {budget_str}, got {adamh_names}"
