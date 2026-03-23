# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug.moe.exp4039_ablate_shared_expert_sweep import (
    ALL_STEPS,
    BATCH_SIZE,
    FLOP_BUDGETS,
    SEQ_LEN,
    _make_model,
    flops_per_token,
    steps_for_budget,
)


def test_each_budget_has_two_arms():
    assert len(ALL_STEPS) == 2 * len(FLOP_BUDGETS)


def test_shared_and_no_shared_differ_only_in_shared_expert():
    for budget in FLOP_BUDGETS:
        shared = _make_model(budget, shared=True)
        no_shared = _make_model(budget, shared=False)
        assert shared.shared_expert_intermediate_dim > 0
        assert no_shared.shared_expert_intermediate_dim == 0
        assert shared.hidden_dim == no_shared.hidden_dim
        assert shared.num_experts == no_shared.num_experts
        assert shared.num_layers == no_shared.num_layers


def test_flop_budgets_are_close_to_target():
    for budget in FLOP_BUDGETS:
        for shared in (True, False):
            model = _make_model(budget, shared=shared)
            fpt = flops_per_token(model)
            num_steps = steps_for_budget(fpt, budget)
            total = 3 * fpt * num_steps * BATCH_SIZE * SEQ_LEN
            ratio = total / budget
            assert 0.9 <= ratio <= 1.1, (
                f"budget={budget:.0e} shared={shared}: total={total:.2e} ratio={ratio:.3f}"
            )


def test_no_shared_trains_more_steps_at_each_budget():
    for budget in FLOP_BUDGETS:
        shared_model = _make_model(budget, shared=True)
        no_shared_model = _make_model(budget, shared=False)
        shared_steps = steps_for_budget(flops_per_token(shared_model), budget)
        no_shared_steps = steps_for_budget(flops_per_token(no_shared_model), budget)
        assert no_shared_steps > shared_steps, (
            f"budget={budget:.0e}: no-shared should need more steps (fewer FLOPs/token)"
        )


def test_models_grow_with_budget():
    prev_hidden = 0
    for budget in sorted(FLOP_BUDGETS):
        model = _make_model(budget, shared=True)
        assert model.hidden_dim >= prev_hidden, (
            f"hidden_dim should grow with budget: {model.hidden_dim} < {prev_hidden}"
        )
        prev_hidden = model.hidden_dim
