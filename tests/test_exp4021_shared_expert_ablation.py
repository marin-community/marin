# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug.moe.exp4021_ablate_shared_expert import (
    BATCH_SIZE,
    FLOP_BUDGET,
    NO_SHARED_MODEL,
    NO_SHARED_STEPS,
    SEQ_LEN,
    SHARED_MODEL,
    SHARED_STEPS,
    _flops_per_token,
)


def test_shared_and_no_shared_models_differ_only_in_shared_expert():
    assert SHARED_MODEL.shared_expert_intermediate_dim > 0
    assert NO_SHARED_MODEL.shared_expert_intermediate_dim == 0
    assert SHARED_MODEL.hidden_dim == NO_SHARED_MODEL.hidden_dim
    assert SHARED_MODEL.num_experts == NO_SHARED_MODEL.num_experts
    assert SHARED_MODEL.num_layers == NO_SHARED_MODEL.num_layers


def test_flop_budgets_are_close_to_target():
    for model, steps in [(SHARED_MODEL, SHARED_STEPS), (NO_SHARED_MODEL, NO_SHARED_STEPS)]:
        fpt = _flops_per_token(model)
        total = 3 * fpt * steps * BATCH_SIZE * SEQ_LEN
        ratio = total / FLOP_BUDGET
        assert 0.95 <= ratio <= 1.05, f"total={total:.2e} vs budget={FLOP_BUDGET:.2e} (ratio={ratio:.3f})"


def test_no_shared_trains_more_tokens():
    assert NO_SHARED_STEPS > SHARED_STEPS, "Without shared expert, model has fewer FLOPs/token so needs more steps"
