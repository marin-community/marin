# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


from experiments.grug.moe.exp4045_ablate_swa_sweep import (
    ALL_STEPS,
    BATCH_SIZE,
    FLOP_BUDGETS,
    SEQ_LEN,
    SLIDING_WINDOW_SIZE,
    _make_model,
    model_flops_per_token,
    steps_for_budget,
)
from experiments.grug.moe.model import GrugModelConfig

import pytest


def test_each_budget_has_two_arms():
    assert len(ALL_STEPS) == 2 * len(FLOP_BUDGETS)


def test_swa_and_full_differ_only_in_sliding_window():
    for budget in FLOP_BUDGETS:
        full = _make_model(budget, swa=False)
        swa = _make_model(budget, swa=True)
        assert full.sliding_window is None
        assert swa.sliding_window == SLIDING_WINDOW_SIZE
        assert full.hidden_dim == swa.hidden_dim
        assert full.num_experts == swa.num_experts
        assert full.num_layers == swa.num_layers
        assert full.shared_expert_intermediate_dim == swa.shared_expert_intermediate_dim


def test_flop_budgets_are_close_to_target():
    for budget in FLOP_BUDGETS:
        for swa in (False, True):
            model = _make_model(budget, swa=swa)
            fpt = model_flops_per_token(model)
            num_steps = steps_for_budget(fpt, budget)
            total = 3 * fpt * num_steps * BATCH_SIZE * SEQ_LEN
            ratio = total / budget
            assert 0.9 <= ratio <= 1.1, f"budget={budget:.0e} swa={swa}: total={total:.2e} ratio={ratio:.3f}"


def test_swa_same_flops_as_full():
    """SWA does not change the FLOP estimate (lm_flops_per_token uses full attention)."""
    for budget in FLOP_BUDGETS:
        full = _make_model(budget, swa=False)
        swa = _make_model(budget, swa=True)
        assert model_flops_per_token(full) == model_flops_per_token(swa)


def test_sliding_window_validation():
    with pytest.raises(ValueError, match="sliding_window must be positive"):
        GrugModelConfig(vocab_size=128, sliding_window=0)
    with pytest.raises(ValueError, match="sliding_window must be positive"):
        GrugModelConfig(vocab_size=128, sliding_window=-1)
