# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.grug.moe_hero_fsdp_constant_lr_tpu.launch import (
    D512_CONSTANT_LR_POINTS,
    D512_LR_MULTIPLIERS,
    D512_STEPS,
    D512_TOKEN_MULTIPLES,
    constant_lr_optimizer,
    d512_model_config,
    select_d512_points,
)

_ISSUE_7856_BASE_PEAK_LR = {
    30: 0.0285752557500679,
    60: 0.0217352363934089,
    150: 0.0151342827466587,
    300: 0.0115098911435345,
    600: 0.0087531499407164,
}


def test_d512_constant_lr_matrix_matches_issue_7856_cells():
    assert len(D512_CONSTANT_LR_POINTS) == 25
    assert len({point.experiment_id for point in D512_CONSTANT_LR_POINTS}) == 25
    assert len({point.run_id for point in D512_CONSTANT_LR_POINTS}) == 25
    assert {(point.token_multiple, point.lr_multiplier, point.num_train_steps) for point in D512_CONSTANT_LR_POINTS} == {
        (token_multiple, lr_multiplier, D512_STEPS[token_multiple])
        for token_multiple in D512_TOKEN_MULTIPLES
        for lr_multiplier in D512_LR_MULTIPLIERS
    }


@pytest.mark.parametrize("token_multiple,expected_peak_lr", _ISSUE_7856_BASE_PEAK_LR.items())
def test_d512_constant_lr_preserves_historical_peak_lr(token_multiple: int, expected_peak_lr: float):
    point = select_d512_points(token_multiples=(token_multiple,), lr_multipliers=(1.0,))[0]
    optimizer = constant_lr_optimizer(point)

    assert optimizer.learning_rate == pytest.approx(expected_peak_lr, rel=1e-12)
    assert optimizer.lr_schedule == "constant"
    assert optimizer.warmup == 0.01
    assert optimizer.use_syrk is False


def test_d512_constant_lr_stays_at_peak_after_warmup():
    point = select_d512_points(token_multiples=(600,), lr_multipliers=(1.0,))[0]
    optimizer = constant_lr_optimizer(point)
    schedule = optimizer.lr_scheduler(point.num_train_steps)
    first_post_warmup_step = int(point.num_train_steps * optimizer.warmup) + 1

    assert float(schedule(0)) < optimizer.learning_rate
    assert float(schedule(first_post_warmup_step)) == pytest.approx(optimizer.learning_rate)
    assert float(schedule(point.num_train_steps // 2)) == pytest.approx(optimizer.learning_rate)
    assert float(schedule(point.num_train_steps - 1)) == pytest.approx(optimizer.learning_rate)


def test_d512_constant_lr_model_matches_historical_architecture():
    model = d512_model_config()

    assert (
        model.hidden_dim,
        model.num_layers,
        model.num_heads,
        model.local_kv_heads,
        model.global_kv_heads,
    ) == (512, 6, 4, 1, 1)
    assert (model.num_experts, model.num_experts_per_token, model.num_shared_experts) == (128, 4, 2)
    assert (model.max_seq_len, model.sliding_window, model.global_every) == (8192, 512, 4)
    assert model.attention_implementation is None
    assert model.moe_implementation is None
