# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig, TiedExpertLrScale


class _ExpertWeights(NamedTuple):
    w_gate: jax.Array


class _ArrayStacked(NamedTuple):
    stacked: _ExpertWeights


class _Parameters(NamedTuple):
    expert_banks: _ArrayStacked
    ordinary: jax.Array


class _TupleParameters(NamedTuple):
    expert_banks: tuple[_ExpertWeights, ...]
    ordinary: jax.Array


@pytest.mark.parametrize(
    ("scale", "expected_ratios"),
    [
        (TiedExpertLrScale.UNSCALED, (1.0, 1.0)),
        (TiedExpertLrScale.SQRT, (1.0 / np.sqrt(2.0), 0.5)),
        (TiedExpertLrScale.LINEAR, (0.5, 0.25)),
    ],
)
def test_array_stacked_expert_lr_scales_by_layer_reuse_count(scale, expected_ratios):
    bank = jnp.arange(1, 65, dtype=jnp.float32).reshape(4, 4, 4)
    params = _Parameters(
        expert_banks=_ArrayStacked(stacked=_ExpertWeights(w_gate=jnp.stack((bank, bank, bank)))),
        ordinary=jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4),
    )
    grads = jax.tree.map(jnp.ones_like, params)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=1e-3,
        adam_lr=1e-3,
        warmup=0.0,
        lr_schedule="constant",
        momentum=0.0,
        nesterov=False,
        expert_bank_for_layer=(0, 1, 1, 2, 2, 2, 2),
        tied_expert_lr_scale=scale,
    ).build(num_train_steps=10)

    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    bank_update_norms = jnp.sqrt(jnp.sum(jnp.square(updates.expert_banks.stacked.w_gate), axis=(1, 2, 3)))
    actual_ratios = bank_update_norms[1:] / bank_update_norms[0]

    np.testing.assert_allclose(actual_ratios, expected_ratios, rtol=2e-3)


def test_unstacked_expert_lr_scales_by_layer_reuse_count():
    bank = jnp.arange(1, 65, dtype=jnp.float32).reshape(4, 4, 4)
    params = _TupleParameters(
        expert_banks=tuple(_ExpertWeights(w_gate=bank) for _ in range(3)),
        ordinary=jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4),
    )
    grads = jax.tree.map(jnp.ones_like, params)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=1e-3,
        adam_lr=1e-3,
        warmup=0.0,
        lr_schedule="constant",
        momentum=0.0,
        nesterov=False,
        expert_bank_for_layer=(0, 1, 1, 2, 2, 2, 2),
        tied_expert_lr_scale=TiedExpertLrScale.SQRT,
    ).build(num_train_steps=10)

    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    bank_update_norms = jnp.stack([jnp.linalg.norm(bank.w_gate) for bank in updates.expert_banks])

    np.testing.assert_allclose(bank_update_norms[1:] / bank_update_norms[0], (1 / np.sqrt(2), 0.5), rtol=2e-3)
