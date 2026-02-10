# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from types import SimpleNamespace

from marin.rl.rl_losses import compute_logprobs, importance_sampling_ratio


class DummyNamedArray:
    def __init__(self, array):
        self.array = array


class DummyModelOutput:
    def __init__(self, array):
        self.array = array


class DummyModel:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, *, input_ids, attn_mask, pos_ids, key):
        return DummyModelOutput(self._logits)


def test_compute_logprobs_one_hot_next_token():
    logits = jnp.zeros((1, 2, 10), dtype=jnp.float32)
    logits = logits.at[0, 0, 1].set(1.0)

    batch = SimpleNamespace(
        input_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    logprobs = compute_logprobs(model, batch, key=None)

    expected_next_logprob = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)[0, 0, 1]

    assert logprobs.shape == (1, 2)
    assert logprobs[0, 0] == pytest.approx(0.0)
    assert logprobs[0, 1] == pytest.approx(float(expected_next_logprob))


def test_current_and_policy_importance_sampling_ratio_equals_one_when_equal():
    current_logprobs = jnp.array([[0.0, 1.0, 2.0, 3.0]])
    policy_logprobs = jnp.array([[0.0, 1.0, 2.0, 3.0]])
    loss_masks = jnp.array([[1.0, 1.0, 1.0, 1.0]])

    ratio = importance_sampling_ratio(
        current_logprobs,
        policy_logprobs,
        loss_masks,
    )

    assert ratio.shape == (1, 4)
    assert ratio[0, 0] == pytest.approx(1.0)
    assert ratio[0, 1] == pytest.approx(1.0)
    assert ratio[0, 2] == pytest.approx(1.0)
    assert ratio[0, 3] == pytest.approx(1.0)
