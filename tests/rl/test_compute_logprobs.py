# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from marin.rl.rl_losses import compute_logprobs, compute_logprobs_and_entropy, importance_sampling_ratio


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


def test_compute_logprobs_and_entropy_can_skip_entropy():
    logits = jnp.zeros((1, 2, 10), dtype=jnp.float32)
    logits = logits.at[0, 0, 1].set(1.0)

    batch = SimpleNamespace(
        input_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    logprobs, entropy = compute_logprobs_and_entropy(model, batch, key=None, compute_entropy=False)

    expected_next_logprob = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)[0, 0, 1]

    assert entropy is None
    assert logprobs.shape == (1, 2)
    assert logprobs[0, 0] == pytest.approx(0.0)
    assert logprobs[0, 1] == pytest.approx(float(expected_next_logprob))


def test_compute_logprobs_and_entropy_uniform_logits():
    vocab_size = 5
    logits = jnp.zeros((1, 3, vocab_size), dtype=jnp.float32)

    batch = SimpleNamespace(
        input_ids=DummyNamedArray(jnp.array([[0, 1, 2]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1, 2]], dtype=jnp.int32)),
        temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    logprobs, entropy = compute_logprobs_and_entropy(model, batch, key=None, compute_entropy=True)

    assert entropy is not None
    assert logprobs.shape == (1, 3)
    assert entropy.shape == (1, 3)
    assert entropy[0, 0] == pytest.approx(0.0)
    assert entropy[0, 1] == pytest.approx(float(jnp.log(vocab_size)))
    assert entropy[0, 2] == pytest.approx(float(jnp.log(vocab_size)))


def test_compute_logprobs_and_entropy_peaked_logits_have_lower_entropy():
    uniform_logits = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    peaked_logits = uniform_logits.at[0, 0, 0].set(8.0)

    batch = SimpleNamespace(
        input_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )

    _uniform_logprobs, uniform_entropy = compute_logprobs_and_entropy(
        DummyModel(uniform_logits), batch, key=None, compute_entropy=True
    )
    _peaked_logprobs, peaked_entropy = compute_logprobs_and_entropy(
        DummyModel(peaked_logits), batch, key=None, compute_entropy=True
    )

    assert uniform_entropy is not None
    assert peaked_entropy is not None
    assert peaked_entropy[0, 1] < uniform_entropy[0, 1]


def test_compute_logprobs_and_entropy_temperature_scales_entropy():
    logits = jnp.zeros((2, 2, 4), dtype=jnp.float32)
    logits = logits.at[:, 0, 0].set(8.0)

    batch = SimpleNamespace(
        input_ids=DummyNamedArray(jnp.array([[0, 1], [0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1], [0, 1]], dtype=jnp.int32)),
        temperature=DummyNamedArray(jnp.array([0.5, 2.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    _logprobs, entropy = compute_logprobs_and_entropy(model, batch, key=None, compute_entropy=True)

    assert entropy is not None
    assert entropy[1, 1] > entropy[0, 1]


def test_compute_logprobs_and_entropy_extreme_logits_are_finite():
    logits = jnp.array([[[1000.0, -1000.0, -1000.0], [0.0, 0.0, 0.0]]], dtype=jnp.float32)

    batch = SimpleNamespace(
        input_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        position_ids=DummyNamedArray(jnp.array([[0, 1]], dtype=jnp.int32)),
        temperature=DummyNamedArray(jnp.array([1.0], dtype=jnp.float32)),
    )
    model = DummyModel(logits)

    _logprobs, entropy = compute_logprobs_and_entropy(model, batch, key=None, compute_entropy=True)

    assert entropy is not None
    assert jnp.isfinite(entropy[0, 1])
    assert entropy[0, 1] == pytest.approx(0.0)


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
