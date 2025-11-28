# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    importance_sampling_ratio = importance_sampling_ratio(
        current_logprobs, 
        policy_logprobs,
        loss_masks, 
    )

    assert importance_sampling_ratio.shape == (1, 4)
    assert importance_sampling_ratio[0, 0] == pytest.approx(1.0)
    assert importance_sampling_ratio[0, 1] == pytest.approx(1.0)
    assert importance_sampling_ratio[0, 2] == pytest.approx(1.0)
    assert importance_sampling_ratio[0, 3] == pytest.approx(1.0)


def test_current_and_policy_importance_sampling_ratio_is_clipped_when_large_difference():
    current_logprobs = jnp.array([[0.0, 1.0, 2.0, 3.0]])
    policy_logprobs = jnp.array([[0.0, 10.0, 2.0, -10.0]])
    loss_masks = jnp.array([[1.0, 1.0, 1.0, 1.0]])

    importance_sampling_ratio = importance_sampling_ratio(
        current_logprobs,
        policy_logprobs,
        loss_masks,
    )

    assert importance_sampling_ratio.shape == (1, 4)
    assert importance_sampling_ratio[0, 0] == pytest.approx(1.0)
    assert importance_sampling_ratio[0, 1] == pytest.approx(0.8)
    assert importance_sampling_ratio[0, 2] == pytest.approx(1.0)
    assert importance_sampling_ratio[0, 3] == pytest.approx(1.2)
