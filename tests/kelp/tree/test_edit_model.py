# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Tests for the AR edit-prediction model."""

import jax
import jax.numpy as jnp
import pytest

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.tree.edit_model import (
    _make_causal_mask,
    ar_loss,
    forward,
    init_edit_params,
)


@pytest.fixture
def tiny_cfg():
    return TreeDiffusionConfig(
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=64,
    )


@pytest.fixture
def params(tiny_cfg):
    key = jax.random.PRNGKey(0)
    return init_edit_params(tiny_cfg, key=key)


def test_init_params_shapes(params, tiny_cfg):
    assert params.token_embed.shape == (tiny_cfg.vocab_size, tiny_cfg.hidden_dim)
    assert params.output_proj.shape == (tiny_cfg.hidden_dim, tiny_cfg.vocab_size)
    assert params.final_norm.shape == (tiny_cfg.hidden_dim,)
    assert len(params.blocks) == tiny_cfg.num_layers


def test_init_params_no_timestep_embed(params):
    """EditModelParams should NOT have a timestep_embed field."""
    assert not hasattr(params, "timestep_embed")


def test_init_params_block_shapes(params, tiny_cfg):
    block = params.blocks[0]
    D = tiny_cfg.hidden_dim
    N = tiny_cfg.num_heads
    H = tiny_cfg.inferred_head_dim
    I = tiny_cfg.intermediate_dim

    assert block.attn.w_q.shape == (D, N * H)
    assert block.attn.w_k.shape == (D, N * H)  # num_kv_heads == num_heads
    assert block.attn.w_v.shape == (D, N * H)
    assert block.attn.w_o.shape == (N * H, D)
    assert block.mlp_gate.shape == (D, I)
    assert block.mlp_up.shape == (D, I)
    assert block.mlp_down.shape == (I, D)
    assert block.rms_attn.shape == (D,)
    assert block.rms_mlp.shape == (D,)


def test_causal_mask():
    mask = _make_causal_mask(4)
    expected = jnp.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert jnp.array_equal(mask, expected)


def test_forward_output_shape(params, tiny_cfg):
    batch_size, seq_len = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)

    logits = forward(params, token_ids, tiny_cfg)
    assert logits.shape == (batch_size, seq_len, tiny_cfg.vocab_size)


def test_forward_output_dtype_is_float32(params, tiny_cfg):
    """Logits should always be float32 regardless of compute_dtype."""
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (1, 8), 1, 100)
    logits = forward(params, token_ids, tiny_cfg)
    assert logits.dtype == jnp.float32


def test_forward_is_causal(params, tiny_cfg):
    """Verify that the model is causal: changing a future token should not
    affect the logits at an earlier position."""
    seq_len = 12
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (1, seq_len), 1, 100)

    logits_a = forward(params, token_ids, tiny_cfg)

    # Change the last token.
    token_ids_b = token_ids.at[0, -1].set(50)
    logits_b = forward(params, token_ids_b, tiny_cfg)

    # All positions except the last should be identical.
    assert jnp.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)
    # The last position should differ.
    assert not jnp.allclose(logits_a[0, -1], logits_b[0, -1], atol=1e-5)


def test_forward_padding_masked(params, tiny_cfg):
    """Padding tokens should not affect non-padding logits."""
    tokens = jnp.array([[10, 20, 30, 0, 0]])  # Last two are padding.
    logits_a = forward(params, tokens, tiny_cfg)

    tokens_b = jnp.array([[10, 20, 30, 50, 60]])  # No padding.
    logits_b = forward(params, tokens_b, tiny_cfg)

    # First 3 positions should be the same (padding doesn't attend).
    assert jnp.allclose(logits_a[0, :3], logits_b[0, :3], atol=1e-5)


def test_ar_loss_returns_scalar_and_metrics(params, tiny_cfg):
    batch_size, seq_len = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)
    # Loss on the last 4 tokens only.
    loss_mask = jnp.zeros((batch_size, seq_len))
    loss_mask = loss_mask.at[:, -4:].set(1.0)

    loss, metrics = ar_loss(params, token_ids, loss_mask, tiny_cfg)

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert "accuracy" in metrics
    assert "perplexity" in metrics
    assert "num_loss_tokens" in metrics


def test_ar_loss_is_positive(params, tiny_cfg):
    batch_size, seq_len = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)
    loss_mask = jnp.ones((batch_size, seq_len))

    loss, _ = ar_loss(params, token_ids, loss_mask, tiny_cfg)
    assert float(loss) > 0


def test_ar_loss_zero_mask_gives_zero_loss(params, tiny_cfg):
    batch_size, seq_len = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)
    loss_mask = jnp.zeros((batch_size, seq_len))

    loss, _ = ar_loss(params, token_ids, loss_mask, tiny_cfg)
    assert float(loss) == 0.0


def test_ar_loss_grad_flows(params, tiny_cfg):
    """Verify gradients flow through the loss."""
    batch_size, seq_len = 1, 8
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 1, 100)
    loss_mask = jnp.ones((batch_size, seq_len))

    def loss_fn(p):
        loss, _ = ar_loss(p, token_ids, loss_mask, tiny_cfg)
        return loss

    grads = jax.grad(loss_fn)(params)
    # Check that gradients are non-zero for at least some params.
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    assert float(grad_norm) > 0
