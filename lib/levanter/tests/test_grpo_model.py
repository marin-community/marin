# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grpo import GrpoConfig, KlGradient, grpo_loss
from levanter.grpo_model import GrpoExample, grpo_model_loss, response_logprobs
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import QwenConfig
from levanter.utils.tree_utils import inference_mode


@pytest.fixture(params=[LlamaConfig, QwenConfig])
def model_batch(request):
    config = request.param(
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        attn_backend=AttentionBackend.VANILLA,
    )
    model = config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(0))
    B, S, T = hax.Axis("batch", 2 * len(jax.devices())), hax.Axis("position", 8), hax.Axis("response", 3)
    tokens = np.tile([[3, 4, 5, 6, 7, 8, 9, 10], [0, 0, 3, 4, 5, 8, 9, 0]], (len(jax.devices()), 1))
    mask = np.tile([[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 0]], (len(jax.devices()), 1))
    response_mask = hax.named(mask[:, -3:].astype(np.float32), (B, T))
    weights = response_mask / response_mask.sum(T) / B.size
    batch = GrpoExample(
        tokens=hax.named(tokens, (B, S)),
        attention_mask=hax.named(mask, (B, S)),
        position_ids=hax.named(np.maximum(mask.cumsum(-1) - 1, 0), (B, S)),
        response_mask=response_mask,
        advantages=response_mask,
        policy_weights=weights,
        kl_weights=weights,
        old_logprobs=hax.full((B, T), -3.5),
        reference_logprobs=None,
    )
    return model, batch


def full_softmax_scores(model, batch):
    model = inference_mode(model, True)
    mask = AttentionMask.causal().with_segment_ids(batch.attention_mask.astype(jnp.int32))
    logits = model(batch.tokens, mask, pos_ids=batch.position_ids).rearrange(("batch", "position", "vocab"))
    tokens = batch.tokens.rearrange(("batch", "position"))
    width = batch.response_mask.axis_size("response")
    # Explicit shifted full-vocabulary reference, independent of fused CE slicing.
    logprobs = jax.nn.log_softmax(logits.array.astype(jnp.float32)[:, :-1], axis=-1)
    selected = jnp.take_along_axis(logprobs, tokens.array[:, 1:, None], axis=-1)[..., 0]
    return hax.named(selected[:, -width:], batch.response_mask.axes)


@pytest.mark.parametrize("block_size", [8, 64])
def test_response_scores_match_full_softmax_and_unpadded_context(model_batch, block_size):
    model, batch = model_batch
    actual = eqx.filter_jit(response_logprobs)(model, batch, block_size=block_size)
    np.testing.assert_allclose(actual.array, full_softmax_scores(model, batch).array, atol=1e-5, rtol=1e-5)
    # Remove left padding from the second trajectory. Its response stays at
    # the end and real tokens keep the same explicit rotary positions.
    row = jax.tree_util.tree_map(
        lambda x: x["batch", hax.ds(1, 1)] if isinstance(x, hax.NamedArray) else x,
        batch,
        is_leaf=lambda x: isinstance(x, hax.NamedArray),
    )
    unpadded = dataclasses.replace(
        row,
        tokens=row.tokens["position", hax.ds(2, 6)],
        attention_mask=row.attention_mask["position", hax.ds(2, 6)],
        position_ids=row.position_ids["position", hax.ds(2, 6)],
    )
    without_padding = eqx.filter_jit(response_logprobs)(model, unpadded, block_size=block_size)
    np.testing.assert_allclose(actual.array[1, :2], without_padding.array[0, :2], atol=1e-5, rtol=1e-5)


def test_grpo_model_value_and_gradients_match_full_softmax(model_batch):
    model, batch = model_batch
    old = response_logprobs(model, batch, block_size=8)
    batch = dataclasses.replace(batch, old_logprobs=old)
    config = GrpoConfig(0.2, 0.2, 0.0, KlGradient.DETACHED)

    def fused_loss(model):
        return grpo_model_loss(
            model, batch, key=jax.random.PRNGKey(9), config=config, accumulation_steps=1, block_size=8
        )[0]

    def reference_loss(model):
        return grpo_loss(
            full_softmax_scores(model, batch),
            batch.old_logprobs,
            None,
            batch.advantages,
            batch.policy_weights,
            batch.kl_weights,
            config=config,
            accumulation_steps=1,
        )[0]

    actual, gradient = eqx.filter_jit(eqx.filter_value_and_grad(fused_loss))(model)
    expected, expected_gradient = eqx.filter_jit(eqx.filter_value_and_grad(reference_loss))(model)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    for observed, reference in zip(
        jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(expected_gradient), strict=True
    ):
        np.testing.assert_allclose(observed, reference, atol=1e-6, rtol=1e-5)
    assert max(float(jnp.max(jnp.abs(g))) for g in jax.tree_util.tree_leaves(gradient)) > 1e-3
