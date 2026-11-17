# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``kda_local_attn_fused_window``: the fused branch (one q/k/v GEMM, the attention kernel's bounds and
banded bias, one output GEMM) computes the same model as ``kda_local_attn_window``'s blocked softmax, over packed
documents and with a trained bias and gain."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer

WINDOW = 4


def _trained_branch(model: Transformer) -> Transformer:
    """Nonzero offset bias, logit scale and output gain, as after some training."""
    attn = model.kda_blocks.stacked.attn.local_attn
    shapes = (attn.offset_bias.shape, attn.logit_scale.shape, attn.out_gain.shape)
    bias, scale, gain = (jax.random.normal(jax.random.key(10 + i), shape) for i, shape in enumerate(shapes))
    return eqx.tree_at(
        lambda m: (
            m.kda_blocks.stacked.attn.local_attn.offset_bias,
            m.kda_blocks.stacked.attn.local_attn.logit_scale,
            m.kda_blocks.stacked.attn.local_attn.out_gain,
        ),
        model,
        (bias, 1.0 + 0.5 * scale, gain),
    )


def test_fused_branch_matches_the_blocked_branch():
    config = tiny_config()
    seq = config.max_seq_len
    segments = jnp.concatenate([jnp.zeros((1, seq // 3), jnp.int32), jnp.ones((1, seq - seq // 3), jnp.int32)], 1)
    segments = jnp.concatenate([segments, jnp.zeros_like(segments)])
    mask = AttentionMask.causal().with_segment_ids(segments)
    tokens = jax.random.randint(jax.random.key(1), (2, seq), 0, config.vocab_size)

    def loss_and_grads(model_config):
        model = _trained_branch(Transformer.init(model_config, key=jax.random.key(0)))
        loss_fn = eqx.filter_value_and_grad(lambda m: jnp.mean(m(tokens, mask)[0].astype(jnp.float32) ** 2))
        return eqx.filter_jit(loss_fn)(model)

    with jax.set_mesh(tiny_mesh()):
        blocked_loss, blocked_grads = loss_and_grads(dataclasses.replace(config, kda_local_attn_window=WINDOW))
        fused_loss, fused_grads = loss_and_grads(dataclasses.replace(config, kda_local_attn_fused_window=WINDOW))

    np.testing.assert_allclose(fused_loss, blocked_loss, rtol=1e-5)
    blocked_attn = blocked_grads.kda_blocks.stacked.attn.local_attn
    fused_attn = fused_grads.kda_blocks.stacked.attn.local_attn
    assert float(jnp.linalg.norm(fused_attn.offset_bias)) > 0
    for leaf in ("w_q", "w_k", "w_v", "w_o", "logit_scale", "offset_bias", "out_gain"):
        np.testing.assert_allclose(
            getattr(fused_attn, leaf), getattr(blocked_attn, leaf), rtol=1e-4, atol=1e-6, err_msg=leaf
        )
