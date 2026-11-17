# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``lm_head_value_embeds``: the MLA value against the gated head-row formula, a tiny model that starts as
the baseline, and no gradient into the lm_head through the value path."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import CausalSelfAttention, Transformer, rms_norm
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def _config():
    return dataclasses.replace(tiny_config(), lm_head_value_embeds=True)


def _mla_attention(model: Transformer) -> CausalSelfAttention:
    return next(layer.attn for layer in model.layers() if isinstance(layer.attn, CausalSelfAttention))


def test_mla_value_adds_the_gated_head_rows():
    config = _config()
    keys = jax.random.split(jax.random.key(3), 4)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        attn = _mla_attention(model)
        width = attn.head_ve_gain.shape[0]
        attn = eqx.tree_at(
            lambda a: (a.head_ve_lambda, a.head_ve_gate, a.head_ve_gain),
            attn,
            (
                jnp.asarray(0.7, jnp.float32),
                jax.random.normal(keys[0], attn.head_ve_gate.shape),
                jax.random.normal(keys[1], (width,)),
            ),
        )
        x = jax.random.normal(keys[2], (2, config.max_seq_len, config.hidden_dim))
        tokens = jax.random.randint(keys[3], (2, config.max_seq_len), 0, config.vocab_size)
        head_ve = model._head_value_embeds(tokens)
        _, _, v = attn._mla_qkv(x, None, None, None, head_ve)
        plain = eqx.tree_at(lambda a: (a.head_ve_lambda, a.head_ve_gate, a.head_ve_gain), attn, replace=(None,) * 3)
        _, _, v_plain = plain._mla_qkv(x, None, None, None, None)

    rows = rms_norm(jnp.asarray(np.asarray(model.output_proj).T[np.asarray(tokens)]))
    gate = jax.nn.sigmoid(np.asarray(x) @ np.asarray(attn.head_ve_gate))[..., None]
    gained = np.asarray(rows)[..., :width] * np.asarray(attn.head_ve_gain)
    e = rearrange(gained, "... (n d) -> ... n d", n=gate.shape[-2])
    np.testing.assert_allclose(v, 0.7 * np.asarray(v_plain) + gate * e, rtol=1e-5, atol=1e-5)


def test_tiny_model_starts_as_baseline_and_sends_no_gradient_to_the_head():
    config = _config()
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, lm_head_value_embeds=False), key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        # Zero-initialised gain: the head rows add nothing at step 0.
        np.testing.assert_array_equal(forward(model, tokens), forward(plain, tokens))

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)
        attn_grads = grads.stacked_blocks.stacked.attn
        assert float(jnp.linalg.norm(attn_grads.head_ve_gain)) > 0
        assert float(jnp.linalg.norm(attn_grads.head_ve_lambda)) > 0

        # With a live gain the hidden states read the head, yet the head gets no gradient from them.
        live = eqx.tree_at(
            lambda m: m.stacked_blocks.stacked.attn.head_ve_gain,
            model,
            jnp.ones_like(model.stacked_blocks.stacked.attn.head_ve_gain),
        )
        moved = eqx.tree_at(lambda m: m.output_proj, live, 2 * live.output_proj)
        assert not np.allclose(forward(live, tokens), forward(moved, tokens))
        _, live_grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(live)

    assert np.isfinite(float(loss))
    np.testing.assert_array_equal(live_grads.output_proj, 0)


def test_head_value_embed_weights_train_on_plain_adam():
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(_config(), key=jax.random.key(0))
    labels = GrugMoeMuonHConfig().create_mask(model).stacked_blocks.stacked.attn
    assert (labels.head_ve_lambda, labels.head_ve_gate, labels.head_ve_gain) == ("adam", "adam", "adam")
