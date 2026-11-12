# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The dense presets are the models they are named after, and remat and QK-norm behave as documented."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from experiments.grug.dense_synthetic.model import GrugModelConfig, Transformer
from experiments.grug.dense_synthetic.train import PRESETS, dense_flops_per_token
from experiments.llama import llama_8b, llama_32b

TINY = dataclasses.replace(PRESETS["tiny"], attention_implementation="reference")


def _mesh() -> Mesh:
    axes = ("replica_dcn", "data", "expert", "model")
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), axes, axis_types=(AxisType.Explicit,) * len(axes))


def _parameter_count(cfg: GrugModelConfig) -> int:
    with jax.set_mesh(_mesh()):
        shapes = jax.eval_shape(lambda: Transformer.init(cfg, key=jax.random.PRNGKey(0)))
    return sum(leaf.size for leaf in jax.tree.leaves(shapes))


@eqx.filter_jit
def _loss_and_grads(model: Transformer, tokens: jax.Array) -> tuple[jax.Array, Transformer]:
    weights = jnp.ones(tokens.shape, dtype=jnp.float32)
    return eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, weights))(model)


def _tokens(cfg: GrugModelConfig) -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(1), (2, cfg.max_seq_len), 0, cfg.vocab_size)


@pytest.mark.parametrize(("size", "reference"), [("8b", llama_8b), ("32b", llama_32b)])
def test_presets_are_marins_llama_shapes(size, reference):
    """8b is Llama-3 8B and 32b is Marin's 32B shape, plus Qwen3's per-head QK-norm weights where enabled."""
    cfg = PRESETS[size]
    qk_norm = 2 * cfg.head_dim * cfg.num_layers if cfg.qk_norm else 0
    assert _parameter_count(cfg) == reference.total_trainable_params(cfg.vocab_size) + qk_norm


@pytest.mark.parametrize(("size", "reference"), [("8b", llama_8b), ("32b", llama_32b)])
def test_presets_charge_levanters_llama_flops(size, reference):
    """MFU is comparable with Levanter's own Llama and Qwen3 runs: the same FLOPs per token at every context."""
    cfg = PRESETS[size]
    for seq_len in (cfg.max_seq_len, 8192):
        assert dense_flops_per_token(cfg, seq_len) == reference.flops_per_token(cfg.vocab_size, seq_len)


def test_remat_modes_keep_the_loss_and_gradients():
    with jax.set_mesh(_mesh()):
        model = Transformer.init(TINY, key=jax.random.PRNGKey(0))
        tokens = _tokens(TINY)
        expected_loss, expected_grads = _loss_and_grads(model, tokens)
        for mode in ("save_dots", "none"):
            remat_model = dataclasses.replace(model, config=dataclasses.replace(TINY, remat_mode=mode))
            loss, grads = _loss_and_grads(remat_model, tokens)
            np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)
            for grad, expected in zip(jax.tree.leaves(grads), jax.tree.leaves(expected_grads), strict=True):
                np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("qk_norm", [True, False])
def test_qk_norm_makes_attention_independent_of_query_and_key_scale(qk_norm):
    """QK-norm divides out a common scale on w_q and w_k; without it, the scale sharpens the attention logits."""
    cfg = dataclasses.replace(TINY, qk_norm=qk_norm)
    with jax.set_mesh(_mesh()):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        scaled = eqx.tree_at(
            lambda m: (m.blocks.stacked.attn.w_q, m.blocks.stacked.attn.w_k), model, replace_fn=lambda w: 30 * w
        )
        tokens = _tokens(cfg)
        loss, _ = _loss_and_grads(model, tokens)
        scaled_loss, _ = _loss_and_grads(scaled, tokens)
    if qk_norm:
        np.testing.assert_allclose(scaled_loss, loss, rtol=1e-5)
    else:
        assert abs(float(scaled_loss) - float(loss)) > 1e-3
