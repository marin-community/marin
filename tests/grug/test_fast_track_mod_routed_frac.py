# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``mod_routed_frac``: a causal per-token score sends tokens at or above the layer's threshold to the
routed experts at the larger top-k and the rest to the shared experts only; the trainer moves the threshold
towards each step's score quantile."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track.model import (
    MOD_SCORE_QUANTILE_KEY,
    MOD_THRESHOLD_EMA,
    GrugModelConfig,
    Transformer,
)
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.train import _apply_mod_thresholds

_HIDDEN = 64
_SEQ = 64
_LAYERS = 2


def _config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=128,
        hidden_dim=_HIDDEN,
        intermediate_dim=_HIDDEN // 2,
        shared_expert_intermediate_dim=_HIDDEN // 2,
        num_shared_experts=1,
        num_experts=8,
        num_experts_per_token=2,
        latent_dim=_HIDDEN // 2,
        num_layers=_LAYERS,
        num_heads=2,
        num_kv_heads=2,
        local_kv_heads=None,
        global_kv_heads=None,
        head_dim=16,
        max_seq_len=_SEQ,
        sliding_window=_SEQ,
        global_every=2,
        capacity_factor=8.0,
        pooled_transport_capacity_factor=8.0,
        initializer_std=0.5 / math.sqrt(_HIDDEN),
        qk_norm=False,
        mod_routed_frac=0.5,
    )


def _mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def test_unrouted_tokens_get_no_routed_output_and_the_score_trains():
    cfg = _config()
    assert cfg.mod_routed_k == 4
    x = jax.random.normal(jax.random.key(1), (2, _SEQ, _HIDDEN), jnp.float32)
    cotangent = jax.random.normal(jax.random.key(2), (2, _SEQ, _HIDDEN), jnp.float32)
    with jax.set_mesh(_mesh()):
        model = Transformer.init(cfg, key=jax.random.key(0))
        mlp = model.layers()[0].mlp
        score_weight = jax.random.normal(jax.random.key(3), (_HIDDEN,), jnp.float32)
        mlp = eqx.tree_at(lambda m: (m.mod_score, m.mod_threshold), mlp, (score_weight, jnp.asarray(0.5, jnp.float32)))

        def loss_fn(m):
            routed, _ = m(x)
            return jnp.sum(routed.astype(jnp.float32) * cotangent)

        routed, stats = eqx.filter_jit(lambda m: m(x))(mlp)
        grads = eqx.filter_jit(eqx.filter_grad(loss_fn))(mlp)

    score = np.asarray(jnp.einsum("bsd,d->bs", x, score_weight))
    token_routed = score >= 0.5
    assert 0 < token_routed.mean() < 1
    norms = np.linalg.norm(np.asarray(routed, np.float32), axis=-1)
    assert (norms[~token_routed] == 0).all()
    assert (norms[token_routed] > 0).all()
    np.testing.assert_allclose(float(stats["outstat/mod_routed_share"]), token_routed.mean(), rtol=1e-6)
    np.testing.assert_allclose(float(stats["mod_score_quantile"]), np.quantile(score, 0.5), rtol=1e-5, atol=1e-6)
    assert float(jnp.linalg.norm(grads.mod_score)) > 0
    assert float(grads.mod_threshold) == 0


def test_trainer_moves_thresholds_towards_the_score_quantile():
    cfg = _config()
    tokens = jax.random.randint(jax.random.key(1), (2, _SEQ), 0, cfg.vocab_size)
    with jax.set_mesh(_mesh()):
        model = Transformer.init(cfg, key=jax.random.key(0))
        model = eqx.tree_at(
            lambda t: [stack.stacked.mlp.mod_score for stack in t.layer_stacks()],
            model,
            [
                jax.random.normal(jax.random.key(4), stack.stacked.mlp.mod_score.shape, jnp.float32)
                for stack in model.layer_stacks()
            ],
        )
        _, metrics = eqx.filter_jit(lambda m: m(tokens))(model)
        quantiles = metrics[MOD_SCORE_QUANTILE_KEY]
        updated = _apply_mod_thresholds(model, model, quantiles)

    assert quantiles.shape == (_LAYERS,)
    assert np.all(np.asarray(quantiles) != 0)
    thresholds = np.asarray([float(layer.mlp.mod_threshold) for layer in updated.layers()])
    np.testing.assert_allclose(thresholds, (1 - MOD_THRESHOLD_EMA) * np.asarray(quantiles), rtol=1e-6)

    labels = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_array))
    assert labels.stacked_blocks.stacked.mlp.mod_score == "adam"
