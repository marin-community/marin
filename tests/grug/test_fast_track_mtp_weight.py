# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``mtp_weight``: a depth-1 MTP head adds a weighted t+2 loss to training (never to evals), and
``mtp_decay_frac`` decays its weight linearly to 0."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track.model import GrugModelConfig, Transformer
from experiments.grug.fast_track.train import mtp_weight_at

_HIDDEN = 64
_SEQ = 64
_WEIGHT = 0.1


def _config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=128,
        hidden_dim=_HIDDEN,
        intermediate_dim=_HIDDEN // 2,
        shared_expert_intermediate_dim=_HIDDEN // 2,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=_HIDDEN // 2,
        num_layers=2,
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
        mtp_weight=_WEIGHT,
        mtp_decay_frac=0.5,
    )


def _mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def test_mtp_loss_is_added_in_training_only_and_trains_its_head():
    cfg = _config()
    tokens = jax.random.randint(jax.random.key(1), (2, _SEQ), 0, cfg.vocab_size)
    loss_weight = jnp.ones((2, _SEQ), jnp.float32)
    with jax.set_mesh(_mesh()):
        model = Transformer.init(cfg, key=jax.random.key(0))

        def train_loss(m, weight):
            return m.next_token_loss(
                tokens, loss_weight, return_router_metrics=True, train_terms=True, mtp_weight=weight
            )

        eval_loss = float(eqx.filter_jit(lambda m: m.next_token_loss(tokens, loss_weight))(model))
        (loss, metrics), grads = eqx.filter_jit(eqx.filter_value_and_grad(train_loss, has_aux=True))(
            model, jnp.asarray(_WEIGHT)
        )
        off_loss, _ = eqx.filter_jit(train_loss)(model, jnp.asarray(0.0))

    mtp_loss = float(metrics["train/aux/mtp_loss"])
    assert mtp_loss > 0
    np.testing.assert_allclose(float(loss), eval_loss + _WEIGHT * mtp_loss, rtol=1e-5)
    np.testing.assert_allclose(float(off_loss), eval_loss, rtol=1e-6)
    assert float(jnp.linalg.norm(grads.w_mtp)) > 0


def test_mtp_weight_decays_linearly_to_zero_and_stays_off():
    cfg = _config()
    weights = [float(mtp_weight_at(cfg, jnp.asarray(step), 100)) for step in (0, 50, 100, 150)]
    np.testing.assert_allclose(weights, [_WEIGHT, _WEIGHT / 2, 0.0, 0.0], rtol=1e-6)
    assert float(mtp_weight_at(cfg, jnp.asarray(80), None)) == np.float32(_WEIGHT)
