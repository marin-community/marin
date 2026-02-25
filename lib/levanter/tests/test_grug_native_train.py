# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jmp
import optax

from levanter.callbacks.watch import WatchConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.model import GrugModelConfig
from levanter.grug_native.train import (
    GrugTrainState,
    _make_train_step,
)


def _build_state(params: jax.Array, optimizer: optax.GradientTransformation) -> GrugTrainState:
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        training_key=jax.random.PRNGKey(0),
        ema_params=params,
    )


def test_train_step_with_watch_matches_base_step(monkeypatch):
    def fake_loss_fn(params, token_ids, loss_weight, cfg, *, mask=None, reduction="mean", logsumexp_weight=None):
        del token_ids, loss_weight, cfg, mask, reduction, logsumexp_weight
        return jnp.mean(jnp.square(params))

    monkeypatch.setattr("levanter.grug_native.train.grug_loss_fn", fake_loss_fn)

    model_config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=4,
    )
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")
    params_base = jnp.array([1.0, -2.0], dtype=jnp.float32)
    params_watch = jnp.array([1.0, -2.0], dtype=jnp.float32)

    state_for_base = _build_state(params_base, optimizer)
    state_for_watch = _build_state(params_watch, optimizer)
    batch = GrugLmExample(
        tokens=jnp.zeros((1, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((1, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    train_step = _make_train_step(model_config, optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    watch_step = _make_train_step(
        model_config,
        optimizer,
        mp,
        z_loss_weight=0.0,
        ema_beta=None,
        watch_config=WatchConfig(
            watch_targets=["grads", "params", "updates"],
            include_norms=True,
            include_per_parameter_norms=True,
            include_histograms=False,
            split_scan_layers=True,
            interval=1,
        ),
    )

    next_base, metrics_base, base_watch_stats = train_step(state_for_base, batch, compute_watch=False)
    next_watch, metrics_watch, watch_stats = watch_step(state_for_watch, batch, compute_watch=True)

    assert int(next_base.step) == 1
    assert int(next_watch.step) == 1
    assert jnp.allclose(next_base.params, next_watch.params)
    assert jnp.allclose(next_base.ema_params, next_watch.ema_params)
    assert jnp.allclose(metrics_base["train/loss"], metrics_watch["train/loss"])
    assert base_watch_stats is None
    assert watch_stats
    assert any(key.startswith("grad/") for key in watch_stats)
    assert any(key.startswith("params/") for key in watch_stats)
    assert any(key.startswith("updates/") for key in watch_stats)
