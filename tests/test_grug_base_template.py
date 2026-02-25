# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax

from levanter.callbacks.watch import WatchConfig
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask as GrugAttentionMask

from experiments.grug.base.train import (
    GrugTrainState,
    _make_train_step,
    run_grug,
)


class DummyModel(eqx.Module):
    w: jax.Array

    def compute_next_token_loss(
        self,
        token_ids: jax.Array,
        loss_weight: jax.Array,
        *,
        mask=None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
    ) -> jax.Array:
        del token_ids, loss_weight, mask, reduction, logsumexp_weight
        return jnp.mean(jnp.square(self.w))


def _build_state(params: DummyModel, optimizer: optax.GradientTransformation) -> GrugTrainState:
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        ema_params=params,
    )


def test_grug_base_train_step_with_watch_matches_base_step():
    optimizer = optax.adam(1e-2)
    mp = jmp.get_policy("f32")

    state_for_base = _build_state(DummyModel(jnp.array([1.0, -2.0], dtype=jnp.float32)), optimizer)
    state_for_watch = _build_state(DummyModel(jnp.array([1.0, -2.0], dtype=jnp.float32)), optimizer)
    batch = GrugLmExample(
        tokens=jnp.zeros((1, 4), dtype=jnp.int32),
        loss_weight=jnp.ones((1, 4), dtype=jnp.float32),
        attn_mask=GrugAttentionMask.causal(),
    )

    base_step = _make_train_step(optimizer, mp, z_loss_weight=0.0, ema_beta=None)
    watch_step = _make_train_step(
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

    next_base, metrics_base, base_watch_stats = base_step(state_for_base, batch, compute_watch=False)
    next_watch, metrics_watch, watch_stats = watch_step(state_for_watch, batch, compute_watch=True)

    assert int(next_base.step) == 1
    assert int(next_watch.step) == 1
    assert jnp.allclose(next_base.params.w, next_watch.params.w)
    assert jnp.allclose(next_base.ema_params.w, next_watch.ema_params.w)
    assert jnp.allclose(metrics_base["train/loss"], metrics_watch["train/loss"])
    assert base_watch_stats is None
    assert watch_stats
    assert any(key.startswith("grad/") for key in watch_stats)
    assert any(key.startswith("params/") for key in watch_stats)
    assert any(key.startswith("updates/") for key in watch_stats)


def test_grug_base_run_metric_contract_skeleton():
    """Guardrail: base trainer keeps core logging/eval surfaces for parity tracking.

    This is intentionally lightweight and source-based; we can replace it with
    full run-level key assertions once the experiment harness is stabilized.
    """

    source = inspect.getsource(run_grug)
    required_snippets = [
        '"train/loss"',
        '"throughput/hook_time"',
        '"throughput/loading_time"',
        "callbacks.log_performance_stats",
        "construct_log_dict",
        "prefix=eval_cfg.prefix",
    ]

    for snippet in required_snippets:
        assert snippet in source
