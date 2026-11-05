# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``layer_output_stats_every``: per-layer output-stream max|h| and rms(h) reach the train metrics, the
final ones match the normed hidden state, and the loss and gradients are those of the flag off."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.belay.model_probe import SEQ, tiny_config, tiny_mesh
from experiments.grug.fast_track.model import LAYER_OUTPUT_STATS_PREFIX, LocalMixer, Transformer


@eqx.filter_jit
def _train_terms(model: Transformer, tokens: jax.Array):
    """The loss, its metrics and gradients, and the final normed hidden state."""

    def loss_fn(m):
        return m.next_token_loss(tokens, jnp.ones(tokens.shape, jnp.float32), return_router_metrics=True)

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    return loss, metrics, grads, model(tokens)[0].astype(jnp.float32)


@pytest.mark.parametrize("attn_res", [True, False])
def test_layer_output_stats_log_without_changing_training(attn_res):
    off = tiny_config()
    if not attn_res:
        # KDA layers run only in the AttnRes loop.
        off = dataclasses.replace(off, attn_res=False, local_mixer=LocalMixer.SLIDING_WINDOW)
    on = dataclasses.replace(off, layer_output_stats_every=100)
    with jax.set_mesh(tiny_mesh()):
        tokens = jax.random.randint(jax.random.key(1), (2, SEQ), 0, off.vocab_size)
        model_off = Transformer.init(off, key=jax.random.key(0))
        model_on = Transformer.init(on, key=jax.random.key(0))
        loss_off, metrics_off, grads_off, _ = _train_terms(model_off, tokens)
        loss_on, metrics_on, grads_on, hidden = _train_terms(model_on, tokens)

    np.testing.assert_array_equal(loss_on, loss_off)
    for g_on, g_off in zip(jax.tree.leaves(grads_on), jax.tree.leaves(grads_off), strict=True):
        np.testing.assert_array_equal(g_on, g_off)

    assert not any(k.startswith(LAYER_OUTPUT_STATS_PREFIX) for k in metrics_off)
    stats = {
        k.removeprefix(LAYER_OUTPUT_STATS_PREFIX): float(v)
        for k, v in metrics_on.items()
        if k.startswith(LAYER_OUTPUT_STATS_PREFIX)
    }
    expected = {f"{s}_L{i}" for s in ("max_abs", "rms") for i in range(on.num_layers)} | {"max_abs_final", "rms_final"}
    assert set(stats) == expected
    for i in range(on.num_layers):
        assert stats[f"max_abs_L{i}"] >= stats[f"rms_L{i}"] > 0

    h = np.asarray(hidden, np.float32)
    np.testing.assert_allclose(stats["max_abs_final"], np.abs(h).max(), rtol=1e-6)
    np.testing.assert_allclose(stats["rms_final"], np.sqrt(np.mean(np.square(h))), rtol=1e-5)
