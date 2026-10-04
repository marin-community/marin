# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.weight_diagnostics import gain_stats, matrix_stats


def test_stable_rank_counts_equal_singular_values_and_one_spike():
    key = jax.random.PRNGKey(0)
    q, _ = jnp.linalg.qr(jax.random.normal(key, (64, 64)))
    flat = q[:, :16] @ jnp.eye(16, 32)  # 16 equal singular values
    spiked = flat.at[:, 0].multiply(4.0)  # one direction 4x larger
    stats_flat = matrix_stats(flat)
    stats_spiked = matrix_stats(spiked)
    np.testing.assert_allclose(float(stats_flat["stable_rank_mean"]), 16.0, rtol=1e-3)
    # (15 + 16) / 16: one singular value of 4, fifteen of 1.
    np.testing.assert_allclose(float(stats_spiked["stable_rank_mean"]), 31 / 16, rtol=1e-3)


def test_channel_ratio_flags_one_outsized_output_channel_per_matrix():
    w = jnp.ones((2, 8, 4)).at[1, :, 2].multiply(5.0)  # two stacked matrices; the second has channel 2 at 5x
    stats = matrix_stats(w)
    np.testing.assert_allclose(float(stats["channel_ratio_max"]), 5.0 / 2.0, rtol=1e-5)  # max 5 over mean 8/4
    np.testing.assert_allclose(float(stats["channel_ratio_mean"]), (1.0 + 2.5) / 2, rtol=1e-5)


def test_gain_stats_measure_distance_from_one():
    stats = gain_stats(jnp.array([1.0, 0.5, 3.0]))
    assert float(stats["gain_dev_max"]) == 2.0 and float(stats["gain_min"]) == 0.5


def test_layer_output_stats_report_max_and_rms_per_layer():
    mesh, model = t._model(ngram_stat_rows=0, layer_output_stats=True)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, t._SEQ), 0, t._VOCAB)
    with jax.set_mesh(mesh):
        _, metrics = eqx.filter_jit(
            lambda m: m.next_token_loss(tokens, jnp.ones(tokens.shape), return_router_metrics=True)
        )(model)
    for i in range(2):
        mx, rms = float(metrics[f"train/aux/outstat/mlp_max_L{i}"]), float(metrics[f"train/aux/outstat/mlp_rms_L{i}"])
        assert mx >= rms > 0
