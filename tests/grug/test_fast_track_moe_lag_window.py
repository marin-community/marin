# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``moe_lag_window``: the routed experts' lag window against a direct sum over lags with its document
masking, and a tiny model that starts as the baseline, trains the gain, and routes the new leaves to their groups."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import MOE_LAG_WINDOW, MoEMLP, Transformer
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def test_lag_window_matches_direct_sum_within_documents():
    config = dataclasses.replace(tiny_config(), moe_lag_window=True)
    batch, seq = 2, 9
    segments = jnp.array([[0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 2, 2, 2]])
    with jax.set_mesh(tiny_mesh()):
        mlp = MoEMLP.init(config, key=jax.random.key(0))
        gain = jax.random.normal(jax.random.key(1), (config.latent_dim,))
        mlp = eqx.tree_at(lambda m: m.lag_gain, mlp, gain)
        narrow = jax.random.normal(jax.random.key(2), (batch * seq, config.hidden_dim // 4))
        got = mlp._lag_window(narrow, batch, segments)

    x, seg = np.asarray(narrow).reshape(batch, seq, -1), np.asarray(segments)
    window = np.zeros((batch, seq, MOE_LAG_WINDOW * x.shape[-1]), np.float32)
    for b in range(batch):
        for t in range(seq):
            for lag in range(1, MOE_LAG_WINDOW + 1):
                s = t - lag
                if s >= 0 and seg[b, s] == seg[b, t]:
                    window[b, t, (lag - 1) * x.shape[-1] : lag * x.shape[-1]] = x[b, s]
    up = window.reshape(batch * seq, -1) @ np.asarray(mlp.lag_up)
    want = np.asarray(gain) * up / np.sqrt(np.mean(up**2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_tiny_model_starts_as_baseline_and_trains_the_gain():
    config = dataclasses.replace(tiny_config(), moe_lag_window=True)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, moe_lag_window=False), key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        # Zero-initialised gain: the window adds nothing at step 0. Not bit-exact: W_down widens the fused input
        # GEMM, which rounds the router and latent columns differently.
        np.testing.assert_allclose(forward(model, tokens), forward(plain, tokens), rtol=1e-3, atol=1e-3)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)

    assert np.isfinite(float(loss))
    for stack in (grads.stacked_blocks, grads.kda_blocks):
        assert float(jnp.linalg.norm(stack.stacked.mlp.lag_gain)) > 0


def test_lag_window_leaves_route_to_muonh_and_adam():
    config = dataclasses.replace(tiny_config(), moe_lag_window=True)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_array))
    labels = {jax.tree_util.keystr(path): label for path, label in jax.tree_util.tree_flatten_with_path(mask)[0]}

    def labels_ending(leaf: str) -> set[str]:
        found = {label for name, label in labels.items() if name.endswith(f".mlp.{leaf}")}
        assert found, leaf
        return found

    assert labels_ending("lag_down") == {"muonh"}
    assert labels_ending("lag_up") == {"muonh"}
    assert labels_ending("lag_gain") == {"adam"}
