# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``mla_dynamic_kv_conv``: the per-token dynamic conv against a direct sum over lags with its document
masking, and a tiny model that starts as the baseline and trains the K and V tap generators."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer, dynamic_short_conv


def test_dynamic_conv_matches_direct_sum_within_documents():
    batch, seq, heads, width, dim, kernel = 2, 9, 3, 4, 5, 4
    keys = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(keys[0], (batch, seq, heads, width))
    source = jax.random.normal(keys[1], (batch, seq, dim))
    generator = jax.random.normal(keys[2], (dim, heads * kernel))
    segments = jnp.array([[0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 2, 2, 2]])

    got = dynamic_short_conv(x, source, generator, segments)

    taps = np.asarray(source @ generator).reshape(batch, seq, heads, kernel)
    x_np, seg = np.asarray(x), np.asarray(segments)
    want = np.zeros_like(x_np)
    for b in range(batch):
        for t in range(seq):
            for lag in range(kernel):
                s = t - lag
                if s >= 0 and seg[b, s] == seg[b, t]:
                    want[b, t] += taps[b, t, :, lag, None] * x_np[b, s]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_tiny_model_starts_as_baseline_and_trains_the_tap_generators():
    config = dataclasses.replace(tiny_config(), mla_dynamic_kv_conv=True)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, mla_dynamic_kv_conv=False), key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        # Zero-initialised generators: the convs add nothing at step 0.
        np.testing.assert_array_equal(forward(model, tokens), forward(plain, tokens))

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)

    assert np.isfinite(float(loss))
    attn_grads = grads.stacked_blocks.stacked.attn
    for name in ("dyn_conv_k", "dyn_conv_v"):
        assert float(jnp.linalg.norm(getattr(attn_grads, name))) > 0, name
