# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``lm_head_bilinear``: the widened head input against a direct computation, a tiny model that starts at
the baseline loss and trains the gain, and the new leaves' optimizer groups."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


def _config():
    return dataclasses.replace(tiny_config(), lm_head_bilinear=True)


def test_head_input_appends_gained_bilinear_features():
    config = _config()
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        gain = jax.random.normal(jax.random.key(1), (config.hidden_dim // 4,))
        model = eqx.tree_at(lambda m: m.head_bilinear_gain, model, gain)
        hidden = jax.random.normal(jax.random.key(2), (2, 5, config.hidden_dim))
        got = model._head_input(hidden)

    h = np.asarray(hidden)
    product = (h @ np.asarray(model.head_bilinear_a)) * (h @ np.asarray(model.head_bilinear_b))
    features = np.asarray(gain) * product / np.sqrt(np.mean(product**2, axis=-1, keepdims=True) + 1e-6)
    assert model.output_proj.shape == (config.hidden_dim + config.hidden_dim // 4, config.vocab_size)
    np.testing.assert_allclose(got, np.concatenate([h, features], axis=-1), rtol=1e-4, atol=1e-5)


def test_tiny_model_starts_at_baseline_loss_and_trains_the_gain():
    config = _config()
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        plain = Transformer.init(dataclasses.replace(config, lm_head_bilinear=False), key=jax.random.key(0))
        # The widened head's first rows are the baseline head; with the zero gain the new rows read zeros.
        plain = eqx.tree_at(lambda m: m.output_proj, plain, model.output_proj[: config.hidden_dim])
        tokens = jax.random.randint(jax.random.key(1), (2, config.max_seq_len), 0, config.vocab_size)
        weight = jnp.ones(tokens.shape, jnp.float32)

        def loss_fn(m):
            return m.next_token_loss(tokens, weight, logsumexp_weight=1e-4)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model)
        baseline = eqx.filter_jit(loss_fn)(plain)

    np.testing.assert_allclose(float(loss), float(baseline), rtol=1e-5)
    assert float(jnp.linalg.norm(grads.head_bilinear_gain)) > 0


def test_bilinear_leaves_route_like_dense_matrices_and_the_head():
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(_config(), key=jax.random.key(0))
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_array))

    assert mask.head_bilinear_a == "muonh"
    assert mask.head_bilinear_b == "muonh"
    assert mask.head_bilinear_gain == "adam"
    assert mask.output_proj == "adamh"
