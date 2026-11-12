# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from levanter.optim.rmnph import RMNPHConfig, scale_with_rmnph
from levanter.main.train_lm import TrainLmConfig, initialize_rmnph_model
from levanter.models.qwen import Qwen3Config


@pytest.mark.parametrize("out_first", [True, False])
def test_rmnph_matches_row_normalization_and_sphere_projection(out_first):
    layer = hax.nn.Linear.init(
        hax.Axis("in", 5), hax.Axis("out", 3), key=jax.random.PRNGKey(0), use_bias=False, out_first=out_first
    )
    grad = dataclasses.replace(
        layer, weight=hax.named(jax.random.normal(jax.random.PRNGKey(1), layer.weight.array.shape), layer.weight.axes)
    )
    transform = scale_with_rmnph(momentum=0.95, nesterov=True, learning_rate=0.015)
    delta, _ = jax.jit(transform.update)(grad, transform.init(layer), layer)
    w = np.asarray(layer.weight.array)
    direction = np.asarray(grad.weight.array)
    axis = -1 if out_first else -2
    direction = direction / np.linalg.norm(direction, axis=axis, keepdims=True)
    candidate = w - 0.015 * direction * np.linalg.norm(w) / np.linalg.norm(direction)
    expected = candidate * np.linalg.norm(w) / np.linalg.norm(candidate)
    actual = np.asarray(layer.weight.array + delta.weight.array)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_embedding_rate_does_not_change_other_groups():
    In, Out, Vocab = hax.Axis("in", 5), hax.Axis("out", 3), hax.Axis("vocab", 7)
    params = {
        "hidden": hax.nn.Linear.init(In, Out, key=jax.random.PRNGKey(0), use_bias=False),
        "lm_head": hax.nn.Linear.init(In, Vocab, key=jax.random.PRNGKey(1), use_bias=False),
        "embedding": hax.nn.Embedding.init(Vocab, In, key=jax.random.PRNGKey(2)),
        "norm": hax.ones(In),
    }
    gradients = jax.tree.map(jnp.ones_like, params)
    results = []
    for rate in [0.01, 0.1]:
        optimizer = RMNPHConfig(embedding_lr=rate, warmup=0, min_lr_ratio=0).build(10)
        updates, _ = jax.jit(optimizer.update)(gradients, optimizer.init(params), params)
        results.append(optax.apply_updates(params, updates))
    for group in ["hidden", "lm_head", "norm"]:
        for a, b in zip(jax.tree.leaves(results[0][group]), jax.tree.leaves(results[1][group])):
            np.testing.assert_array_equal(a, b)
    assert not jnp.allclose(results[0]["embedding"].weight.array, results[1]["embedding"].weight.array)


def test_radius_scales_only_hyperball_groups():
    model_config = Qwen3Config(
        num_layers=1,
        hidden_dim=32,
        intermediate_dim=64,
        num_heads=4,
        num_kv_heads=4,
        gradient_checkpointing=False,
        tie_word_embeddings=False,
    )
    config = TrainLmConfig(model=model_config, optimizer=RMNPHConfig(), hyperball_radius_scale=0.8)
    original = model_config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(3))
    scaled = initialize_rmnph_model(config, hax.Axis("vocab", 32), key=jax.random.PRNGKey(3))
    labels = config.optimizer.create_mask(original)

    def check(label, before, after):
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
            if label in ("rmnph", "adamh"):
                np.testing.assert_allclose(b, 0.8 * a, rtol=1e-5, atol=1e-6)
            else:
                np.testing.assert_array_equal(a, b)
        return label

    jax.tree.map(check, labels, original, scaled)
