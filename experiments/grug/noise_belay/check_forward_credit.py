# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from experiments.grug.noise_belay.forward_credit import (
    attnres_readout_credit,
    capped_token_loss,
    causal_convolution_credit,
    causal_convolution_parameter_credit,
    direction_frame,
    head_coordinate_credit,
    readout_parameter_credit,
    relu2_parameter_credit,
    sqrt_softplus_router_credit,
    token_readout_credit,
)

# These cases require the pinned runtime prepared by prepare.py.
pytestmark = pytest.mark.manual


@pytest.mark.parametrize("width", [512, 768, 1280])
def test_readout_cycle_recovers_quadratic_credit_at_belay_widths(width):
    rng = np.random.default_rng(7)
    point = jnp.asarray(rng.normal(size=(3, width)), dtype=jnp.float32)
    linear = jnp.asarray(rng.normal(size=(3, width)), dtype=jnp.float32)
    frame = direction_frame(width, jax.random.PRNGKey(13))

    def loss(value):
        return jnp.sum(0.5 * value**2 + value * linear, axis=-1) / width

    # A quadratic's central differences equal its derivatives at any step.
    # This checks frame decoding, including the non-power-of-two normalization.
    actual = token_readout_credit(loss, point, frame, relative_step=0.1, chunk_size=37)
    np.testing.assert_allclose(actual, (point + linear) / width, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("z_loss_weight", [0.0, 0.0001])
def test_head_credit_matches_explicit_perturbed_forward_losses(z_loss_weight):
    raw = np.random.default_rng(19).normal(0, 12, (3, 17))
    labels = np.array([0, 16, -1])
    sigma = 0.1
    cap = 10.0
    expected = np.zeros_like(raw)

    def loss(logits, target):
        capped = cap * np.tanh(logits / cap)
        partition = logsumexp(capped)
        return partition - capped[target] + z_loss_weight * partition**2

    for token, label in enumerate(labels):
        if label < 0:
            continue
        for coordinate in range(raw.shape[-1]):
            upper, lower = raw[token].copy(), raw[token].copy()
            upper[coordinate] += sigma
            lower[coordinate] -= sigma
            expected[token, coordinate] = (loss(upper, label) - loss(lower, label)) / (2 * sigma)
    actual = jax.jit(head_coordinate_credit)(
        jnp.asarray(raw),
        jnp.asarray(labels),
        sigma=sigma,
        soft_cap=cap,
        z_loss_weight=z_loss_weight,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("width", [16, 24])
def test_attnres_readout_reuses_projections_without_changing_forward_probes(width):
    rng = np.random.default_rng(31)
    point = jnp.asarray(rng.normal(size=(3, width)), jnp.float32)
    sources = jnp.asarray(rng.normal(size=(4, 3, width)), jnp.float32)
    query = jnp.asarray(rng.normal(0, 0.2, size=width), jnp.float32)
    head = jnp.asarray(rng.normal(0, 0.1, size=(width, 33)), jnp.float32)
    gain = jnp.asarray(rng.uniform(0.5, 1.5, size=width), jnp.float32)
    labels = jnp.array([0, 32, -1])
    frame = direction_frame(width, jax.random.PRNGKey(5))

    def direct_loss(moved):
        all_sources = jnp.concatenate([sources, moved[None]], axis=0)
        keys = all_sources / jnp.sqrt(jnp.mean(all_sources**2, axis=-1, keepdims=True) + 1e-5)
        logits = jnp.einsum("ntd,d->nt", keys, query)
        mixed = jnp.sum(jax.nn.softmax(logits, axis=0)[..., None] * all_sources, axis=0)
        normalized = mixed / jnp.sqrt(jnp.mean(mixed**2, axis=-1, keepdims=True) + 1e-5)
        return capped_token_loss((normalized * gain) @ head, labels, 10.0, 0.0001)

    expected = token_readout_credit(direct_loss, point, frame, relative_step=0.05, chunk_size=7)
    actual = attnres_readout_credit(
        point,
        sources,
        query,
        head,
        gain,
        labels,
        frame,
        relative_step=0.05,
        chunk_size=7,
        epsilon=1e-5,
        soft_cap=10.0,
        z_loss_weight=0.0001,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("width,query_scale,source_scale", [(16, 0.0, 1e-3), (24, 0.2, 1.0)])
def test_readout_parameter_credit_matches_float64_forward_oracle(width, query_scale, source_scale):
    rng = np.random.default_rng(78)
    sources = rng.normal(0, source_scale, (4, 5, width))
    query = rng.normal(0, query_scale, width)
    head = rng.normal(0, 0.1, (width, 19))
    gain = rng.uniform(0.5, 1.5, width)
    labels = np.array([0, 18, -1, 4, 7])
    weights = np.array([1.0, 0.25, 1.0, 0.0, 1.0])
    directions = direction_frame(width, jax.random.PRNGKey(4))
    actual = jax.jit(readout_parameter_credit, static_argnames=("direction_chunk",))(
        jnp.asarray(sources),
        jnp.asarray(query),
        jnp.asarray(head),
        jnp.asarray(gain),
        jnp.asarray(labels),
        jnp.asarray(weights),
        directions,
        head_step=0.1,
        query_step=0.005,
        direction_chunk=7,
        epsilon=1e-5,
        soft_cap=10.0,
        z_loss_weight=0.0001,
    )

    def readout(current_query):
        keys = sources / np.sqrt(np.mean(sources**2, axis=-1, keepdims=True) + 1e-5)
        logits = keys @ current_query
        source_weights = np.exp(logits - logsumexp(logits, axis=0, keepdims=True))
        mixed = np.sum(source_weights[..., None] * sources, axis=0)
        normalized = mixed / np.sqrt(np.mean(mixed**2, axis=-1, keepdims=True) + 1e-5)
        return normalized, (normalized * gain) @ head

    def loss(raw):
        capped = 10 * np.tanh(raw / 10)
        partition = logsumexp(capped, axis=-1)
        value = partition - capped[np.arange(5), np.maximum(labels, 0)] + 0.0001 * partition**2
        return np.where(labels >= 0, value, 0)

    normalized, raw = readout(query)
    logit_credit = np.zeros_like(raw)
    for coordinate in range(19):
        move = np.zeros_like(raw)
        move[:, coordinate] = 0.1
        logit_credit[:, coordinate] = (loss(raw + move) - loss(raw - move)) / 0.2 * weights
    expected_head = (normalized * gain).T @ logit_credit
    expected_gain = np.sum(normalized * (logit_credit @ head.T), axis=0)
    expected_query = np.zeros_like(query)
    for row in np.asarray(directions, dtype=np.float64):
        upper = loss(readout(query + 0.005 * row)[1])
        lower = loss(readout(query - 0.005 * row)[1])
        expected_query += np.sum((upper - lower) * weights) / 0.01 * row / len(directions)
    for observed, expected in zip(actual, (expected_head, expected_gain, expected_query), strict=True):
        np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("width", [16, 24])
def test_relu2_parameter_credit_matches_forward_cycle_with_zero_crossings(width):
    rng = np.random.default_rng(12)
    inputs = rng.normal(size=(5, 7))
    preactivation = rng.normal(size=(5, width))
    # Deliberately cross the kink; replacing finite slopes with derivatives fails.
    preactivation[:, :3] = [0.0, -0.001, 0.001]
    preactivation[-1] = 0
    weight = rng.normal(0, 0.2, (width, 9))
    credit = rng.normal(size=(5, 9)) * np.array([1, 0, 0.25, 1, 1])[:, None]
    sigma = 0.05
    steps = sigma * np.maximum(np.sqrt(np.mean(preactivation**2, axis=-1)), 1e-6)
    frame = np.asarray(direction_frame(width, jax.random.PRNGKey(6)), np.float64)
    hidden_credit = np.zeros_like(preactivation)
    for direction in frame:
        upper = np.maximum(preactivation + steps[:, None] * direction, 0) ** 2 @ weight
        lower = np.maximum(preactivation - steps[:, None] * direction, 0) ** 2 @ weight
        score = np.sum((upper - lower) * credit, axis=-1) / (2 * steps)
        hidden_credit += score[:, None] * direction / len(frame)
    expected_up = inputs.T @ hidden_credit
    # Independent coordinate perturbations of the linear output projection.
    expected_down = np.zeros_like(weight)
    hidden = np.maximum(preactivation, 0) ** 2
    for i in range(width):
        for j in range(9):
            move = np.zeros_like(weight)
            move[i, j] = 0.1
            upper = np.sum((hidden @ (weight + move)) * credit)
            lower = np.sum((hidden @ (weight - move)) * credit)
            expected_down[i, j] = (upper - lower) / 0.2
    observed = jax.jit(relu2_parameter_credit)(
        jnp.asarray(inputs),
        jnp.asarray(preactivation),
        jnp.asarray(weight),
        jnp.asarray(credit),
        relative_step=sigma,
    )
    np.testing.assert_allclose(observed.up, expected_up, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(observed.down, expected_down, atol=1e-5, rtol=1e-4)


def test_causal_convolution_credit_matches_forward_perturbations_at_document_edges():
    rng = np.random.default_rng(87)
    point = rng.normal(size=(2, 9, 4))
    weight = rng.normal(size=(4, 4))
    measured = rng.normal(size=point.shape)
    segments = np.array([[0, 0, 1, 1, 1, 2, 2, 2, 2], [0, 1, 1, 1, 2, 2, 3, 3, 3]])

    def score(value, kernel=weight):
        total = 0.0
        for b in range(2):
            for t in range(9):
                for lag in range(min(t + 1, 4)):
                    if segments[b, t] == segments[b, t - lag]:
                        total += np.sum(kernel[lag] * value[b, t - lag] * measured[b, t])
        return total

    expected = np.zeros_like(point)
    for index in np.ndindex(point.shape):
        move = np.zeros_like(point)
        move[index] = 0.03
        expected[index] = (score(point + move) - score(point - move)) / 0.06
    actual = jax.jit(causal_convolution_credit)(jnp.asarray(measured), jnp.asarray(weight), jnp.asarray(segments))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)

    expected_kernel = np.zeros_like(weight)
    for index in np.ndindex(weight.shape):
        move = np.zeros_like(weight)
        move[index] = 0.03
        expected_kernel[index] = (score(point, weight + move) - score(point, weight - move)) / 0.06
    actual_kernel = jax.jit(causal_convolution_parameter_credit, static_argnames=("kernel_size",))(
        jnp.asarray(point),
        jnp.asarray(measured),
        jnp.asarray(segments),
        kernel_size=len(weight),
    )
    np.testing.assert_allclose(actual_kernel, expected_kernel, atol=1e-5, rtol=1e-4)


def test_router_credit_matches_full_forward_with_drop_renormalization():
    logits = np.array(
        [
            [0.2, -1.1, 0.8, 2.0],
            [-2, 0.1, 0.7, 1.4],
            [0.2, 0.3, -0.2, 1.0],
            [-80, -75, -90, -72],
        ],
        np.float64,
    )
    scores = np.array(
        [
            [0.2, -0.8, 1.1, 0.6],
            [2, -3, 0.1, -0.8],
            [0.4, 0.1, 1.2, -0.2],
            [-0.4, 0.6, 0.1, 2],
        ],
        np.float64,
    )
    kept = np.array([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 1]], bool)
    sigma = 0.005

    def evaluate(point):
        mass = np.sqrt(np.maximum(np.logaddexp(0, point), 1e-30))
        weights = 2.5 * mass / (mass.sum(-1, keepdims=True) + 1e-9)
        surviving = np.where(kept, weights, 0).sum(-1, keepdims=True)
        scale = np.where(
            surviving > 0,
            weights.sum(-1, keepdims=True) / np.where(surviving > 0, surviving, 1),
            1,
        )
        return (np.where(kept, weights * scores, 0).sum(-1, keepdims=True) * scale)[:, 0]

    expected = np.zeros_like(logits)
    for column in range(logits.shape[1]):
        displacement = np.zeros_like(logits)
        displacement[:, column] = sigma
        expected[:, column] = (evaluate(logits + displacement) - evaluate(logits - displacement)) / (2 * sigma)
    actual = jax.jit(
        lambda x: sqrt_softplus_router_credit(x, jnp.asarray(scores), jnp.asarray(kept), sigma=sigma, renorm_sum=2.5)
    )(jnp.asarray(logits))
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-5)
