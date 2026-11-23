# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.logsumexp import batch_invariant_log_softmax, batch_invariant_logsumexp

# Actual TPUv4 predictor logits from the two-layer Snowball first-token STOP
# counterexample. Ordinary logsumexp shifted all 23 logprobs by one FP32 ULP
# between one predictor row and the 129-row uninterrupted continuation.
STOP_LOGITS = np.array(
    [
        0.052859075367450714,
        0.0981975719332695,
        -0.07207846641540527,
        0.046331483870744705,
        -0.031706441193819046,
        -0.016008514910936356,
        -0.07458018511533737,
        0.005578896030783653,
        0.02227604016661644,
        0.025030873715877533,
        -0.02058310993015766,
        0.07940946519374847,
        -0.03836846351623535,
        -0.04019194468855858,
        0.09802177548408508,
        0.03237268328666687,
        0.08273816108703613,
        -0.01608993485569954,
        -0.01230352371931076,
        0.03370737284421921,
        0.01677270606160164,
        -0.059621259570121765,
        0.02533578872680664,
    ],
    dtype=np.float32,
)


def test_stop_logprobs_match_across_row_counts_and_positions():
    normalize = jax.jit(batch_invariant_log_softmax)
    expected = np.asarray(normalize(jnp.asarray(STOP_LOGITS)))
    for rows in (1, 4, 7, 8, 9, 128, 129):
        for index in {0, rows - 1}:
            values = np.random.default_rng(rows).normal(size=(rows, STOP_LOGITS.size)).astype(np.float32)
            values[index] = STOP_LOGITS
            actual = np.asarray(normalize(jnp.asarray(values)))[index]
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    reference = STOP_LOGITS.astype(np.float64)
    reference -= np.log(np.exp(reference).sum())
    np.testing.assert_allclose(expected, reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("vocab", [23, 128, 257])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_log_normalizer_values_and_gradients_match_reference(vocab, dtype):
    logits = jax.random.normal(jax.random.key(vocab), (2, 9, vocab)).astype(dtype)
    actual, grad = jax.jit(jax.value_and_grad(lambda x: batch_invariant_logsumexp(x).sum()))(logits)
    expected, reference_grad = jax.jit(
        jax.value_and_grad(lambda x: jax.nn.logsumexp(x.astype(jnp.float32), -1).sum())
    )(logits)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad.astype(jnp.float32), reference_grad.astype(jnp.float32), rtol=1e-4, atol=1e-5)


def test_normalizer_gathers_vocabulary_and_preserves_row_sharding():
    devices = np.array(jax.devices())
    vocab_shards = 2 if len(devices) % 2 == 0 else 1
    mesh = Mesh(devices.reshape((-1, vocab_shards)), ("data", "model"), axis_types=(AxisType.Explicit,) * 2)
    values = np.random.default_rng(0).normal(size=(8 * len(devices), 128)).astype(np.float32)
    reference = np.asarray(jax.jit(batch_invariant_log_softmax)(jnp.asarray(values)))
    with jax.set_mesh(mesh):
        logits = jax.device_put(values, NamedSharding(mesh, P("data", "model")))
        actual = jax.jit(batch_invariant_log_softmax)(logits)
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), reference.view(np.uint32))
        normalizers = jax.jit(batch_invariant_logsumexp)(logits)
        assert normalizers.sharding.spec == P("data")


def test_logprobs_with_masked_vocabulary_match_reference():
    logits = jnp.array([[2.0, -jnp.inf, 1.0], [-jnp.inf, 0.0, -jnp.inf]])
    actual = jax.jit(batch_invariant_log_softmax)(logits)
    np.testing.assert_allclose(actual, jax.nn.log_softmax(logits), rtol=1e-5, atol=1e-7)


def test_selected_logprobs_match_separately_normalized_distribution():
    for rows in (1, 4, 129):
        logits = jnp.asarray(np.random.default_rng(rows).normal(size=(rows, 23)).astype(np.float32))
        targets = jnp.arange(rows) % 23
        selected = jax.jit(
            lambda x, ids: jnp.take_along_axis(x, ids[:, None], axis=-1).squeeze(-1) - batch_invariant_logsumexp(x)
        )(logits, targets)
        distribution = jax.jit(batch_invariant_log_softmax)(logits)
        np.testing.assert_array_equal(
            np.asarray(selected).view(np.uint32),
            np.asarray(jnp.take_along_axis(distribution, targets[:, None], axis=-1).squeeze(-1)).view(np.uint32),
        )


def test_logprobs_and_derivatives_match_through_nested_vmap():
    values = np.random.default_rng(129).normal(size=(3, 43, STOP_LOGITS.size)).astype(np.float32)
    values[0, 0] = STOP_LOGITS
    logits = jnp.asarray(values)
    direct = jax.jit(batch_invariant_log_softmax)(logits)
    mapped = jax.jit(jax.vmap(jax.vmap(batch_invariant_log_softmax)))(logits)
    np.testing.assert_array_equal(np.asarray(mapped).view(np.uint32), np.asarray(direct).view(np.uint32))
    objective = lambda x: jax.vmap(jax.vmap(batch_invariant_logsumexp))(x).sum()
    reference = lambda x: jax.nn.logsumexp(x, axis=-1).sum()
    np.testing.assert_allclose(jax.grad(objective)(logits), jax.grad(reference)(logits), rtol=1e-4, atol=1e-5)
    tangent = jnp.sin(logits)
    np.testing.assert_allclose(
        jax.jvp(objective, (logits,), (tangent,))[1],
        jax.jvp(reference, (logits,), (tangent,))[1],
        rtol=1e-5,
        atol=1e-5,
    )
