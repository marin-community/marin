# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss


def _reference_linear_cross_entropy(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    weight: jax.Array,
    *,
    reduction: str,
    logsumexp_weight: float | None = None,
) -> jax.Array:
    flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
    flat_labels = labels.reshape((-1,)).astype(jnp.int32)
    flat_weight = weight.reshape((-1,))
    logits = flat_hidden @ lm_head
    logsumexp = jax.nn.logsumexp(logits, axis=-1)
    label_logits = logits[jnp.arange(flat_labels.shape[0]), flat_labels]
    loss = logsumexp - label_logits
    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        loss = loss + logsumexp_weight * (logsumexp**2)
    loss = loss * flat_weight.astype(loss.dtype)

    if reduction == "none":
        return loss.reshape(labels.shape)
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction == "mean":
        denom = jnp.sum(flat_weight)
        return jnp.where(denom != 0, jnp.sum(loss) / denom, jnp.zeros_like(denom))
    raise ValueError(f"Unknown reduction: {reduction}")


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_fused_linear_softmax_cross_entropy_loss_matches_reference_without_mesh(reduction: str):
    hidden = jnp.linspace(-0.7, 0.9, 2 * 3 * 4, dtype=jnp.float32).reshape((2, 3, 4))
    lm_head = jnp.linspace(-0.4, 0.6, 4 * 7, dtype=jnp.float32).reshape((4, 7))
    labels = jnp.array([[0, 2, 6], [4, 1, 3]], dtype=jnp.int32)
    weight = jnp.array([[1.0, 0.5, 1.0], [0.25, 0.0, 1.0]], dtype=jnp.float32)

    actual = fused_linear_softmax_cross_entropy_loss(
        hidden,
        lm_head,
        labels,
        weight=weight,
        reduction=reduction,
        logsumexp_weight=0.01,
        implementation="xla",
    )
    expected = _reference_linear_cross_entropy(
        hidden,
        lm_head,
        labels,
        weight,
        reduction=reduction,
        logsumexp_weight=0.01,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("implementation", ["xla", ("pallas_gpu", "xla")])
def test_cross_entropy_model_sharded_vocab_matches_reference_and_gradients(implementation: str | tuple[str, ...]):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two devices; run with XLA_FLAGS=--xla_force_host_platform_device_count=2")

    hidden = jnp.linspace(-0.8, 0.7, 2 * 3 * 4, dtype=jnp.float32).reshape((2, 3, 4))
    lm_head = jnp.linspace(-0.5, 0.4, 4 * 8, dtype=jnp.float32).reshape((4, 8))
    labels = jnp.array([[0, 3, 7], [6, 1, 4]], dtype=jnp.int32)
    weight = jnp.array([[1.0, 0.25, 1.0], [0.5, 0.0, 1.0]], dtype=jnp.float32)

    mesh = Mesh(
        np.array(devices[:2], dtype=object).reshape((1, 2)),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    def sharded_loss(hidden_value: jax.Array, lm_head_value: jax.Array) -> jax.Array:
        return fused_linear_softmax_cross_entropy_loss(
            hidden_value,
            lm_head_value,
            labels_sharded,
            weight=weight_sharded,
            reduction="mean",
            logsumexp_weight=0.02,
            implementation=implementation,
        )

    def reference_loss(hidden_value: jax.Array, lm_head_value: jax.Array) -> jax.Array:
        return _reference_linear_cross_entropy(
            hidden_value,
            lm_head_value,
            labels,
            weight,
            reduction="mean",
            logsumexp_weight=0.02,
        )

    with jax.set_mesh(mesh):
        hidden_sharded = jax.device_put(hidden, NamedSharding(mesh, P("data", None, None)))
        labels_sharded = jax.device_put(labels, NamedSharding(mesh, P("data", None)))
        weight_sharded = jax.device_put(weight, NamedSharding(mesh, P("data", None)))
        lm_head_sharded = jax.device_put(lm_head, NamedSharding(mesh, P(None, "model")))

        warning_context = pytest.warns(RuntimeWarning)
        if implementation == "xla":
            warning_context = nullcontext()
        with warning_context:
            actual, (actual_hidden_grad, actual_lm_head_grad) = jax.jit(
                jax.value_and_grad(sharded_loss, argnums=(0, 1))
            )(hidden_sharded, lm_head_sharded)

    expected, (expected_hidden_grad, expected_lm_head_grad) = jax.value_and_grad(reference_loss, argnums=(0, 1))(
        hidden,
        lm_head,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_hidden_grad, expected_hidden_grad, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_lm_head_grad, expected_lm_head_grad, rtol=1e-5, atol=1e-5)
