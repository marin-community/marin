# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import haliax as hax
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty, maybe_fused_next_token_loss

pytestmark = pytest.mark.tpu

Batch = hax.Axis("batch", 4)
Position = hax.Axis("position", 2)
Embed = hax.Axis("embed", 4)
Vocab = hax.Axis("vocab", 8)


class LossInputs(NamedTuple):
    embeddings: np.ndarray
    head: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    mapping: dict[str, str]
    placed: tuple[jax.Array, jax.Array]


def _arrays(mesh, contract_sharded):
    x = (np.arange(32, dtype=np.float32).reshape(4, 2, 4) - 15) / 23
    head = np.cos(np.arange(32, dtype=np.float32).reshape(8, 4)) / 3
    labels = np.arange(8, dtype=np.int32).reshape(4, 2)
    weights = np.asarray([[1, 0], [0.25, 1], [0, 1], [0.5, 1]], dtype=np.float32)
    mapping = {"batch": "data", "embed" if contract_sharded else "vocab": "model"}
    x_spec = P("data", None, "model" if contract_sharded else None)
    head_spec = P(None, "model") if contract_sharded else P("model", None)
    return LossInputs(
        x,
        head,
        labels,
        weights,
        mapping,
        (
            jax.device_put(x, NamedSharding(mesh, x_spec)),
            jax.device_put(head, NamedSharding(mesh, head_spec)),
        ),
    )


def _reference(x, head, labels, weights, reduction, zloss, softcap):
    logits = jnp.einsum("bse,ve->bsv", x, head, precision=jax.lax.Precision.HIGHEST)
    if softcap is not None:
        logits = softcap * jnp.tanh(logits / softcap)
    logz = jax.nn.logsumexp(logits, axis=-1)
    target = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    values = (logz - target + zloss * logz**2) * weights
    if reduction == "position":
        return jnp.sum(values, axis=1) / jnp.sum(weights, axis=1)
    if reduction == "sum":
        return jnp.sum(values)
    return values


@pytest.mark.parametrize(
    "shape,contract_sharded", [((2, 2), False), ((1, 4), False), ((2, 2), True)], ids=["vocab2", "vocab4", "contract2"]
)
@pytest.mark.parametrize(
    "reduction,zloss,softcap,argmax",
    [("none", 0.0, None, True), ("position", 0.1, 1.2, False), ("sum", 0.05, None, False)],
    ids=["per_position", "weighted_reduction", "sum_zloss"],
)
def test_global_vocab_loss_and_vjp_match_full_reference(shape, contract_sharded, reduction, zloss, softcap, argmax):
    if len(jax.devices()) < 4:
        pytest.skip("Requires four JAX devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(shape), ("data", "model"))
    x, head, labels, weights, mapping, placed = _arrays(mesh, contract_sharded)
    label = hax.named(jax.device_put(labels, NamedSharding(mesh, P("data", None))), (Batch, Position))
    weight = hax.named(jax.device_put(weights, NamedSharding(mesh, P("data", None))), (Batch, Position))

    def actual(a, b):
        output = fused_cross_entropy_loss_and_logsumexp_penalty(
            hax.named(a, (Batch, Position, Embed)),
            hax.named(b, (Vocab, Embed)),
            Contract=Embed,
            Label=Vocab,
            target_y=label,
            weight=weight,
            reduction=None if reduction == "none" else (hax.mean if reduction == "position" else hax.sum),
            reduction_axis=Position if reduction == "position" else None,
            logsumexp_weight=zloss,
            logit_soft_cap=softcap,
            block_size=2,
            dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            return_argmax=argmax,
        )
        return output[0].array if argmax else output.array

    with jax.set_mesh(mesh), hax.axis_mapping(mapping):
        value, pullback = jax.vjp(jax.jit(actual), *placed)
        cotangent = jnp.asarray(np.linspace(0.2, 1.2, value.size, dtype=np.float32).reshape(value.shape))
        gradients = pullback(cotangent)
        if argmax:

            def get_argmax(a, b):
                return fused_cross_entropy_loss_and_logsumexp_penalty(
                    hax.named(a, (Batch, Position, Embed)),
                    hax.named(b, (Vocab, Embed)),
                    Embed,
                    Vocab,
                    label,
                    reduction=None,
                    weight=weight,
                    block_size=2,
                    dtype=jnp.float32,
                    precision=jax.lax.Precision.HIGHEST,
                    return_argmax=True,
                )[1].array

            actual_argmax = jax.jit(get_argmax)(*placed)

    def reference(a, b):
        return _reference(a, b, labels, weights, reduction, zloss, softcap)

    expected, ref_pullback = jax.vjp(jax.jit(reference), jnp.asarray(x), jnp.asarray(head))
    expected_gradients = ref_pullback(cotangent)
    actual_values = np.asarray(value)
    expected_values = np.asarray(expected)
    np.testing.assert_allclose(actual_values, expected_values, atol=1e-5, rtol=1e-4)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(np.asarray(actual_gradient), np.asarray(expected_gradient), atol=1e-5, rtol=1e-4)
    if argmax:
        np.testing.assert_array_equal(np.asarray(actual_argmax), np.argmax(np.einsum("bse,ve->bsv", x, head), axis=-1))


def test_shifted_global_targets_keep_full_vocab_loss_and_gradients():
    if len(jax.devices()) < 4:
        pytest.skip("Requires four JAX devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(1, 4), ("data", "model"))
    x, head, tokens, _, mapping, placed = _arrays(mesh, False)
    token_array = hax.named(jax.device_put(tokens, NamedSharding(mesh, P(None, None))), (Batch, Position))
    weights = np.asarray([[1, 0]] * 4, dtype=np.float32)

    def actual(a, b):
        return maybe_fused_next_token_loss(
            Position,
            Embed,
            Vocab,
            hax.named(a, (Batch, Position, Embed)),
            hax.named(b, (Vocab, Embed)),
            token_array,
            reduction=None,
            logsumexp_weight=0.0,
            block_size=2,
            precision=jax.lax.Precision.HIGHEST,
        ).array

    with jax.set_mesh(mesh), hax.axis_mapping(mapping):
        loss, backward = jax.vjp(jax.jit(actual), *placed)
        gradients = backward(jnp.ones_like(loss))

    def reference(a, b):
        return _reference(a, b, np.roll(tokens, -1, axis=-1), weights, "none", 0.0, None)

    expected, ref_backward = jax.vjp(jax.jit(reference), jnp.asarray(x), jnp.asarray(head))
    ref_gradients = ref_backward(jnp.ones_like(expected))
    np.testing.assert_allclose(np.asarray(loss), np.asarray(expected), atol=1e-5, rtol=1e-4)
    for actual_gradient, expected_gradient in zip(gradients, ref_gradients, strict=True):
        np.testing.assert_allclose(np.asarray(actual_gradient), np.asarray(expected_gradient), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("contract_sharded", [False, True], ids=["vocab", "contract"])
def test_explicit_sharded_loss_and_gradients_match_full_reference(contract_sharded):
    if len(jax.devices()) < 4:
        pytest.skip("Requires four JAX devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    with jax.set_mesh(mesh):
        inputs = _arrays(mesh, contract_sharded)
        label = hax.named(jax.device_put(inputs.labels, NamedSharding(mesh, P("data", None))), (Batch, Position))
        weight = hax.named(jax.device_put(inputs.weights, NamedSharding(mesh, P("data", None))), (Batch, Position))

        def actual(embeddings, head):
            return fused_cross_entropy_loss_and_logsumexp_penalty(
                hax.named(embeddings, (Batch, Position, Embed)),
                hax.named(head, (Vocab, Embed)),
                Contract=Embed,
                Label=Vocab,
                target_y=label,
                weight=weight,
                reduction=hax.sum,
                logsumexp_weight=0.1,
                block_size=2,
                dtype=jnp.float32,
                precision=jax.lax.Precision.HIGHEST,
            ).scalar()

        with hax.axis_mapping(inputs.mapping):
            value, gradients = jax.value_and_grad(jax.jit(actual), argnums=(0, 1))(*inputs.placed)

    def reference(embeddings, head):
        return _reference(embeddings, head, inputs.labels, inputs.weights, "sum", 0.1, None)

    expected, expected_gradients = jax.value_and_grad(jax.jit(reference), argnums=(0, 1))(
        jnp.asarray(inputs.embeddings), jnp.asarray(inputs.head)
    )
    np.testing.assert_allclose(np.asarray(value), np.asarray(expected), atol=1e-5, rtol=1e-4)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(np.asarray(actual_gradient), np.asarray(expected_gradient), atol=1e-5, rtol=1e-4)
