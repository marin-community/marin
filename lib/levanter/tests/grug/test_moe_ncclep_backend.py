# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AbstractMesh, PartitionSpec as P

from levanter.grug._moe.ep_ncclep import _apply_recv_weights, moe_mlp_ep_ncclep, ncclep_receive_capacity


class _FakePrimitive:
    def __init__(self, bind):
        self.inner_primitive = SimpleNamespace(bind=bind)


@dataclass(frozen=True)
class _FakeEpLayerConfig:
    top_k: int
    dispatch_output_per_expert_alignment: int = 0


def test_ncclep_receive_capacity_matches_ep8_training_shape() -> None:
    capacity = ncclep_receive_capacity(
        global_tokens=131_072,
        top_k=4,
        ep_size=8,
    )

    assert capacity == 524_288


@pytest.mark.parametrize(
    ("global_tokens", "top_k", "ep_size"),
    [
        (0, 4, 8),
        (131_072, 0, 8),
        (131_072, 4, 0),
        (131_071, 4, 8),
    ],
)
def test_ncclep_receive_capacity_rejects_invalid_layouts(
    global_tokens: int,
    top_k: int,
    ep_size: int,
) -> None:
    with pytest.raises(ValueError):
        ncclep_receive_capacity(global_tokens, top_k, ep_size)


def test_ncclep_recv_weighting_masks_unused_rows_from_values_and_gradients() -> None:
    expert_out = jnp.array(
        [
            [2.0, 3.0],
            [5.0, 7.0],
            [jnp.nan, jnp.nan],
            [jnp.nan, jnp.nan],
        ],
        dtype=jnp.float32,
    )
    recv_weights = jnp.array([0.5, 0.25, jnp.nan, 1.0], dtype=jnp.float32)
    token_counts = jnp.array([2], dtype=jnp.int32)

    def weighted_sum(outputs, weights):
        return _apply_recv_weights(outputs, weights, token_counts).sum()

    weighted = _apply_recv_weights(expert_out, recv_weights, token_counts)
    output_gradient, weight_gradient = jax.grad(weighted_sum, argnums=(0, 1))(expert_out, recv_weights)

    assert jnp.array_equal(weighted, jnp.array([[1.0, 1.5], [1.25, 1.75], [0.0, 0.0], [0.0, 0.0]]))
    assert jnp.array_equal(
        output_gradient,
        jnp.array([[0.5, 0.5], [0.25, 0.25], [0.0, 0.0], [0.0, 0.0]]),
    )
    assert jnp.array_equal(weight_gradient, jnp.array([5.0, 12.0, 0.0, 0.0]))


def test_ncclep_backend_passes_rank_local_shapes_to_transformer_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prepare(routes, **_kwargs):
        assert routes.shape == (2, 1)
        return [jnp.zeros((1, 1), jnp.int32), jnp.zeros((1, 32), jnp.uint8)]

    def dispatch(_handle, _routes, tokens, weights, *, recv_capacity_per_rank, **_kwargs):
        assert tokens.shape == (2, 4)
        return [
            jnp.zeros((1, recv_capacity_per_rank, 4), tokens.dtype),
            jnp.zeros((1, recv_capacity_per_rank), weights.dtype),
        ]

    def dispatch_bwd(_handle, token_cotangent, weight_cotangent, *, out_leading_shape, **_kwargs):
        assert out_leading_shape == (2,)
        return [
            jnp.zeros((2, 4), token_cotangent.dtype),
            jnp.zeros((2, 1), weight_cotangent.dtype),
        ]

    def combine(_handle, expert_out, *, out_leading_shape, **_kwargs):
        assert out_leading_shape == (2,)
        return jnp.zeros((2, 4), expert_out.dtype)

    def combine_bwd(_handle, output_cotangent, *, recv_capacity_per_rank, **_kwargs):
        return jnp.zeros((1, recv_capacity_per_rank, 4), output_cotangent.dtype)

    fake_cpp_ep = SimpleNamespace(
        EpPreparePrimitive=_FakePrimitive(prepare),
        EpDispatchPrimitive=_FakePrimitive(dispatch),
        EpDispatchBwdPrimitive=_FakePrimitive(dispatch_bwd),
        EpCombinePrimitive=_FakePrimitive(combine),
        EpCombineBwdPrimitive=_FakePrimitive(combine_bwd),
        _on_collective_stream=lambda fn: fn,
    )
    monkeypatch.setitem(sys.modules, "transformer_engine.jax.cpp_extensions.ep", fake_cpp_ep)
    monkeypatch.setitem(
        sys.modules,
        "transformer_engine.jax.ep",
        SimpleNamespace(EpLayerConfig=_FakeEpLayerConfig),
    )

    mesh = AbstractMesh((8,), ("expert",))
    routes = jax.ShapeDtypeStruct((16, 1), jnp.int32)
    tokens = jax.ShapeDtypeStruct((16, 4), jnp.bfloat16)
    weights = jax.ShapeDtypeStruct((16, 1), jnp.float32)
    w_up_gate = jax.ShapeDtypeStruct((8, 4, 4), jnp.bfloat16)
    w_down = jax.ShapeDtypeStruct((8, 2, 4), jnp.bfloat16)

    def loss(tokens, weights, w_up_gate, w_down):
        output, _ = moe_mlp_ep_ncclep(
            tokens,
            routes,
            weights,
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=8,
            capacity_factor=1.25,
            mesh=mesh,
            batch_spec=P("expert", None),
        )
        return output.astype(jnp.float32).sum()

    with jax.sharding.use_abstract_mesh(mesh):
        _, gradients = jax.eval_shape(
            jax.value_and_grad(loss, argnums=(0, 1, 2, 3)),
            tokens,
            weights,
            w_up_gate,
            w_down,
        )

    assert tuple(gradient.shape for gradient in gradients) == (
        tokens.shape,
        weights.shape,
        w_up_gate.shape,
        w_down.shape,
    )
