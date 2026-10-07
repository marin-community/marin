# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.routing_top_k import is_rocm_backend, routing_top_k, triton_top_k_indices

_SPECIAL_VALUES = np.array([0.0, -0.0, np.inf, -np.inf, 1.0, -1.0, np.nan, -np.nan, 1e-30], dtype=np.float32)


def _router_matrix(kind: str, tokens: int, experts: int) -> jax.Array:
    key = jax.random.key(0)
    if kind == "normal":
        return jax.random.normal(key, (tokens, experts), jnp.float32)
    if kind == "ties":
        return jnp.round(jax.random.normal(key, (tokens, experts), jnp.float32) * 2) / 2
    if kind == "constant":
        return jnp.zeros((tokens, experts), jnp.float32)
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.choice(_SPECIAL_VALUES, size=(tokens, experts)))


def _assert_same_indices_as_lax(x: jax.Array, k: int, *, interpret: bool) -> None:
    _, expected = jax.lax.top_k(x, k)
    actual = jax.jit(lambda x: triton_top_k_indices(x, k, interpret=interpret))(x)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("kind", ["normal", "ties", "constant", "specials"])
@pytest.mark.parametrize("k", [1, 5])
def test_triton_top_k_indices_match_lax_top_k_in_interpreter(kind: str, k: int):
    _assert_same_indices_as_lax(_router_matrix(kind, 48, 16), k, interpret=True)


@pytest.mark.parametrize("kind", ["normal", "ties", "constant", "specials"])
def test_triton_top_k_indices_match_lax_top_k_on_rocm(kind: str):
    if not is_rocm_backend():
        pytest.skip("Pallas-Triton routing top-k runs on ROCm")
    _assert_same_indices_as_lax(_router_matrix(kind, 4096, 256), 5, interpret=False)


def test_routing_top_k_on_token_sharded_mesh_matches_lax_top_k_on_rocm():
    if not is_rocm_backend():
        pytest.skip("Pallas-Triton routing top-k runs on ROCm")
    mesh = Mesh(np.asarray(jax.devices()), axis_names=("data",), axis_types=(AxisType.Explicit,))
    x = _router_matrix("ties", 4096 * len(jax.devices()), 256)
    expected_values, expected_indices = jax.lax.top_k(x, 5)
    expected_grad = jax.grad(lambda x: jnp.sum(jax.lax.top_k(x, 5)[0] ** 2))(x)

    def routed_sum_of_squares(x):
        return jnp.sum(routing_top_k(x, 5, mesh=mesh, batch_axes=("data",))[0] ** 2)

    with jax.set_mesh(mesh):
        x_sharded = jax.device_put(x, NamedSharding(mesh, P("data", None)))
        values, indices = jax.jit(lambda x: routing_top_k(x, 5, mesh=mesh, batch_axes=("data",)))(x_sharded)
        grad = jax.jit(jax.grad(routed_sum_of_squares))(x_sharded)
    np.testing.assert_array_equal(np.asarray(indices), np.asarray(expected_indices))
    np.testing.assert_array_equal(np.asarray(values), np.asarray(expected_values))
    np.testing.assert_array_equal(np.asarray(grad), np.asarray(expected_grad))
