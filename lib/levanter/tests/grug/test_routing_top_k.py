# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from haliax.jax_utils import is_amd_instinct
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import levanter.grug._moe.routing_top_k as routing_top_k_module
from levanter.grug._moe.routing_top_k import routing_top_k, triton_top_k_indices

_SPECIAL_VALUES = np.array([0.0, -0.0, np.inf, -np.inf, 1.0, -1.0, np.nan, -np.nan, 1e-30], dtype=np.float32)
_K = 5


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


def _assert_same_indices_as_lax(x: jax.Array, k: int) -> None:
    _, expected = jax.lax.top_k(x, k)
    actual = jax.jit(lambda x: triton_top_k_indices(x, k))(x)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("kind", ["normal", "ties", "constant", "specials"])
@pytest.mark.parametrize("k", [1, _K])
def test_triton_top_k_indices_match_lax_top_k(kind: str, k: int):
    # 48 tokens run as three 16-row blocks; on CPU the kernel runs in the Pallas interpreter.
    _assert_same_indices_as_lax(_router_matrix(kind, 48, 16), k)


@pytest.mark.parametrize(
    ("instinct", "experts", "dtype", "uses_kernel"),
    [
        (True, 16, jnp.float32, True),
        (True, 24, jnp.float32, False),
        (True, 16, jnp.bfloat16, False),
        (False, 16, jnp.float32, False),
    ],
    ids=["instinct_power_of_two", "instinct_non_power_of_two", "instinct_bfloat16", "not_instinct"],
)
def test_routing_top_k_matches_lax_top_k_values_and_grads(
    monkeypatch, instinct: bool, experts: int, dtype: jnp.dtype, uses_kernel: bool
):
    monkeypatch.setattr(routing_top_k_module, "is_amd_instinct", lambda: instinct)
    # Levanter CI runs on one CPU device, so there the shard_map has a single shard; set
    # XLA_FLAGS=--xla_force_host_platform_device_count=4 to run the kernel on several token shards.
    mesh = Mesh(np.asarray(jax.devices()), axis_names=("data",), axis_types=(AxisType.Explicit,))
    x = _router_matrix("ties", 16 * len(jax.devices()), experts).astype(dtype)

    def routed(x):
        return routing_top_k(x, _K, mesh=mesh, batch_axes=("data",))

    def sum_of_squares(top_k):
        return lambda x: jnp.sum(top_k(x)[0] ** 2)

    expected_values, expected_indices = jax.lax.top_k(x, _K)
    expected_grad = jax.grad(sum_of_squares(lambda x: jax.lax.top_k(x, _K)))(x)
    with jax.set_mesh(mesh):
        x_sharded = jax.device_put(x, NamedSharding(mesh, P("data", None)))
        grad_fn = jax.grad(sum_of_squares(routed))
        assert ("pallas_call" in str(jax.make_jaxpr(grad_fn)(x_sharded))) == uses_kernel
        values, indices = jax.jit(routed)(x_sharded)
        grad = jax.jit(grad_fn)(x_sharded)
    np.testing.assert_array_equal(np.asarray(indices), np.asarray(expected_indices))
    np.testing.assert_array_equal(np.asarray(values), np.asarray(expected_values))
    np.testing.assert_array_equal(np.asarray(grad), np.asarray(expected_grad))


@pytest.mark.parametrize("kind", ["normal", "ties", "constant", "specials"])
def test_triton_top_k_indices_match_lax_top_k_on_amd_instinct(kind: str):
    if not is_amd_instinct():
        pytest.skip("the compiled Pallas-Triton router top-k runs on AMD Instinct")
    _assert_same_indices_as_lax(_router_matrix(kind, 4096, 256), _K)
