# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FP8 over-the-wire MoE dispatch/combine collectives."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.ep_common import _shard_a2a_params
from levanter.grug._moe.fp8_wire import fp8_all_gather, fp8_psum_scatter, fp8_ragged_a2a


def _expert_mesh_or_skip(min_shards: int = 2) -> Mesh:
    devices = jax.devices()
    if len(devices) < min_shards:
        pytest.skip(f"needs >= {min_shards} devices")
    mesh_devices = np.array(devices).reshape(len(devices))
    return Mesh(mesh_devices, axis_names=("expert",), axis_types=(AxisType.Explicit,))


def _relfrob(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _shmap(fn, mesh, n_args=1):
    return shard_map(
        fn,
        mesh=mesh,
        in_specs=tuple(P("expert") for _ in range(n_args)),
        out_specs=P("expert"),
        check_vma=False,
    )


def test_fp8_all_gather_forward_matches_native_within_quant_tolerance():
    mesh = _expert_mesh_or_skip()
    n = len(jax.devices())
    x = jax.random.normal(jax.random.key(0), (n * 8, 16), dtype=jnp.bfloat16)
    with jax.set_mesh(mesh):
        xs = jax.device_put(x, NamedSharding(mesh, P("expert")))
        got = jax.jit(_shmap(lambda v: fp8_all_gather(v, "expert"), mesh))(xs)
        want = jax.jit(_shmap(lambda v: jax.lax.all_gather(v, "expert", tiled=True), mesh))(xs)
    # E4M3 per-tensor quantization error; well under 2^-3 relative for gaussian data.
    assert _relfrob(got, want) < 0.05


def test_fp8_all_gather_backward_is_exact_native_reduction():
    # The gradient must be the exact bf16 psum_scatter (straight-through across
    # QDQ): reductions never run in FP8.
    mesh = _expert_mesh_or_skip()
    n = len(jax.devices())
    key = jax.random.key(1)
    x = jax.random.normal(key, (n * 8, 16), dtype=jnp.float32)
    cot = jax.random.normal(jax.random.key(2), (n * n * 8, 16), dtype=jnp.float32)

    with jax.set_mesh(mesh):
        xs = jax.device_put(x, NamedSharding(mesh, P("expert")))
        cots = jax.device_put(cot, NamedSharding(mesh, P("expert")))

        def loss_fp8(v, c):
            return jnp.sum(fp8_all_gather(v, "expert") * c)[None]

        def loss_native(v, c):
            return jnp.sum(jax.lax.all_gather(v, "expert", tiled=True) * c)[None]

        g_fp8 = jax.jit(jax.grad(lambda v: jnp.sum(_shmap(loss_fp8, mesh, n_args=2)(v, cots))))(xs)
        g_native = jax.jit(jax.grad(lambda v: jnp.sum(_shmap(loss_native, mesh, n_args=2)(v, cots))))(xs)
    np.testing.assert_allclose(np.asarray(g_fp8), np.asarray(g_native), rtol=1e-6)


def test_fp8_psum_scatter_forward_is_exact_native():
    # Contract: the forward reduction is bit-identical to the native psum_scatter
    # (only the backward all_gather carries FP8).
    mesh = _expert_mesh_or_skip()
    n = len(jax.devices())
    y = jax.random.normal(jax.random.key(3), (n * n * 8, 16), dtype=jnp.float32)
    with jax.set_mesh(mesh):
        ys = jax.device_put(y, NamedSharding(mesh, P("expert")))
        got = jax.jit(_shmap(lambda v: fp8_psum_scatter(v, "expert"), mesh))(ys)
        want = jax.jit(_shmap(lambda v: jax.lax.psum_scatter(v, "expert", scatter_dimension=0, tiled=True), mesh))(ys)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6)


def test_fp8_psum_scatter_backward_within_e5m2_tolerance():
    mesh = _expert_mesh_or_skip()
    n = len(jax.devices())
    y = jax.random.normal(jax.random.key(4), (n * n * 8, 16), dtype=jnp.float32)
    cot = jax.random.normal(jax.random.key(5), (n * 8, 16), dtype=jnp.float32)
    with jax.set_mesh(mesh):
        ys = jax.device_put(y, NamedSharding(mesh, P("expert")))
        cots = jax.device_put(cot, NamedSharding(mesh, P("expert")))

        def loss_fp8(v, c):
            return jnp.sum(fp8_psum_scatter(v, "expert") * c)[None]

        def loss_native(v, c):
            return jnp.sum(jax.lax.psum_scatter(v, "expert", scatter_dimension=0, tiled=True) * c)[None]

        g_fp8 = jax.jit(jax.grad(lambda v: jnp.sum(_shmap(loss_fp8, mesh, n_args=2)(v, cots))))(ys)
        g_native = jax.jit(jax.grad(lambda v: jnp.sum(_shmap(loss_native, mesh, n_args=2)(v, cots))))(ys)
    # E5M2 gradient wire: 2 mantissa bits => ~1e-1 worst-case relative rounding.
    assert _relfrob(g_fp8, g_native) < 0.15


def test_fp8_all_gather_scaling_is_token_independent():
    # Per-token scaling: perturbing one token must not change any other token's
    # dequantized value. A scale shared across rows would couple tokens (and leak
    # later sequence positions into earlier ones through the quantization grid).
    mesh = _expert_mesh_or_skip()
    n = len(jax.devices())
    x = jax.random.normal(jax.random.key(6), (n * 8, 16), dtype=jnp.float32)
    x_perturbed = x.at[0, :].multiply(100.0)  # local row 0 of shard 0

    with jax.set_mesh(mesh):
        gather = jax.jit(_shmap(lambda v: fp8_all_gather(v, "expert"), mesh))
        got = np.asarray(gather(jax.device_put(x, NamedSharding(mesh, P("expert")))))
        got_perturbed = np.asarray(gather(jax.device_put(x_perturbed, NamedSharding(mesh, P("expert")))))

    # Each shard's [n*8, 16] gather stacks into [n*n*8, 16]; the perturbed input
    # row appears at position 0 of every shard's block.
    affected = np.zeros(got.shape[0], dtype=bool)
    affected[np.arange(n) * (n * 8)] = True
    assert not np.array_equal(got[affected], got_perturbed[affected])
    np.testing.assert_array_equal(got[~affected], got_perturbed[~affected])


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="ragged_all_to_all has no CPU lowering")
def test_fp8_ragged_a2a_forward_and_backward_within_wire_tolerance():
    # Forward parity within E4M3 tolerance and, through the custom VJP's reverse
    # ragged_all_to_all (transposed counts, per-token E5M2), gradient parity
    # within E5M2 tolerance against the native collective.
    mesh = _expert_mesh_or_skip()
    n = len(jax.devices())
    rows_per_peer = 4
    total = n * rows_per_peer
    counts = jnp.full((n, n), rows_per_peer, dtype=jnp.int32)
    x = jax.random.normal(jax.random.key(7), (n * total, 16), dtype=jnp.float32)
    cot = jax.random.normal(jax.random.key(8), (n * total, 16), dtype=jnp.float32)

    def wire(v):
        return fp8_ragged_a2a(v, counts, jax.lax.axis_index("expert"), total, "expert")

    def native(v):
        offsets = _shard_a2a_params(counts, jax.lax.axis_index("expert"))
        buf = jnp.zeros((total, v.shape[1]), dtype=v.dtype)
        return jax.lax.ragged_all_to_all(v, buf, *offsets, axis_name="expert")

    with jax.set_mesh(mesh):
        xs = jax.device_put(x, NamedSharding(mesh, P("expert")))
        cots = jax.device_put(cot, NamedSharding(mesh, P("expert")))

        got = jax.jit(_shmap(wire, mesh))(xs)
        want = jax.jit(_shmap(native, mesh))(xs)

        def loss(fn):
            def inner(v, c):
                return jnp.sum(fn(v) * c)[None]

            return jax.jit(jax.grad(lambda v: jnp.sum(_shmap(inner, mesh, n_args=2)(v, cots))))

        g_wire = loss(wire)(xs)
        g_native = loss(native)(xs)

    assert _relfrob(got, want) < 0.05  # E4M3 forward
    assert _relfrob(g_wire, g_native) < 0.15  # E5M2 backward
