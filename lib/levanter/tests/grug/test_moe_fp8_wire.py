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
from levanter.grug.grug_moe import moe_mlp


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


def _ep_moe_mesh_or_skip(expert: int = 2) -> Mesh:
    devices = jax.devices()
    if len(devices) < expert or len(devices) % expert != 0:
        pytest.skip(f"needs >= {expert} devices with expert-divisible count")
    mesh_devices = np.array(devices).reshape(len(devices) // expert, expert, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def test_moe_mlp_ring_fp8_wire_parity():
    # End-to-end: values and input gradients stay within FP8 wire tolerance of
    # the bf16-wire path (E4M3 dispatch fwd, E5M2 combine-transpose bwd).
    # d/i are TPU-tile-sized (128): the reference arm's grad runs the megablox
    # ragged_dot on TPU, whose Pallas kernel needs 128-divisible trailing dims.
    mesh = _ep_moe_mesh_or_skip()
    t, d, i, e, k = 64, 128, 128, 16, 4
    key = jax.random.key(0)
    kx, ks, kw, k1, k2, kc = jax.random.split(key, 6)
    batch = P(("data", "expert"))
    with jax.set_mesh(mesh):
        x = jax.device_put(jax.random.normal(kx, (t, d), dtype=jnp.float32), NamedSharding(mesh, batch))
        sel = jax.device_put(
            jnp.argsort(jax.random.uniform(ks, (t, e)), axis=-1)[:, :k].astype(jnp.int32),
            NamedSharding(mesh, batch),
        )
        cw = jax.device_put(jax.nn.softmax(jax.random.normal(kw, (t, k))), NamedSharding(mesh, batch))
        w13 = jax.device_put(jax.random.normal(k1, (e, d, 2 * i)) * 0.05, NamedSharding(mesh, P("expert", None, None)))
        w2 = jax.device_put(jax.random.normal(k2, (e, i, d)) * 0.05, NamedSharding(mesh, P("expert", None, None)))
        cot = jax.device_put(jax.random.normal(kc, (t, d), dtype=jnp.float32), NamedSharding(mesh, batch))

        def loss(x_, *, wire):
            out = moe_mlp(x_, sel, cw, w13, w2, implementation="ring", mesh=mesh, fp8_wire=wire)
            return jnp.sum(out * cot)

        out_wire = moe_mlp(x, sel, cw, w13, w2, implementation="ring", mesh=mesh, fp8_wire=True)
        out_ref = moe_mlp(x, sel, cw, w13, w2, implementation="ring", mesh=mesh)
        g_wire = jax.jit(jax.grad(lambda v: loss(v, wire=True)))(x)
        g_ref = jax.jit(jax.grad(lambda v: loss(v, wire=False)))(x)

    assert _relfrob(out_wire, out_ref) < 0.1
    assert _relfrob(g_wire, g_ref) < 0.2


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="ragged_all_to_all has no CPU lowering")
def test_moe_mlp_ragged_a2a_fp8_wire_parity():
    mesh = _ep_moe_mesh_or_skip()
    t, d, i, e, k = 64, 32, 16, 16, 4
    key = jax.random.key(0)
    kx, ks, kw, k1, k2 = jax.random.split(key, 5)
    batch = P(("data", "expert"))
    with jax.set_mesh(mesh):
        x = jax.device_put(jax.random.normal(kx, (t, d), dtype=jnp.float32), NamedSharding(mesh, batch))
        sel = jax.device_put(
            jnp.argsort(jax.random.uniform(ks, (t, e)), axis=-1)[:, :k].astype(jnp.int32),
            NamedSharding(mesh, batch),
        )
        cw = jax.device_put(jax.nn.softmax(jax.random.normal(kw, (t, k))), NamedSharding(mesh, batch))
        w13 = jax.device_put(jax.random.normal(k1, (e, d, 2 * i)) * 0.05, NamedSharding(mesh, P("expert", None, None)))
        w2 = jax.device_put(jax.random.normal(k2, (e, i, d)) * 0.05, NamedSharding(mesh, P("expert", None, None)))

        out_wire = moe_mlp(x, sel, cw, w13, w2, implementation="ragged_all_to_all", mesh=mesh, fp8_wire=True)
        out_ref = moe_mlp(x, sel, cw, w13, w2, implementation="ragged_all_to_all", mesh=mesh)
    assert _relfrob(out_wire, out_ref) < 0.1


def test_moe_mlp_rejects_fp8_wire_without_expert_axis():
    # Contract: fp8_wire only exists on the EP dispatch/combine collectives; a
    # silent bf16 fallback would let runs believe they measure the FP8 wire.
    devices = jax.devices()
    mesh_devices = np.array(devices).reshape(len(devices), 1)
    mesh = Mesh(mesh_devices, axis_names=("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    t, d, i, e, k = 16, 8, 4, 4, 2
    key = jax.random.key(0)
    kx, ks, kw, k1, k2 = jax.random.split(key, 5)
    with jax.set_mesh(mesh):
        x = jax.device_put(jax.random.normal(kx, (t, d), dtype=jnp.float32), NamedSharding(mesh, P("data")))
        sel = jax.device_put(
            jnp.argsort(jax.random.uniform(ks, (t, e)), axis=-1)[:, :k].astype(jnp.int32),
            NamedSharding(mesh, P("data")),
        )
        cw = jax.device_put(jax.nn.softmax(jax.random.normal(kw, (t, k))), NamedSharding(mesh, P("data")))
        w13 = jax.device_put(jax.random.normal(k1, (e, d, 2 * i)) * 0.05, NamedSharding(mesh, P(None, None, None)))
        w2 = jax.device_put(jax.random.normal(k2, (e, i, d)) * 0.05, NamedSharding(mesh, P(None, None, None)))
        with pytest.raises(NotImplementedError, match="fp8_wire"):
            moe_mlp(x, sel, cw, w13, w2, implementation="ring", mesh=mesh, fp8_wire=True)
