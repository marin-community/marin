# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


"""Run the four-device cases on v4-8, v5p-8, or v6e-4.

uv run --package marin-levanter --extra tpu --group test pytest -n 0 -m manual \
    lib/levanter/tests/kernels/test_fixed_schedule_linear.py -k four_device
"""

import hashlib

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from levanter.grug.sharding import compact_grug_mesh
from levanter.kernels.pallas.fixed_schedule_linear import fixed_schedule_linear

pytestmark = pytest.mark.manual


@pytest.fixture(autouse=True)
def require_supported_tpu(request):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas fixed-schedule tests require an actual TPU runtime")
    devices = jax.devices()
    if devices[0].device_kind.lower() not in ("tpu v4", "tpu v5", "tpu v5p", "tpu v6 lite", "tpu v6e"):
        pytest.skip("requires TPU v4, v5p, or v6e")
    if "four_device" not in request.node.name:
        if len(devices) != 1 or len(jax.local_devices()) != 1:
            pytest.skip("requires one visible TPU device")
    if "four_device" in request.node.name:
        with jax.set_mesh(compact_grug_mesh()):
            yield
    else:
        yield


def _raw_bits(value: jax.Array) -> np.ndarray:
    host = np.asarray(value)
    return host.view(np.uint16 if host.dtype.itemsize == 2 else np.uint32)


def _inputs(m: int, d: int, input_dtype: jnp.dtype, weight_dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    x = jnp.sin(jnp.arange(m * d, dtype=jnp.float32).reshape(m, d) * 0.013)
    w = jnp.cos(jnp.arange(d * 256, dtype=jnp.float32).reshape(d, 256) * 0.007)
    x = x.at[:, 0].set(10000.0 + 256.0 * jnp.arange(m, dtype=jnp.float32))
    return x.astype(input_dtype), w.astype(weight_dtype)


def _four_device_shardings() -> tuple[Mesh, NamedSharding, NamedSharding]:
    devices = jax.devices()
    if len(devices) != 4 or len(jax.local_devices()) != 4:
        pytest.skip("requires four same-process TPU devices")
    mesh = compact_grug_mesh()
    row_spec = P(("replica_dcn", "data", "expert"), None)
    return mesh, NamedSharding(mesh, row_spec), NamedSharding(mesh, P(None, None))


@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (jnp.float32, jnp.float32),
        (jnp.float32, jnp.bfloat16),
        (jnp.bfloat16, jnp.float32),
        (jnp.bfloat16, jnp.bfloat16),
    ],
)
@pytest.mark.parametrize("m", [1, 127, 128, 129])
@pytest.mark.parametrize("d", [128, 256])
def test_fixed_schedule_preserves_frozen_rows_across_tile_boundaries(m, d, input_dtype, weight_dtype):
    target, weight = _inputs(1, d, input_dtype, weight_dtype)
    peers, _ = _inputs(m, d, input_dtype, weight_dtype)
    target_hash = hashlib.sha256(np.asarray(target).tobytes()).hexdigest()
    weight_hash = hashlib.sha256(np.asarray(weight).tobytes()).hexdigest()
    peer_hash = hashlib.sha256(np.asarray(peers).tobytes()).hexdigest()
    target_output = fixed_schedule_linear(target, weight)
    assert bool(jnp.isfinite(target).all())
    assert bool(jnp.isfinite(weight).all())
    assert bool(jnp.isfinite(peers).all())
    assert bool(jnp.isfinite(target_output).all())

    positions = tuple(dict.fromkeys((0, m - 1, 127, 128)))
    for position in positions:
        if position >= m:
            continue
        variant = peers.at[position].set(target[0])
        observed = fixed_schedule_linear(variant, weight)[position]
        assert bool(jnp.isfinite(observed).all())
        np.testing.assert_array_equal(_raw_bits(observed), _raw_bits(target_output[0]))

    assert hashlib.sha256(np.asarray(target).tobytes()).hexdigest() == target_hash
    assert hashlib.sha256(np.asarray(weight).tobytes()).hexdigest() == weight_hash
    assert hashlib.sha256(np.asarray(peers).tobytes()).hexdigest() == peer_hash


def _highest_reference(x: jax.Array, w: jax.Array) -> jax.Array:
    # Preserve weight-dtype rounding before FP32 accumulation.
    x_compute = jax.lax.optimization_barrier(x.astype(w.dtype))
    return jax.lax.dot_general(
        x_compute.astype(jnp.float32),
        w.astype(jnp.float32),
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (jnp.float32, jnp.float32),
        (jnp.float32, jnp.bfloat16),
        (jnp.bfloat16, jnp.float32),
        (jnp.bfloat16, jnp.bfloat16),
    ],
)
@pytest.mark.parametrize("d", [128, 256])
def test_fixed_schedule_compiled_custom_vjp_preserves_primal_and_matches_reference_gradients(
    d, input_dtype, weight_dtype
):
    x, w = _inputs(5, d, input_dtype, weight_dtype)
    cotangent = jnp.linspace(-1.0, 1.0, 5 * 256, dtype=jnp.float32).reshape(5, 256).astype(x.dtype)

    def public_vjp(x_, w_):
        primal, pullback = jax.vjp(fixed_schedule_linear, x_, w_)
        dx, dw = pullback(cotangent)
        return primal, dx, dw

    fixed_primal = jax.jit(fixed_schedule_linear)(x, w)
    ad_primal, fixed_dx, fixed_dw = jax.jit(public_vjp)(x, w)

    def reference_loss(x_, w_):
        return jnp.sum(_highest_reference(x_, w_) * cotangent)

    reference_dx, reference_dw = jax.jit(jax.grad(reference_loss, argnums=(0, 1)))(x, w)
    reference_primal = jax.jit(_highest_reference)(x, w)

    np.testing.assert_array_equal(_raw_bits(ad_primal), _raw_bits(fixed_primal))
    np.testing.assert_allclose(fixed_dx, reference_dx, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(fixed_dw, reference_dw, atol=1e-5, rtol=1e-4)
    assert bool(jnp.isfinite(fixed_primal).all())
    assert bool(jnp.isfinite(reference_primal).all())
    assert bool(jnp.isfinite(fixed_dx).all())
    assert bool(jnp.isfinite(fixed_dw).all())
    assert bool(jnp.isfinite(reference_dx).all())
    assert bool(jnp.isfinite(reference_dw).all())
    assert bool(jnp.isfinite(ad_primal).all())


@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (jnp.float32, jnp.float32),
        (jnp.float32, jnp.bfloat16),
        (jnp.bfloat16, jnp.float32),
        (jnp.bfloat16, jnp.bfloat16),
    ],
)
@pytest.mark.parametrize("m", [4, 128, 512])
@pytest.mark.parametrize("d", [128, 256])
def test_fixed_schedule_four_device_explicit_mesh_matches_replicated_rows(m, d, input_dtype, weight_dtype):
    _, row_sharding, replicated_sharding = _four_device_shardings()
    x, w = _inputs(m, d, input_dtype, weight_dtype)
    row_x = jax.device_put(x, row_sharding)
    replicated_x = jax.device_put(x, replicated_sharding)
    replicated_w = jax.device_put(w, replicated_sharding)

    row_output = jax.jit(fixed_schedule_linear)(row_x, replicated_w)
    replicated_output = jax.jit(fixed_schedule_linear)(replicated_x, replicated_w)
    np.testing.assert_array_equal(_raw_bits(row_output), _raw_bits(replicated_output))
    assert row_output.sharding == row_sharding
    assert replicated_output.sharding == replicated_sharding
    assert bool(jnp.isfinite(row_output).all())


@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (jnp.float32, jnp.float32),
        (jnp.float32, jnp.bfloat16),
        (jnp.bfloat16, jnp.float32),
        (jnp.bfloat16, jnp.bfloat16),
    ],
)
@pytest.mark.parametrize("d", [128, 256])
def test_fixed_schedule_four_device_custom_vjp_matches_independent_reference(d, input_dtype, weight_dtype):
    mesh, row_sharding, replicated_sharding = _four_device_shardings()
    x, w = _inputs(128, d, input_dtype, weight_dtype)
    row_x = jax.device_put(x, row_sharding)
    replicated_x = jax.device_put(x, replicated_sharding)
    replicated_w = jax.device_put(w, replicated_sharding)
    cotangent_host = jnp.linspace(-1.0, 1.0, 128 * 256, dtype=jnp.float32).reshape(128, 256).astype(input_dtype)
    row_cotangent = jax.device_put(cotangent_host, row_sharding)
    replicated_cotangent = jax.device_put(cotangent_host, replicated_sharding)

    def public_vjp(x_, w_, cotangent_):
        primal, pullback = jax.vjp(fixed_schedule_linear, x_, w_)
        dx, dw = pullback(cotangent_)
        return primal, dx, dw

    primal, dx, dw = jax.jit(public_vjp)(row_x, replicated_w, row_cotangent)
    replicated_primal, replicated_dx, replicated_dw = jax.jit(public_vjp)(
        replicated_x, replicated_w, replicated_cotangent
    )

    def reference_loss(x_, w_, cotangent):
        return jnp.sum(_highest_reference(x_, w_) * cotangent)

    def reference_weight_gradient(activations, weights, cotangent):
        # Differentiate a native dot in FP32 on each row shard. Casting the
        # local gradients to the storage dtype before their sum can round twice.
        partial_dw = jax.grad(reference_loss, argnums=1)(
            activations,
            weights.astype(jnp.float32),
            cotangent.astype(jnp.float32),
        )
        return jax.lax.psum(partial_dw, "data").astype(weights.dtype)

    reference_grad = jax.jit(jax.grad(reference_loss, argnums=(0, 1)))
    row_reference_dx, _ = reference_grad(row_x, replicated_w, row_cotangent)
    replicated_reference_dx, replicated_reference_dw = reference_grad(replicated_x, replicated_w, replicated_cotangent)
    rounded_x = jax.device_put(np.asarray(x).astype(weight_dtype).astype(np.float32), row_sharding)
    row_reference_dw = jax.jit(
        jax.shard_map(
            reference_weight_gradient,
            mesh=mesh,
            in_specs=(row_sharding.spec, replicated_sharding.spec, row_sharding.spec),
            out_specs=replicated_sharding.spec,
            check_vma=False,  # Keep the autodiff reduction local to each shard.
        )
    )(rounded_x, replicated_w, row_cotangent)

    np.testing.assert_array_equal(_raw_bits(primal), _raw_bits(replicated_primal))
    np.testing.assert_allclose(dx, row_reference_dx, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(dw, row_reference_dw, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(replicated_dx, replicated_reference_dx, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(replicated_dw, replicated_reference_dw, atol=1e-5, rtol=1e-4)
    assert primal.dtype == dx.dtype == input_dtype
    assert dw.dtype == weight_dtype
    assert primal.sharding == row_sharding
    assert dx.sharding == row_sharding
    assert dw.sharding == replicated_sharding
    assert bool(jnp.isfinite(primal).all()) and bool(jnp.isfinite(dx).all()) and bool(jnp.isfinite(dw).all())


def test_fixed_schedule_four_device_accepts_a_differently_named_row_axis():
    _four_device_shardings()
    mesh = Mesh(np.asarray(jax.devices()), ("rows",), axis_types=(AxisType.Explicit,))
    rows = NamedSharding(mesh, P("rows", None))
    replicated = NamedSharding(mesh, P(None, None))
    x, w = _inputs(128, 128, jnp.float32, jnp.bfloat16)

    with jax.set_mesh(mesh):
        project = jax.jit(fixed_schedule_linear)
        w = jax.device_put(w, replicated)
        row_output = project(jax.device_put(x, rows), w)
        replicated_output = project(jax.device_put(x, replicated), w)
    np.testing.assert_array_equal(_raw_bits(row_output), _raw_bits(replicated_output))
    assert row_output.sharding == rows


def test_fixed_schedule_four_device_lowered_forward_backward_have_shard_map_without_all_gather():
    _, row_sharding, replicated_sharding = _four_device_shardings()
    x, w = _inputs(128, 128, jnp.bfloat16, jnp.float32)
    x = jax.device_put(x, row_sharding)
    w = jax.device_put(w, replicated_sharding)

    def public_vjp(x_, w_):
        primal, pullback = jax.vjp(fixed_schedule_linear, x_, w_)
        return primal, *pullback(jnp.ones_like(primal))

    for function in (jax.jit(fixed_schedule_linear), jax.jit(public_vjp)):
        jaxpr_text = str(jax.make_jaxpr(function)(x, w))
        stablehlo_text = function.lower(x, w).as_text()
        assert "shard_map" in jaxpr_text
        assert "all_gather" not in jaxpr_text
        assert "all_gather" not in stablehlo_text


@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [(jnp.bfloat16, jnp.bfloat16), (jnp.float32, jnp.bfloat16), (jnp.float32, jnp.float32)],
)
@pytest.mark.parametrize("d,n", [(385, 1023), (641, 2560)])
def test_fixed_schedule_four_device_frozen_row_across_batches(input_dtype, weight_dtype, d, n):
    _, row_sharding, replicated_sharding = _four_device_shardings()
    rng = np.random.default_rng(23)
    target = rng.standard_normal((1, d), dtype=np.float32).astype(input_dtype)
    weights = rng.standard_normal((d, n), dtype=np.float32).astype(weight_dtype)
    fixed = jax.jit(fixed_schedule_linear)
    w = jax.device_put(weights, replicated_sharding)
    expected = np.asarray(fixed(jax.device_put(target, replicated_sharding), w))[0]

    for m in (4, 4 * 127, 4 * 257, 4 * 513):
        batch = rng.standard_normal((m, d), dtype=np.float32).astype(input_dtype)
        # Move the row across devices and across local tile boundaries.
        positions = np.array([0, m // 2 - 1, m - 1])
        batch[positions] = target
        actual = np.asarray(fixed(jax.device_put(batch, row_sharding), w))[positions]
        raw_dtype = np.uint16 if input_dtype == jnp.bfloat16 else np.uint32
        np.testing.assert_array_equal(actual.view(raw_dtype), np.broadcast_to(expected.view(raw_dtype), actual.shape))
