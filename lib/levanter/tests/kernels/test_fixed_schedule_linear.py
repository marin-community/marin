# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from levanter.grug.sharding import compact_grug_mesh
from levanter.kernels.pallas.fixed_schedule_linear import fixed_schedule_linear


@pytest.fixture(autouse=True)
def require_tpu_v4(request):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas fixed-schedule tests require an actual TPU runtime")
    devices = jax.devices()
    assert devices and "tpu v4" in devices[0].device_kind.lower()
    if "four_device" not in request.node.name:
        if len(devices) != 1 or len(jax.local_devices()) != 1:
            pytest.skip("the historical single-device cases require one visible TPUv4 device")
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
        pytest.skip("requires four same-process TPUv4 devices")
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
    _, row_sharding, replicated_sharding = _four_device_shardings()
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

    def reference_loss(x_, w_):
        return jnp.sum(_highest_reference(x_, w_) * replicated_cotangent)

    reference_dx, reference_dw = jax.jit(jax.grad(reference_loss, argnums=(0, 1)))(replicated_x, replicated_w)

    np.testing.assert_array_equal(_raw_bits(primal), _raw_bits(replicated_primal))
    np.testing.assert_allclose(dx, reference_dx, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(dw, reference_dw, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(dx, replicated_dx, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(dw, replicated_dw, atol=1e-5, rtol=1e-4)
    assert primal.dtype == dx.dtype == input_dtype
    assert dw.dtype == weight_dtype
    assert primal.sharding == row_sharding
    assert dx.sharding == row_sharding
    assert dw.sharding == replicated_sharding
    assert bool(jnp.isfinite(primal).all()) and bool(jnp.isfinite(dx).all()) and bool(jnp.isfinite(dw).all())


@pytest.mark.parametrize("d", [128, 256])
def test_fixed_schedule_four_device_reference_preserves_weight_dtype_rounding(d):
    _, row_sharding, replicated_sharding = _four_device_shardings()
    x, w = _inputs(128, d, jnp.float32, jnp.bfloat16)
    x_host, w_host = np.asarray(x), np.asarray(w)
    # XLA never receives original FP32 x on this independent oracle path.
    q_host = x_host.astype(w_host.dtype)
    rounded_host = q_host.astype(np.float32)
    assert np.count_nonzero(x_host.view(np.uint32) != rounded_host.view(np.uint32)) > 0
    np.testing.assert_array_equal(rounded_host.view(np.uint32) & 0xFFFF, 0)
    output_dtype = x.dtype
    replicated_x = jax.device_put(x, replicated_sharding)
    replicated_w = jax.device_put(w, replicated_sharding)
    q = jax.device_put(q_host, replicated_sharding)
    cotangent = jnp.linspace(-1.0, 1.0, 128 * 256, dtype=jnp.float32).reshape(128, 256)
    replicated_cotangent = jax.device_put(cotangent, replicated_sharding)

    def boundary_forward(q_, w_):
        return jax.lax.dot_general(
            q_.astype(jnp.float32),
            w_.astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ).astype(output_dtype)

    def boundary_vjp(q_, w_, dy_):
        primal, pullback = jax.vjp(boundary_forward, q_, w_)
        return primal, *pullback(dy_)

    def reference_vjp(x_, w_, dy_):
        primal, pullback = jax.vjp(_highest_reference, x_, w_)
        return primal, *pullback(dy_)

    def public_vjp(x_, w_, dy_):
        primal, pullback = jax.vjp(fixed_schedule_linear, x_, w_)
        return primal, *pullback(dy_)

    boundary_primal, dq, expected_dw = jax.jit(boundary_vjp)(q, replicated_w, replicated_cotangent)
    # Close the original x->q cast transpose on the host, preserving dq rounding.
    expected_dx = np.asarray(dq).astype(x_host.dtype)
    reference_primal, reference_dx, reference_dw = jax.jit(lambda x_, w_, dy_: jax.jit(reference_vjp)(x_, w_, dy_))(
        replicated_x, replicated_w, replicated_cotangent
    )
    np.testing.assert_allclose(reference_primal, boundary_primal, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(reference_dx, expected_dx, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(reference_dw, expected_dw, atol=1e-5, rtol=1e-4)
    assert reference_dx.dtype == x.dtype
    assert reference_dw.dtype == w.dtype

    for activation_sharding in (row_sharding, replicated_sharding):
        public_x = jax.device_put(x, activation_sharding)
        public_cotangent = jax.device_put(cotangent, activation_sharding)
        _, public_dx, public_dw = jax.jit(public_vjp)(public_x, replicated_w, public_cotangent)
        np.testing.assert_allclose(public_dx, expected_dx, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(public_dw, expected_dw, atol=1e-5, rtol=1e-4)


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
