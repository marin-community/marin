# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.testing.devices import use_cpu_devices
from levanter.kernels.pallas import xsa as xsa_module
from levanter.kernels.pallas.xsa import XsaUnsupportedError, xsa_forward, xsa_pallas_tpu, xsa_reference


def _oracle(y, v):
    y32 = y.astype(jnp.float32)
    v32 = v.astype(jnp.float32)
    dot = np.sum(np.asarray(y32 * v32), axis=-1, keepdims=True)
    norm = np.sum(np.asarray(v32 * v32), axis=-1, keepdims=True)
    return np.asarray(y32) - dot / (norm + 1e-6) * np.asarray(v32)


def test_xsa_reference_matches_independent_oracle_and_batch_shape():
    y = (jnp.arange(3 * 2 * 128, dtype=jnp.float32).reshape(3, 2, 128) % 17) / 11 - 0.7
    v = (jnp.arange(3 * 2 * 128, dtype=jnp.float32).reshape(3, 2, 128) % 13) / 7 + 0.2
    actual = xsa_forward(y, v)
    np.testing.assert_allclose(actual, _oracle(y, v), rtol=1e-6, atol=1e-6)
    flattened = xsa_reference(y.reshape(6, 128), v.reshape(6, 128)).reshape(y.shape)
    np.testing.assert_allclose(actual, flattened, rtol=0, atol=0)


def test_xsa_reference_custom_vjp_matches_reference_gradient():
    y = jnp.asarray([[0.2, -0.7, 1.1], [0.4, 0.3, -0.9]], dtype=jnp.float32)
    v = jnp.asarray([[1.3, 0.1, -0.4], [-0.2, 0.8, 0.6]], dtype=jnp.float32)
    cotangent = jnp.asarray([[0.5, -0.3, 0.8], [0.2, 0.9, -0.6]], dtype=jnp.float32)

    def loss(a, b):
        return jnp.sum(xsa_forward(a, b) * cotangent)

    value, actual = jax.value_and_grad(loss, argnums=(0, 1))(y, v)
    epsilon = 1e-3

    def finite_difference(value, *, wrt):
        result = np.zeros_like(np.asarray(value))
        for index in np.ndindex(result.shape):
            plus = np.asarray(value).copy()
            minus = np.asarray(value).copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            args_plus = (plus, np.asarray(v)) if wrt == 0 else (np.asarray(y), plus)
            args_minus = (minus, np.asarray(v)) if wrt == 0 else (np.asarray(y), minus)
            result[index] = (float(loss(*args_plus)) - float(loss(*args_minus))) / (2 * epsilon)
        return result

    assert np.isfinite(float(value))
    np.testing.assert_allclose(actual[0], finite_difference(y, wrt=0), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(actual[1], finite_difference(v, wrt=1), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_xsa_reference_preserves_supported_storage_dtype(dtype):
    y = jnp.asarray([[0.2, -0.7, 1.1], [0.4, 0.3, -0.9]], dtype=dtype)
    v = jnp.asarray([[1.3, 0.1, -0.4], [-0.2, 0.8, 0.6]], dtype=dtype)
    actual = xsa_reference(y, v)
    assert actual.dtype == dtype
    expected = _oracle(y, v).astype(np.dtype(dtype))
    if dtype == jnp.bfloat16:
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint16), expected.view(np.uint16))
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_xsa_pallas_rejects_actual_width_before_flatten(monkeypatch):
    monkeypatch.setattr(xsa_module, "xsa_pallas_tpu", lambda *_: pytest.fail("width was flattened before validation"))
    y = jnp.ones((2, 256), dtype=jnp.float32)
    v = jnp.ones((2, 256), dtype=jnp.float32)
    with pytest.raises(XsaUnsupportedError, match="head width 128"):
        xsa_forward(y, v, "pallas_tpu")


def test_xsa_pallas_tpu_is_explicit_on_cpu():
    y = jnp.ones((1, 128), dtype=jnp.float32)
    v = jnp.ones((1, 128), dtype=jnp.float32)
    if jax.default_backend() != "cpu":
        return
    with pytest.raises(XsaUnsupportedError):
        xsa_pallas_tpu(y, v)


@pytest.mark.timeout(120)
def test_xsa_explicit_layout_and_vjp_reference_boundary_regression(monkeypatch):
    """Exercise real four-CPU layouts with a reference terminal, not the Pallas kernel."""
    monkeypatch.setattr(xsa_module, "xsa_pallas_tpu", xsa_reference)
    with use_cpu_devices(4):
        names = ("replica_dcn", "data", "context", "expert", "model")
        mesh = Mesh(np.asarray(jax.devices()).reshape(1, 4, 1, 1, 1), names, axis_types=(AxisType.Explicit,) * 5)
        spec = P(("replica_dcn", "data"), "expert", None, "model")
        sharding = NamedSharding(mesh, spec)
        host_y = np.sin(np.arange(1024, dtype=np.float32) * 0.013).reshape(4, 2, 1, 128)
        host_v = np.cos(np.arange(1024, dtype=np.float32) * 0.017).reshape(4, 2, 1, 128) + 0.1
        with jax.set_mesh(mesh):
            y = jax.device_put(host_y, sharding)
            v = jax.device_put(host_v, sharding)
            forward = lambda a, b: xsa_module.xsa_forward(a, b, "pallas_tpu")
            actual = jax.jit(forward)(y, v)
            expected = jax.jit(xsa_module.xsa_reference)(y, v)
            assert actual.sharding.spec == spec
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            cotangent = jax.device_put(np.sin(host_v * 0.7), sharding)
            dy, dv = jax.jit(jax.grad(lambda a, b: jnp.sum(forward(a, b) * cotangent), argnums=(0, 1)))(y, v)
            assert dy.sharding.spec == dv.sharding.spec == spec
            norm = np.sum(host_v * host_v, axis=-1, keepdims=True) + 1e-6
            dot = np.sum(host_y * host_v, axis=-1, keepdims=True)
            g = np.asarray(cotangent)
            gv = np.sum(g * host_v, axis=-1, keepdims=True)
            expected_dy = g - host_v * gv / norm
            expected_dv = -(g * dot / norm + host_y * gv / norm - 2 * host_v * dot * gv / (norm * norm))
            np.testing.assert_allclose(np.asarray(dy), expected_dy, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.asarray(dv), expected_dv, rtol=1e-5, atol=1e-5)
            canonical_spec = P("data", None, None, None)
            canonical_v = jax.sharding.reshard(v, NamedSharding(mesh, canonical_spec))
            mixed = jax.jit(forward)(y, canonical_v)
            assert mixed.sharding.spec == spec
            np.testing.assert_array_equal(np.asarray(mixed), np.asarray(expected))
            mixed_dy, mixed_dv = jax.jit(jax.grad(lambda a, b: jnp.sum(forward(a, b) * cotangent), argnums=(0, 1)))(
                y, canonical_v
            )
            assert mixed_dy.sharding.spec == spec
            assert mixed_dv.sharding.spec == canonical_spec
            np.testing.assert_allclose(np.asarray(mixed_dy), expected_dy, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.asarray(mixed_dv), expected_dv, rtol=1e-5, atol=1e-5)

        for axis in ("context", "expert", "model"):
            sizes = tuple(4 if name == axis else 1 for name in names)
            unsupported = Mesh(np.asarray(jax.devices()).reshape(sizes), names, axis_types=(AxisType.Explicit,) * 5)
            with jax.set_mesh(unsupported):
                bad = jax.device_put(host_y, NamedSharding(unsupported, P(axis, None, None, None)))
                with pytest.raises(xsa_module.XsaUnsupportedError):
                    xsa_module.xsa_forward(bad, bad, "pallas_tpu")


@pytest.mark.tpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_xsa_tpu_bitwise_batch_and_permutation_invariance(dtype):
    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        pytest.skip("requires four real TPU devices")
    names = ("replica_dcn", "data", "context", "expert", "model")
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(1, 4, 1, 1, 1),
        names,
        axis_types=(jax.sharding.AxisType.Explicit,) * 5,
    )
    spec = jax.sharding.PartitionSpec(("replica_dcn", "data"), "expert", None, "model")
    sharding = jax.sharding.NamedSharding(mesh, spec)
    host_y = np.sin(np.arange(2048 * 128, dtype=np.float32) * 0.013).reshape(2048, 128).astype(np.dtype(dtype))
    host_v = (np.cos(np.arange(2048 * 128, dtype=np.float32) * 0.017) + 0.1).reshape(2048, 128).astype(np.dtype(dtype))
    outputs = []
    with jax.set_mesh(mesh):
        apply = jax.jit(lambda y, v: xsa_forward(y, v, "pallas_tpu"))
        for rows in (8, 1024, 2048):
            shape = (4, rows // 8, 2, 128)
            y = jax.device_put(host_y[:rows].reshape(shape), sharding)
            v = jax.device_put(host_v[:rows].reshape(shape), sharding)
            actual = apply(y, v)
            assert actual.sharding.spec == spec
            output = np.asarray(actual).reshape(rows, 128)
            outputs.append(output)
            if dtype == jnp.float32:
                np.testing.assert_allclose(output, _oracle(host_y[:rows], host_v[:rows]), rtol=1e-5, atol=1e-5)
            permutation = np.random.default_rng(rows).permutation(rows)
            permuted = apply(
                jax.device_put(host_y[:rows][permutation].reshape(shape), sharding),
                jax.device_put(host_v[:rows][permutation].reshape(shape), sharding),
            )
            bits = np.uint32 if dtype == jnp.float32 else np.uint16
            np.testing.assert_array_equal(
                np.asarray(permuted).reshape(rows, 128).view(bits), output[permutation].view(bits)
            )
    for rows, output in zip((8, 1024, 2048), outputs, strict=True):
        bits = np.uint32 if dtype == jnp.float32 else np.uint16
        np.testing.assert_array_equal(output.view(bits), outputs[-1][:rows].view(bits))
