# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.pallas.rms_norm import batch_invariant_rms_norm, batch_invariant_rms_norm_reference
from levanter.kernels.pallas.rms_norm.pallas_tpu import PallasRmsNormUnsupportedError


_EPS = 1e-5


def _independent_reference(x, weight, eps):
    x32 = x.astype(jnp.float32)
    return (
        x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps) * weight.astype(jnp.float32)
    ).astype(x.dtype)


def _bits(value):
    host = np.asarray(value)
    return host.view(np.uint16 if host.dtype.itemsize == 2 else np.uint32)


@pytest.mark.parametrize("implementation", ["reference", pytest.param("pallas_tpu", marks=pytest.mark.tpu)])
@pytest.mark.parametrize("hidden", [128, 2560, 4096])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_batch_invariant_rms_norm_is_bitwise_row_invariant(implementation, hidden, dtype):
    key_x, key_w = jax.random.split(jax.random.key(hidden))
    x = jax.random.normal(key_x, (1025, hidden), dtype=dtype)
    weight = jax.random.normal(key_w, (hidden,), dtype=dtype)
    if implementation == "pallas_tpu":
        if jax.default_backend() != "tpu":
            pytest.skip("requires a TPU backend")
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[:1]), ("data",), axis_types=(jax.sharding.AxisType.Explicit,)
        )
        x = jax.device_put(x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None)))
        weight = jax.device_put(weight, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))

    def apply(x, weight):
        return batch_invariant_rms_norm(x, weight, _EPS, implementation=implementation)

    compiled = jax.jit(apply)
    full = compiled(x, weight)
    if dtype == jnp.float32:
        np.testing.assert_allclose(
            np.asarray(full), np.asarray(_independent_reference(x, weight, _EPS)), rtol=1e-5, atol=1e-5
        )
    for start, rows in ((0, 1), (128, 127), (384, 129), (511, 513)):
        output = compiled(x[start : start + rows], weight)
        np.testing.assert_array_equal(_bits(output), _bits(full[start : start + rows]))


@pytest.mark.tpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pallas_rms_norm_preserves_row_bits_across_device_counts(dtype):
    if jax.default_backend() != "tpu" or jax.local_device_count() < 4:
        pytest.skip("requires four local TPU devices")
    devices = np.asarray(jax.local_devices()[:4])
    hidden, rows_per_device = 2560, 641
    rng = np.random.default_rng(18)
    values = rng.standard_normal((4 * rows_per_device, hidden), dtype=np.float32)
    rows = np.arange(4) * rows_per_device + np.array([0, 127, 512, 640])
    values[rows] = values[0]
    scales = rng.uniform(0.5, 1.5, hidden).astype(np.float32)

    @jax.jit
    def apply(x, weight):
        return batch_invariant_rms_norm(x, weight, _EPS, implementation="pallas_tpu")

    def run(mesh, batch):
        x = jax.device_put(batch, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None)))
        weight = jax.device_put(scales, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
        return _bits(apply(x.astype(dtype), weight.astype(dtype)))

    explicit = (jax.sharding.AxisType.Explicit,)
    alone = run(jax.sharding.Mesh(devices[:1], ("data",), axis_types=explicit), values[:1])
    sharded = run(jax.sharding.Mesh(devices, ("data",), axis_types=explicit), values)
    np.testing.assert_array_equal(sharded[rows], np.broadcast_to(alone, (4, hidden)))


@pytest.mark.parametrize("shape", [(4, 128, 128), (32, 1, 256), (4, 128, 2560)])
def test_batch_invariant_rms_norm_leading_layout_and_jitted_vjp_primal_match(shape):
    key_x, key_w = jax.random.split(jax.random.key(sum(shape)))
    x = jax.random.normal(key_x, shape, dtype=jnp.float32)
    weight = jax.random.normal(key_w, (shape[-1],), dtype=jnp.float32)
    plain = jax.jit(lambda value: batch_invariant_rms_norm(value, weight, _EPS, implementation="reference"))(x)
    differentiated = jax.jit(
        lambda value: jax.vjp(
            lambda inner: batch_invariant_rms_norm(inner, weight, _EPS, implementation="reference"), value
        )[0]
    )(x)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(differentiated))


def test_batch_invariant_rms_norm_matches_independent_reference_and_fixed_schedule_gradient():
    key_x, key_w = jax.random.split(jax.random.key(17))
    x = jax.random.normal(key_x, (3, 128), dtype=jnp.float32)
    weight = jax.random.normal(key_w, (128,), dtype=jnp.float32)
    actual = batch_invariant_rms_norm(x, weight, _EPS, implementation="reference")
    expected = _independent_reference(x, weight, _EPS)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def actual_loss(value, scale):
        return jnp.sum(batch_invariant_rms_norm(value, scale, _EPS, implementation="reference") ** 2)

    def expected_loss(value, scale):
        return jnp.sum(_independent_reference(value, scale, _EPS) ** 2)

    def fixed_reference_loss(value, scale):
        return jnp.sum(batch_invariant_rms_norm_reference(value, scale, eps=_EPS) ** 2)

    np.testing.assert_allclose(
        np.asarray(actual_loss(x, weight)), np.asarray(expected_loss(x, weight)), rtol=1e-5, atol=1e-5
    )
    actual_grad = jax.grad(actual_loss, argnums=(0, 1))(x, weight)
    expected_grad = jax.grad(fixed_reference_loss, argnums=(0, 1))(x, weight)
    for actual_component, expected_component in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(np.asarray(actual_component), np.asarray(expected_component), rtol=1e-5, atol=1e-5)


def test_batch_invariant_rms_norm_public_pallas_backend_is_explicitly_tpu_only():
    x = jnp.ones((1, 128), dtype=jnp.float32)
    weight = jnp.ones((128,), dtype=jnp.float32)
    if jax.default_backend() == "tpu":
        pytest.skip("CPU-only contract check is not applicable on TPU.")
    with pytest.raises(NotImplementedError, match="requires a TPU backend"):
        batch_invariant_rms_norm(x, weight, _EPS, implementation="pallas_tpu")


@pytest.mark.tpu
def test_batch_invariant_rms_norm_named_sharding_contract_on_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("NamedSharding Pallas behavior requires a TPU backend.")
    one_device_mesh = jax.sharding.Mesh(np.asarray([jax.devices()[0]]), ("data",))
    one_device_x_sharding = jax.sharding.NamedSharding(one_device_mesh, jax.sharding.PartitionSpec("data", None))
    one_device_weight_sharding = jax.sharding.NamedSharding(one_device_mesh, jax.sharding.PartitionSpec(None))
    weight = jax.device_put(jnp.linspace(0.5, 1.5, 128, dtype=jnp.float32), one_device_weight_sharding)
    for rows in (1, 128, 129, 512):
        values = jnp.arange(rows * 128, dtype=jnp.float32).reshape(rows, 128) / 128.0
        x = jax.device_put(values, one_device_x_sharding)
        output = batch_invariant_rms_norm(x, weight, _EPS, implementation="pallas_tpu")
        expected = _independent_reference(x, weight, _EPS)
        np.testing.assert_allclose(np.asarray(output), np.asarray(expected), rtol=1e-5, atol=1e-5)
    hlo = str(
        jax.jit(lambda value: batch_invariant_rms_norm(value, weight, _EPS, implementation="pallas_tpu"))
        .lower(x)
        .compile()
        .as_text()
    )
    assert "all-gather" not in hlo.lower()

    if len(jax.devices()) > 1:
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
        weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
        replicated_x = jax.device_put(
            jnp.ones((len(jax.devices()), 128), dtype=jnp.float32),
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, None)),
        )
        multidevice_weight = jax.device_put(jnp.ones((128,), dtype=jnp.float32), weight_sharding)
        with pytest.raises(PallasRmsNormUnsupportedError, match="row-sharded"):
            batch_invariant_rms_norm(replicated_x, multidevice_weight, _EPS, implementation="pallas_tpu")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.tpu
def test_batch_invariant_rms_norm_explicit_three_dimensional_vjp_preserves_sharding(dtype):
    if jax.default_backend() != "tpu" or len(jax.devices()) < 4:
        pytest.skip("requires four TPU devices for the production row-sharded layout")

    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:4]).reshape((1, 4, 1, 1)),
        ("replica_dcn", "data", "context", "expert"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
    x_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("replica_dcn", "data"), "expert", None))
    weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
    values = jnp.arange(4 * 128 * 256, dtype=jnp.float32).reshape(4, 128, 256) / 256.0
    scales = jnp.linspace(0.5, 1.5, 256, dtype=jnp.float32)
    x = jax.device_put(values.astype(dtype), x_sharding)
    weight = jax.device_put(scales.astype(dtype), weight_sharding)

    def apply(value, scale):
        return batch_invariant_rms_norm(value, scale, _EPS, implementation="pallas_tpu")

    primal, pullback = jax.vjp(apply, x, weight)
    dx, dw = pullback(jnp.ones_like(primal))
    expected_primal = _independent_reference(x, weight, _EPS)
    _, expected_pullback = jax.vjp(lambda value, scale: _independent_reference(value, scale, _EPS), x, weight)
    expected_dx, expected_dw = expected_pullback(jnp.ones_like(expected_primal))
    np.testing.assert_allclose(np.asarray(primal), np.asarray(expected_primal), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(dx), np.asarray(expected_dx), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(dw), np.asarray(expected_dw), rtol=1e-5, atol=1e-5)
    assert dx.dtype == x.dtype
    assert dw.dtype == weight.dtype
    assert dx.sharding == x_sharding
    assert dw.sharding == weight_sharding

    hlo = str(jax.jit(apply).lower(x, weight).compile().as_text()).lower()
    assert "all-gather" not in hlo
