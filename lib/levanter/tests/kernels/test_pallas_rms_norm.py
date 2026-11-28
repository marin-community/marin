# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import levanter.kernels.pallas.rms_norm.pallas_tpu as pallas_tpu

from levanter.kernels.pallas.rms_norm import batch_invariant_rms_norm, batch_invariant_rms_norm_reference
from levanter.kernels.pallas.rms_norm.pallas_tpu import (
    PallasRmsNormUnsupportedError,
    _normalize_partition_spec,
    _sharded_local_pallas,
    batch_invariant_rms_norm_pallas_tpu,
)


def _independent_reference(x, weight, eps):
    x32 = x.astype(jnp.float32)
    return (
        x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps) * weight.astype(jnp.float32)
    ).astype(x.dtype)


@pytest.mark.parametrize("hidden", [128, 256, 2560])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_batch_invariant_rms_norm_reference_is_bitwise_row_invariant(hidden, dtype):
    key_x, key_w = jax.random.split(jax.random.key(hidden))
    x = jax.random.normal(key_x, (512, hidden), dtype=dtype)
    weight = jax.random.normal(key_w, (hidden,), dtype=dtype)
    full = np.asarray(batch_invariant_rms_norm(x, weight, 1e-5, implementation="reference"))
    for rows in (1, 8, 31, 32, 127, 128, 129):
        output = np.asarray(batch_invariant_rms_norm(x[:rows], weight, 1e-5, implementation="reference"))
        np.testing.assert_array_equal(output, full[:rows])


@pytest.mark.parametrize("shape", [(4, 128, 128), (32, 1, 256), (4, 128, 2560)])
def test_batch_invariant_rms_norm_leading_layout_and_jitted_vjp_primal_match(shape):
    key_x, key_w = jax.random.split(jax.random.key(sum(shape)))
    x = jax.random.normal(key_x, shape, dtype=jnp.float32)
    weight = jax.random.normal(key_w, (shape[-1],), dtype=jnp.float32)
    plain = jax.jit(lambda value: batch_invariant_rms_norm(value, weight, 1e-5, implementation="reference"))(x)
    differentiated = jax.jit(
        lambda value: jax.vjp(
            lambda inner: batch_invariant_rms_norm(inner, weight, 1e-5, implementation="reference"), value
        )[0]
    )(x)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(differentiated))


def test_batch_invariant_rms_norm_matches_independent_reference_and_fixed_schedule_gradient():
    key_x, key_w = jax.random.split(jax.random.key(17))
    x = jax.random.normal(key_x, (3, 128), dtype=jnp.float32)
    weight = jax.random.normal(key_w, (128,), dtype=jnp.float32)
    actual = batch_invariant_rms_norm(x, weight, 1e-5, implementation="reference")
    expected = _independent_reference(x, weight, 1e-5)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def actual_loss(value, scale):
        return jnp.sum(batch_invariant_rms_norm(value, scale, 1e-5, implementation="reference") ** 2)

    def expected_loss(value, scale):
        return jnp.sum(_independent_reference(value, scale, 1e-5) ** 2)

    def fixed_reference_loss(value, scale):
        return jnp.sum(batch_invariant_rms_norm_reference(value, scale, eps=1e-5) ** 2)

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
        batch_invariant_rms_norm(x, weight, 1e-5, implementation="pallas_tpu")


@pytest.mark.parametrize("rows", [0, 513])
def test_batch_invariant_rms_norm_has_one_shared_flattened_row_domain(rows):
    x = jnp.ones((rows, 128), dtype=jnp.float32)
    weight = jnp.ones((128,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="flattened rows"):
        batch_invariant_rms_norm(x, weight, 1e-5, implementation="reference")
    with pytest.raises(PallasRmsNormUnsupportedError, match="flattened rows"):
        batch_invariant_rms_norm_pallas_tpu(x, weight, eps=1e-5)


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
        output = batch_invariant_rms_norm(x, weight, 1e-5, implementation="pallas_tpu")
        expected = _independent_reference(x, weight, 1e-5)
        np.testing.assert_allclose(np.asarray(output), np.asarray(expected), rtol=1e-5, atol=1e-5)
    hlo = str(
        jax.jit(lambda value: batch_invariant_rms_norm(value, weight, 1e-5, implementation="pallas_tpu"))
        .lower(x)
        .compiler_ir()
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
            batch_invariant_rms_norm(replicated_x, multidevice_weight, 1e-5, implementation="pallas_tpu")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
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
        return batch_invariant_rms_norm(value, scale, 1e-5, implementation="pallas_tpu")

    primal, pullback = jax.vjp(apply, x, weight)
    dx, dw = pullback(jnp.ones_like(primal))
    expected_primal = _independent_reference(x, weight, 1e-5)
    _, expected_pullback = jax.vjp(lambda value, scale: _independent_reference(value, scale, 1e-5), x, weight)
    expected_dx, expected_dw = expected_pullback(jnp.ones_like(expected_primal))
    np.testing.assert_allclose(np.asarray(primal), np.asarray(expected_primal), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(dx), np.asarray(expected_dx), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(dw), np.asarray(expected_dw), rtol=1e-5, atol=1e-5)
    assert dx.dtype == x.dtype
    assert dw.dtype == weight.dtype
    assert dx.sharding == x_sharding
    assert dw.sharding == weight_sharding

    hlo = str(jax.jit(apply).lower(x, weight).compiler_ir()).lower()
    assert "all-gather" not in hlo


def test_batch_invariant_rms_norm_named_mesh_survives_public_jit_and_vjp(monkeypatch):
    device = jax.devices()[0]
    mesh = jax.sharding.Mesh(np.asarray([device]), ("data",))
    x_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    x = jax.device_put(jnp.ones((1, 128), dtype=jnp.float32), x_sharding)
    weight = jax.device_put(jnp.ones((128,), dtype=jnp.float32), weight_sharding)
    monkeypatch.setattr(pallas_tpu, "_validate_tpu_inputs", lambda value, scale: None)
    monkeypatch.setattr(pallas_tpu, "_local_pallas", lambda value, scale, *, eps: value * scale)

    plain = jax.jit(lambda value: batch_invariant_rms_norm(value, weight, 1e-5, implementation="pallas_tpu"))(x)
    differentiated = jax.jit(
        lambda value: jax.vjp(
            lambda inner: batch_invariant_rms_norm(inner, weight, 1e-5, implementation="pallas_tpu"), value
        )[0]
    )(x)
    dynamic = jax.jit(lambda value, scale: batch_invariant_rms_norm(value, scale, 1e-5, implementation="pallas_tpu"))(
        x, weight
    )
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(differentiated))
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(dynamic))


def test_batch_invariant_rms_norm_normalizes_short_specs_and_rejects_invalid_layouts(monkeypatch):
    assert _normalize_partition_spec(jax.sharding.PartitionSpec("data"), 2) == jax.sharding.PartitionSpec("data", None)
    assert _normalize_partition_spec(jax.sharding.PartitionSpec(), 1) == jax.sharding.PartitionSpec(None)
    with pytest.raises(PallasRmsNormUnsupportedError, match="exceeds operand rank"):
        _normalize_partition_spec(jax.sharding.PartitionSpec("data", "model", "extra"), 2)

    device = jax.devices()[0]
    mesh = jax.sharding.Mesh(np.asarray([device]), ("data",))
    other_mesh = jax.sharding.Mesh(np.asarray([device]), ("other",))
    x_short = jax.device_put(
        jnp.ones((1, 128), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")),
    )
    replicated_weight = jax.device_put(
        jnp.ones((128,), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
    monkeypatch.setattr(pallas_tpu, "_local_pallas", lambda value, scale, *, eps: value)
    np.testing.assert_array_equal(
        np.asarray(_sharded_local_pallas(x_short, replicated_weight, eps=1e-5)), np.ones((1, 128), dtype=np.float32)
    )

    invalid_cases = (
        (
            x_short,
            jax.device_put(
                jnp.ones((128,), dtype=jnp.float32),
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")),
            ),
            "row-sharded",
        ),
        (
            jax.device_put(
                jnp.ones((1, 128), dtype=jnp.float32),
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "data")),
            ),
            replicated_weight,
            "row-sharded",
        ),
        (
            x_short,
            jnp.ones((128,), dtype=jnp.float32),
            "requires explicit NamedSharding",
        ),
        (
            x_short,
            jax.device_put(
                jnp.ones((128,), dtype=jnp.float32),
                jax.sharding.NamedSharding(other_mesh, jax.sharding.PartitionSpec()),
            ),
            "same mesh",
        ),
    )
    for invalid_x, invalid_weight, message in invalid_cases:
        with pytest.raises(PallasRmsNormUnsupportedError, match=message):
            _sharded_local_pallas(invalid_x, invalid_weight, eps=1e-5)


def test_batch_invariant_rms_norm_rejects_same_shape_different_concrete_meshes(monkeypatch):
    if len(jax.devices()) < 2:
        pytest.skip("requires two visible devices to construct distinct concrete meshes")

    devices = np.asarray(jax.devices()[:2])
    mesh = jax.sharding.Mesh(devices, ("data",))
    reordered_mesh = jax.sharding.Mesh(devices[::-1], ("data",))
    assert mesh.shape == reordered_mesh.shape
    assert mesh.axis_names == reordered_mesh.axis_names
    assert mesh != reordered_mesh

    x = jax.device_put(
        jnp.ones((2, 128), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")),
    )
    weight = jax.device_put(
        jnp.ones((128,), dtype=jnp.float32),
        jax.sharding.NamedSharding(reordered_mesh, jax.sharding.PartitionSpec()),
    )
    monkeypatch.setattr(pallas_tpu, "_local_pallas", lambda value, scale, *, eps: value)
    with pytest.raises(PallasRmsNormUnsupportedError, match="same mesh"):
        _sharded_local_pallas(x, weight, eps=1e-5)
