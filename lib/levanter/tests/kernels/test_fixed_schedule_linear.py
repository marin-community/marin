# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.pallas.fixed_schedule_linear import fixed_schedule_linear
from levanter.models.snowball import SnowballAttention, SnowballConfig


@pytest.fixture(autouse=True)
def require_single_v4_device():
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas fixed-schedule tests require an actual TPU runtime")
    devices = jax.devices()
    assert devices and "tpu v4" in devices[0].device_kind.lower()
    assert len(devices) == 1
    assert len(jax.local_devices()) == 1


def _raw_bits(value: jax.Array) -> np.ndarray:
    host = np.asarray(value)
    return host.view(np.uint16 if host.dtype.itemsize == 2 else np.uint32)


def _inputs(m: int, d: int, input_dtype: jnp.dtype, weight_dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    x = jnp.sin(jnp.arange(m * d, dtype=jnp.float32).reshape(m, d) * 0.013)
    w = jnp.cos(jnp.arange(d * 256, dtype=jnp.float32).reshape(d, 256) * 0.007)
    x = x.at[:, 0].set(10000.0 + 256.0 * jnp.arange(m, dtype=jnp.float32))
    return x.astype(input_dtype), w.astype(weight_dtype)


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
    x_compute = x.astype(w.dtype)
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
def test_fixed_schedule_compiled_custom_vjp_primal_and_gradient_match_reference(d, input_dtype, weight_dtype):
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


def test_snowball_shared_qkv_caller_accepts_fp32_weights_and_bf16_activations():
    cfg = SnowballConfig(
        hidden_dim=128,
        intermediate_dim=128,
        shared_expert_intermediate_dim=128,
        num_experts=1,
        num_experts_per_token=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=128,
        projection_implementation="fixed_schedule",
    )
    attention = SnowballAttention(
        w_q=jnp.ones((128, 128), dtype=jnp.float32),
        w_k=jnp.ones((128, 128), dtype=jnp.float32),
        w_v=jnp.ones((128, 128), dtype=jnp.float32),
        w_o=jnp.ones((128, 128), dtype=jnp.float32),
        attn_gate=jnp.zeros((128, 1), dtype=jnp.float32),
        cfg=cfg,
    )
    q, k, v = attention._project_qkv(jnp.ones((1, 1, 128), dtype=jnp.bfloat16), disable_rope=True)

    assert q.dtype == jnp.bfloat16
    assert k.dtype == jnp.bfloat16
    assert v.dtype == jnp.bfloat16
