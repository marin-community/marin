# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.attention import AttentionMask
from levanter.models.snowball import (
    SnowballConfig,
    _fixed_shared_attention,
    _fixed_shared_attention_forward,
    _fixed_shared_attention_with_vjp_bwd,
)


def _config() -> SnowballConfig:
    return SnowballConfig(
        vocab_size=23,
        hidden_dim=128,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=128,
        max_seq_len=256,
        shared_attention_implementation="fixed_schedule",
    )


def _fixture(dtype: jnp.dtype) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(51109)
    q = jnp.asarray(rng.normal(0, 0.7, (4, 256, 2, 128)), dtype=dtype)
    k = jnp.asarray(rng.normal(0, 0.7, (4, 256, 1, 128)), dtype=dtype)
    v = jnp.asarray(rng.normal(0, 0.5, (4, 256, 1, 128)), dtype=dtype)
    cotangent = np.zeros(q.shape, dtype=np.float32)
    # Distinct head/feature weights avoid cancellation, including both sides of a page boundary.
    cotangent[:, [0, 1, 127, 128, 255]] = rng.normal(size=(4, 5, 2, 128))
    return q, k, v, jnp.asarray(cotangent)


def _oracle(q: jax.Array, k: jax.Array, v: jax.Array, sliding_window: int | None = None) -> jax.Array:
    # Expand in BF16 before promotion: the cast-aware adjoint rounds each query-head
    # contribution to BF16 before reducing contributions for the shared KV head.
    repeats = q.shape[2] // k.shape[2]
    k = jnp.repeat(k.astype(jnp.bfloat16), repeats, axis=2).astype(jnp.float32)
    v = jnp.repeat(v.astype(jnp.bfloat16), repeats, axis=2).astype(jnp.float32)
    scale = jnp.asarray(q.shape[-1] ** -0.5, dtype=jnp.float32)
    return jax.nn.dot_product_attention(
        q.astype(jnp.float32) * scale,
        k,
        v,
        scale=1.0,
        is_causal=True,
        local_window_size=None if sliding_window is None else (sliding_window - 1, 0),
        implementation="xla",
    )


def _oracle_gradients(
    q: jax.Array, k: jax.Array, v: jax.Array, cotangent: jax.Array, sliding_window: int | None = None
) -> tuple[jax.Array, ...]:
    return jax.grad(
        lambda q_, k_, v_: jnp.sum(_oracle(q_, k_, v_, sliding_window) * cotangent),
        argnums=(0, 1, 2),
    )(q, k, v)


def _assert_gradients(actual: tuple[jax.Array, ...], expected: tuple[jax.Array, ...]) -> None:
    for name, observed, reference in zip(("q", "k", "v"), actual, expected, strict=True):
        assert observed.shape == reference.shape
        assert observed.dtype == reference.dtype
        observed = np.asarray(observed, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        assert np.isfinite(observed).all(), name
        magnitude = np.abs(reference)
        # The previous fixture admitted an identically zero dK. Each input now has
        # a substantial signal, and zero derivatives exceed both error bounds.
        assert magnitude.max() > 0.05, name
        error = np.abs(observed - reference)
        metrics = f"{name}: max absolute error={error.max()}, mean absolute error={error.mean()}"
        # BF16 adjoint casts can straddle a rounding boundary between independently
        # evaluated attention formulas. Bound both peak and average relative error.
        assert error.max() <= 1e-3 * magnitude.max(), metrics
        assert error.mean() <= 1e-4 * magnitude.mean(), metrics


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("sliding_window", [None, 128])
def test_fixed_schedule_reverse_callback_matches_xla_attention(dtype: jnp.dtype, sliding_window: int | None) -> None:
    q, k, v, cotangent = _fixture(dtype)
    mask = AttentionMask.causal(sliding_window=sliding_window)
    with jax.default_matmul_precision("highest"):
        actual = _fixed_shared_attention_with_vjp_bwd(mask, _config(), (q, k, v), cotangent)
        expected = _oracle_gradients(q, k, v, cotangent, sliding_window)
    _assert_gradients(actual, expected)


def test_fixed_schedule_reverse_callback_restores_training_sequence_layout() -> None:
    q, k, v, cotangent = _fixture(jnp.float32)
    devices = jax.devices()[: min(4, len(jax.devices()))]
    mesh = Mesh(
        np.asarray(devices).reshape(1, len(devices), 1, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    training_spec = P(("replica_dcn", "data"), "expert", None, None)
    sharding = NamedSharding(mesh, training_spec)
    cfg = _config()
    mask = AttentionMask.causal()
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        q, k, v, cotangent = (jax.device_put(value, sharding) for value in (q, k, v, cotangent))
        actual = eqx.filter_jit(
            lambda q_, k_, v_, ct: _fixed_shared_attention_with_vjp_bwd(mask, cfg, (q_, k_, v_), ct)
        )(q, k, v, cotangent)
        # Compare compiled attention on the same devices; eager TPU operations can
        # round BF16 adjoints differently from the fused compiled reference.
        reference_sharding = NamedSharding(mesh, P("data", None, None, None))
        reference_inputs = tuple(jax.sharding.reshard(value, reference_sharding) for value in (q, k, v, cotangent))
        expected = eqx.filter_jit(_oracle_gradients)(*reference_inputs)
    _assert_gradients(actual, expected)
    for gradient in actual:
        assert gradient.sharding.spec == training_spec


@pytest.mark.tpu
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.timeout(300)
def test_fixed_schedule_compiled_adapter_preserves_primal_and_gradients_on_tpu_v4(checkpoint: bool) -> None:
    if jax.default_backend() != "tpu" or len(jax.devices()) < 4:
        pytest.skip("fixed shared attention requires four TPUv4 devices")
    if any(device.device_kind != "TPU v4" for device in jax.devices()[:4]):
        pytest.skip("the bounded shared-attention fixture is qualified for TPUv4")

    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(4, 1, 1, 1),
        ("data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    sharding = NamedSharding(mesh, P("data", None, None, None))
    cfg = _config()
    mask = AttentionMask.causal()
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        q, k, v, cotangent = (jax.device_put(value, sharding) for value in _fixture(jnp.float32))

        def objective(q_: jax.Array, k_: jax.Array, v_: jax.Array) -> tuple[jax.Array, jax.Array]:
            output = _fixed_shared_attention(q_, k_, v_, mask, cfg=cfg)
            return jnp.sum(output * cotangent), output

        differentiated = jax.checkpoint(objective) if checkpoint else objective
        (_, primal), actual = eqx.filter_jit(jax.value_and_grad(differentiated, argnums=(0, 1, 2), has_aux=True))(
            q, k, v
        )
        accepted_forward = eqx.filter_jit(
            lambda q_, k_, v_: _fixed_shared_attention_forward(q_, k_, v_, mask, cfg=cfg)
        )(q, k, v)
        expected = eqx.filter_jit(_oracle_gradients)(q, k, v, cotangent)

    np.testing.assert_array_equal(
        np.asarray(primal).view(np.uint32),
        np.asarray(accepted_forward).view(np.uint32),
    )
    assert np.isfinite(np.asarray(primal)).all()
    assert primal.sharding.spec == P("data", None, None, None)
    for gradient in actual:
        assert gradient.sharding.spec == P("data", None, None, None)
    _assert_gradients(actual, expected)
