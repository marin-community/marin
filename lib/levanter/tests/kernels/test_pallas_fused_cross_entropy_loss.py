# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import warnings

import jax
import jax.numpy as jnp
import pytest

from levanter.kernels.pallas import autotune_cache_utils
from levanter.kernels.pallas.fused_cross_entropy_loss import api as fused_api
from levanter.kernels.pallas.fused_cross_entropy_loss import pallas_tpu
from levanter.kernels.pallas.fused_cross_entropy_loss import pallas_gpu
from levanter.kernels.pallas.fused_cross_entropy_loss import tuned_block_sizes
from levanter.kernels.pallas.fused_cross_entropy_loss import xla as fused_xla
from levanter.kernels.pallas.fused_cross_entropy_loss.reference import (
    linear_softmax_cross_entropy_loss_reference,
    linear_softmax_cross_entropy_loss_streaming,
)
from levanter.kernels.pallas.fused_cross_entropy_loss.tuned_block_sizes import infer_block_sizes
from levanter.kernels.pallas.fused_cross_entropy_loss.xla import (
    _linear_softmax_cross_entropy_loss_streaming_custom_vjp,
)


def _make_toy_inputs():
    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (2, 3, 4), dtype=jnp.float32)
    w = jax.random.normal(key_w, (4, 5), dtype=jnp.float32)
    y = jax.random.randint(key_y, (2, 3), 0, 5, dtype=jnp.int32)
    return x, w, y


def test_fused_cross_entropy_xla_matches_reference():
    x, w, y = _make_toy_inputs()

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x.reshape(6, 4),
        y.reshape(6),
        w,
        reduction=None,
        logsumexp_weight=0.0,
        implementation="xla",
    )

    loss_ref, _ = linear_softmax_cross_entropy_loss_reference(
        x.reshape(6, 4),
        y.reshape(6),
        w,
    )

    assert jnp.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_xla_return_argmax_matches_reference():
    x, w, y = _make_toy_inputs()
    x = x.reshape(6, 4)
    y = y.reshape(6)

    loss, argmax = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        logsumexp_weight=0.0,
        logit_soft_cap=1.3,
        implementation="xla",
        return_argmax=True,
    )

    loss_ref, _ = linear_softmax_cross_entropy_loss_reference(
        x,
        y,
        w,
        logit_soft_cap=1.3,
    )
    logits = jax.lax.dot_general(x, w, (((1,), (0,)), ((), ())))
    logits = jnp.tanh(logits / 1.3) * 1.3
    argmax_ref = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    assert jnp.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)
    assert jnp.array_equal(argmax, argmax_ref)


def test_fused_cross_entropy_reduction_and_penalty():
    x, w, y = _make_toy_inputs()
    weight = jnp.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    logsumexp_weight = 0.2
    logit_soft_cap = 1.7

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x.reshape(6, 4),
        y.reshape(6),
        w,
        reduction="mean",
        weight=weight,
        logsumexp_weight=logsumexp_weight,
        logit_soft_cap=logit_soft_cap,
        implementation="xla",
    )

    loss_ref, lse_ref = linear_softmax_cross_entropy_loss_reference(
        x.reshape(6, 4),
        y.reshape(6),
        w,
        logit_soft_cap=logit_soft_cap,
    )
    loss_ref = loss_ref + logsumexp_weight * (lse_ref**2)
    loss_ref = (loss_ref * weight).sum() / weight.sum()

    assert jnp.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_grad_matches_reference():
    x, w, y = _make_toy_inputs()
    logsumexp_weight = 0.3

    def loss_api(x_raw, w_raw):
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw.reshape(6, 4),
            y.reshape(6),
            w_raw,
            reduction="mean",
            logsumexp_weight=logsumexp_weight,
            implementation="xla",
        )

    def loss_ref(x_raw, w_raw):
        loss_ref, lse_ref = linear_softmax_cross_entropy_loss_reference(
            x_raw.reshape(6, 4),
            y.reshape(6),
            w_raw,
        )
        loss_ref = loss_ref + logsumexp_weight * (lse_ref**2)
        return loss_ref.mean()

    gx_api, gw_api = jax.grad(loss_api, argnums=(0, 1))(x, w)
    gx_ref, gw_ref = jax.grad(loss_ref, argnums=(0, 1))(x, w)

    # TPU reductions can differ by a few ulps vs the reference path; keep
    # CPU/GPU strict while allowing slight TPU noise.
    grad_tol = 3e-5 if jax.default_backend() == "tpu" else 1e-5

    assert jnp.allclose(gx_api, gx_ref, atol=grad_tol, rtol=grad_tol)
    assert jnp.allclose(gw_api, gw_ref, atol=grad_tol, rtol=grad_tol)


def test_bwd_delta_supertile_soft_cap_scales_label_term():
    x = jnp.array([[4.0, -3.0], [2.5, 3.5]], dtype=jnp.float32)
    w_supertile = jnp.array(
        [
            [2.0, -1.0, 0.5, 3.0],
            [-2.0, 3.0, -3.0, 1.5],
        ],
        dtype=jnp.float32,
    )
    labels = jnp.array([0, 2], dtype=jnp.int32)
    dout_loss = jnp.array([1.0, 0.75], dtype=jnp.float32)
    dout_lse = jnp.array([0.4, -0.2], dtype=jnp.float32)
    dout_loss_plus_lse = dout_loss + dout_lse
    logit_soft_cap = 0.7

    logits_raw = jax.lax.dot_general(x, w_supertile, (((1,), (0,)), ((), ())))
    tanh_val = jnp.tanh(logits_raw / logit_soft_cap)
    logits = tanh_val * logit_soft_cap
    cap_deriv = 1.0 - tanh_val**2
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    probs = jnp.exp(logits - lse[:, None])

    expected_delta = dout_loss_plus_lse[:, None] * probs
    expected_delta = expected_delta.at[jnp.arange(labels.shape[0]), labels].add(-dout_loss)
    expected_delta = expected_delta * cap_deriv

    wrong_delta = dout_loss_plus_lse[:, None] * probs * cap_deriv
    wrong_delta = wrong_delta.at[jnp.arange(labels.shape[0]), labels].add(-dout_loss)

    actual_delta = jax.jit(
        lambda x_arg, labels_arg, w_arg, lse_arg, dout_sum_arg, dout_arg: (
            pallas_tpu._linear_softmax_cross_entropy_loss_bwd_xla_delta_supertile(
                dout_loss_plus_lse=dout_sum_arg,
                dout_loss=dout_arg,
                lse=lse_arg,
                x=x_arg,
                labels=labels_arg,
                w_supertile=w_arg,
                v_start=jnp.asarray(0, dtype=labels_arg.dtype),
                v_dim=w_arg.shape[1],
                dtype=jnp.float32,
                logit_soft_cap=logit_soft_cap,
                precision=None,
            )
        )
    )(x, labels, w_supertile, lse, dout_loss_plus_lse, dout_loss)

    assert jnp.max(jnp.abs(expected_delta - wrong_delta)) > 0.1
    assert jnp.allclose(actual_delta, expected_delta, atol=1e-6, rtol=1e-6)


def test_xla_streaming_custom_vjp_grad_matches_streaming_autodiff():
    x, w, y = _make_toy_inputs()
    logsumexp_weight = 0.2
    logit_soft_cap = 1.5
    block_size = 4

    def loss_custom(x_raw, w_raw):
        loss, lse = _linear_softmax_cross_entropy_loss_streaming_custom_vjp(
            block_size,
            6,
            jnp.float32,
            logit_soft_cap,
            None,
            x_raw.reshape(6, 4),
            y.reshape(6),
            w_raw,
        )
        return (loss + logsumexp_weight * (lse**2)).mean()

    def loss_streaming(x_raw, w_raw):
        loss, lse = linear_softmax_cross_entropy_loss_streaming(
            x_raw.reshape(6, 4),
            y.reshape(6),
            w_raw,
            block_size=block_size,
            dtype=jnp.float32,
            logit_soft_cap=logit_soft_cap,
        )
        return (loss + logsumexp_weight * (lse**2)).mean()

    gx_custom, gw_custom = jax.grad(loss_custom, argnums=(0, 1))(x, w)
    gx_stream, gw_stream = jax.grad(loss_streaming, argnums=(0, 1))(x, w)

    assert jnp.allclose(gx_custom, gx_stream, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(gw_custom, gw_stream, atol=1e-5, rtol=1e-5)


def test_xla_streaming_custom_vjp_grad_matches_streaming_autodiff_with_batch_blocking(
    monkeypatch: pytest.MonkeyPatch,
):
    key = jax.random.PRNGKey(1)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (256, 32), dtype=jnp.float32)
    w = jax.random.normal(key_w, (32, 96), dtype=jnp.float32)
    y = jax.random.randint(key_y, (256,), 0, 96, dtype=jnp.int32)
    logsumexp_weight = 0.2
    logit_soft_cap = 1.5
    block_size = 32

    monkeypatch.setattr(fused_xla, "infer_xla_b_block_size", lambda b, v_block_size: 64)

    def loss_custom(x_raw, w_raw):
        loss, lse = _linear_softmax_cross_entropy_loss_streaming_custom_vjp(
            block_size,
            64,
            jnp.float32,
            logit_soft_cap,
            None,
            x_raw,
            y,
            w_raw,
        )
        return (loss + logsumexp_weight * (lse**2)).mean()

    def loss_streaming(x_raw, w_raw):
        loss, lse = linear_softmax_cross_entropy_loss_streaming(
            x_raw,
            y,
            w_raw,
            block_size=block_size,
            dtype=jnp.float32,
            logit_soft_cap=logit_soft_cap,
        )
        return (loss + logsumexp_weight * (lse**2)).mean()

    gx_custom, gw_custom = jax.grad(loss_custom, argnums=(0, 1))(x, w)
    gx_stream, gw_stream = jax.grad(loss_streaming, argnums=(0, 1))(x, w)

    assert jnp.allclose(gx_custom, gx_stream, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(gw_custom, gw_stream, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_xla_uses_explicit_batch_block_size(monkeypatch: pytest.MonkeyPatch):
    x, w, y = _make_toy_inputs()
    x = x.reshape(6, 4)
    y = y.reshape(6)
    captured: dict[str, int] = {}

    monkeypatch.setattr(fused_xla, "infer_xla_b_block_size", lambda b, v_block_size: 6)

    def fake_custom_vjp(block_size, batch_block_size, dtype, logit_soft_cap, precision, x_arg, labels_arg, w_arg):
        del dtype, logit_soft_cap, precision, x_arg, labels_arg, w_arg
        captured["block_size"] = block_size
        captured["batch_block_size"] = batch_block_size
        return jnp.zeros((6,), dtype=jnp.float32), jnp.zeros((6,), dtype=jnp.float32)

    monkeypatch.setattr(fused_xla, "_linear_softmax_cross_entropy_loss_streaming_custom_vjp", fake_custom_vjp)

    fused_xla.linear_softmax_cross_entropy_loss_xla(
        x,
        y,
        w,
        block_sizes=fused_api.BlockSizes(b_block_size=2, h_block_size=4, v_block_size=4),
        dtype=jnp.float32,
    )

    assert captured == {"block_size": 4, "batch_block_size": 2}


def test_fused_cross_entropy_xla_caps_requested_batch_block_size_to_legal_divisor(
    monkeypatch: pytest.MonkeyPatch,
):
    x, w, y = _make_toy_inputs()
    x = x.reshape(6, 4)
    y = y.reshape(6)
    captured: dict[str, int] = {}

    monkeypatch.setattr(fused_xla, "infer_xla_b_block_size", lambda b, v_block_size: 6)

    def fake_custom_vjp(block_size, batch_block_size, dtype, logit_soft_cap, precision, x_arg, labels_arg, w_arg):
        del dtype, logit_soft_cap, precision, x_arg, labels_arg, w_arg
        captured["block_size"] = block_size
        captured["batch_block_size"] = batch_block_size
        return jnp.zeros((6,), dtype=jnp.float32), jnp.zeros((6,), dtype=jnp.float32)

    monkeypatch.setattr(fused_xla, "_linear_softmax_cross_entropy_loss_streaming_custom_vjp", fake_custom_vjp)

    fused_xla.linear_softmax_cross_entropy_loss_xla(
        x,
        y,
        w,
        block_sizes=fused_api.BlockSizes(b_block_size=1024, h_block_size=4, v_block_size=4),
        dtype=jnp.float32,
    )

    assert captured == {"block_size": 4, "batch_block_size": 6}


def test_fused_cross_entropy_xla_infer_uses_tuned_batch_block_size_when_available(
    monkeypatch: pytest.MonkeyPatch,
):
    x, w, y = _make_toy_inputs()
    x = x.reshape(6, 4)
    y = y.reshape(6)
    captured: dict[str, int] = {}

    monkeypatch.setattr(fused_xla, "infer_xla_v_block_size", lambda b, h, v, dtype: 4)
    monkeypatch.setattr(fused_xla, "infer_xla_b_block_size", lambda b, v_block_size: 6)
    monkeypatch.setattr(
        fused_xla,
        "infer_block_sizes_with_tuned_match",
        lambda *args, **kwargs: (fused_api.BlockSizes(b_block_size=3, h_block_size=4, v_block_size=8), True),
    )

    def fake_custom_vjp(block_size, batch_block_size, dtype, logit_soft_cap, precision, x_arg, labels_arg, w_arg):
        del dtype, logit_soft_cap, precision, x_arg, labels_arg, w_arg
        captured["block_size"] = block_size
        captured["batch_block_size"] = batch_block_size
        return jnp.zeros((6,), dtype=jnp.float32), jnp.zeros((6,), dtype=jnp.float32)

    monkeypatch.setattr(fused_xla, "_linear_softmax_cross_entropy_loss_streaming_custom_vjp", fake_custom_vjp)

    fused_xla.linear_softmax_cross_entropy_loss_xla(
        x,
        y,
        w,
        dtype=jnp.float32,
    )

    assert captured == {"block_size": 4, "batch_block_size": 3}


def test_fused_cross_entropy_xla_infer_falls_back_when_tuned_batch_block_size_is_unsafe(
    monkeypatch: pytest.MonkeyPatch,
):
    x, w, y = _make_toy_inputs()
    x = x.reshape(6, 4)
    y = y.reshape(6)

    monkeypatch.setattr(fused_xla, "infer_xla_v_block_size", lambda b, h, v, dtype: 4_194_304)
    monkeypatch.setattr(fused_xla, "infer_xla_b_block_size", lambda b, v_block_size: 6)
    monkeypatch.setattr(
        fused_xla,
        "infer_block_sizes_with_tuned_match",
        lambda *args, **kwargs: (fused_api.BlockSizes(b_block_size=1024, h_block_size=4, v_block_size=8), True),
    )

    loss, lse = fused_xla.linear_softmax_cross_entropy_loss_xla(
        x,
        y,
        w,
        dtype=jnp.float32,
    )
    loss_ref, lse_ref = linear_softmax_cross_entropy_loss_reference(
        x,
        y,
        w,
        dtype=jnp.float32,
    )

    assert jnp.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(lse, lse_ref, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_xla_rejects_unsafe_explicit_batch_block_size():
    x = jnp.zeros((1024, 4), dtype=jnp.float32)
    w = jnp.zeros((4, 4_194_304), dtype=jnp.float32)
    y = jnp.zeros((1024,), dtype=jnp.int32)

    with pytest.raises(ValueError, match="int32 word-count limit"):
        fused_xla.linear_softmax_cross_entropy_loss_xla(
            x,
            y,
            w,
            block_sizes=fused_api.BlockSizes(b_block_size=1024, h_block_size=4, v_block_size=4_194_304),
            dtype=jnp.float32,
        )


def test_fused_cross_entropy_xla_return_argmax_matches_reference_with_batch_blocking(
    monkeypatch: pytest.MonkeyPatch,
):
    x, w, y = _make_toy_inputs()
    x = x.reshape(6, 4)
    y = y.reshape(6)

    monkeypatch.setattr(fused_xla, "infer_xla_b_block_size", lambda b, v_block_size: 2)

    loss, _, argmax = fused_xla.linear_softmax_cross_entropy_loss_xla(
        x,
        y,
        w,
        block_sizes=fused_api.BlockSizes(b_block_size=2, h_block_size=4, v_block_size=4),
        dtype=jnp.float32,
        logit_soft_cap=1.3,
        return_argmax=True,
    )

    loss_ref, _, argmax_ref = linear_softmax_cross_entropy_loss_reference(
        x,
        y,
        w,
        dtype=jnp.float32,
        logit_soft_cap=1.3,
        return_argmax=True,
    )

    loss_tol = 6e-5 if jax.default_backend() == "tpu" else 1e-5

    assert jnp.allclose(loss, loss_ref, atol=loss_tol, rtol=loss_tol)
    assert jnp.array_equal(argmax, argmax_ref)


def test_fused_cross_entropy_pallas_requires_tpu():
    if jax.default_backend() == "tpu":
        pytest.skip("requires non-TPU backend")

    x = jnp.zeros((128, 128), dtype=jnp.float32)
    w = jnp.zeros((128, 128), dtype=jnp.float32)
    y = jnp.zeros((128,), dtype=jnp.int32)

    with pytest.raises(pallas_tpu.PallasUnsupportedError):
        fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x,
            y,
            w,
            reduction=None,
            implementation="pallas_tpu",
        )


def test_infer_block_sizes_adapts_to_supported_divisors():
    block_sizes = infer_block_sizes(
        b=512,
        h=128,
        v=4096,
        dtype=jnp.float32,
        device_kind="TPU v5e",
    )

    assert block_sizes.b_block_size == 512
    assert block_sizes.h_block_size == 128


def test_infer_block_sizes_preserves_defaults_without_128_aligned_divisors():
    block_sizes = infer_block_sizes(
        b=96,
        h=64,
        v=4096,
        dtype=jnp.float32,
        device_kind="TPU v5e",
    )

    assert block_sizes.b_block_size == 1024
    assert block_sizes.h_block_size == 512


def test_infer_num_tensorcores_uses_device_kind(monkeypatch):
    fake_device = type("FakeDevice", (), {"device_kind": "TPU v4"})()
    monkeypatch.setattr(pallas_tpu.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(pallas_tpu.jax, "devices", lambda: [fake_device])
    assert pallas_tpu._infer_num_tensorcores() == 2

    fake_v5e_device = type("FakeDevice", (), {"device_kind": "TPU v5e"})()
    monkeypatch.setattr(pallas_tpu.jax, "devices", lambda: [fake_v5e_device])
    assert pallas_tpu._infer_num_tensorcores() == 1


def test_device_key_tpu_v5_lite_maps_to_v5e():
    assert tuned_block_sizes._device_key("TPU v5 lite") == "TPU v5e"
    assert tuned_block_sizes._device_key("TPU v5litepod") == "TPU v5e"


def test_pallas_tpu_backward_uses_pallas_by_default(monkeypatch):
    monkeypatch.delenv("LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH", raising=False)
    captured: dict[str, bool] = {}

    def fake_make_custom_vjp(
        b_block_size,
        h_block_size,
        v_block_size,
        dtype,
        logit_soft_cap,
        precision,
        use_bwd_xla_streaming,
    ):
        del b_block_size, h_block_size, v_block_size, dtype, logit_soft_cap, precision
        captured["use_bwd_xla_streaming"] = use_bwd_xla_streaming

        def _fake_fn(x, labels, w):
            del labels, w
            zeros = jnp.zeros((x.shape[0],), dtype=x.dtype)
            return zeros, zeros

        return _fake_fn

    monkeypatch.setattr(pallas_tpu, "_make_custom_vjp", fake_make_custom_vjp)
    x = jnp.zeros((128, 128), dtype=jnp.float32)
    w = jnp.zeros((128, 128), dtype=jnp.float32)
    y = jnp.zeros((128,), dtype=jnp.int32)
    pallas_tpu.linear_softmax_cross_entropy_loss_pallas(x, y, w, block_sizes=fused_api.BlockSizes.get_default())
    assert captured["use_bwd_xla_streaming"] is False


def test_pallas_tpu_backward_can_force_xla_streaming(monkeypatch):
    monkeypatch.setenv("LEVANTER_PALLAS_TPU_BWD_USE_XLA_STREAMING_BENCH", "1")
    captured: dict[str, bool] = {}

    def fake_make_custom_vjp(
        b_block_size,
        h_block_size,
        v_block_size,
        dtype,
        logit_soft_cap,
        precision,
        use_bwd_xla_streaming,
    ):
        del b_block_size, h_block_size, v_block_size, dtype, logit_soft_cap, precision
        captured["use_bwd_xla_streaming"] = use_bwd_xla_streaming

        def _fake_fn(x, labels, w):
            del labels, w
            zeros = jnp.zeros((x.shape[0],), dtype=x.dtype)
            return zeros, zeros

        return _fake_fn

    monkeypatch.setattr(pallas_tpu, "_make_custom_vjp", fake_make_custom_vjp)
    x = jnp.zeros((128, 128), dtype=jnp.float32)
    w = jnp.zeros((128, 128), dtype=jnp.float32)
    y = jnp.zeros((128,), dtype=jnp.int32)
    pallas_tpu.linear_softmax_cross_entropy_loss_pallas(x, y, w, block_sizes=fused_api.BlockSizes.get_default())
    assert captured["use_bwd_xla_streaming"] is True


def test_default_implementation_on_cpu_skips_expected_tpu_warning():
    if jax.default_backend() == "tpu":
        pytest.skip("requires non-TPU backend")

    x = jnp.zeros((32, 64), dtype=jnp.float32)
    w = jnp.zeros((64, 128), dtype=jnp.float32)
    y = jnp.zeros((32,), dtype=jnp.int32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x,
            y,
            w,
            reduction=None,
        )

    assert not any("requires TPU backend" in str(warning.message) for warning in caught)


def test_fused_cross_entropy_default_matches_reference():
    backend = jax.default_backend()
    if backend == "tpu":
        batch, hidden, vocab = 128, 128, 256
        block_sizes = fused_api.BlockSizes(
            b_block_size=128,
            h_block_size=128,
            v_block_size=128,
        )
    else:
        batch, hidden, vocab = 32, 64, 128
        block_sizes = None

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    logsumexp_weight = 0.2
    logit_soft_cap = 2.0

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        logsumexp_weight=logsumexp_weight,
        block_sizes=block_sizes,
        dtype=jnp.float32,
        logit_soft_cap=logit_soft_cap,
    )

    loss_ref, lse_ref = linear_softmax_cross_entropy_loss_reference(
        x,
        y,
        w,
        dtype=jnp.float32,
        logit_soft_cap=logit_soft_cap,
    )
    loss_ref = loss_ref + logsumexp_weight * (lse_ref**2)

    assert jnp.allclose(loss, loss_ref, atol=1e-4, rtol=1e-4)


def test_fused_cross_entropy_default_grad_matches_reference():
    backend = jax.default_backend()
    if backend == "tpu":
        batch, hidden, vocab = 128, 128, 256
        block_sizes = fused_api.BlockSizes(
            b_block_size=128,
            h_block_size=128,
            v_block_size=128,
        )
    else:
        batch, hidden, vocab = 16, 32, 64
        block_sizes = None

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    logsumexp_weight = 0.2
    logit_soft_cap = 1.5

    def loss_default(x_raw, w_raw):
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            logsumexp_weight=logsumexp_weight,
            block_sizes=block_sizes,
            dtype=jnp.float32,
            logit_soft_cap=logit_soft_cap,
        )

    def loss_ref(x_raw, w_raw):
        loss_val, lse_val = linear_softmax_cross_entropy_loss_reference(
            x_raw,
            y,
            w_raw,
            dtype=jnp.float32,
            logit_soft_cap=logit_soft_cap,
        )
        loss_val = loss_val + logsumexp_weight * (lse_val**2)
        return loss_val.mean()

    gx_default, gw_default = jax.grad(loss_default, argnums=(0, 1))(x, w)
    gx_ref, gw_ref = jax.grad(loss_ref, argnums=(0, 1))(x, w)

    assert jnp.allclose(gx_default, gx_ref, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(gw_default, gw_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("implementation", "required_backend", "block_sizes"),
    [
        ("pallas_tpu", "tpu", fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128)),
        ("pallas_gpu", "gpu", None),
    ],
)
def test_fused_cross_entropy_pallas_matches_reference(
    implementation: str,
    required_backend: str,
    block_sizes: fused_api.BlockSizes | None,
):
    if jax.default_backend() != required_backend:
        pytest.skip(f"requires {required_backend.upper()} backend")

    x = jnp.zeros((128, 128), dtype=jnp.float32)
    w = jnp.zeros((128, 128), dtype=jnp.float32)
    y = jnp.zeros((128,), dtype=jnp.int32)

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        implementation=implementation,
        block_sizes=block_sizes,
    )

    loss_ref, _ = linear_softmax_cross_entropy_loss_reference(
        x,
        y,
        w,
        dtype=jnp.float32,
    )
    assert jnp.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_pallas_gpu_matches_reference_non_multiple():
    if jax.default_backend() != "gpu":
        pytest.skip("requires GPU backend")

    key = jax.random.PRNGKey(1)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (6, 7), dtype=jnp.float32)
    w = jax.random.normal(key_w, (7, 9), dtype=jnp.float32)
    y = jax.random.randint(key_y, (6,), 0, 9, dtype=jnp.int32)
    block_sizes = fused_api.BlockSizes(b_block_size=16, h_block_size=16, v_block_size=16)

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        implementation="pallas_gpu",
        block_sizes=block_sizes,
    )

    loss_ref, _ = linear_softmax_cross_entropy_loss_reference(
        x,
        y,
        w,
        dtype=jnp.float32,
    )
    assert jnp.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)


def test_fused_cross_entropy_pallas_gpu_return_argmax_matches_reference():
    if jax.default_backend() != "gpu":
        pytest.skip("requires GPU backend")

    key = jax.random.PRNGKey(5)
    key_x, key_w, key_y = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (6, 7), dtype=jnp.float32)
    w = jax.random.normal(key_w, (7, 9), dtype=jnp.float32)
    y = jax.random.randint(key_y, (6,), 0, 9, dtype=jnp.int32)
    block_sizes = fused_api.BlockSizes(b_block_size=16, h_block_size=16, v_block_size=16)

    loss, argmax = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        logsumexp_weight=0.0,
        logit_soft_cap=1.1,
        implementation="pallas_gpu",
        block_sizes=block_sizes,
        return_argmax=True,
    )

    loss_ref, _ = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        logsumexp_weight=0.0,
        logit_soft_cap=1.1,
        implementation="xla",
        block_sizes=block_sizes,
        return_argmax=True,
    )
    logits = jax.lax.dot_general(x, w, (((1,), (0,)), ((), ())))
    logits = jnp.tanh(logits / 1.1) * 1.1
    argmax_ref = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    assert jnp.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)
    assert jnp.array_equal(argmax, argmax_ref)


def test_fused_cross_entropy_pallas_gpu_requires_gpu():
    if jax.default_backend() == "gpu":
        pytest.skip("requires non-GPU backend")

    x = jnp.zeros((128, 128), dtype=jnp.float32)
    w = jnp.zeros((128, 128), dtype=jnp.float32)
    y = jnp.zeros((128,), dtype=jnp.int32)

    with pytest.raises(pallas_gpu.PallasUnsupportedError):
        fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x,
            y,
            w,
            reduction=None,
            implementation="pallas_gpu",
        )


def test_fused_cross_entropy_pallas_gpu_custom_backward_grad_matches_xla():
    if jax.default_backend() != "gpu":
        pytest.skip("requires GPU backend")
    device_kind = jax.devices()[0].device_kind.lower()
    if "gb10" not in device_kind:
        pytest.skip("requires GB10 device to exercise custom backward path")

    batch, hidden, vocab = 1024, 32, 65537
    key = jax.random.PRNGKey(23)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.bfloat16)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.bfloat16)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    # Match the v-block used by GB10 pallas_gpu forward fallback for B>=1024, V>=65536.
    xla_block_sizes = fused_api.BlockSizes(v_block_size=2048)

    def loss_pallas(x_raw: jax.Array, w_raw: jax.Array) -> jax.Array:
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            logsumexp_weight=0.2,
            dtype=jnp.float32,
            implementation="pallas_gpu",
        )

    def loss_xla(x_raw: jax.Array, w_raw: jax.Array) -> jax.Array:
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            logsumexp_weight=0.2,
            block_sizes=xla_block_sizes,
            dtype=jnp.float32,
            implementation="xla",
        )

    gx_pallas, gw_pallas = jax.grad(loss_pallas, argnums=(0, 1))(x, w)
    gx_xla, gw_xla = jax.grad(loss_xla, argnums=(0, 1))(x, w)

    gx_max_abs = jnp.max(jnp.abs(gx_pallas.astype(jnp.float32) - gx_xla.astype(jnp.float32)))
    gw_max_abs = jnp.max(jnp.abs(gw_pallas.astype(jnp.float32) - gw_xla.astype(jnp.float32)))
    assert gx_max_abs <= 5e-3
    assert gw_max_abs <= 5e-3


def test_fused_cross_entropy_pallas_gpu_grad_tracing_non_gb10_path(
    monkeypatch: pytest.MonkeyPatch,
):
    if jax.default_backend() != "gpu":
        pytest.skip("requires GPU backend")

    monkeypatch.setattr(pallas_gpu, "_device_kind", lambda: "nvidia h100")

    batch, hidden, vocab = 13, 19, 37
    key = jax.random.PRNGKey(31)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)
    block_sizes = fused_api.BlockSizes(b_block_size=16, h_block_size=16, v_block_size=16)

    def loss_fn(x_raw: jax.Array, w_raw: jax.Array) -> jax.Array:
        loss, _ = pallas_gpu.linear_softmax_cross_entropy_loss_pallas_gpu(
            x_raw,
            y,
            w_raw,
            block_sizes=block_sizes,
            dtype=jnp.float32,
        )
        return loss.sum()

    jax.make_jaxpr(jax.grad(loss_fn, argnums=(0, 1)))(x, w)


def test_pallas_autotune_used_when_tuned_match_missing(monkeypatch: pytest.MonkeyPatch):
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    inferred = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128)
    autotuned = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=256)
    seen_block_sizes: list[fused_api.BlockSizes | None] = []

    def fake_impl(x_raw, labels_raw, w_raw, *, block_sizes=None, **_kwargs):
        del labels_raw
        seen_block_sizes.append(block_sizes)
        batch = x_raw.shape[0]
        return jnp.zeros((batch,), dtype=jnp.float32), jnp.zeros((batch,), dtype=jnp.float32)

    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "pallas_tpu", fake_impl)
    monkeypatch.setattr(
        fused_api,
        "infer_block_sizes_with_tuned_match",
        lambda *args, **kwargs: (inferred, False),
    )
    monkeypatch.setattr(
        fused_api,
        "_autotune_block_sizes_on_miss",
        lambda **kwargs: autotuned,
    )

    fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        implementation="pallas_tpu",
    )

    assert seen_block_sizes == [autotuned]


def test_pallas_autotune_skipped_when_tuned_match_exists(monkeypatch: pytest.MonkeyPatch):
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    inferred = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128)
    seen_block_sizes: list[fused_api.BlockSizes | None] = []

    def fake_impl(x_raw, labels_raw, w_raw, *, block_sizes=None, **_kwargs):
        del labels_raw
        seen_block_sizes.append(block_sizes)
        batch = x_raw.shape[0]
        return jnp.zeros((batch,), dtype=jnp.float32), jnp.zeros((batch,), dtype=jnp.float32)

    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "pallas_tpu", fake_impl)
    monkeypatch.setattr(
        fused_api,
        "infer_block_sizes_with_tuned_match",
        lambda *args, **kwargs: (inferred, True),
    )
    monkeypatch.setattr(
        fused_api,
        "_autotune_block_sizes_on_miss",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("autotune should not be called")),
    )

    fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        implementation="pallas_tpu",
    )

    assert seen_block_sizes == [inferred]


def test_autotune_benchmark_wraps_in_shard_map_when_named_sharding_present(
    monkeypatch: pytest.MonkeyPatch,
):
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    mesh = object()
    x_sharding = type("FakeNamedSharding", (), {"mesh": mesh, "spec": ("data", None)})()
    y_sharding = type("FakeNamedSharding", (), {"mesh": mesh, "spec": ("data",)})()
    w_sharding = type("FakeNamedSharding", (), {"mesh": mesh, "spec": (None, "model")})()

    def fake_named_sharding_of(value):
        if value is x:
            return x_sharding
        if value is y:
            return y_sharding
        if value is w:
            return w_sharding
        return None

    calls: list[tuple[object, tuple[object, ...], object, bool]] = []

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
        calls.append((mesh, in_specs, out_specs, check_vma))
        return fn

    monkeypatch.setattr(fused_api.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(fused_api, "_named_sharding_of", fake_named_sharding_of)
    monkeypatch.setattr(fused_api.jax, "shard_map", fake_shard_map)

    wrapped = fused_api._maybe_wrap_loss_in_shard_map_for_benchmark(
        lambda x_value, labels_value, w_value: x_value[:, 0] + labels_value.astype(x_value.dtype) + w_value[0, 0],
        x=x,
        labels=y,
        w=w,
    )

    out = wrapped(x, y, w)
    assert out.shape == (4,)
    assert calls == [(mesh, (x_sharding.spec, y_sharding.spec, w_sharding.spec), y_sharding.spec, False)]


def test_pallas_tpu_vmem_compile_error_falls_back_to_xla_when_requested(monkeypatch: pytest.MonkeyPatch):
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    called = {"pallas": 0, "xla": 0}
    inferred = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128)
    vmem_error = RuntimeError(
        "RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. " "Ran out of memory in memory space vmem."
    )

    def fake_pallas(*args, **kwargs):
        del args, kwargs
        called["pallas"] += 1
        raise vmem_error

    def fake_xla(x_raw, labels_raw, w_raw, **kwargs):
        del labels_raw, w_raw, kwargs
        called["xla"] += 1
        batch = x_raw.shape[0]
        return jnp.zeros((batch,), dtype=jnp.float32), jnp.zeros((batch,), dtype=jnp.float32)

    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "pallas_tpu", fake_pallas)
    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "xla", fake_xla)
    monkeypatch.setattr(
        fused_api,
        "infer_block_sizes_with_tuned_match",
        lambda *args, **kwargs: (inferred, True),
    )
    monkeypatch.setattr(fused_api, "_VMEM_COMPILE_FALLBACK_WARNINGS_EMITTED", set())

    with pytest.warns(RuntimeWarning, match="vmem compile OOM"):
        fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x,
            y,
            w,
            reduction=None,
            implementation=("pallas_tpu", "xla"),
        )

    assert called["pallas"] == 1
    assert called["xla"] == 1


def test_pallas_tpu_vmem_compile_error_uses_remaining_requested_order(monkeypatch: pytest.MonkeyPatch):
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    called = {"pallas": 0, "reference": 0, "xla": 0}
    block_sizes = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=256)
    vmem_error = RuntimeError(
        "RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. " "Ran out of memory in memory space vmem."
    )

    def fake_pallas(*args, **kwargs):
        del args, kwargs
        called["pallas"] += 1
        raise vmem_error

    def fake_reference(x_raw, labels_raw, w_raw, **kwargs):
        del labels_raw, w_raw
        called["reference"] += 1
        assert kwargs["block_sizes"] == block_sizes
        batch = x_raw.shape[0]
        return jnp.ones((batch,), dtype=jnp.float32), jnp.zeros((batch,), dtype=jnp.float32)

    def fake_xla(*args, **kwargs):
        del args, kwargs
        called["xla"] += 1
        raise AssertionError("xla should not run before earlier requested implementations")

    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "pallas_tpu", fake_pallas)
    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "reference", fake_reference)
    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "xla", fake_xla)
    monkeypatch.setattr(fused_api, "_VMEM_COMPILE_FALLBACK_WARNINGS_EMITTED", set())

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        block_sizes=block_sizes,
        implementation=("pallas_tpu", "reference", "xla"),
    )

    assert jnp.array_equal(loss, jnp.ones((4,), dtype=jnp.float32))
    assert called == {"pallas": 1, "reference": 1, "xla": 0}


@pytest.mark.parametrize("implementation", ["pallas_tpu", ("pallas_tpu", "xla")])
def test_pallas_tpu_non_vmem_runtime_error_still_raises(
    monkeypatch: pytest.MonkeyPatch,
    implementation: str | tuple[str, str],
):
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    inferred = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128)
    called = {"xla": 0}

    def fake_pallas(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("some other runtime error")

    def fake_xla(*args, **kwargs):
        del args, kwargs
        called["xla"] += 1
        raise AssertionError("xla should not run for non-vmem explicit failures")

    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "pallas_tpu", fake_pallas)
    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "xla", fake_xla)
    monkeypatch.setattr(
        fused_api,
        "infer_block_sizes_with_tuned_match",
        lambda *args, **kwargs: (inferred, True),
    )

    with pytest.raises(RuntimeError, match="some other runtime error"):
        fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x,
            y,
            w,
            reduction=None,
            implementation=implementation,
        )

    assert called["xla"] == 0


def test_pallas_autotune_cache_reuses_winner(monkeypatch: pytest.MonkeyPatch):
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    inferred = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128)
    faster = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=256)
    slower = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=512)

    calls = {"bench": 0}
    monkeypatch.setattr(fused_api, "_autotune_enabled", lambda: True)
    monkeypatch.setattr(fused_api, "_ensure_autotune_cache_loaded", lambda: None)
    monkeypatch.setattr(fused_api, "_persist_autotune_cache", lambda: None)
    monkeypatch.setattr(
        fused_api,
        "_candidate_block_sizes",
        lambda impl_name, inferred_block_sizes, **kwargs: [inferred_block_sizes, slower, faster],
    )

    def fake_benchmark(**kwargs):
        calls["bench"] += 1
        candidate = kwargs["candidate"]
        return 1.0 if candidate.v_block_size == 256 else 2.0

    monkeypatch.setattr(fused_api, "_benchmark_block_sizes_candidate", fake_benchmark)
    fused_api._AUTOTUNE_BLOCK_SIZE_CACHE.clear()

    def fake_impl(x_raw, labels_raw, w_raw, **kwargs):
        del x_raw, labels_raw, w_raw, kwargs
        return jnp.zeros((4,), dtype=jnp.float32), jnp.zeros((4,), dtype=jnp.float32)

    winner_1 = fused_api._autotune_block_sizes_on_miss(
        impl_name="pallas_tpu",
        fn=fake_impl,
        x=x,
        labels=y,
        w=w,
        inferred=inferred,
        dtype=jnp.float32,
        logit_soft_cap=None,
        precision=None,
        return_argmax=False,
    )
    winner_2 = fused_api._autotune_block_sizes_on_miss(
        impl_name="pallas_tpu",
        fn=fake_impl,
        x=x,
        labels=y,
        w=w,
        inferred=inferred,
        dtype=jnp.float32,
        logit_soft_cap=None,
        precision=None,
        return_argmax=False,
    )

    assert winner_1 == faster
    assert winner_2 == faster
    assert calls["bench"] == 3


def test_fused_ce_autotune_cache_url_uses_jax_cache_subdir(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        autotune_cache_utils,
        "get_jax_compilation_cache_dir",
        lambda: "gs://my-cache-root/compiled",
    )

    cache_url = fused_api._kernel_autotune_cache_url()

    assert (
        cache_url
        == "gs://my-cache-root/compiled/levanter_kernel_autotune/fused_cross_entropy_loss/block_sizes_v1.json"
    )


def test_fused_ce_autotune_jaxpr_hash_is_stable_for_same_inputs():
    x = jnp.ones((4, 8), dtype=jnp.float32)
    w = jnp.ones((8, 16), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)
    inferred = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128)

    def fake_impl(x_raw, labels_raw, w_raw, *, block_sizes=None, **_kwargs):
        del labels_raw, block_sizes
        batch = x_raw.shape[0]
        logits = jnp.einsum("bh,hv->bv", x_raw, w_raw)
        return jnp.sum(logits, axis=-1), jnp.zeros((batch,), dtype=jnp.float32)

    hash_1 = fused_api._autotune_jaxpr_hash(
        fn=fake_impl,
        inferred=inferred,
        x=x,
        labels=y,
        w=w,
        dtype=jnp.float32,
        logit_soft_cap=None,
        precision=None,
        return_argmax=False,
    )
    hash_2 = fused_api._autotune_jaxpr_hash(
        fn=fake_impl,
        inferred=inferred,
        x=x,
        labels=y,
        w=w,
        dtype=jnp.float32,
        logit_soft_cap=None,
        precision=None,
        return_argmax=False,
    )

    assert hash_1 is not None
    assert hash_1 == hash_2


def test_fused_cross_entropy_pallas_bwd_matches_reference():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")
    device_kind = jax.devices()[0].device_kind.lower()
    if device_kind != "tpu v5":
        pytest.skip("requires TPU v5")

    hidden, vocab = 128, 256

    logsumexp_weight = 0.2
    logit_soft_cap = 1.5

    def make_inputs(batch):
        key = jax.random.PRNGKey(batch)
        key_x, key_w, key_y = jax.random.split(key, 3)
        x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
        w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
        y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)
        return x, w, y

    def loss_ref(x_raw, w_raw, y_raw):
        loss_val, lse_val = linear_softmax_cross_entropy_loss_reference(
            x_raw,
            y_raw,
            w_raw,
            dtype=jnp.float32,
            logit_soft_cap=logit_soft_cap,
        )
        loss_val = loss_val + logsumexp_weight * (lse_val**2)
        return loss_val.mean()

    batch = 4096
    block_sizes = fused_api.BlockSizes(b_block_size=1024, h_block_size=128, v_block_size=128)
    x, w, y = make_inputs(batch)
    gx_ref, gw_ref = jax.grad(loss_ref, argnums=(0, 1))(x, w, y)

    def loss_pallas(x_raw, y_raw, w_raw):
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y_raw,
            w_raw,
            reduction="mean",
            logsumexp_weight=logsumexp_weight,
            block_sizes=block_sizes,
            dtype=jnp.float32,
            logit_soft_cap=logit_soft_cap,
            implementation="pallas_tpu",
        )

    loss_pallas(x, y, w).block_until_ready()

    gx_pallas, gw_pallas = jax.grad(loss_pallas, argnums=(0, 2))(x, y, w)
    assert jnp.allclose(gx_pallas, gx_ref, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(gw_pallas, gw_ref, atol=1e-4, rtol=1e-4)


def test_infer_block_sizes_respects_local_batch_and_hidden_divisibility():
    block_sizes = infer_block_sizes(
        b=512,
        h=768,
        v=128_256,
        dtype=jnp.bfloat16,
        device_kind="TPU v5p",
    )
    assert block_sizes.b_block_size % 128 == 0
    assert block_sizes.h_block_size % 128 == 0
    assert 512 % block_sizes.b_block_size == 0
    assert 768 % block_sizes.h_block_size == 0


@pytest.mark.parametrize(
    ("b", "h", "v", "expected"),
    [
        (1024, 512, 16_384, fused_api.BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=1024)),
        (8_192, 4_096, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=512)),
        (65_536, 512, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=2048)),
        (16_384, 2_048, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512)),
    ],
)
def test_infer_block_sizes_tpu_v5p_updated_tuning(
    b: int,
    h: int,
    v: int,
    expected: fused_api.BlockSizes,
):
    block_sizes = infer_block_sizes(
        b=b,
        h=h,
        v=v,
        dtype=jnp.bfloat16,
        device_kind="TPU v5p",
    )
    assert block_sizes == expected


@pytest.mark.parametrize(
    ("b", "h", "v", "expected"),
    [
        (1024, 512, 16_384, fused_api.BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=1024)),
        (16_384, 1024, 128_256, fused_api.BlockSizes(b_block_size=8192, h_block_size=256, v_block_size=1024)),
        (65_536, 512, 128_256, fused_api.BlockSizes(b_block_size=4096, h_block_size=512, v_block_size=2048)),
        (262_144, 1024, 128_256, fused_api.BlockSizes(b_block_size=8192, h_block_size=256, v_block_size=1024)),
        (8_192, 4_096, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=512)),
        (16_384, 2_048, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512)),
    ],
)
def test_infer_block_sizes_tpu_v5e_updated_tuning(
    b: int,
    h: int,
    v: int,
    expected: fused_api.BlockSizes,
):
    block_sizes = infer_block_sizes(
        b=b,
        h=h,
        v=v,
        dtype=jnp.bfloat16,
        device_kind="TPU v5 lite",
    )
    assert block_sizes == expected


@pytest.mark.parametrize(
    ("b", "h", "v", "expected"),
    [
        (1024, 512, 16_384, fused_api.BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=2048)),
        (16_384, 1024, 128_256, fused_api.BlockSizes(b_block_size=8192, h_block_size=1024, v_block_size=1024)),
        (65_536, 512, 128_256, fused_api.BlockSizes(b_block_size=4096, h_block_size=512, v_block_size=2048)),
        (262_144, 1024, 128_256, fused_api.BlockSizes(b_block_size=8192, h_block_size=256, v_block_size=1024)),
        (8_192, 4_096, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=512)),
        (16_384, 2_048, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512)),
    ],
)
def test_infer_block_sizes_tpu_v6_updated_tuning(
    b: int,
    h: int,
    v: int,
    expected: fused_api.BlockSizes,
):
    block_sizes = infer_block_sizes(
        b=b,
        h=h,
        v=v,
        dtype=jnp.bfloat16,
        device_kind="TPU v6e",
    )
    assert block_sizes == expected


@pytest.mark.parametrize(
    ("b", "h", "v", "expected"),
    [
        (1024, 512, 16_384, fused_api.BlockSizes(b_block_size=1024, h_block_size=256, v_block_size=512)),
        (16_384, 1024, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=1024, v_block_size=256)),
        (65_536, 512, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=512)),
        (262_144, 1024, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=1024, v_block_size=256)),
        (8_192, 4_096, 128_256, fused_api.BlockSizes(b_block_size=8192, h_block_size=512, v_block_size=1024)),
        (16_384, 2_048, 128_256, fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024)),
    ],
)
def test_infer_block_sizes_tpu_v4_updated_tuning(
    b: int,
    h: int,
    v: int,
    expected: fused_api.BlockSizes,
):
    block_sizes = infer_block_sizes(
        b=b,
        h=h,
        v=v,
        dtype=jnp.bfloat16,
        device_kind="TPU v4",
    )
    assert block_sizes == expected


def test_infer_block_sizes_tpu_v4_huge_batch_small_h_jits_with_pallas():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")
    device_kind = jax.devices()[0].device_kind.lower()
    if "v4" not in device_kind:
        pytest.skip("requires TPU v4 backend")

    b, h, v = 262_144, 1024, 128_256
    block_sizes = infer_block_sizes(
        b=b,
        h=h,
        v=v,
        dtype=jnp.bfloat16,
        device_kind=device_kind,
    )

    def loss_fn(x: jax.Array, w: jax.Array, y: jax.Array) -> jax.Array:
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x,
            y,
            w,
            reduction="mean",
            logsumexp_weight=0.0,
            block_sizes=block_sizes,
            dtype=jnp.float32,
            implementation="pallas_tpu",
        )

    compiled = (
        jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
        .lower(
            jax.ShapeDtypeStruct((b, h), jnp.bfloat16),
            jax.ShapeDtypeStruct((h, v), jnp.bfloat16),
            jax.ShapeDtypeStruct((b,), jnp.int32),
        )
        .compile()
    )

    assert compiled is not None


def test_infer_block_sizes_huge_batch_without_scoped_vmem_flag_warns_and_uses_safe_v(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    monkeypatch.setattr(tuned_block_sizes, "_WARNED_HUGE_BATCH_SAFE_FALLBACK", False)

    with pytest.warns(RuntimeWarning, match="Using safer fused CE huge-batch block sizes"):
        block_sizes = infer_block_sizes(
            b=262_144,
            h=4096,
            v=128_256,
            dtype=jnp.bfloat16,
            device_kind="TPU v5p",
        )
    assert block_sizes.v_block_size == 256


def test_infer_block_sizes_uses_widest_operand_dtype_bucket(monkeypatch: pytest.MonkeyPatch):
    tuned = dict(tuned_block_sizes.TUNED_BLOCK_SIZES["TPU v5p"])
    tuned[("bfloat16", "large-batch-medium-h")] = fused_api.BlockSizes(
        b_block_size=1024,
        h_block_size=256,
        v_block_size=256,
    )
    tuned[("float32", "large-batch-medium-h")] = fused_api.BlockSizes(
        b_block_size=1024,
        h_block_size=1024,
        v_block_size=768,
    )
    monkeypatch.setitem(tuned_block_sizes.TUNED_BLOCK_SIZES, "TPU v5p", tuned)

    mixed_block_sizes, has_tuned_match = tuned_block_sizes.infer_block_sizes_with_tuned_match(
        40_960,
        2_048,
        128_256,
        dtype=jnp.bfloat16,
        x_dtype=jnp.bfloat16,
        w_dtype=jnp.float32,
        device_kind="TPU v5p",
    )
    bf16_block_sizes = tuned_block_sizes.infer_block_sizes(
        40_960,
        2_048,
        128_256,
        dtype=jnp.bfloat16,
        x_dtype=jnp.bfloat16,
        w_dtype=jnp.bfloat16,
        device_kind="TPU v5p",
    )
    float32_block_sizes = tuned_block_sizes.infer_block_sizes(
        40_960,
        2_048,
        128_256,
        dtype=jnp.bfloat16,
        x_dtype=jnp.float32,
        w_dtype=jnp.float32,
        device_kind="TPU v5p",
    )

    assert has_tuned_match is True
    assert mixed_block_sizes == float32_block_sizes
    assert mixed_block_sizes != bf16_block_sizes


def test_shape_bucket_name_large_batch_medium_h_boundary():
    assert (
        tuned_block_sizes.shape_bucket_name(32_767, 2_048, 128_256, device_kind="TPU v5p") == "medium-batch-medium-h"
    )
    assert tuned_block_sizes.shape_bucket_name(32_768, 2_048, 128_256, device_kind="TPU v5p") == "large-batch-medium-h"


def test_pallas_autotune_only_runs_when_block_sizes_are_not_explicit(monkeypatch: pytest.MonkeyPatch):
    inferred = fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=512)
    autotuned = fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=768)
    explicit = fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024)
    autotune_calls: list[fused_api.BlockSizes] = []
    seen_block_sizes: list[fused_api.BlockSizes | None] = []

    monkeypatch.setattr(fused_api, "infer_block_sizes_with_tuned_match", lambda *args, **kwargs: (inferred, False))

    def fake_autotune(**kwargs):
        autotune_calls.append(kwargs["inferred"])
        return autotuned

    def fake_impl(x, labels, w, *, block_sizes, **kwargs):
        del labels, w, kwargs
        seen_block_sizes.append(block_sizes)
        zeros = jnp.zeros((x.shape[0],), dtype=jnp.float32)
        return zeros, zeros

    monkeypatch.setattr(fused_api, "_autotune_block_sizes_on_miss", fake_autotune)
    monkeypatch.setitem(fused_api.IMPLEMENTATIONS, "pallas_tpu", fake_impl)

    x = jnp.ones((1024, 512), dtype=jnp.bfloat16)
    w = jnp.ones((512, 4096), dtype=jnp.float32)
    labels = jnp.zeros((1024,), dtype=jnp.int32)

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        labels,
        w,
        reduction="mean",
        dtype=jnp.float32,
        implementation="pallas_tpu",
    )

    assert autotune_calls == [inferred]
    assert seen_block_sizes == [autotuned]
    assert float(loss) == 0.0

    autotune_calls.clear()
    seen_block_sizes.clear()

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        labels,
        w,
        reduction="mean",
        dtype=jnp.float32,
        implementation="pallas_tpu",
        block_sizes=explicit,
    )

    assert autotune_calls == []
    assert seen_block_sizes == [explicit]
    assert float(loss) == 0.0


def test_infer_block_sizes_skips_invalid_tuned_entry(monkeypatch: pytest.MonkeyPatch):
    tuned = dict(tuned_block_sizes.TUNED_BLOCK_SIZES["TPU v5p"])
    tuned[("bfloat16", "llama3-ish")] = fused_api.BlockSizes(
        b_block_size=3072,
        h_block_size=512,
        v_block_size=1024,
    )
    monkeypatch.setitem(tuned_block_sizes.TUNED_BLOCK_SIZES, "TPU v5p", tuned)

    block_sizes = infer_block_sizes(
        b=4096,
        h=4096,
        v=128_256,
        dtype=jnp.bfloat16,
        device_kind="TPU v5p",
    )
    assert block_sizes == fused_api.BlockSizes(b_block_size=1024, h_block_size=512, v_block_size=1024)


def test_fused_cross_entropy_default_non_divisible_vocab_matches_reference():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")

    hidden, vocab, batch = 128, 130, 128
    block_sizes = fused_api.BlockSizes(b_block_size=512, h_block_size=128, v_block_size=128)

    key = jax.random.PRNGKey(0)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    def loss_default(x_raw, w_raw):
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            block_sizes=block_sizes,
            dtype=jnp.float32,
        )

    def loss_ref(x_raw, w_raw):
        loss_val, _ = linear_softmax_cross_entropy_loss_reference(
            x_raw,
            y,
            w_raw,
            dtype=jnp.float32,
        )
        return loss_val.mean()

    loss_default_val = loss_default(x, w)
    loss_ref_val = loss_ref(x, w)
    gx_default, gw_default = jax.grad(loss_default, argnums=(0, 1))(x, w)
    gx_ref, gw_ref = jax.grad(loss_ref, argnums=(0, 1))(x, w)

    assert jnp.allclose(loss_default_val, loss_ref_val, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(gx_default, gx_ref, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(gw_default, gw_ref, atol=1e-4, rtol=1e-4)


def test_fused_cross_entropy_pallas_non_divisible_vocab_dx_matches_xla():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")

    hidden, vocab, batch = 256, 130, 128
    block_sizes = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128)

    key = jax.random.PRNGKey(17)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    def loss_pallas(x_raw: jax.Array, w_raw: jax.Array) -> jax.Array:
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            block_sizes=block_sizes,
            dtype=jnp.float32,
            implementation="pallas_tpu",
        )

    def loss_xla(x_raw: jax.Array, w_raw: jax.Array) -> jax.Array:
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            block_sizes=block_sizes,
            dtype=jnp.float32,
            implementation="xla",
        )

    gx_pallas, gw_pallas = jax.grad(loss_pallas, argnums=(0, 1))(x, w)
    gx_xla, gw_xla = jax.grad(loss_xla, argnums=(0, 1))(x, w)

    assert jnp.allclose(gx_pallas, gx_xla, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(gw_pallas, gw_xla, atol=1e-4, rtol=1e-4)


def test_fused_cross_entropy_pallas_backward_matches_xla():
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")

    hidden, vocab, batch = 256, 2048, 512
    block_sizes = fused_api.BlockSizes(b_block_size=512, h_block_size=128, v_block_size=128)

    key = jax.random.PRNGKey(29)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    def loss_pallas(x_raw: jax.Array, w_raw: jax.Array) -> jax.Array:
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            block_sizes=block_sizes,
            dtype=jnp.float32,
            implementation="pallas_tpu",
        )

    def loss_xla(x_raw: jax.Array, w_raw: jax.Array) -> jax.Array:
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            block_sizes=block_sizes,
            dtype=jnp.float32,
            implementation="xla",
        )

    gx_pallas, gw_pallas = jax.grad(loss_pallas, argnums=(0, 1))(x, w)
    gx_xla, gw_xla = jax.grad(loss_xla, argnums=(0, 1))(x, w)

    assert jnp.allclose(gx_pallas, gx_xla, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(gw_pallas, gw_xla, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("implementation", ["pallas_tpu"])
def test_fused_cross_entropy_pallas_backward_matches_xla_infer_blocks(implementation: str):
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU backend")

    hidden, vocab, batch = 256, 2048, 512
    block_sizes = fused_api.BlockSizes(b_block_size=512, h_block_size=128, v_block_size=128)

    key = jax.random.PRNGKey(37)
    key_x, key_w, key_y = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (batch, hidden), dtype=jnp.float32)
    w = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    y = jax.random.randint(key_y, (batch,), 0, vocab, dtype=jnp.int32)

    def loss_linear_ce(x_raw: jax.Array, w_raw: jax.Array) -> jax.Array:
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            block_sizes=block_sizes,
            dtype=jnp.float32,
            implementation=implementation,
        )

    def loss_xla(x_raw: jax.Array, w_raw: jax.Array) -> jax.Array:
        return fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
            x_raw,
            y,
            w_raw,
            reduction="mean",
            block_sizes=block_sizes,
            dtype=jnp.float32,
            implementation="xla",
        )

    gx_linear_ce, gw_linear_ce = jax.grad(loss_linear_ce, argnums=(0, 1))(x, w)
    gx_xla, gw_xla = jax.grad(loss_xla, argnums=(0, 1))(x, w)

    assert jnp.allclose(gx_linear_ce, gx_xla, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(gw_linear_ce, gw_xla, atol=1e-4, rtol=1e-4)
