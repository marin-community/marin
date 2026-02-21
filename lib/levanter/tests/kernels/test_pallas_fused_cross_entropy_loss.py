# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import warnings
import jax
import jax.numpy as jnp
import pytest

from levanter.kernels.pallas.fused_cross_entropy_loss import api as fused_api
from levanter.kernels.pallas.fused_cross_entropy_loss import pallas_tpu
from levanter.kernels.pallas.fused_cross_entropy_loss import pallas_gpu
from levanter.kernels.pallas.fused_cross_entropy_loss import tuned_block_sizes
from levanter.kernels.pallas.fused_cross_entropy_loss.reference import (
    linear_softmax_cross_entropy_loss_reference,
)
from levanter.kernels.pallas.fused_cross_entropy_loss.tuned_block_sizes import infer_block_sizes


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

    assert jnp.allclose(gx_api, gx_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(gw_api, gw_ref, atol=1e-5, rtol=1e-5)


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


def test_fused_cross_entropy_pallas_gpu_matches_reference():
    if jax.default_backend() != "gpu":
        pytest.skip("requires GPU backend")

    x = jnp.zeros((128, 128), dtype=jnp.float32)
    w = jnp.zeros((128, 128), dtype=jnp.float32)
    y = jnp.zeros((128,), dtype=jnp.int32)

    loss = fused_api.fused_cross_entropy_loss_and_logsumexp_penalty(
        x,
        y,
        w,
        reduction=None,
        implementation="pallas_gpu",
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


def test_infer_block_sizes_huge_batch_with_scoped_vmem_flag_uses_fast_v(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_tpu_scoped_vmem_limit_kib=50000")
    monkeypatch.setattr(tuned_block_sizes, "_WARNED_HUGE_BATCH_SAFE_FALLBACK", False)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        block_sizes = infer_block_sizes(
            b=262_144,
            h=4096,
            v=128_256,
            dtype=jnp.bfloat16,
            device_kind="TPU v5p",
        )
    assert len(recorded) == 0
    assert block_sizes.v_block_size == 1024


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
    block_sizes = fused_api.BlockSizes(b_block_size=128, h_block_size=128, v_block_size=128)

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
