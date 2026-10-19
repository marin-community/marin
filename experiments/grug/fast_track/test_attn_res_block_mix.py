# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The AttnRes history kernels (``attn_res_block_mix``, ``attn_res_source_grads``) against plain JAX, and the
scanned AttnRes layers with them against the XLA path (same loss, gradients and gate logs)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_scan_layers as scan_test
from experiments.grug.fast_track.attn_res_block_mix import (
    block_mix_local,
    block_mix_reference,
    block_mix_weight_grads_local,
)
from experiments.grug.fast_track.attn_res_source_grads import accumulate_source_grads_local

_N, _G, _T, _D = 5, 2, 32, 512


def _history(num_valid: int):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    sources = jax.random.normal(k1, (_N, _T, _D), jnp.float32).astype(jnp.bfloat16)
    logits = jax.random.normal(k2, (_N, _G, _T), jnp.float32)
    logits = jnp.where((jnp.arange(_N) < num_valid)[:, None, None], logits, -jnp.inf)
    weights = jax.nn.softmax(logits, axis=0)  # exactly zero on the unfilled slots
    d_mixed = jax.random.normal(k3, (_G, _T, _D), jnp.float32).astype(jnp.bfloat16)
    return sources, weights, d_mixed


@pytest.mark.parametrize("num_valid", [1, 3, _N])
def test_block_mix_matches_the_dense_sum(num_valid):
    sources, weights, _ = _history(num_valid)
    mixed = block_mix_local(sources, weights, jnp.int32(num_valid), interpret=True)
    expected = block_mix_reference(sources, weights)
    # The float32 sums may round (FMA contraction) differently: at most one bfloat16 ulp apart.
    np.testing.assert_allclose(np.asarray(mixed, np.float32), np.asarray(expected, np.float32), rtol=2**-7, atol=1e-6)


@pytest.mark.parametrize("num_valid", [1, 3, _N])
def test_block_mix_weight_grads_match_autodiff(num_valid):
    sources, weights, d_mixed = _history(num_valid)
    _, vjp = jax.vjp(lambda w: block_mix_reference(sources, w), weights)
    (expected,) = vjp(d_mixed)
    d_weights = block_mix_weight_grads_local(sources, d_mixed, jnp.int32(num_valid), interpret=True)
    valid = np.arange(_N) < num_valid
    np.testing.assert_allclose(np.asarray(d_weights)[valid], np.asarray(expected)[valid], rtol=1e-5, atol=1e-3)
    # Unfilled slots get zero (their weights are exactly zero, so their weight gradient never reaches a logit).
    np.testing.assert_array_equal(np.asarray(d_weights)[~valid], 0.0)


@pytest.mark.parametrize("num_valid", [2, _N])
def test_source_grads_skip_only_unfilled_slots(num_valid):
    sources, weights, d_mixed = _history(num_valid)
    d_sources = sources  # any bf16 [N, T, D] stack
    gate_weights = tuple(weights[:, g] for g in range(_G))
    d_mixes = tuple(d_mixed[g].astype(jnp.float32) for g in range(_G))
    out = accumulate_source_grads_local(d_sources, gate_weights, d_mixes, jnp.int32(num_valid), interpret=True)
    acc = d_sources.astype(jnp.float32)
    for w, dm in zip(gate_weights, d_mixes, strict=True):
        acc = acc + w[..., None] * dm[None]
    expected = np.asarray(acc.astype(jnp.bfloat16), np.float32)
    np.testing.assert_allclose(np.asarray(out, np.float32), expected, rtol=2**-7, atol=1e-6)
    # Unfilled slots are left exactly as they were.
    np.testing.assert_array_equal(np.asarray(out[num_valid:]), np.asarray(d_sources[num_valid:]))


@pytest.mark.timeout(300)
def test_scanned_layers_with_the_history_kernels_match_xla():
    overrides = dict(num_layers=8, attn_res_num_blocks=4, scan_layers=True, attn_res_two_phase=True)
    loss, grads, metrics = scan_test._loss_grads_metrics(**overrides)
    fused_loss, fused_grads, fused_metrics = scan_test._loss_grads_metrics(
        attn_res_fused_block_mix=True, attn_res_fused_source_grads=True, **overrides
    )
    _, exact_grads, _ = scan_test._loss_grads_metrics(
        jnp.float64, num_layers=8, attn_res_num_blocks=4, attn_res_two_phase=True
    )
    np.testing.assert_allclose(fused_loss, loss, rtol=1e-6)
    # The kernels sum in a different order, so float32 gradients are compared by their error against float64.
    for xla, fused, exact in zip(grads, fused_grads, exact_grads, strict=True):
        xla_err = np.max(np.abs(xla - exact))
        assert np.max(np.abs(fused - exact)) <= 2 * xla_err + 1e-6 * np.max(np.abs(exact)) + 1e-12
    gate_logs = sorted(k for k in metrics if k.startswith("train/attn_res/w_g"))
    assert gate_logs
    for key in gate_logs:
        np.testing.assert_allclose(np.asarray(fused_metrics[key]), np.asarray(metrics[key]), rtol=1e-5, atol=1e-7)
