# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-side tests for the MXFP8 expert-MLP op glue (no GPU / no cutlass needed).

Covers the pieces of ``experiments.grug.moe.mxfp8`` that are pure JAX/numpy:
the 256-aligned padded dispatch layout, the traced wgrad atom-block
permutation (vs the validated host-side reference), the gate/up interleave
permutations (vs the fused-kernel bench formula), and the config knob
validation. Kernel-facing behavior is validated on GB200 by
``standalone/test_mxfp8_op_gpu.py``.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe import mxfp8
from experiments.grug.moe.model import GrugFp8Config, resolve_fp8_config

# mxfp8_grouped only becomes importable after the mxfp8 import extends sys.path.
_quantize = importlib.import_module("mxfp8_grouped.quantize")
sf_wgrad_col_layout = _quantize.sf_wgrad_col_layout
sfa_row_gather_indices = _quantize.sfa_row_gather_indices

GROUP_CASES = [
    [3, 0, 511, 256, 30],  # ragged, zero-token expert, unaligned sizes
    [0, 0, 0, 7],  # nearly empty
    [1024],  # single expert
    [256] * 4,  # already aligned
    [1, 1, 1, 1, 1, 1, 1, 1],  # all tiny
]


@pytest.mark.parametrize("groups", GROUP_CASES)
def test_padded_dispatch_layout_properties(groups):
    capacity = sum(groups)
    g = jnp.asarray(groups, jnp.int32)
    layout = mxfp8.padded_dispatch_layout(g, capacity=capacity)

    gp = np.asarray(layout.padded_group_sizes)
    offs = np.asarray(layout.offs)
    # Every padded group is 256-aligned, at least as large as its raw group,
    # and the buffer is exactly covered (offs[-1] == padded_rows: the fused
    # kernels' contract).
    assert (gp % mxfp8.GROUP_PAD == 0).all()
    assert (gp >= np.asarray(groups)).all()
    assert gp.sum() == layout.padded_rows
    np.testing.assert_array_equal(offs, np.cumsum(gp))
    assert offs[-1] == layout.padded_rows
    assert layout.padded_rows == mxfp8.padded_row_count(capacity, len(groups))
    assert layout.padded_rows % mxfp8.GROUP_PAD == 0

    # Rows land contiguously at each expert's padded start, in order.
    starts_pad = offs - gp
    starts_raw = np.cumsum([0, *groups[:-1]])
    expected_unpad = np.concatenate(
        [starts_pad[e] + np.arange(n) for e, n in enumerate(groups)] or [np.zeros(0, np.int64)]
    )
    np.testing.assert_array_equal(np.asarray(layout.unpad_idx), expected_unpad)
    del starts_raw


@pytest.mark.parametrize("groups", GROUP_CASES)
def test_pad_unpad_roundtrip_and_zero_fill(groups):
    capacity = sum(groups)
    if capacity == 0:
        pytest.skip("empty dispatch buffer")
    g = jnp.asarray(groups, jnp.int32)
    layout = mxfp8.padded_dispatch_layout(g, capacity=capacity)
    x = jnp.arange(1, capacity * 4 + 1, dtype=jnp.float32).reshape(capacity, 4)
    x_pad = mxfp8._pad_rows(x, layout)
    # Roundtrip: unpadding recovers every raw row; pad rows are exact zeros
    # (they must contribute nothing through GEMMs and wgrads).
    np.testing.assert_array_equal(np.asarray(jnp.take(x_pad, layout.unpad_idx, axis=0)), np.asarray(x))
    invalid = ~np.asarray(layout.src_valid)
    assert (np.asarray(x_pad)[invalid] == 0).all()


@pytest.mark.parametrize("groups", GROUP_CASES)
@pytest.mark.parametrize("rows", [256, 1280, 2560])
def test_wgrad_block_perm_matches_host_reference(groups, rows):
    # The traced permutation must reproduce the validated host-side
    # sf_wgrad_col_layout block order for 256-aligned groups (what the fused
    # kernels emit and the wgrad kernel consumes).
    g = jnp.asarray(groups, jnp.int32)
    capacity = sum(groups)
    layout = mxfp8.padded_dispatch_layout(g, capacity=capacity)
    padded_groups = [int(v) for v in np.asarray(layout.padded_group_sizes)]

    col_ref, perm_ref = sf_wgrad_col_layout(padded_groups, rows)
    # 256-aligned groups need no column padding: the gather map is the
    # identity (the op relies on this to skip data-dependent gathers).
    np.testing.assert_array_equal(col_ref, np.arange(layout.padded_rows // 32))
    np.testing.assert_array_equal(sfa_row_gather_indices(padded_groups), np.arange(layout.padded_rows))

    perm = mxfp8.wgrad_block_perm(layout.padded_group_sizes, rows=rows, total_tokens=layout.padded_rows)
    np.testing.assert_array_equal(np.asarray(perm), perm_ref)


def test_interleave_matches_bench_formula():
    # interleave_w13 from bench_mxfp8_fused (the layout the swiglu kernel's
    # correctness run was validated against), reproduced from concat layout.
    e, d, f = 3, 64, 96
    blk = 32
    rng = np.random.default_rng(0)
    w13 = rng.standard_normal((e, d, 2 * f)).astype(np.float32)
    w_gate, w_up = w13[:, :, :f], w13[:, :, f:]
    gate_t = np.swapaxes(w_gate, 1, 2).reshape(e, f // blk, blk, d)
    up_t = np.swapaxes(w_up, 1, 2).reshape(e, f // blk, blk, d)
    ref = np.stack((gate_t, up_t), axis=2).reshape(e, 2 * f, d)
    np.testing.assert_array_equal(np.asarray(mxfp8._interleave_w13(jnp.asarray(w13))), ref)


def test_deinterleave_grad_inverts_interleave():
    e, d, f = 2, 32, 64
    n2 = 2 * f
    rng = np.random.default_rng(1)
    dw_concat = rng.standard_normal((e, d, n2)).astype(np.float32)
    # A gradient over interleaved columns j carries the concat column ic[j];
    # de-interleaving must therefore restore the concat layout exactly.
    dw_interleaved = dw_concat[:, :, mxfp8.interleave_perm(n2)]
    np.testing.assert_array_equal(np.asarray(mxfp8._deinterleave_w13_grad(jnp.asarray(dw_interleaved))), dw_concat)


def test_padded_row_count_is_an_upper_bound():
    rng = np.random.default_rng(2)
    for _ in range(50):
        e = int(rng.integers(1, 12))
        capacity = int(rng.integers(1, 5000))
        cuts = np.sort(rng.integers(0, capacity + 1, size=e - 1))
        groups = np.diff(np.concatenate([[0], cuts, [capacity]]))
        padded = sum(-(-int(g) // mxfp8.GROUP_PAD) * mxfp8.GROUP_PAD for g in groups)
        assert padded <= mxfp8.padded_row_count(capacity, e)


def test_op_is_hashable_static_value():
    # MoeExpertMlpOp contract: threaded through shard_map as a static closure
    # constant, so it must hash and compare by value.
    assert mxfp8.MxFp8MoeMlpOp() == mxfp8.MxFp8MoeMlpOp()
    assert hash(mxfp8.MxFp8MoeMlpOp(producer="xla")) == hash(mxfp8.MxFp8MoeMlpOp(producer="xla"))
    assert mxfp8.MxFp8MoeMlpOp(producer="xla") != mxfp8.MxFp8MoeMlpOp(producer="cute")


def test_op_rejects_non_blackwell_devices():
    if jax.devices()[0].platform == "gpu":
        pytest.skip("GPU host: covered by the GB200 ladder")
    op = mxfp8.MxFp8MoeMlpOp(producer="xla")
    x = jnp.zeros((256, 128), jnp.bfloat16)
    w13 = jnp.zeros((2, 128, 256), jnp.bfloat16)
    w2 = jnp.zeros((2, 128, 128), jnp.bfloat16)
    with pytest.raises(RuntimeError, match="sm100"):
        op(x, w13, w2, jnp.array([128, 128], jnp.int32))


def test_fp8_config_recipe_validation():
    cfg = GrugFp8Config(recipe="mxfp8", wire=False)
    assert cfg.grouped and cfg.dense
    # wire=None resolves per recipe, so a bare mxfp8 config is valid...
    assert GrugFp8Config(recipe="mxfp8").wire is None
    # ...but explicitly requesting fp8 wire with mxfp8 is rejected.
    with pytest.raises(ValueError, match="wire=False"):
        GrugFp8Config(recipe="mxfp8", wire=True)
    with pytest.raises(ValueError, match="grouped=True"):
        GrugFp8Config(recipe="mxfp8", wire=False, grouped=False)
    with pytest.raises(ValueError, match="recipe"):
        GrugFp8Config(recipe="blockwise")


def test_resolve_fp8_config_explicit_recipes():
    # Explicit recipes resolve without probing the device arch.
    per_tensor = resolve_fp8_config(GrugFp8Config(recipe="per_tensor"))
    assert per_tensor.recipe == "per_tensor" and per_tensor.wire is True
    mx = resolve_fp8_config(GrugFp8Config(recipe="mxfp8"))
    assert mx.recipe == "mxfp8" and mx.wire is False
    # Explicit wire passes through.
    assert resolve_fp8_config(GrugFp8Config(recipe="per_tensor", wire=False)).wire is False


def test_resolve_fp8_config_auto_requires_gpu():
    if jax.devices()[0].platform == "gpu":
        pytest.skip("GPU host: auto-resolution is exercised by the GPU ladder")
    with pytest.raises(RuntimeError, match="recipe='auto'"):
        resolve_fp8_config(GrugFp8Config())
