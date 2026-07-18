# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stateless MXFP8 whole-expert-MLP op on the fused Blackwell kernels (MXFP8-004c).

Implements ``levanter.grug._moe.common.MoeExpertMlpOp`` for the grug MoE EP
backends using the vendored cudnn-frontend fused MXFP8 grouped kernels
(``standalone/mxfp8_fused``, validated in MXFP8-004a). Unlike the per-tensor
``Fp8RaggedDotOp`` recipe, MXFP8 block scaling is computed on the fly, so the
op carries no state (no amax history, no ``OverwriteWithGradient``) and is a
plain hashable value threaded through ``shard_map`` statically (MXFP8-H3).

Op boundary: ``grouped_gemm_swiglu_quant`` fuses the w13 GEMM, SwiGLU, and the
dual-orientation quantize of the hidden activation into one kernel, so the
natural custom-VJP unit is the WHOLE expert MLP (w13 -> SwiGLU -> w2), not two
independent ragged dots:

- fwd: pad dispatch rows to the kernels' 256-row group contract, dual-quantize
  x, quantize the (interleaved) weights, then ``swiglu_quant`` + the bf16-out
  quantizing GEMM for w2.
- bwd: dual-quantize the cotangent, ``dswiglu_quant`` (dgrad-w2 GEMM + dSwiGLU
  + dual quantize of dc), bf16-out GEMM for dgrad-w13, and two 2Dx2D wgrads
  fed entirely by column-quantized tensors.
- residuals: the QUANTIZED column-orientation tensors the wgrads need
  (``x_col``/``h_col`` + swizzled SFs) plus the bf16 pre-activation ``c13``
  that dSwiGLU consumes; the bf16 hidden activation ``h`` is never stored
  (that is the point of the fused quantizing epilogue). Weights are saved
  as-is (aliases of the live parameters) and their dgrad-orientation copies
  are re-quantized in bwd, trading a cheap producer pass for not holding fp8
  weight copies across the whole forward.

Padding: EP backends hand the op a fixed-capacity dispatch buffer with raw
cumsum group layout; the op re-packs rows into a 256-aligned padded buffer
(zero-filled pad rows contribute exact zeros through every leg) and hands the
kernels padded END offsets. All scale-factor swizzles reuse the bit-exact
builders from ``mxfp8_grouped.quantize`` with traced index maps; under
256-padding the row/column gather maps degenerate to identities and only the
wgrad atom-block permutation stays group-dependent.

Producers: the dual-orientation quantize of x / cotangent uses the MXFP8-002c
CuTe kernel when the node's libNVVM can compile it (probed once in a
subprocess -- compile failures on the FFI thread are process-fatal) and falls
back to the corrected XLA path otherwise.

The vendored kernels live under ``standalone/`` (imported via sys.path, as the
benches do); this module itself stays importable without ``cutlass`` -- kernel
wrappers are imported lazily inside the traced pipeline (sanctioned local
imports: optional GPU-only dependency).
"""

import functools
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from haliax.jax_utils import tree_checkpoint_name

_STANDALONE_DIR = Path(__file__).resolve().parent / "standalone"
if str(_STANDALONE_DIR) not in sys.path:
    sys.path.insert(0, str(_STANDALONE_DIR))

from mxfp8_grouped.quantize import (  # noqa: E402
    _round_up,
    build_sfb_fast,
    dual_quantize_activation,
    quantize_mxfp8,
)

# Every expert's row group is padded to a multiple of this in the kernel-facing
# buffer (the fused kernels' FIX_PAD_SIZE contract).
GROUP_PAD = 256
# gate/up column interleave block width of the fused SwiGLU epilogue.
_GATE_UP_BLK = 32

# Checkpoint name on the fwd-orientation quantized weight copies (+ swizzled
# SFs). Weights change once per train step, but under full remat the forward
# pipeline runs twice (primal + recompute), re-quantizing them. A remat policy
# that saves this name (GrugFp8Config.mxfp8_save_qweights) computes the fwd
# weight quantize + w13 interleave once per step; numerics are bit-identical
# either way. Costs ~fp8 copies of the local expert weights across the bwd.
MXFP8_QWEIGHT_CHECKPOINT_NAME = "grug_moe_mxfp8_qweights"

# Tiler winners from the MXFP8-004a g6/g8 ablations (jobs mxfp8-004a-g6..g11).
_SWIGLU_TILER = ((256, 256), (2, 1))
_DSWIGLU_TILER = ((256, 256), (2, 1))  # + vectorized_f32=True
_GEMM_TILER = ((256, 256), (2, 1))
_WGRAD_TILER = ((256, 256), (2, 2))

_PRODUCER_CHOICES = ("auto", "cute", "xla")


def _fused_kernels():
    # Local import: cutlass (CuTe DSL) only exists in GPU environments; this
    # module must stay importable by model.py everywhere.
    from mxfp8_fused import adapter as fused_adapter  # noqa: PLC0415

    return fused_adapter


def _require_sm100() -> None:
    dev = jax.devices()[0]
    cc = getattr(dev, "compute_capability", None)
    if cc is None or not str(cc).startswith("10"):
        raise RuntimeError(
            "GrugFp8Config recipe='mxfp8' requires a Blackwell (sm100) GPU for the fused "
            f"block-scaled grouped kernels; got {dev!r} with compute_capability={cc!r}. "
            "Use recipe='auto' (resolves per_tensor on Hopper) or fp8=None instead."
        )


# --------------------------------------------------------------------------- #
# 256-aligned padded dispatch layout
# --------------------------------------------------------------------------- #


class PaddedDispatch(NamedTuple):
    """Traced index maps between the raw dispatch buffer and the padded buffer."""

    offs: jax.Array  # (E,) int32 padded END offsets; offs[-1] == padded_rows
    padded_group_sizes: jax.Array  # (E,) int32, each a multiple of GROUP_PAD
    src_idx: jax.Array  # (Mp,) int32 raw source row per padded row (clipped)
    src_valid: jax.Array  # (Mp,) bool: padded row carries a real raw row
    unpad_idx: jax.Array  # (C,) int32 padded position of each raw row
    padded_rows: int  # static buffer size Mp


def padded_row_count(capacity: int, num_experts: int) -> int:
    """Static row count of the padded buffer for a raw buffer of ``capacity`` rows.

    Upper bound of ``sum_e round_up(g_e, GROUP_PAD)`` over all group splits
    with ``sum_e g_e == capacity``, rounded up so the buffer itself is
    GROUP_PAD-aligned (the last group absorbs the slack).
    """
    return _round_up(capacity + (GROUP_PAD - 1) * num_experts, GROUP_PAD)


def padded_dispatch_layout(group_sizes: jax.Array, *, capacity: int) -> PaddedDispatch:
    """Compute the padded-buffer layout for traced per-expert ``group_sizes``.

    ``group_sizes`` must sum to ``capacity`` (both EP backends guarantee this:
    their final group absorbs the dispatch buffer's own padding rows, which
    are zeros). The final PADDED group additionally absorbs the static slack
    so ``offs[-1] == padded_rows`` exactly, as the fused kernels require.
    """
    (e,) = group_sizes.shape
    mp = padded_row_count(capacity, e)
    g = group_sizes.astype(jnp.int32)
    gp = ((g + GROUP_PAD - 1) // GROUP_PAD) * GROUP_PAD
    ends = jnp.cumsum(gp)
    offs = jnp.concatenate([ends[:-1], jnp.array([mp], jnp.int32)])
    gp = jnp.concatenate([offs[:1], jnp.diff(offs)]) if e > 1 else offs
    pad_starts = offs - gp

    raw_ends = jnp.cumsum(g)
    raw_starts = raw_ends - g

    p = jnp.arange(mp, dtype=jnp.int32)
    p_expert = jnp.searchsorted(offs, p, side="right").astype(jnp.int32)
    j = p - pad_starts[p_expert]
    src_valid = j < g[p_expert]
    src_idx = jnp.clip(raw_starts[p_expert] + j, 0, capacity - 1).astype(jnp.int32)

    i = jnp.arange(capacity, dtype=jnp.int32)
    i_expert = jnp.minimum(jnp.searchsorted(raw_ends, i, side="right"), e - 1).astype(jnp.int32)
    unpad_idx = (pad_starts[i_expert] + (i - raw_starts[i_expert])).astype(jnp.int32)

    return PaddedDispatch(
        offs=offs,
        padded_group_sizes=gp,
        src_idx=src_idx,
        src_valid=src_valid,
        unpad_idx=unpad_idx,
        padded_rows=mp,
    )


def _pad_rows(t: jax.Array, layout: PaddedDispatch) -> jax.Array:
    """Re-pack raw dispatch rows into the padded buffer (pad rows zero-filled)."""
    return jnp.where(layout.src_valid[:, None], jnp.take(t, layout.src_idx, axis=0), 0)


def wgrad_block_perm(padded_group_sizes: jax.Array, *, rows: int, total_tokens: int) -> jax.Array:
    """Traced atom-block permutation for the 2Dx2D wgrad SF layout.

    Traced equivalent of ``sf_wgrad_col_layout(...)[1]`` for group sizes that
    are already multiples of GROUP_PAD (so no per-group column padding exists
    and the gather map is the identity): reorders the whole-matrix 128x4
    atom-block grid (row-block-major) into per-expert chunks
    ``[expert][row_block][col_block]``.
    """
    rb = _round_up(rows, 128) // 128
    cb_total = total_tokens // 128
    gcb = padded_group_sizes // 128  # column blocks per expert
    blocks = gcb * rb
    block_ends = jnp.cumsum(blocks)
    block_starts = block_ends - blocks
    cb_ends = jnp.cumsum(gcb)
    cb_starts = cb_ends - gcb

    p = jnp.arange(rb * cb_total, dtype=jnp.int32)
    e_idx = jnp.searchsorted(block_ends, p, side="right").astype(jnp.int32)
    q = p - block_starts[e_idx]
    gcb_safe = jnp.maximum(gcb[e_idx], 1)  # zero-block experts receive no p
    r = q // gcb_safe
    c_local = q % gcb_safe
    return (r * cb_total + cb_starts[e_idx] + c_local).astype(jnp.int32)


# --------------------------------------------------------------------------- #
# gate/up interleave (fused SwiGLU column contract)
# --------------------------------------------------------------------------- #


@functools.cache
def interleave_perm(n2: int) -> np.ndarray:
    """Interleaved column position -> concat column index (gate first half)."""
    assert n2 % (2 * _GATE_UP_BLK) == 0, f"N2={n2} must be a multiple of {2 * _GATE_UP_BLK}"
    f = n2 // 2
    j = np.arange(n2)
    fb, r = j // (2 * _GATE_UP_BLK), j % (2 * _GATE_UP_BLK)
    is_up = (r >= _GATE_UP_BLK).astype(np.int64)
    return (fb * _GATE_UP_BLK + (r % _GATE_UP_BLK) + is_up * f).astype(np.int32)


@functools.cache
def deinterleave_perm(n2: int) -> np.ndarray:
    """Concat column index -> interleaved column position (inverse of interleave_perm)."""
    perm = interleave_perm(n2)
    inv = np.empty(n2, dtype=np.int32)
    inv[perm] = np.arange(n2, dtype=np.int32)
    return inv


def _interleave_w13(w13: jax.Array) -> jax.Array:
    """(E, D, N2) concat gate/up -> (E, N2, D) fused-kernel B buffer (32-col interleave)."""
    n2 = w13.shape[2]
    return jnp.swapaxes(w13, 1, 2)[:, jnp.asarray(interleave_perm(n2)), :]


def _deinterleave_w13_grad(dw13i: jax.Array) -> jax.Array:
    """(E, D, N2) wgrad output over interleaved columns -> concat gate/up layout."""
    n2 = dw13i.shape[2]
    return dw13i[:, :, jnp.asarray(deinterleave_perm(n2))]


# --------------------------------------------------------------------------- #
# dual-orientation activation producers (CuTe kernel with XLA fallback)
# --------------------------------------------------------------------------- #


@functools.cache
def cute_producer_available() -> bool:
    """Compile-probe the MXFP8-002c CuTe dual quantizer in a SUBPROCESS.

    cutlass_call compile failures happen on an FFI thread and are fatal to the
    calling process: GB200 nodes are heterogeneous for libNVVM and the
    quantizer kernel dies with NVVM_ERROR_COMPILATION on most of them (the
    fused GEMM kernels are all-inline-PTX and compile everywhere). Probe once
    per process in a child; on failure the op uses the XLA producer.
    """
    code = (
        f"import sys; sys.path.insert(0, {str(_STANDALONE_DIR)!r})\n"
        "import jax, jax.numpy as jnp\n"
        "from mxfp8_grouped.quantize_cute import dual_quantize_mxfp8_cute\n"
        "x = jnp.ones((256, 128), jnp.bfloat16)\n"
        "jax.block_until_ready(dual_quantize_mxfp8_cute(x))\n"
        "print('CUTE_PROBE_OK')\n"
    )
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0 and "CUTE_PROBE_OK" in proc.stdout


def _resolve_producer(producer: str) -> str:
    if producer not in _PRODUCER_CHOICES:
        raise ValueError(f"producer must be one of {_PRODUCER_CHOICES}, got {producer!r}")
    if producer == "auto":
        return "cute" if cute_producer_available() else "xla"
    return producer


def _dual_quantize(t, row_idx, col_idx, block_perm, producer: str):
    if producer == "cute":
        from mxfp8_grouped.quantize_cute import dual_quantize_activation_cute  # noqa: PLC0415

        return dual_quantize_activation_cute(t, row_idx, col_idx, block_perm)
    return dual_quantize_activation(t, row_idx, col_idx, block_perm)


# --------------------------------------------------------------------------- #
# fwd / bwd pipelines
# --------------------------------------------------------------------------- #


def _identity_maps(layout: PaddedDispatch) -> tuple[jax.Array, jax.Array]:
    """SFA row map and wgrad column map: identities under GROUP_PAD alignment."""
    row_idx = jnp.arange(layout.padded_rows, dtype=jnp.int32)
    col_idx = jnp.arange(layout.padded_rows // 32, dtype=jnp.int32)
    return row_idx, col_idx


def _forward_pipeline(op: "MxFp8MoeMlpOp", x, w13, w2, group_sizes) -> dict:
    """Run the fwd legs; returns all intermediates (consumed by vjp fwd and tests)."""
    _require_sm100()
    if x.dtype != jnp.bfloat16:
        raise ValueError(f"mxfp8 expert MLP expects bf16 dispatch activations, got {x.dtype}")
    capacity, d = x.shape
    e, dk, n2 = w13.shape
    ew, f, dw = w2.shape
    if dk != d or dw != d or ew != e or n2 != 2 * f:
        raise ValueError(f"inconsistent MLP shapes: x={x.shape}, w13={w13.shape}, w2={w2.shape}")
    if d % 128 != 0 or f % 128 != 0:
        raise ValueError(f"mxfp8 kernels require 128-aligned dims; got hidden={d}, intermediate={f}")
    producer = _resolve_producer(op.producer)
    fused = _fused_kernels()

    layout = padded_dispatch_layout(group_sizes, capacity=capacity)
    row_idx, col_idx = _identity_maps(layout)
    perm_d = wgrad_block_perm(layout.padded_group_sizes, rows=d, total_tokens=layout.padded_rows)

    x_pad = _pad_rows(x, layout)
    x_q, x_sf, x_col, x_sfc = _dual_quantize(x_pad, row_idx, col_idx, perm_d, producer)

    w13i = _interleave_w13(w13)
    w13i_q, s13f = quantize_mxfp8(w13i)  # (E, N2, D) along D: fwd B operand
    sfb13f = build_sfb_fast(s13f)
    w2f_q, s2f = quantize_mxfp8(jnp.swapaxes(w2, 1, 2))  # (E, D, F) along F: fwd-w2 B operand
    sfb2f = build_sfb_fast(s2f)
    # Tag unconditionally; whether the copies are saved across the remat
    # recompute is the remat policy's decision (MXFP8_QWEIGHT_CHECKPOINT_NAME).
    w13i_q, sfb13f, w2f_q, sfb2f = tree_checkpoint_name((w13i_q, sfb13f, w2f_q, sfb2f), MXFP8_QWEIGHT_CHECKPOINT_NAME)

    c13, h, h_col, sfh_row, sfh_col = fused.mxfp8_fused_swiglu(
        x_q, x_sf, w13i_q, sfb13f, layout.offs, mma_tiler_mn=_SWIGLU_TILER[0], cluster_shape_mn=_SWIGLU_TILER[1]
    )
    y_pad = fused.mxfp8_fused_gemm(
        h, sfh_row, w2f_q, sfb2f, layout.offs, mma_tiler_mn=_GEMM_TILER[0], cluster_shape_mn=_GEMM_TILER[1]
    )
    y = jnp.take(y_pad, layout.unpad_idx, axis=0)

    return {
        "layout": layout,
        "producer": producer,
        "x_q": x_q,
        "x_sf": x_sf,
        "x_col": x_col,
        "x_sfc": x_sfc,
        "w13i_q": w13i_q,
        "sfb13f": sfb13f,
        "w2f_q": w2f_q,
        "sfb2f": sfb2f,
        "c13": c13,
        "h": h,
        "h_col": h_col,
        "sfh_row": sfh_row,
        "sfh_col": sfh_col,
        "y_pad": y_pad,
        "y": y,
    }


def _backward_pipeline(op: "MxFp8MoeMlpOp", res, g) -> dict:
    """Run the bwd legs; returns all intermediates (consumed by vjp bwd and tests)."""
    x_col, x_sfc, h_col, sfh_col, c13, w13, w2, group_sizes = res
    capacity, d = g.shape
    n2 = c13.shape[1]
    f = n2 // 2
    producer = _resolve_producer(op.producer)
    fused = _fused_kernels()

    layout = padded_dispatch_layout(group_sizes, capacity=capacity)
    row_idx, col_idx = _identity_maps(layout)
    perm_d = wgrad_block_perm(layout.padded_group_sizes, rows=d, total_tokens=layout.padded_rows)

    g_pad = _pad_rows(g.astype(jnp.bfloat16), layout)
    g_q, g_sf, g_col, g_sfc = _dual_quantize(g_pad, row_idx, col_idx, perm_d, producer)

    # Fresh dgrad-orientation weight copies, re-quantized from the live params
    # (never transpose-requantize the fwd fp8 copies).
    w13i = _interleave_w13(w13)
    w13dg_q, s13d = quantize_mxfp8(jnp.swapaxes(w13i, 1, 2))  # (E, D, N2) along N2: dgrad-w13 B
    sfb13d = build_sfb_fast(s13d)
    w2dg_q, s2d = quantize_mxfp8(w2)  # (E, F, D) along D: dgrad-w2 B (dswiglu leg)
    sfb2d = build_sfb_fast(s2d)

    dc, dc_col, sfdc_row, sfdc_col = fused.mxfp8_fused_dswiglu(
        g_q,
        g_sf,
        w2dg_q,
        sfb2d,
        c13,
        layout.offs,
        mma_tiler_mn=_DSWIGLU_TILER[0],
        cluster_shape_mn=_DSWIGLU_TILER[1],
        vectorized_f32=True,
    )
    dx_pad = fused.mxfp8_fused_gemm(
        dc, sfdc_row, w13dg_q, sfb13d, layout.offs, mma_tiler_mn=_GEMM_TILER[0], cluster_shape_mn=_GEMM_TILER[1]
    )
    dx = jnp.take(dx_pad, layout.unpad_idx, axis=0)

    dw13i = fused.mxfp8_fused_wgrad(
        x_col,
        x_sfc,
        dc_col,
        sfdc_col.reshape(n2, -1),
        layout.offs,
        mma_tiler_mn=_WGRAD_TILER[0],
        cluster_shape_mn=_WGRAD_TILER[1],
    )
    dw2 = fused.mxfp8_fused_wgrad(
        h_col,
        sfh_col.reshape(f, -1),
        g_col,
        g_sfc,
        layout.offs,
        mma_tiler_mn=_WGRAD_TILER[0],
        cluster_shape_mn=_WGRAD_TILER[1],
    )
    dw13 = _deinterleave_w13_grad(dw13i)

    return {
        "layout": layout,
        "producer": producer,
        "g_q": g_q,
        "g_sf": g_sf,
        "g_col": g_col,
        "g_sfc": g_sfc,
        "w13dg_q": w13dg_q,
        "sfb13d": sfb13d,
        "w2dg_q": w2dg_q,
        "sfb2d": sfb2d,
        "dc": dc,
        "dc_col": dc_col,
        "sfdc_row": sfdc_row,
        "sfdc_col": sfdc_col,
        "dx_pad": dx_pad,
        "dx": dx,
        "dw13i": dw13i,
        "dw13": dw13,
        "dw2": dw2,
    }


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _mxfp8_expert_mlp(op: "MxFp8MoeMlpOp", x, w13, w2, group_sizes):
    inter = _forward_pipeline(op, x, w13, w2, group_sizes)
    return inter["y"]


def _mxfp8_expert_mlp_fwd(op, x, w13, w2, group_sizes):
    inter = _forward_pipeline(op, x, w13, w2, group_sizes)
    res = (inter["x_col"], inter["x_sfc"], inter["h_col"], inter["sfh_col"], inter["c13"], w13, w2, group_sizes)
    return inter["y"], res


def _mxfp8_expert_mlp_bwd(op, res, g):
    inter = _backward_pipeline(op, res, g)
    _, _, _, _, _, w13, w2, _ = res
    dx = inter["dx"].astype(g.dtype)
    dw13 = inter["dw13"].astype(w13.dtype)
    dw2 = inter["dw2"].astype(w2.dtype)
    return dx, dw13, dw2, None


_mxfp8_expert_mlp.defvjp(_mxfp8_expert_mlp_fwd, _mxfp8_expert_mlp_bwd)


@dataclass(frozen=True)
class MxFp8MoeMlpOp:
    """Stateless MXFP8 expert-MLP op (implements ``MoeExpertMlpOp``).

    ``producer`` selects the dual-orientation activation quantizer: ``"auto"``
    probes the CuTe kernel once per process and falls back to XLA on nodes
    whose libNVVM cannot compile it; ``"cute"``/``"xla"`` force one path.
    """

    producer: Literal["auto", "cute", "xla"] = "auto"

    def __call__(self, x, w13, w2, group_sizes):
        return _mxfp8_expert_mlp(self, x, w13, w2, group_sizes)
