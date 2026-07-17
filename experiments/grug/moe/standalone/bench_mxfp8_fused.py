# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-004a: fused cudnn-frontend MXFP8 MoE layer-quad bench (GB200, sm100).

Benchmarks the vendored fused kernels (``mxfp8_fused/``) as the full grug
row-13 MoE layer fwd+bwd (D=2560, F=1280, E=64, gated: w13 = interleaved
gate/up (E, N2=2560, D)):

  leg1 fwd w13   swiglu_quant: x_q @ w13i -> c bf16 + h quantized BOTH ways
  leg2 fwd w2    quant kernel (bf16 out): h_q @ w2 -> y
  leg3 bwd       dswiglu_quant: g2_q @ w2^T + dSwiGLU(c) -> dc quantized BOTH ways
  leg4 dgrad w13 quant kernel (bf16 out): dc_q @ w13^T -> dx
  leg5 wgrad w13 wgrad kernel: x_col^T (x) dc_col -> dw13i
  leg6 wgrad w2  wgrad kernel: h_col^T (x) g2_col -> dw2

Remaining external producers (timed): dual-orientation quantize of x and g2
(both the XLA and the 002c CuTe producer), plus the per-step weight producers
(amortized). h and dc need NO producer -- that is the fusion win.

Correctness phase (reduced M): every fused output vs a reference built from
the validated MXFP8-002 pieces (per-expert dense matmul on dequantized
operands + XLA SwiGLU/dSwiGLU + the corrected round-up e8m0 quantizer),
including byte-level match stats on the quantized outputs and swizzled SFs.
The dSwiGLU reference formula itself is validated against jax.vjp.

Timing phase (full M): per-leg medians, producer costs, and the layer-quad
total vs the bf16 dense yardstick at the REAL layer shapes (note: the old
6.95 ms baseline from MXFP8-002b counted w13 at half width, K2560->N1280;
the real gated layer is K2560->N2560).

Usage: python bench_mxfp8_fused.py [--tokens 262144] [--check-tokens 32768]
         [--iters 50] [--phases check,time] [--out out.json]
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cutlass
import jax
import jax.numpy as jnp
import numpy as np
from mxfp8_fused.adapter import (
    mxfp8_fused_dswiglu,
    mxfp8_fused_gemm,
    mxfp8_fused_swiglu,
    mxfp8_fused_wgrad,
)
from mxfp8_grouped.adapter import (
    E4M3_MAX,
    SF_VEC_SIZE,
    build_sf_wgrad_fast,
    build_sfa_fast,
    build_sfb_fast,
    cast_to_e8m0_with_rounding_up,
    dual_quantize_activation,
    e8m0_to_f32,
    quantize_mxfp8,
    quantize_mxfp8_tokens,
    sf_wgrad_col_layout,
    sfa_row_gather_indices,
)
from mxfp8_grouped.quantize_cute import dual_quantize_activation_cute

E = 64
D = 2560  # model dim
F = 1280  # intermediate dim (post-SwiGLU)
N2 = 2 * F  # interleaved gate/up width
BLK = 32  # gate/up interleave block width
# The fused kernel's SF scale is amax * (1/448 rounded to f32) * norm_const;
# reproducing the multiply (not a divide by 448) keeps the reference bit-true.
RCP_LIMIT = np.float32(1.0 / 448.0)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _ceil_div(a, b) * b


# --------------------------------------------------------------------------- #
# interleave + reference math
# --------------------------------------------------------------------------- #


def gate_up_idx(n2: int) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.arange(n2).reshape(n2 // BLK, BLK)
    return blocks[0::2].reshape(-1), blocks[1::2].reshape(-1)  # gate = even, up = odd


def interleave_w13(w_gate, w_up):
    """(E, D, F) x2 -> (E, N2, D): B-operand layout, gate/up rows interleaved 32-wide."""
    e, d, f = w_gate.shape
    gate_t = jnp.swapaxes(w_gate, 1, 2).reshape(e, f // BLK, BLK, d)
    up_t = jnp.swapaxes(w_up, 1, 2).reshape(e, f // BLK, BLK, d)
    return jnp.stack((gate_t, up_t), axis=2).reshape(e, 2 * f, d)


def swiglu_interleaved(c):
    """Reference SwiGLU on the interleaved pre-activation c[..., N2] -> [..., N2//2]."""
    gate_idx, up_idx = gate_up_idx(c.shape[-1])
    gate = c[..., gate_idx]
    up = c[..., up_idx]
    return up * gate * jax.nn.sigmoid(gate)


def dswiglu_interleaved(dh, c):
    """Reference dSwiGLU: dh[..., N2//2] cotangent, c[..., N2] pre-activation -> dc[..., N2]."""
    n2 = c.shape[-1]
    gate_idx, up_idx = gate_up_idx(n2)
    gate = c[..., gate_idx]
    up = c[..., up_idx]
    sig = jax.nn.sigmoid(gate)
    dgate = dh * up * sig * (1.0 + gate * (1.0 - sig))
    dup = dh * gate * sig
    dc = jnp.zeros(c.shape, dh.dtype)
    dc = dc.at[..., gate_idx].set(dgate)
    return dc.at[..., up_idx].set(dup)


def validate_dswiglu_formula():
    """Cross-check dswiglu_interleaved against jax.vjp of swiglu_interleaved."""
    key = jax.random.PRNGKey(7)
    c = jax.random.normal(key, (64, 128), jnp.float32)
    dh = jax.random.normal(jax.random.PRNGKey(8), (64, 64), jnp.float32)
    _, vjp = jax.vjp(swiglu_interleaved, c)
    (dc_ad,) = vjp(dh)
    dc_ref = dswiglu_interleaved(dh, c)
    err = float(jnp.max(jnp.abs(dc_ad - dc_ref)))
    assert err < 1e-5, f"dSwiGLU reference formula disagrees with jax.vjp: max abs {err}"
    print(f"dswiglu formula vs jax.vjp: max abs diff {err:.2e} OK", flush=True)


def fused_quant_rows(v):
    """Reference of the fused kernels' row-orientation quantize (along the last axis)."""
    m, n = v.shape
    vb = v.astype(jnp.float32).reshape(m, n // SF_VEC_SIZE, SF_VEC_SIZE)
    amax = jnp.max(jnp.abs(vb), axis=-1)
    sf = cast_to_e8m0_with_rounding_up(amax * RCP_LIMIT)
    q = jnp.clip(vb / e8m0_to_f32(sf)[..., None], -E4M3_MAX, E4M3_MAX).astype(jnp.float8_e4m3fn)
    return q.reshape(m, n), sf


def fused_quant_cols(v):
    """Reference of the fused kernels' column-orientation quantize (along tokens)."""
    m, n = v.shape
    vb = v.astype(jnp.float32).reshape(m // SF_VEC_SIZE, SF_VEC_SIZE, n)
    amax = jnp.max(jnp.abs(vb), axis=1)
    sf = cast_to_e8m0_with_rounding_up(amax * RCP_LIMIT)
    q = jnp.clip(vb / e8m0_to_f32(sf)[:, None], -E4M3_MAX, E4M3_MAX).astype(jnp.float8_e4m3fn)
    return q.reshape(m, n), sf


def unswizzle_sf_row(flat, rows: int, cols: int):
    """Inverse of the whole-matrix 32x4x4 swizzle -> raw (rows, cols) uint8 scales."""
    rb, cb = _round_up(rows, 128) // 128, _round_up(cols, 4) // 4
    z = jax.lax.bitcast_convert_type(flat.reshape(-1), jnp.uint8)
    z = z.reshape(rb * cb, 32, 4, 4).swapaxes(1, 2).reshape(rb, cb, 128, 4)
    return z.transpose(0, 2, 1, 3).reshape(rb * 128, cb * 4)[:rows, :cols]


def deq_rows(q, sf_raw):
    m, n = q.shape
    qb = q.astype(jnp.float32).reshape(m, n // SF_VEC_SIZE, SF_VEC_SIZE)
    return (qb * e8m0_to_f32(sf_raw)[..., None]).reshape(m, n)


def ragged_ref(a32, b32_ekn, group_sizes):
    """Per-expert f32 dense matmul reference (2Dx3D)."""
    outs = []
    start = 0
    for ei, g in enumerate(group_sizes):
        if g > 0:
            outs.append(a32[start : start + g] @ b32_ekn[ei])
        start += g
    return jnp.concatenate(outs, axis=0)


def wgrad_ref(a32, b32, group_sizes, k, n):
    outs = []
    start = 0
    for g in group_sizes:
        if g > 0:
            outs.append(a32[start : start + g].T @ b32[start : start + g])
        else:
            outs.append(jnp.zeros((k, n), jnp.float32))
        start += g
    return jnp.stack(outs)


def rel_frob(a, ref):
    a = jnp.asarray(a, jnp.float32)
    ref = jnp.asarray(ref, jnp.float32)
    return float(jnp.linalg.norm(a - ref) / jnp.linalg.norm(ref))


def byte_match(a, b) -> float:
    au = jax.lax.bitcast_convert_type(a.reshape(-1), jnp.uint8)
    bu = jax.lax.bitcast_convert_type(b.reshape(-1), jnp.uint8)
    return float(jnp.mean((au == bu).astype(jnp.float32)))


def timed(fn, args, iters, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# --------------------------------------------------------------------------- #
# input construction
# --------------------------------------------------------------------------- #


def make_inputs(m: int, seed: int = 0):
    """Random layer inputs + all quantized operands and swizzle index maps."""
    key = jax.random.PRNGKey(seed)
    kx, kg1, kg3, kw2, kg = jax.random.split(key, 5)
    x = jax.random.normal(kx, (m, D), jnp.bfloat16)
    w_gate = jax.random.normal(kg1, (E, D, F), jnp.bfloat16) / (D**0.5)
    w_up = jax.random.normal(kg3, (E, D, F), jnp.bfloat16) / (D**0.5)
    w2 = jax.random.normal(kw2, (E, F, D), jnp.bfloat16) / (F**0.5)
    g2 = jax.random.normal(kg, (m, D), jnp.bfloat16)

    groups = [m // E] * E
    offs = jnp.cumsum(jnp.asarray(groups, jnp.int32)).astype(jnp.int32)

    row_idx = jnp.asarray(sfa_row_gather_indices(groups))
    col_idx_np, perm_d_np = sf_wgrad_col_layout(groups, D)
    _, perm_f_np = sf_wgrad_col_layout(groups, F)
    _, perm_n2_np = sf_wgrad_col_layout(groups, N2)
    col_idx = jnp.asarray(col_idx_np)
    perms = {"d": jnp.asarray(perm_d_np), "f": jnp.asarray(perm_f_np), "n2": jnp.asarray(perm_n2_np)}

    w13i = interleave_w13(w_gate, w_up)  # (E, N2, D)

    def weights_fn(w13i_, w2_):
        w13i_q, s13f = quantize_mxfp8(w13i_)  # along D (fwd B operand)
        w13dg_q, s13d = quantize_mxfp8(jnp.swapaxes(w13i_, 1, 2))  # (E, D, N2) along N2 (dgrad B)
        w2f_q, s2f = quantize_mxfp8(jnp.swapaxes(w2_, 1, 2))  # (E, D, F) along F (fwd-w2 B)
        w2dg_q, s2d = quantize_mxfp8(w2_)  # (E, F, D) along D (dswiglu B)
        return (
            w13i_q,
            build_sfb_fast(s13f),
            w13dg_q,
            build_sfb_fast(s13d),
            w2f_q,
            build_sfb_fast(s2f),
            w2dg_q,
            build_sfb_fast(s2d),
        )

    weights = jax.jit(weights_fn)(w13i, w2)
    acts_fn = jax.jit(lambda t: dual_quantize_activation(t, row_idx, col_idx, perms["d"]))
    x_ops = acts_fn(x)  # (q_row, sf_row, q_col, sf_col)
    g_ops = acts_fn(g2)
    jax.block_until_ready((weights, x_ops, g_ops))
    return {
        "m": m,
        "groups": groups,
        "offs": offs,
        "x": x,
        "g2": g2,
        "w13i": w13i,
        "w2": w2,
        "weights": weights,
        "x_ops": x_ops,
        "g_ops": g_ops,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "perms": perms,
        "acts_fn": acts_fn,
        "weights_fn": weights_fn,
    }


def run_legs(inp, tilers):
    """Run the six fused legs once, returning all outputs (jitted callables kept)."""
    offs = inp["offs"]
    w13i_q, sfb13f, w13dg_q, sfb13d, w2f_q, sfb2f, w2dg_q, sfb2d = inp["weights"]
    x_q, x_sf, x_col, x_sfc = inp["x_ops"]
    g_q, g_sf, g_col, g_sfc = inp["g_ops"]
    m = inp["m"]

    swiglu_kw = dict(mma_tiler_mn=tilers["swiglu"][0], cluster_shape_mn=tilers["swiglu"][1])
    gemm_kw = dict(mma_tiler_mn=tilers["gemm"][0], cluster_shape_mn=tilers["gemm"][1])
    dsw_kw = dict(mma_tiler_mn=tilers["dswiglu"][0], cluster_shape_mn=tilers["dswiglu"][1])
    wg_kw = dict(mma_tiler_mn=tilers["wgrad"][0], cluster_shape_mn=tilers["wgrad"][1])

    legs = {}
    legs["swiglu"] = jax.jit(lambda a, asf, b, bsf, o: mxfp8_fused_swiglu(a, asf, b, bsf, o, **swiglu_kw))
    legs["gemm_w2"] = jax.jit(lambda a, asf, b, bsf, o: mxfp8_fused_gemm(a, asf, b, bsf, o, **gemm_kw))
    legs["dswiglu"] = jax.jit(lambda a, asf, b, bsf, c, o: mxfp8_fused_dswiglu(a, asf, b, bsf, c, o, **dsw_kw))
    legs["gemm_w13"] = jax.jit(lambda a, asf, b, bsf, o: mxfp8_fused_gemm(a, asf, b, bsf, o, **gemm_kw))
    legs["wgrad"] = jax.jit(lambda a, asf, b, bsf, o: mxfp8_fused_wgrad(a, asf, b, bsf, o, **wg_kw))

    c13, h, h_col, sfh_row, sfh_col = legs["swiglu"](x_q, x_sf, w13i_q, sfb13f, offs)
    y = legs["gemm_w2"](h, sfh_row, w2f_q, sfb2f, offs)
    dc, dc_col, sfdc_row, sfdc_col = legs["dswiglu"](g_q, g_sf, w2dg_q, sfb2d, c13, offs)
    dx = legs["gemm_w13"](dc, sfdc_row, w13dg_q, sfb13d, offs)
    sfh_col2d = sfh_col.reshape(F, -1)
    sfdc_col2d = sfdc_col.reshape(N2, -1)
    dw13 = legs["wgrad"](x_col, x_sfc, dc_col, sfdc_col2d, offs)
    dw2 = legs["wgrad"](h_col, sfh_col2d, g_col, g_sfc, offs)
    outs = {
        "c13": c13,
        "h": h,
        "h_col": h_col,
        "sfh_row": sfh_row,
        "sfh_col": sfh_col,
        "y": y,
        "dc": dc,
        "dc_col": dc_col,
        "sfdc_row": sfdc_row,
        "sfdc_col": sfdc_col,
        "dx": dx,
        "dw13": dw13,
        "dw2": dw2,
    }
    jax.block_until_ready(outs)
    args = {
        "swiglu": (x_q, x_sf, w13i_q, sfb13f, offs),
        "gemm_w2": (h, sfh_row, w2f_q, sfb2f, offs),
        "dswiglu": (g_q, g_sf, w2dg_q, sfb2d, c13, offs),
        "gemm_w13": (dc, sfdc_row, w13dg_q, sfb13d, offs),
        "wgrad_w13": (x_col, x_sfc, dc_col, sfdc_col2d, offs),
        "wgrad_w2": (h_col, sfh_col2d, g_col, g_sfc, offs),
    }
    return legs, args, outs


# --------------------------------------------------------------------------- #
# correctness
# --------------------------------------------------------------------------- #


def check_phase(m: int, tilers, results: dict):
    print(f"\n== correctness phase (M={m}) ==", flush=True)
    validate_dswiglu_formula()
    inp = make_inputs(m)
    legs, args, outs = run_legs(inp, tilers)
    groups = inp["groups"]
    checks = {}

    def rec(name, **kv):
        checks[name] = kv
        parts = "  ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items())
        print(f"  {name}: {parts}", flush=True)

    w13i_q, sfb13f, w13dg_q, sfb13d, w2f_q, sfb2f, w2dg_q, sfb2d = inp["weights"]
    x_q, x_sf, x_col, x_sfc = inp["x_ops"]
    g_q, g_sf, g_col, g_sfc = inp["g_ops"]

    # Dequantized operands (ground truth = what the hardware actually multiplies).
    x_deq = deq_rows(x_q, quantize_mxfp8(inp["x"])[1])
    w13_deq = deq_rows(w13i_q.reshape(E * N2, D), quantize_mxfp8(inp["w13i"])[1].reshape(E * N2, -1)).reshape(E, N2, D)

    # ---- leg1: fused swiglu ----
    c_ref = ragged_ref(x_deq, jnp.swapaxes(w13_deq, 1, 2), groups)  # (m, N2) f32
    err_c = rel_frob(outs["c13"], c_ref.astype(jnp.bfloat16))
    h_ref = swiglu_interleaved(c_ref)  # f32
    hq_ref, sh_ref = fused_quant_rows(h_ref)
    hc_ref, sc_ref = fused_quant_cols(h_ref)
    sfh_row_ref = build_sfa_fast(sh_ref, inp["row_idx"])
    sfh_col_ref = build_sf_wgrad_fast(sc_ref.T, inp["col_idx"], inp["perms"]["f"])
    rec(
        "leg1_swiglu",
        c_relfrob=err_c,
        h_deq_relfrob=rel_frob(deq_rows(outs["h"], unswizzle_sf_row(outs["sfh_row"], m, F // 32)), deq_rows(hq_ref, sh_ref)),
        h_bytes=byte_match(outs["h"], hq_ref),
        hcol_bytes=byte_match(outs["h_col"], hc_ref),
        sfrow_bytes=byte_match(outs["sfh_row"], sfh_row_ref),
        sfcol_bytes=byte_match(outs["sfh_col"], sfh_col_ref),
    )
    assert err_c < 1e-3, f"leg1 c rel-frob {err_c}"

    # ---- leg2: fwd w2 (operands = the kernel's own h + sfh) ----
    h_deq_kernel = deq_rows(outs["h"], unswizzle_sf_row(outs["sfh_row"], m, F // 32))
    w2f_deq = deq_rows(w2f_q.reshape(E * D, F), quantize_mxfp8(jnp.swapaxes(inp["w2"], 1, 2))[1].reshape(E * D, -1)).reshape(
        E, D, F
    )
    y_ref = ragged_ref(h_deq_kernel, jnp.swapaxes(w2f_deq, 1, 2), groups)
    err_y = rel_frob(outs["y"], y_ref.astype(jnp.bfloat16))
    rec("leg2_gemm_w2", y_relfrob=err_y)
    assert err_y < 1e-3, f"leg2 y rel-frob {err_y}"

    # ---- leg3: fused dswiglu (c = kernel's own c13, bf16) ----
    g_deq = deq_rows(g_q, quantize_mxfp8(inp["g2"])[1])
    w2dg_deq = deq_rows(w2dg_q.reshape(E * F, D), quantize_mxfp8(inp["w2"])[1].reshape(E * F, -1)).reshape(E, F, D)
    dh_ref = ragged_ref(g_deq, jnp.swapaxes(w2dg_deq, 1, 2), groups)  # (m, F) f32
    dc_ref = dswiglu_interleaved(dh_ref, outs["c13"].astype(jnp.float32))
    dcq_ref, sdc_ref = fused_quant_rows(dc_ref)
    dcc_ref, sdcc_ref = fused_quant_cols(dc_ref)
    sfdc_row_ref = build_sfa_fast(sdc_ref, inp["row_idx"])
    sfdc_col_ref = build_sf_wgrad_fast(sdcc_ref.T, inp["col_idx"], inp["perms"]["n2"])
    dc_deq_kernel = deq_rows(outs["dc"], unswizzle_sf_row(outs["sfdc_row"], m, N2 // 32))
    rec(
        "leg3_dswiglu",
        dc_deq_relfrob=rel_frob(dc_deq_kernel, deq_rows(dcq_ref, sdc_ref)),
        dc_bytes=byte_match(outs["dc"], dcq_ref),
        dccol_bytes=byte_match(outs["dc_col"], dcc_ref),
        sfrow_bytes=byte_match(outs["sfdc_row"], sfdc_row_ref),
        sfcol_bytes=byte_match(outs["sfdc_col"], sfdc_col_ref),
    )

    # ---- leg4: dgrad w13 (operands = kernel's own dc + sfdc) ----
    w13dg_deq = deq_rows(
        w13dg_q.reshape(E * D, N2), quantize_mxfp8(jnp.swapaxes(inp["w13i"], 1, 2))[1].reshape(E * D, -1)
    ).reshape(E, D, N2)
    dx_ref = ragged_ref(dc_deq_kernel, jnp.swapaxes(w13dg_deq, 1, 2), groups)
    err_dx = rel_frob(outs["dx"], dx_ref.astype(jnp.bfloat16))
    rec("leg4_gemm_w13", dx_relfrob=err_dx)
    assert err_dx < 1e-3, f"leg4 dx rel-frob {err_dx}"

    # ---- leg5/6: wgrads (operands = the actual col-quantized tensors fed in) ----
    def deq_cols(q, sf_raw):
        mm, nn = q.shape
        qb = q.astype(jnp.float32).reshape(mm // SF_VEC_SIZE, SF_VEC_SIZE, nn)
        return (qb * e8m0_to_f32(sf_raw)[:, None]).reshape(mm, nn)

    x_col_deq = deq_cols(x_col, quantize_mxfp8_tokens(inp["x"])[1])
    dcc_kernel_deq = deq_cols(outs["dc_col"], sdcc_ref)  # kernel bytes, reference scales (validated above)
    dw13_ref = wgrad_ref(x_col_deq, dcc_kernel_deq, groups, D, N2)
    err_dw13 = rel_frob(outs["dw13"], dw13_ref.astype(jnp.bfloat16))
    hc_kernel_deq = deq_cols(outs["h_col"], sc_ref)
    g_col_deq = deq_cols(g_col, quantize_mxfp8_tokens(inp["g2"])[1])
    dw2_ref = wgrad_ref(hc_kernel_deq, g_col_deq, groups, F, D)
    err_dw2 = rel_frob(outs["dw2"], dw2_ref.astype(jnp.bfloat16))
    rec("leg5_wgrad_w13", dw13_relfrob=err_dw13)
    rec("leg6_wgrad_w2", dw2_relfrob=err_dw2)
    assert err_dw13 < 2e-3, f"leg5 dw13 rel-frob {err_dw13}"
    assert err_dw2 < 2e-3, f"leg6 dw2 rel-frob {err_dw2}"

    results["checks"] = checks
    del inp, legs, args, outs
    print("correctness phase done", flush=True)


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #


def time_phase(m: int, tilers, iters: int, warmup: int, results: dict):
    print(f"\n== timing phase (M={m}) ==", flush=True)
    inp = make_inputs(m)
    legs, args, outs = run_legs(inp, tilers)
    del outs
    flops = {
        "swiglu": 2 * m * D * N2,
        "gemm_w2": 2 * m * F * D,
        "dswiglu": 2 * m * D * F,
        "gemm_w13": 2 * m * N2 * D,
        "wgrad_w13": 2 * m * D * N2,
        "wgrad_w2": 2 * m * F * D,
    }
    leg_fn = {
        "swiglu": legs["swiglu"],
        "gemm_w2": legs["gemm_w2"],
        "dswiglu": legs["dswiglu"],
        "gemm_w13": legs["gemm_w13"],
        "wgrad_w13": legs["wgrad"],
        "wgrad_w2": legs["wgrad"],
    }
    times = {}
    for name in ("swiglu", "gemm_w2", "dswiglu", "gemm_w13", "wgrad_w13", "wgrad_w2"):
        t = timed(leg_fn[name], args[name], iters, warmup)
        times[name] = {"ms": t * 1e3, "tfs": flops[name] / t / 1e12}
        print(f"  {name:10s} {t * 1e3:8.3f} ms  {flops[name] / t / 1e12:7.1f} TF/s", flush=True)
    gemm_total = sum(v["ms"] for v in times.values())

    # ---- producers ----
    prod = {}
    prod["x_dual_xla_ms"] = timed(inp["acts_fn"], (inp["x"],), iters, warmup) * 1e3
    cute_fn = jax.jit(lambda t: dual_quantize_activation_cute(t, inp["row_idx"], inp["col_idx"], inp["perms"]["d"]))
    cute_ok = True
    try:
        jax.block_until_ready(cute_fn(inp["x"]))
    except Exception as exc:  # noqa: BLE001 - report and fall back to the XLA producer
        cute_ok = False
        print(f"  cute producer unavailable: {type(exc).__name__}: {exc}", flush=True)
    if cute_ok:
        prod["x_dual_cute_ms"] = timed(cute_fn, (inp["x"],), iters, warmup) * 1e3
    prod["weights_ms"] = timed(jax.jit(inp["weights_fn"]), (inp["w13i"], inp["w2"]), iters, warmup) * 1e3
    per_mb_key = "x_dual_cute_ms" if cute_ok else "x_dual_xla_ms"
    prod["per_microbatch_ms"] = 2 * prod[per_mb_key]  # x and g2, identical shape (M, D)
    print(
        f"  producers: x dual XLA {prod['x_dual_xla_ms']:.3f} ms"
        + (f", CuTe {prod['x_dual_cute_ms']:.3f} ms" if cute_ok else "")
        + f"; weights (amortized) {prod['weights_ms']:.3f} ms",
        flush=True,
    )

    # ---- bf16 dense yardsticks at the REAL shapes ----
    xb = inp["x"]
    wb_big = inp["w13i"][0].T  # (D, N2)
    wb_small = inp["w2"][0]  # (F, D)
    hb = jax.random.normal(jax.random.PRNGKey(3), (m, F), jnp.bfloat16)
    dense = jax.jit(lambda a, b: jnp.einsum("mk,kn->mn", a, b, preferred_element_type=jnp.bfloat16))
    t_big = timed(dense, (xb, wb_big), iters, warmup)
    t_small = timed(dense, (hb, wb_small), iters, warmup)
    bf16 = {
        "w13_dense_ms": t_big * 1e3,
        "w13_dense_tfs": 2 * m * D * N2 / t_big / 1e12,
        "w2_dense_ms": t_small * 1e3,
        "w2_dense_tfs": 2 * m * F * D / t_small / 1e12,
        "quad_ms": 3 * (t_big + t_small) * 1e3,
    }
    print(
        f"  bf16 dense: w13-shape {bf16['w13_dense_ms']:.3f} ms ({bf16['w13_dense_tfs']:.0f} TF/s), "
        f"w2-shape {bf16['w2_dense_ms']:.3f} ms ({bf16['w2_dense_tfs']:.0f} TF/s); 6-GEMM quad {bf16['quad_ms']:.3f} ms",
        flush=True,
    )

    quad = {
        "gemm_ms": gemm_total,
        "producer_ms": prod["per_microbatch_ms"],
        "total_ms": gemm_total + prod["per_microbatch_ms"],
        "bf16_quad_ms": bf16["quad_ms"],
    }
    quad["speedup_gemm_only"] = bf16["quad_ms"] / quad["gemm_ms"]
    quad["speedup_total"] = bf16["quad_ms"] / quad["total_ms"]
    quad["speedup_with_weight_producers"] = bf16["quad_ms"] / (quad["total_ms"] + prod["weights_ms"])
    print("\n== fused layer quad ==", flush=True)
    print(f"  6 fused legs           {quad['gemm_ms']:8.3f} ms")
    print(f"  + producers (x,g2 dual){quad['producer_ms']:8.3f} ms  -> total {quad['total_ms']:.3f} ms")
    print(f"  bf16 6-GEMM yardstick  {quad['bf16_quad_ms']:8.3f} ms")
    print(
        f"  speedup: gemm-only {quad['speedup_gemm_only']:.3f}x | total {quad['speedup_total']:.3f}x | "
        f"incl. per-step weight producers {quad['speedup_with_weight_producers']:.3f}x",
        flush=True,
    )
    results["times"] = times
    results["producers"] = prod
    results["bf16"] = bf16
    results["quad"] = quad


def parse_tiler(s: str):
    a, b, c, d = (int(v) for v in s.split(","))
    return ((a, b), (c, d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=262144)
    p.add_argument("--check-tokens", type=int, default=32768)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--phases", default="check,time")
    p.add_argument("--swiglu-tiler", default="256,256,2,1", help="mma_m,mma_n,cluster_m,cluster_n")
    p.add_argument("--dswiglu-tiler", default="256,256,2,1")
    p.add_argument("--gemm-tiler", default="128,256,1,1")
    p.add_argument("--wgrad-tiler", default="128,256,1,1")
    p.add_argument("--out", default="bench_mxfp8_fused.json")
    a = p.parse_args()
    phases = set(a.phases.split(","))
    tilers = {
        "swiglu": parse_tiler(a.swiglu_tiler),
        "dswiglu": parse_tiler(a.dswiglu_tiler),
        "gemm": parse_tiler(a.gemm_tiler),
        "wgrad": parse_tiler(a.wgrad_tiler),
    }

    dev = jax.devices()[0]
    print(f"device: {dev.device_kind} (cc {dev.compute_capability}), jax {jax.__version__}, cutlass {cutlass.__version__}")
    results = {
        "device": str(dev.device_kind),
        "jax": jax.__version__,
        "cutlass": cutlass.__version__,
        "tokens": a.tokens,
        "check_tokens": a.check_tokens,
        "experts": E,
        "tilers": {k: [list(v[0]), list(v[1])] for k, v in tilers.items()},
    }
    if "check" in phases:
        check_phase(a.check_tokens, tilers, results)
    if "time" in phases:
        time_phase(a.tokens, tilers, results)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
