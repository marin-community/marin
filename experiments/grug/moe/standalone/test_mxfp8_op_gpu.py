# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-004c: GPU (GB200) validation of the MxFp8MoeMlpOp expert-MLP op.

Three phases, all with NON-UNIFORM routing (heavy skew + zero-token experts —
the padded-offset path the MXFP8-004a bench never exercised):

1. contract: jax.eval_shape of the op's fwd + vjp (shape/dtype contracts).
2. dequant (reduced M): every kernel leg of the op pipeline vs a reference on
   DEQUANTIZED operands (per-expert f32 dense matmuls + the validated
   XLA SwiGLU/dSwiGLU/quantizer references), gates <1e-3 GEMM legs / <2e-3
   wgrads — mirrors bench_mxfp8_fused's check_phase through the op's actual
   padding + traced SF layouts.
3. blackbox (full M=262144): op fwd+bwd via jax.vjp vs a bf16-input f32
   reference MLP — rel-Frobenius ~4e-2 class on output and all three grads
   (expected mxfp8 quantization noise), plus exact-zero wgrads for
   zero-token experts.

Usage: python test_mxfp8_op_gpu.py [--tokens 262144] [--check-tokens 65536]
         [--producer auto|cute|xla] [--phases contract,dequant,blackbox]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))

import jax
import jax.numpy as jnp
import numpy as np
from bench_mxfp8_fused import (
    dswiglu_interleaved,
    fused_quant_cols,
    fused_quant_rows,
    ragged_ref,
    rel_frob,
    swiglu_interleaved,
    unswizzle_sf_row,
    wgrad_ref,
)
from mxfp8_grouped.quantize import (
    SF_VEC_SIZE,
    dequantize_mxfp8,
    e8m0_to_f32,
    quantize_mxfp8,
    quantize_mxfp8_tokens,
)

from experiments.grug.moe.mxfp8 import (
    MxFp8MoeMlpOp,
    _backward_pipeline,
    _forward_pipeline,
    _interleave_w13,
    _pad_rows,
    padded_dispatch_layout,
)

D = 2560
F = 1280
N2 = 2 * F
E = 64


def make_group_sizes(m: int, seed: int = 0) -> list[int]:
    """Skewed non-uniform groups summing to m, with zero-token experts."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(E, 0.5))
    g = np.floor(w * m).astype(np.int64)
    g[3] = 0  # explicit zero-token experts
    g[57] = 0
    g[11] += m - g.sum()  # dump the remainder on one expert (heavy skew)
    assert g.sum() == m and (g >= 0).all()
    return [int(v) for v in g]


def make_inputs(m: int, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    kx, k13, k2, kg = jax.random.split(key, 4)
    x = jax.random.normal(kx, (m, D), jnp.bfloat16)
    w13 = (jax.random.normal(k13, (E, D, N2), jnp.bfloat16) / (D**0.5)).astype(jnp.bfloat16)
    w2 = (jax.random.normal(k2, (E, F, D), jnp.bfloat16) / (F**0.5)).astype(jnp.bfloat16)
    cot = jax.random.normal(kg, (m, D), jnp.bfloat16)
    groups = make_group_sizes(m, seed)
    gs = jnp.asarray(groups, jnp.int32)
    return x, w13, w2, cot, groups, gs


def ref_mlp_factory(groups):
    def ref_mlp(x, w13, w2):
        outs = []
        start = 0
        for ei, gsize in enumerate(groups):
            xs = x[start : start + gsize].astype(jnp.float32)
            h = xs @ w13[ei].astype(jnp.float32)
            gate, up = h[:, :F], h[:, F:]
            outs.append((jax.nn.silu(gate) * up) @ w2[ei].astype(jnp.float32))
            start += gsize
        return jnp.concatenate(outs, axis=0)

    return ref_mlp


def deq_rows(q, sf_raw):
    m, n = q.shape
    qb = q.astype(jnp.float32).reshape(m, n // SF_VEC_SIZE, SF_VEC_SIZE)
    return (qb * e8m0_to_f32(sf_raw)[..., None]).reshape(m, n)


def deq_cols(q, sf_raw):
    m, n = q.shape
    qb = q.astype(jnp.float32).reshape(m // SF_VEC_SIZE, SF_VEC_SIZE, n)
    return (qb * e8m0_to_f32(sf_raw)[:, None]).reshape(m, n)


def check_contract(op, results):
    print("\n== contract phase (eval_shape) ==", flush=True)
    m = 4096
    x, w13, w2, cot, _groups, gs = make_inputs(m)

    def fwd_and_grads(x_, w13_, w2_):
        y, vjp = jax.vjp(lambda a, b, c: op(a, b, c, gs), x_, w13_, w2_)
        return y, vjp(cot)

    shapes = jax.eval_shape(fwd_and_grads, x, w13, w2)
    y_s, (dx_s, dw13_s, dw2_s) = shapes
    assert y_s.shape == (m, D) and y_s.dtype == jnp.bfloat16, y_s
    assert dx_s.shape == x.shape and dx_s.dtype == x.dtype, dx_s
    assert dw13_s.shape == w13.shape and dw13_s.dtype == w13.dtype, dw13_s
    assert dw2_s.shape == w2.shape and dw2_s.dtype == w2.dtype, dw2_s
    print("  fwd+vjp shape/dtype contract OK", flush=True)
    results["contract"] = "ok"


def check_dequant(op, m: int, results):
    print(f"\n== dequant phase (M={m}) ==", flush=True)
    x, w13, w2, cot, _groups, gs = make_inputs(m)

    def _arrays_only(d):
        return {k: v for k, v in d.items() if k not in ("layout", "producer")}

    inter = jax.jit(lambda *a: _arrays_only(_forward_pipeline(op, *a)))(x, w13, w2, gs)
    res = (inter["x_col"], inter["x_sfc"], inter["h_col"], inter["sfh_col"], inter["c13"], w13, w2, gs)
    bint = jax.jit(lambda r, g: _arrays_only(_backward_pipeline(op, r, g)))(res, cot)
    jax.block_until_ready((inter["y"], bint["dx"]))

    layout = padded_dispatch_layout(gs, capacity=m)
    padded_groups = [int(v) for v in np.asarray(layout.padded_group_sizes)]
    mp = layout.padded_rows
    x_pad = _pad_rows(x, layout)
    g_pad = _pad_rows(cot, layout)
    w13i = _interleave_w13(w13)

    checks = {}

    def rec(name, value, gate):
        checks[name] = value
        status = "OK" if value < gate else f"FAIL (gate {gate})"
        print(f"  {name}: {value:.3e} {status}", flush=True)
        assert value < gate, f"{name}: {value} >= {gate}"

    # -- leg1: c13 vs dequantized x/w13i --
    x_deq = dequantize_mxfp8(*quantize_mxfp8(x_pad))
    w13i_deq = dequantize_mxfp8(*quantize_mxfp8(w13i))
    c_ref = ragged_ref(x_deq, jnp.swapaxes(w13i_deq, 1, 2), padded_groups)
    rec("c13_vs_dequant", rel_frob(inter["c13"], c_ref.astype(jnp.bfloat16)), 1e-3)

    # -- leg2: y_pad vs dequantized kernel-h/w2f --
    h_deq_kernel = deq_rows(inter["h"], unswizzle_sf_row(inter["sfh_row"], mp, F // 32))
    w2f_deq = dequantize_mxfp8(*quantize_mxfp8(jnp.swapaxes(w2, 1, 2)))
    y_ref = ragged_ref(h_deq_kernel, jnp.swapaxes(w2f_deq, 1, 2), padded_groups)
    rec("y_vs_dequant", rel_frob(inter["y_pad"], y_ref.astype(jnp.bfloat16)), 1e-3)

    # h quantize itself vs the XLA reference quantizer on the reference h.
    h_ref = swiglu_interleaved(c_ref)
    hq_ref, sh_ref = fused_quant_rows(h_ref)
    rec("h_deq_vs_ref_quant", rel_frob(h_deq_kernel, deq_rows(hq_ref, sh_ref)), 2e-3)

    # -- leg3+4: dc chain and dx vs dequantized operands --
    g_deq = dequantize_mxfp8(*quantize_mxfp8(g_pad))
    w2dg_deq = dequantize_mxfp8(*quantize_mxfp8(w2))
    dh_ref = ragged_ref(g_deq, jnp.swapaxes(w2dg_deq, 1, 2), padded_groups)
    dc_ref = dswiglu_interleaved(dh_ref, inter["c13"].astype(jnp.float32))
    dcq_ref, sdc_ref = fused_quant_rows(dc_ref)
    dc_deq_kernel = deq_rows(bint["dc"], unswizzle_sf_row(bint["sfdc_row"], mp, N2 // 32))
    rec("dc_deq_vs_ref", rel_frob(dc_deq_kernel, deq_rows(dcq_ref, sdc_ref)), 2e-3)

    w13dg_deq = dequantize_mxfp8(*quantize_mxfp8(jnp.swapaxes(w13i, 1, 2)))
    dx_ref = ragged_ref(dc_deq_kernel, jnp.swapaxes(w13dg_deq, 1, 2), padded_groups)
    rec("dx_vs_dequant", rel_frob(bint["dx_pad"], dx_ref.astype(jnp.bfloat16)), 1e-3)

    # -- leg5/6: wgrads vs dequantized column operands (reference scales) --
    _, sdcc_ref = fused_quant_cols(dc_ref)
    x_col_deq = deq_cols(inter["x_col"], quantize_mxfp8_tokens(x_pad)[1])
    dcc_kernel_deq = deq_cols(bint["dc_col"], sdcc_ref)
    dw13i_ref = wgrad_ref(x_col_deq, dcc_kernel_deq, padded_groups, D, N2)
    rec("dw13i_vs_dequant", rel_frob(bint["dw13i"], dw13i_ref.astype(jnp.bfloat16)), 2e-3)

    _, sc_ref = fused_quant_cols(h_ref)
    hc_kernel_deq = deq_cols(inter["h_col"], sc_ref)
    g_col_deq = deq_cols(bint["g_col"], quantize_mxfp8_tokens(g_pad)[1])
    dw2_ref = wgrad_ref(hc_kernel_deq, g_col_deq, padded_groups, F, D)
    rec("dw2_vs_dequant", rel_frob(bint["dw2"], dw2_ref.astype(jnp.bfloat16)), 2e-3)

    results["dequant"] = checks


def check_blackbox(op, m: int, results):
    print(f"\n== blackbox phase (M={m}) ==", flush=True)
    x, w13, w2, cot, groups, gs = make_inputs(m)

    @jax.jit
    def op_fwd_bwd(x_, w13_, w2_):
        y, vjp = jax.vjp(lambda a, b, c: op(a, b, c, gs), x_, w13_, w2_)
        return y, vjp(cot)

    y_op, (dx_op, dw13_op, dw2_op) = op_fwd_bwd(x, w13, w2)
    jax.block_until_ready(y_op)

    ref_mlp = ref_mlp_factory(groups)

    @jax.jit
    def ref_fwd_bwd(x_, w13_, w2_):
        y, vjp = jax.vjp(ref_mlp, x_, w13_, w2_)
        return y, vjp(cot.astype(jnp.float32))

    y_ref, (dx_ref, dw13_ref, dw2_ref) = ref_fwd_bwd(x, w13, w2)

    errs = {
        "out": rel_frob(y_op, y_ref),
        "dx": rel_frob(dx_op, dx_ref),
        "dw13": rel_frob(dw13_op, dw13_ref),
        "dw2": rel_frob(dw2_op, dw2_ref),
    }
    for name, err in errs.items():
        print(f"  {name} rel-frob vs bf16 ref: {err:.3e}", flush=True)
        assert err < 0.1, f"{name}: {err} (expected ~4e-2 mxfp8 class)"

    for ei, gsize in enumerate(groups):
        if gsize == 0:
            assert float(jnp.abs(dw13_op[ei]).max()) == 0.0, f"zero-token expert {ei} has nonzero dw13"
            assert float(jnp.abs(dw2_op[ei]).max()) == 0.0, f"zero-token expert {ei} has nonzero dw2"
    print("  zero-token expert wgrads exactly zero OK", flush=True)
    results["blackbox"] = errs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=262144)
    p.add_argument("--check-tokens", type=int, default=65536)
    p.add_argument("--producer", default="auto", choices=["auto", "cute", "xla"])
    p.add_argument("--phases", default="contract,dequant,blackbox")
    p.add_argument("--out", default="test_mxfp8_op_gpu.json")
    a = p.parse_args()
    phases = set(a.phases.split(","))

    dev = jax.devices()[0]
    print(f"device: {dev.device_kind} (cc {dev.compute_capability}), jax {jax.__version__}", flush=True)
    op = MxFp8MoeMlpOp(producer=a.producer)

    results = {"device": str(dev.device_kind), "producer": a.producer}
    if "contract" in phases:
        check_contract(op, results)
    if "dequant" in phases:
        check_dequant(op, a.check_tokens, results)
    if "blackbox" in phases:
        check_blackbox(op, a.tokens, results)

    print("\nRESULTS " + json.dumps(results, sort_keys=True), flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
