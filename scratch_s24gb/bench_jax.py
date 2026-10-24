# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One GB200: our production expert kernels (QuACK grouped GEMMs, bf16 and MXFP8; Pallas MXFP8 quantizers) at the
lc1 GB200 recipe shapes, kernel-only time from the XLA GPU profile (``get_profiled_instructions_proto``)."""

import json
import os
import struct
import sys
import tempfile
import time
import traceback

import jax
import jax.experimental.profiler
import jax.numpy as jnp
from levanter.grug._moe.quack_moe_cute import (
    quack_act_grouped_gemm,
    quack_dact_grouped_gemm,
    quack_grouped_gemm,
    quack_grouped_wgrad,
)
from levanter.grug._moe.quack_mxfp8 import (
    _MN_MAJOR,
    _dense_gemm,
    _quantize_cols,
    mx_weights,
    mxfp8_drelu2_grouped_gemm,
    mxfp8_grouped_gemm,
    mxfp8_relu2_grouped_gemm,
    padded_scale_rows,
)
from levanter.grug._moe.sonic_cute import _QUACK_GATED_KW, _QUACK_GROUPED_KW, _QUACK_WGRAD_KW
from levanter.grug.mxfp8 import mx_quantize_rows
from levanter.grug.mxfp8_pallas import (
    mx_quantize_cols_pallas,
    mx_quantize_rows_cols_pallas,
    mx_quantize_rows_pallas,
)

E = 6
L, F, H = 3072, 4608, 6144
ITERS = 10


def emit(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


def _varint(buf, i):
    shift = val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return val, i


def _parse_costs(buf):
    """ProfiledInstructionsProto -> {name: cost_us} (field 1 = repeated InstructionCost{1: name, 2: cost_us})."""
    out, i = {}, 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wt = key >> 3, key & 7
        if wt == 2:
            n, i = _varint(buf, i)
            sub = buf[i : i + n]
            i += n
            if field == 1:
                name, cost, j = None, None, 0
                while j < len(sub):
                    k2, j = _varint(sub, j)
                    f2, w2 = k2 >> 3, k2 & 7
                    if w2 == 2:
                        m, j = _varint(sub, j)
                        if f2 == 1:
                            name = sub[j : j + m].decode()
                        j += m
                    elif w2 == 1:
                        if f2 == 2:
                            cost = struct.unpack("<d", sub[j : j + 8])[0]
                        j += 8
                    elif w2 == 0:
                        _, j = _varint(sub, j)
                    elif w2 == 5:
                        j += 4
                if name is not None:
                    out[name] = cost
        elif wt == 0:
            _, i = _varint(buf, i)
        elif wt == 1:
            i += 8
        elif wt == 5:
            i += 4
    return out


def kernel_us(fn, *args):
    """Device time per call: sum of the profiled instruction costs (each the mean over ITERS calls)."""
    f = jax.jit(fn)
    jax.block_until_ready(f(*args))
    jax.block_until_ready(f(*args))
    t0 = time.perf_counter()
    for _ in range(ITERS):
        r = f(*args)
    jax.block_until_ready(r)
    wall = (time.perf_counter() - t0) / ITERS * 1e6
    d = tempfile.mkdtemp()
    jax.profiler.start_trace(d)
    for _ in range(ITERS):
        r = f(*args)
    jax.block_until_ready(r)
    jax.profiler.stop_trace()
    runs = []
    for root, _dirs, files in os.walk(d):
        if any(x.endswith(".xplane.pb") for x in files):
            runs.append(root)
    costs = _parse_costs(jax.experimental.profiler.get_profiled_instructions_proto(runs[0])) if runs else {}
    total = sum(v for v in costs.values() if v)
    top = sorted(costs.items(), key=lambda kv: -(kv[1] or 0))[:3]
    return total, wall, top


def run(name, fn, *args, **meta):
    try:
        us, wall, top = kernel_us(fn, *args)
        emit(name=name, us=us, wall_us=wall, top=top, **meta)
    except Exception:
        emit(name=name, error=traceback.format_exc()[-1500:], **meta)


def bench(m_e, out_dim=L, roles=("up", "down", "dpost", "dX", "dWu", "dWd")):
    rows = E * m_e
    meta = dict(m_e=m_e, rows=rows, out_dim=out_dim)
    k = jax.random.split(jax.random.key(0), 6)
    bf = jnp.bfloat16
    x = jax.random.normal(k[0], (rows, L), bf)
    w_up = (jax.random.normal(k[1], (E, L, F), jnp.float32) / L**0.5).astype(bf)
    w_down = (jax.random.normal(k[2], (E, F, out_dim), jnp.float32) / F**0.5).astype(bf)
    dy = jax.random.normal(k[3], (rows, out_dim), bf)
    cu = jnp.arange(E + 1, dtype=jnp.int32) * m_e
    pre, post = jax.jit(lambda a, b, c: quack_act_grouped_gemm(a, b, c, activation="relu_sq", **_QUACK_GATED_KW))(
        x, w_up, cu
    )
    d_pre = jax.random.normal(k[4], (rows, F), bf)
    # bf16 (the recipe's wgrads; and the bf16 baseline of the activation-path GEMMs)
    if "up" in roles:
        run(
            "bf16_up",
            lambda a, b, c: quack_act_grouped_gemm(a, b, c, activation="relu_sq", **_QUACK_GATED_KW),
            x,
            w_up,
            cu,
            **meta,
        )
    if "down" in roles:
        run(
            "bf16_down",
            lambda a, b, c: quack_grouped_gemm(a, b, c, b_major="n", **_QUACK_GROUPED_KW),
            post,
            w_down,
            cu,
            **meta,
        )
    if "dpost" in roles:
        run(
            "bf16_dpost",
            lambda a, b, p, c: quack_dact_grouped_gemm(a, b, p, c, activation="relu_sq", **_QUACK_GATED_KW),
            dy,
            w_down,
            pre,
            cu,
            **meta,
        )
    if "dX" in roles:
        run(
            "bf16_dX",
            lambda a, b, c: quack_grouped_gemm(a, b, c, b_major="k", **_QUACK_GROUPED_KW),
            d_pre,
            w_up,
            cu,
            **meta,
        )
    if "dWu" in roles:
        run("bf16_dWu", lambda a, b, c: quack_grouped_wgrad(a, b, c, **_QUACK_WGRAD_KW), x, d_pre, cu, **meta)
    if "dWd" in roles:
        run("bf16_dWd", lambda a, b, c: quack_grouped_wgrad(a, b, c, **_QUACK_WGRAD_KW), post, dy, cu, **meta)
    # MXFP8 (the recipe's activation path)
    wu_n, wd_n = jax.jit(lambda w: mx_weights(w, "n"))(w_up), jax.jit(lambda w: mx_weights(w, "n"))(w_down)
    wu_k, wd_k = jax.jit(lambda w: mx_weights(w, "k"))(w_up), jax.jit(lambda w: mx_weights(w, "k"))(w_down)
    xq, xe = jax.jit(mx_quantize_rows)(x)
    xs = jax.jit(padded_scale_rows)(xe, cu)
    _pre, post_q, post_s = jax.jit(lambda *a: mxfp8_relu2_grouped_gemm(*a, b_major="n"))(xq, xs, *wu_n, cu)
    dyq, dye = jax.jit(mx_quantize_rows)(dy)
    dys = jax.jit(padded_scale_rows)(dye, cu)
    if "up" in roles:
        run("mxfp8_up", lambda *a: mxfp8_relu2_grouped_gemm(*a, b_major="n"), xq, xs, *wu_n, cu, **meta)
    if "down" in roles:
        run("mxfp8_down", lambda *a: mxfp8_grouped_gemm(*a, b_major="n"), post_q, post_s, *wd_n, cu, **meta)
    _post, dpq, dps, _dot = jax.jit(mxfp8_drelu2_grouped_gemm)(dyq, dys, *wd_k, pre, cu)
    if "dpost" in roles:
        run("mxfp8_dpost", mxfp8_drelu2_grouped_gemm, dyq, dys, *wd_k, pre, cu, **meta)
    if "dX" in roles:
        run("mxfp8_dX", lambda *a: mxfp8_grouped_gemm(*a, b_major="k"), dpq, dps, *wu_k, cu, **meta)
    # MXFP8 weight gradient of one expert (dense block-scaled, K = tokens), x E for the layer; reference only
    if "dWd" in roles:
        p1, d1 = post[:m_e], dy[:m_e]
        a_cols = jax.jit(_quantize_cols)(p1)
        b_cols = jax.jit(_quantize_cols)(d1)

        def wg(av, asc, bv, bsc):
            return _dense_gemm(av, asc, bv, bsc, a_spec=_MN_MAJOR, b_spec=_MN_MAJOR, rows=F, cols=out_dim)

        run("mxfp8_dWd_one_expert", wg, *a_cols, *b_cols, **meta)
    del x, w_up, w_down, dy, pre, post, d_pre


def bench_quant(R):
    meta = dict(R=R, K=F)
    x = jnp.square(jax.nn.relu(jax.random.normal(jax.random.key(1), (R, F), jnp.bfloat16)))
    if x.size < 2**31:
        run("pallas_quant_rows", mx_quantize_rows_pallas, x, **meta)
        run("pallas_quant_cols", mx_quantize_cols_pallas, x, **meta)
        run("pallas_quant_rows_cols", mx_quantize_rows_cols_pallas, x, **meta)
    run("xla_copy", lambda a: a * 2, x, **meta)


def main():
    emit(name="env", jax=jax.__version__, devices=str(jax.devices()))
    for R in (24576, 98304, 393216):
        try:
            bench_quant(R)
        except Exception:
            emit(name="quant", R=R, error=traceback.format_exc()[-1500:])
    for m_e in (4096, 16384, 65536, 109312):
        try:
            bench(m_e)
        except Exception:
            emit(name="bench", m_e=m_e, error=traceback.format_exc()[-1500:])
    try:
        bench(109312, out_dim=H, roles=("down", "dpost", "dWd"))
    except Exception:
        emit(name="bench", m_e=109312, out_dim=H, error=traceback.format_exc()[-1500:])


if __name__ == "__main__":
    sys.exit(main())
