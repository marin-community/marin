"""Raw-Triton (jax-triton) prototype of the fused RMSNorm + GatedNorm forward, against Pallas and XLA.

Times, at the hero per-GPU shape [65536, 6144] rank 128: a plain copy kernel (the streaming
ceiling), the two raw-Triton kernels over a small config sweep, the Pallas-Triton kernels, and
XLA's reference; checks the Triton outputs against the Pallas path (same rounding points).

Usage: python grn_triton.py
"""

import glob
import itertools
import os
import sys

import jax
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl

from levanter.kernels.pallas.gated_rms_norm import GatedRmsNormBlockSizes, gated_rms_norm, gated_rms_norm_reference
from levanter.kernels.pallas.gated_rms_norm.pallas_gpu import gated_rms_norm_pallas_fwd_local

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop-260930-mfu30"))
from overlap import load, plane_events  # noqa: E402

T, D, R = 65536, 6144, 128
EPS = 1e-5


@triton.jit
def _rnd(v):
    return v.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _logistic(v):
    e = _rnd(tl.exp(-v))
    return _rnd(1.0 / _rnd(1.0 + e))


@triton.jit
def _copy_kernel(x_ptr, o_ptr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    rows = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    offs = rows[:, None] * D + cols[None, :]
    tl.store(o_ptr + offs, tl.load(x_ptr + offs))


@triton.jit
def _stats_kernel(
    x_ptr, w_ptr, wd_ptr, h_ptr, s_ptr, rstd_ptr, eps,
    D: tl.constexpr, R: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, NS: tl.constexpr,
):
    rows = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    rk = tl.arange(0, R)
    acc = tl.zeros([BT, R], dtype=tl.float32)
    sq = tl.zeros([BT, BD], dtype=tl.float32)
    for k in tl.range(0, D, BD, num_stages=NS):
        cols = k + tl.arange(0, BD)
        x = tl.load(x_ptr + rows[:, None] * D + cols[None, :]).to(tl.float32)
        w = tl.load(w_ptr + cols).to(tl.float32)
        sq += x * x
        xw = (x * w[None, :]).to(tl.bfloat16)
        wd = tl.load(wd_ptr + cols[:, None] * R + rk[None, :])
        acc = tl.dot(xw, wd, acc)
    rstd = tl.rsqrt(tl.sum(sq, axis=1) * (1.0 / D) + eps)
    h = _rnd(acc * rstd[:, None])
    s = _rnd(h * _logistic(h))
    ro = rows[:, None] * R + rk[None, :]
    tl.store(h_ptr + ro, h.to(tl.bfloat16))
    tl.store(s_ptr + ro, s.to(tl.bfloat16))
    tl.store(rstd_ptr + rows, rstd)


@triton.jit
def _output_kernel(
    x_ptr, w_ptr, rstd_ptr, s_ptr, wu_ptr, out_ptr, gate_ptr,
    D: tl.constexpr, R: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
):
    rows = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    rk = tl.arange(0, R)
    s = tl.load(s_ptr + rows[:, None] * R + rk[None, :])
    wu = tl.load(wu_ptr + rk[:, None] * D + cols[None, :])
    gate = _logistic(_rnd(tl.dot(s, wu)))
    offs = rows[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + offs).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows)
    w = tl.load(w_ptr + cols).to(tl.float32)
    y = _rnd(x * rstd[:, None] * w[None, :])
    tl.store(out_ptr + offs, (y * gate).to(tl.bfloat16))
    tl.store(gate_ptr + offs, gate.to(tl.bfloat16))


def triton_fwd(x, w, wd, wu, *, bt_s, bd_s, ns, warps_s, bt_o, bd_o, warps_o, stages_o):
    h, s, rstd = jt.triton_call(
        x, w, wd, kernel=_stats_kernel,
        out_shape=[jax.ShapeDtypeStruct((T, R), jnp.bfloat16)] * 2 + [jax.ShapeDtypeStruct((T,), jnp.float32)],
        grid=(T // bt_s,), num_warps=warps_s, num_stages=1, eps=EPS, D=D, R=R, BT=bt_s, BD=bd_s, NS=ns,
    )
    out, gate = jt.triton_call(
        x, w, rstd, s, wu, kernel=_output_kernel,
        out_shape=[jax.ShapeDtypeStruct((T, D), jnp.bfloat16)] * 2,
        grid=(T // bt_o, D // bd_o), num_warps=warps_o, num_stages=stages_o, D=D, R=R, BT=bt_o, BD=bd_o,
    )
    return out, gate, h, rstd


def profile(fn, args, tag, reps=5):
    out = fn(*args)
    jax.block_until_ready(out)
    d = f"/tmp/grn_triton/{tag}"
    jax.profiler.start_trace(d)
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    jax.profiler.stop_trace()
    planes = {p.name: p for p in load(sorted(glob.glob(f"{d}/**/*.xplane.pb", recursive=True))[-1]).planes}
    by = {}
    for lname, name, s, e, st in plane_events(planes["/device:GPU:0"]):
        if lname.startswith("Stream"):
            by[name[:40]] = by.get(name[:40], 0.0) + (e - s) * 1e-9 / reps
    return by


def main():
    k = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k[0], (T, D), jnp.bfloat16)
    w = (1 + 0.2 * jax.random.normal(k[1], (D,))).astype(jnp.bfloat16)
    wd = (0.02 * jax.random.normal(k[2], (D, R))).astype(jnp.bfloat16)
    wu = (0.2 * jax.random.normal(k[3], (R, D))).astype(jnp.bfloat16)
    gb = T * D * 2 / 1e9
    for bt, bd in ((64, 128), (32, 256), (16, 512)):
        cp = jax.jit(lambda a: jt.triton_call(a, kernel=_copy_kernel, out_shape=jax.ShapeDtypeStruct((T, D), jnp.bfloat16),
                                              grid=(T // bt, D // bd), num_warps=4, D=D, BT=bt, BD=bd))
        by = profile(cp, (x,), f"copy{bt}_{bd}")
        t = sum(by.values())
        print(f"copy tile {bt}x{bd}: {t:.3f} ms ({2 * gb / t:.2f} TB/s)", flush=True)
    by = profile(jax.jit(lambda *a: gated_rms_norm_reference(*a, eps=EPS)), (x, w, wd, wu), "ref")
    print(f"XLA reference {sum(by.values()):.3f} ms", flush=True)
    pal = jax.jit(lambda *a: gated_rms_norm_pallas_fwd_local(*a, eps=EPS, block_sizes=GatedRmsNormBlockSizes()))
    by = profile(pal, (x, w, wd, wu), "pallas")
    print(f"Pallas pair {sum(by.values()):.3f} ms: {by}", flush=True)
    want = [np.asarray(a, np.float32) for a in pal(x, w, wd, wu)]
    configs = list(itertools.product((32, 64, 128), (64, 128), (2, 3, 4), (4, 8)))
    best = {}
    for bt_s, bd_s, ns, warps_s in configs:
        fn = jax.jit(lambda *a, c=(bt_s, bd_s, ns, warps_s): triton_fwd(
            *a, bt_s=c[0], bd_s=c[1], ns=c[2], warps_s=c[3], bt_o=64, bd_o=128, warps_o=4, stages_o=1))
        try:
            by = profile(fn, (x, w, wd, wu), f"s{bt_s}_{bd_s}_{ns}_{warps_s}")
        except Exception as e:  # record configs that fail to compile
            print(f"stats {bt_s},{bd_s},{ns},{warps_s}: ERROR {str(e)[:160]}", flush=True)
            continue
        t = sum(v for n, v in by.items() if "stats" in n)
        best.setdefault("stats", []).append((t, (bt_s, bd_s, ns, warps_s)))
        print(f"stats bt={bt_s} bd={bd_s} ns={ns} warps={warps_s}: {t:.3f} ms ({gb / t:.2f} TB/s)", flush=True)
    for bt_o, bd_o, warps_o, stages_o in itertools.product((32, 64, 128), (64, 128, 256), (4, 8), (1, 2)):
        fn = jax.jit(lambda *a, c=(bt_o, bd_o, warps_o, stages_o): triton_fwd(
            *a, bt_s=64, bd_s=64, ns=3, warps_s=4, bt_o=c[0], bd_o=c[1], warps_o=c[2], stages_o=c[3]))
        try:
            by = profile(fn, (x, w, wd, wu), f"o{bt_o}_{bd_o}_{warps_o}_{stages_o}")
        except Exception as e:  # record configs that fail to compile
            print(f"output {bt_o},{bd_o},{warps_o},{stages_o}: ERROR {str(e)[:160]}", flush=True)
            continue
        t = sum(v for n, v in by.items() if "output" in n)
        best.setdefault("output", []).append((t, (bt_o, bd_o, warps_o, stages_o)))
        print(f"output bt={bt_o} bd={bd_o} warps={warps_o} stages={stages_o}: {t:.3f} ms ({3 * gb / t:.2f} TB/s)", flush=True)
    (ts, cs), (to, co) = min(best["stats"]), min(best["output"])
    print(f"BEST stats {cs} {ts:.3f} ms, output {co} {to:.3f} ms, total {ts + to:.3f} ms", flush=True)
    got = [np.asarray(a, np.float32) for a in jax.jit(lambda *a: triton_fwd(
        *a, bt_s=cs[0], bd_s=cs[1], ns=cs[2], warps_s=cs[3], bt_o=co[0], bd_o=co[1], warps_o=co[2], stages_o=co[3]))(x, w, wd, wu)]
    for name, g, p in zip(("out", "gate", "h", "rstd"), got, want):
        print(f"triton vs pallas {name}: max|d| {np.max(np.abs(g - p)):.3e} (|p|max {np.max(np.abs(p)):.2e}) "
              f"frac differing {np.mean(g != p):.2e}", flush=True)


if __name__ == "__main__":
    main()
