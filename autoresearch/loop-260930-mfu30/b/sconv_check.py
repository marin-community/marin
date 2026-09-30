# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness and timing of the streaming Triton short conv against the Pallas kernel and the reference.

On one GB200 at the hero per-layer shapes ([16, 4096, C], C in {6144, 1536}, bf16, kernel 4) and a set
of segment layouts: output and dx bitwise against `short_conv_reference` (exact rounding) and the
Pallas kernel, dw against a float64 oracle and the Pallas kernel, then median forward and backward
time per call for each implementation.

Usage (one GB200): python autoresearch/loop-260930-mfu30/b/sconv_check.py [--quick]
"""

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from levanter.kernels.pallas.short_conv import short_conv, short_conv_reference

WIDTH = 4


def _segments(kind: str, batch: int, seq: int, rng) -> np.ndarray | None:
    if kind == "none":
        return None
    if kind == "packed":
        # Documents of random lengths; boundaries land anywhere, including chunk edges.
        seg = np.zeros((batch, seq), np.int32)
        for b in range(batch):
            cuts = np.sort(rng.choice(np.arange(1, seq), size=12, replace=False))
            cuts = np.concatenate([cuts, [63, 64, 65, 127, 128]])
            seg[b] = np.searchsorted(np.sort(np.unique(cuts)), np.arange(seq), side="right")
        return seg
    if kind == "short_docs":
        # Documents of length 1 to 3 everywhere: every tap can cross a boundary.
        lengths = rng.integers(1, 4, size=(batch, seq))
        seg = np.zeros((batch, seq), np.int32)
        for b in range(batch):
            seg[b] = np.repeat(np.arange(seq), lengths[b])[:seq]
        return seg
    if kind == "padded":
        # A packed prefix, then padding positions carrying segment id -1 (the out-of-range id).
        seg = np.zeros((batch, seq), np.int32)
        for b in range(batch):
            valid = int(rng.integers(seq // 2, seq))
            seg[b, : valid // 2] = 0
            seg[b, valid // 2 : valid] = 1
            seg[b, valid:] = -1
        return seg
    raise ValueError(kind)


def _run(impl, weight, x, seg, ct):
    def f(w, x):
        return short_conv(w, x, seg, implementation=impl)

    out, pullback = jax.vjp(f, weight, x)
    dw, dx = pullback(ct)
    return out, dx, dw


def _dw_oracle(x, seg, ct):
    x64 = np.asarray(x, np.float64)
    ct64 = np.asarray(ct, np.float64)
    batch, seq, channels = x64.shape
    out = np.zeros((WIDTH, channels))
    for lag in range(WIDTH):
        shifted = np.zeros_like(x64)
        shifted[:, lag:] = x64[:, : seq - lag]
        if seg is not None and lag:
            s = np.asarray(seg)
            prev = np.full_like(s, -1)
            prev[:, lag:] = s[:, : seq - lag]
            shifted = np.where((prev == s)[..., None], shifted, 0.0)
        out[lag] = np.sum(ct64 * shifted, axis=(0, 1))
    return out


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    scale = float(np.max(np.abs(b))) or 1.0
    return float(np.max(np.abs(a - b))) / scale


def _time(fn, *args, iters=20):
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    shapes = [(2, 512, 1536)] if args.quick else [(16, 4096, 6144), (16, 4096, 1536)]
    failures = 0
    for batch, seq, channels in shapes:
        keys = jax.random.split(jax.random.key(batch * seq + channels), 3)
        x = jax.random.normal(keys[0], (batch, seq, channels), jnp.bfloat16)
        weight = (jax.random.normal(keys[1], (WIDTH, channels)) * 0.5).astype(jnp.bfloat16)
        ct = jax.random.normal(keys[2], (batch, seq, channels), jnp.bfloat16)
        for kind in ("none", "packed", "short_docs", "padded"):
            seg_np = _segments(kind, batch, seq, rng)
            seg = None if seg_np is None else jnp.asarray(seg_np)
            ref_out = jax.jit(lambda w, x, s=seg: short_conv_reference(w, x, s))(weight, x)
            _, ref_pull = jax.vjp(lambda w, x, s=seg: short_conv_reference(w, x, s), weight, x)
            _ref_dw, ref_dx = jax.jit(ref_pull)(ct)
            tri = jax.jit(lambda w, x, c, s=seg: _run("triton_gpu", w, x, s, c))(weight, x, ct)
            pal = jax.jit(lambda w, x, c, s=seg: _run("pallas_gpu", w, x, s, c))(weight, x, ct)
            oracle = _dw_oracle(x, seg_np, ct)
            record = dict(
                shape=[batch, seq, channels],
                segments=kind,
                out_equal_reference=bool(np.array_equal(np.asarray(tri[0]), np.asarray(ref_out))),
                out_equal_pallas=bool(np.array_equal(np.asarray(tri[0]), np.asarray(pal[0]))),
                dx_equal_reference=bool(np.array_equal(np.asarray(tri[1]), np.asarray(ref_dx))),
                dx_equal_pallas=bool(np.array_equal(np.asarray(tri[1]), np.asarray(pal[1]))),
                dw_rel_oracle=_rel(np.asarray(tri[2], np.float32), oracle),
                dw_rel_oracle_pallas=_rel(np.asarray(pal[2], np.float32), oracle),
                dw_rel_pallas=_rel(np.asarray(tri[2], np.float32), np.asarray(pal[2], np.float32)),
                finite=bool(np.isfinite(np.asarray(tri[2], np.float32)).all()),
            )
            ok = (
                record["out_equal_reference"]
                and record["dx_equal_reference"]
                and record["finite"]
                and record["dw_rel_oracle"] <= max(2 * record["dw_rel_oracle_pallas"], 2.0**-8)
            )
            record["ok"] = ok
            failures += not ok
            print(json.dumps(record), flush=True)
        seg = jnp.asarray(_segments("packed", batch, seq, rng))
        timing = {}
        for impl in ("reference", "pallas_gpu", "triton_gpu"):
            fwd = jax.jit(lambda w, x, i=impl: short_conv(w, x, seg, implementation=i))
            _, pull = jax.vjp(lambda w, x, i=impl: short_conv(w, x, seg, implementation=i), weight, x)
            bwd = jax.jit(pull)
            timing[impl] = dict(fwd_ms=_time(fwd, weight, x), bwd_ms=_time(bwd, ct))
        tensor_gb = batch * seq * channels * 2 / 1e9
        print(
            json.dumps(
                dict(
                    shape=[batch, seq, channels],
                    timing_ms=timing,
                    tensor_gb=tensor_gb,
                    floor_ms_at_7tbps=dict(fwd=2 * tensor_gb / 7.0, bwd=3 * tensor_gb / 7.0),
                )
            ),
            flush=True,
        )
    print(json.dumps(dict(failures=failures)), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
