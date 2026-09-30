# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check and time the grouped dh GEMM with the SwiGLU backward fused into its epilogue.

Compares `quack_grouped_dswiglu_gemm` with the current path (QuACK dh GEMM, then the XLA SwiGLU
backward and row dot) at one chunk of the hero's per-shard shapes: active rows only, relative to the
reference's largest magnitude, and the median time of each path.

Usage (one GB200): python autoresearch/loop-260930-mfu30/b/dswiglu_check.py
"""

import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug._moe.common import _swiglu_gate_up_backward, _unpack_pairs_u32
from levanter.grug._moe.quack_moe_cute import (
    quack_gated_grouped_gemm,
    quack_grouped_dswiglu_gemm,
    quack_grouped_gemm,
)
from levanter.grug._moe.sonic_cute import _QUACK_GATED_KW, _QUACK_GROUPED_KW

CAPACITY, HIDDEN, INTER, EXPERTS = 301466, 3072, 3072, 3
GROUPS = np.array([88000, 86000, 88144], dtype=np.int32)


def _time(fn, *args, iters=20):
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _rel(actual, expected):
    a = np.asarray(actual, np.float32)
    e = np.asarray(expected, np.float32)
    scale = float(np.max(np.abs(e))) or 1.0
    diff = np.abs(a - e)
    return dict(
        max_rel=float(np.max(diff)) / scale, mean_rel=float(np.mean(diff)) / scale, finite=bool(np.isfinite(a).all())
    )


def main():
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(GROUPS)]), jnp.int32)
    active = int(cu[-1])
    keys = jax.random.split(jax.random.key(0), 4)
    bf16 = jnp.bfloat16
    x = jax.random.normal(keys[0], (CAPACITY, HIDDEN), bf16)
    w13_il = (jax.random.normal(keys[1], (EXPERTS, HIDDEN, 2 * INTER)) * 0.02).astype(bf16)
    w2 = (jax.random.normal(keys[2], (EXPERTS, INTER, HIDDEN)) * 0.02).astype(bf16)
    dy = (jax.random.normal(keys[3], (CAPACITY, HIDDEN)) * 0.01).astype(bf16)
    gu, _h = jax.jit(lambda x: quack_gated_grouped_gemm(x, w13_il, cu, return_preact=True, **_QUACK_GATED_KW))(x)

    @jax.jit
    def reference(dy, gu):
        dh = quack_grouped_gemm(dy, w2, cu, b_major="k", **_QUACK_GROUPED_KW)
        gate, up = _unpack_pairs_u32(gu)
        h = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        return _swiglu_gate_up_backward(gu, dh), jnp.sum(dh.astype(jnp.float32) * h, axis=-1)

    @jax.jit
    def fused(dy, gu):
        return quack_grouped_dswiglu_gemm(dy, w2, gu, cu, **_QUACK_GROUPED_KW)

    ref_d_gu, ref_dot = reference(dy, gu)
    got_d_gu, got_dot = fused(dy, gu)
    record = dict(
        active_rows=active,
        d_gate_up=_rel(got_d_gu[:active], ref_d_gu[:active]),
        row_dot=_rel(got_dot[:active], ref_dot[:active]),
        reference_ms=_time(reference, dy, gu) * 1e3,
        fused_ms=_time(fused, dy, gu) * 1e3,
        dh_gemm_only_ms=_time(jax.jit(lambda dy: quack_grouped_gemm(dy, w2, cu, b_major="k", **_QUACK_GROUPED_KW)), dy)
        * 1e3,
    )
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
