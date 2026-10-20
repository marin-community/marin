# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""S7 global-layer attention on one GB200: MLA (MHA 48/48) + Inkling band 512.

Compares the Pallas forward (``gpu_pallas_fa4``) with the native SM100 forward (``gpu_fa4_cute_sm100``), both
paired with the native SM100 Inkling backward, against S6's 6-KV GQA. Per-GPU shape B16 S4096 D128, causal.
Prints per-call fwd and fwd+bwd times and the max deviation of the native path from the Pallas one.

    python -m experiments.grug.fast_track.bench_relpos_attention
"""

import time

import jax
import jax.numpy as jnp
from levanter.grug.attention import AttentionMask, attention
from levanter.grug.attention._fa4_cute import fa4_cute_segment_bounds
from levanter.grug.attention._inkling_relpos import compact_to_band

BATCH, SEQ, Q_HEADS, HEAD_DIM, REL_EXTENT = 16, 4096, 48, 128, 512
ITERATIONS = 20


def _mask() -> AttentionMask:
    lower_bounds, valid = fa4_cute_segment_bounds(
        AttentionMask(is_causal=True), batch_size=BATCH, seq_len=SEQ, sliding_window=None
    )
    return AttentionMask(is_causal=True, sliding_window=None).with_fa4_bounds(lower_bounds, valid)


def _inputs(kv_heads: int):
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(keys[0], (BATCH, SEQ, Q_HEADS, HEAD_DIM), jnp.bfloat16)
    k = jax.random.normal(keys[1], (BATCH, SEQ, kv_heads, HEAD_DIM), jnp.bfloat16)
    v = jax.random.normal(keys[2], (BATCH, SEQ, kv_heads, HEAD_DIM), jnp.bfloat16)
    compact = (jax.random.normal(keys[3], (BATCH, Q_HEADS, SEQ, REL_EXTENT)) * 0.5).astype(jnp.bfloat16)
    cotangent = jax.random.normal(keys[4], q.shape, jnp.bfloat16)
    return q, k, v, compact_to_band(compact), cotangent


def _milliseconds(fn, *args) -> float:
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        result = fn(*args)
    jax.block_until_ready(result)
    return (time.perf_counter() - start) / ITERATIONS * 1e3


def _run(name: str, implementation: str, kv_heads: int, use_band: bool):
    mask = _mask()
    q, k, v, band, cotangent = _inputs(kv_heads)

    def forward(q, k, v, band):
        return attention(q, k, v, mask, implementation=implementation, rel_bias=band if use_band else None)

    def loss(q, k, v, band):
        return jnp.sum(forward(q, k, v, band).astype(jnp.float32) * cotangent.astype(jnp.float32))

    fwd = jax.jit(forward)
    grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3) if use_band else (0, 1, 2)))
    out = fwd(q, k, v, band)
    grads = grad(q, k, v, band)
    fwd_ms, grad_ms = _milliseconds(fwd, q, k, v, band), _milliseconds(grad, q, k, v, band)
    print(f"TIME {name}: fwd {fwd_ms:.2f} ms fwd+bwd {grad_ms:.2f} ms", flush=True)
    return out, grads


def _max_deviation(name: str, actual, expected) -> None:
    for label, a, e in zip(("out", "dq", "dk", "dv", "dband"), actual, expected, strict=True):
        a, e = a.astype(jnp.float32), e.astype(jnp.float32)
        rel = float(jnp.max(jnp.abs(a - e)) / jnp.max(jnp.abs(e)))
        print(f"DIFF {name} {label}: max-normalized {rel:.3e}", flush=True)


def main() -> None:
    print(jax.devices(), flush=True)
    _run("S6 GQA 48/6 sm100 noband", "gpu_fa4_cute_sm100", 6, False)
    _run("MHA 48/48 sm100 noband", "gpu_fa4_cute_sm100", Q_HEADS, False)
    _run("MHA 48/48 pallas noband", "gpu_pallas_fa4", Q_HEADS, False)
    pallas_out, pallas_grads = _run("MLA 48/48 pallas band512", "gpu_pallas_fa4", Q_HEADS, True)
    native_out, native_grads = _run("MLA 48/48 sm100 band512", "gpu_fa4_cute_sm100", Q_HEADS, True)
    _max_deviation("sm100 vs pallas", (native_out, *native_grads), (pallas_out, *pallas_grads))


if __name__ == "__main__":
    main()
