# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep segmented FA4/CuTe tile configurations at Grug hero attention shapes.

``_segmented_kernel_config`` pins compute capability 10.x at head dimension 128 to a 64x64
forward tile, a choice measured on d5120 shapes. The FSDP hero is d2048 and spends 15 of 18
layers on a 512-token sliding window, where each query tile covers only nine key tiles and
per-tile overhead stops amortizing. This sweep times both windows against the reference
configuration and checks that every candidate reproduces its outputs and gradients.
"""

import argparse
import dataclasses
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask
from levanter.grug.attention import _fa4_cute as fa4_cute
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig, flash4_cute_kernel_config

REFERENCE_TILE = (64, 64)
CANDIDATE_FORWARD_TILES = ((64, 64), (128, 64), (128, 128), (256, 64))
CANDIDATE_BACKWARD_TILES = ((64, 64), (128, 64))
CANDIDATE_NUM_THREADS = (128, 256)


def _segment_ids(batch: int, seq_len: int, documents: int) -> jax.Array:
    """Evenly spaced document boundaries, matching the corpus density of ~5 per 4096 tokens."""
    boundaries = np.linspace(0, seq_len, documents + 1).astype(np.int32)[1:-1]
    ids = np.zeros((batch, seq_len), dtype=np.int32)
    for b in range(batch):
        # Stagger each row so tiles do not share identical boundaries.
        ids[b] = np.searchsorted(boundaries, np.arange(seq_len) - b % 64, side="right")
    return jnp.asarray(ids)


def _loss(q, k, v, mask):
    return jnp.sum(fa4_cute.gpu_fa4_cute_attention(q, k, v, mask).astype(jnp.float32) ** 2)


def _bench(config: Flash4CuteKernelConfig, q, k, v, mask, steps: int, warmup: int):
    with patch.object(fa4_cute, "_segmented_kernel_config", lambda head_dim: config):
        forward = jax.jit(lambda q, k, v: fa4_cute.gpu_fa4_cute_attention(q, k, v, mask))
        grad = jax.jit(jax.grad(_loss, argnums=(0, 1, 2)))

        start = time.perf_counter()
        out = forward(q, k, v)
        out.block_until_ready()
        grads = grad(q, k, v, mask)
        jax.block_until_ready(grads)
        compile_time = time.perf_counter() - start

        for _ in range(warmup):
            forward(q, k, v).block_until_ready()
        start = time.perf_counter()
        for _ in range(steps):
            forward(q, k, v).block_until_ready()
        forward_time = (time.perf_counter() - start) / steps

        for _ in range(warmup):
            jax.block_until_ready(grad(q, k, v, mask))
        start = time.perf_counter()
        for _ in range(steps):
            jax.block_until_ready(grad(q, k, v, mask))
        grad_time = (time.perf_counter() - start) / steps

    return out, grads, forward_time, grad_time - forward_time, compile_time


def _max_diff(a, b) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32, help="Per-GPU batch; the hero runs 32.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--documents", type=int, default=5)
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=2e-2)
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise SystemExit("bench_fa4_cute_tile_sweep requires the JAX GPU backend.")

    key = jax.random.key(0)
    shape_q = (args.batch, args.seq_len, args.q_heads, args.head_dim)
    shape_kv = (args.batch, args.seq_len, args.kv_heads, args.head_dim)
    q = jax.random.normal(key, shape_q, dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), shape_kv, dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), shape_kv, dtype=jnp.bfloat16)
    segment_ids = _segment_ids(args.batch, args.seq_len, args.documents)

    arch = fa4_cute._gpu_compute_arch()
    base = flash4_cute_kernel_config(args.head_dim, arch=arch)
    print(f"arch=sm{arch} base_forward_tile={base.forward_tile} base_backward_tile={base.backward_tile}")
    print(f"shape: batch={args.batch} seq={args.seq_len} q_heads={args.q_heads} head_dim={args.head_dim}")

    candidates = []
    for forward_tile in CANDIDATE_FORWARD_TILES:
        for backward_tile in CANDIDATE_BACKWARD_TILES:
            for num_threads in CANDIDATE_NUM_THREADS:
                if (forward_tile[0] * 2) % num_threads or (backward_tile[0] * 2) % num_threads:
                    continue
                candidates.append(
                    dataclasses.replace(
                        base, forward_tile=forward_tile, backward_tile=backward_tile, num_threads=num_threads
                    )
                )

    for window_name, window in (("sliding", args.sliding_window), ("causal", None)):
        mask = AttentionMask.causal(sliding_window=window).with_segment_ids(segment_ids)
        reference = dataclasses.replace(
            base, forward_tile=REFERENCE_TILE, backward_tile=REFERENCE_TILE, num_threads=128
        )
        ref_out, ref_grads, ref_fwd, ref_bwd, _ = _bench(reference, q, k, v, mask, args.steps, args.warmup)
        print(f"\n=== {window_name} (window={window}) ===")
        print(
            f"{'forward_tile':>14} {'backward_tile':>14} {'thr':>4} {'fwd ms':>8} {'bwd ms':>8} "
            f"{'total ms':>9} {'vs ref':>8} {'max|d|':>9}"
        )
        print(
            f"{str(REFERENCE_TILE):>14} {str(REFERENCE_TILE):>14} {128:>4} {ref_fwd * 1e3:8.3f} "
            f"{ref_bwd * 1e3:8.3f} {(ref_fwd + ref_bwd) * 1e3:9.3f} {'1.00x':>8} {'ref':>9}"
        )

        for config in candidates:
            if (config.forward_tile, config.backward_tile, config.num_threads) == (
                REFERENCE_TILE,
                REFERENCE_TILE,
                128,
            ):
                continue
            try:
                out, grads, fwd, bwd, _ = _bench(config, q, k, v, mask, args.steps, args.warmup)
            except Exception as exc:  # unsupported tile/thread/smem combination
                print(
                    f"{str(config.forward_tile):>14} {str(config.backward_tile):>14} "
                    f"{config.num_threads:>4} {'-':>8} {'-':>8} {'-':>9} {'-':>8}  {type(exc).__name__}"
                )
                continue
            diff = max([_max_diff(out, ref_out)] + [_max_diff(g, r) for g, r in zip(grads, ref_grads)])
            speedup = (ref_fwd + ref_bwd) / (fwd + bwd)
            flag = "" if diff <= args.tolerance else "  MISMATCH"
            print(
                f"{str(config.forward_tile):>14} {str(config.backward_tile):>14} {config.num_threads:>4} "
                f"{fwd * 1e3:8.3f} {bwd * 1e3:8.3f} {(fwd + bwd) * 1e3:9.3f} {speedup:7.2f}x "
                f"{diff:9.2e}{flag}"
            )


if __name__ == "__main__":
    main()
