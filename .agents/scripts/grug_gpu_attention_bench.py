# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Correctness and timing harness for Grug GPU attention."""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask, gpu_cudnn_attention, gpu_xla_attention, reference_attention
from levanter.grug.flex_attention import gpu_flex_pallas_attention, tokamax_flash_attention


def _dtype_from_name(name: str) -> jnp.dtype:
    match name:
        case "bf16":
            return jnp.bfloat16
        case "fp16":
            return jnp.float16
        case "fp32":
            return jnp.float32
        case _:
            raise ValueError(f"Unsupported dtype: {name}")


def _make_qkv(batch: int, seq: int, q_heads: int, kv_heads: int, head_dim: int, dtype: jnp.dtype):
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (batch, seq, q_heads, head_dim), dtype=dtype)
    k = jax.random.normal(k_key, (batch, seq, kv_heads, head_dim), dtype=dtype)
    v = jax.random.normal(v_key, (batch, seq, kv_heads, head_dim), dtype=dtype)
    return q, k, v


def _packed_segment_ids(batch: int, seq: int, segments: int) -> jax.Array:
    boundaries = jnp.minimum(jnp.arange(seq) * segments // seq, segments - 1)
    per_batch = []
    for batch_index in range(batch):
        per_batch.append(jnp.roll(boundaries, batch_index % max(1, seq // max(1, segments))))
    return jnp.stack(per_batch).astype(jnp.int32)


@dataclass(frozen=True)
class CudnnMatrixCase:
    name: str
    mask: AttentionMask | None
    q_heads: int
    kv_heads: int


def _cudnn_matrix_cases(batch: int, seq: int) -> list[CudnnMatrixCase]:
    segment_1d = jnp.arange(seq, dtype=jnp.int32) // max(1, seq // 4)
    segment_2d = _packed_segment_ids(batch=batch, seq=seq, segments=4)
    masks = [
        ("no_mask", None),
        ("causal", AttentionMask.causal()),
        ("sliding", AttentionMask().with_sliding_window(16)),
        ("segment_1d", AttentionMask().with_segment_ids(segment_1d)),
        ("segment_2d", AttentionMask().with_segment_ids(segment_2d)),
        ("causal_segment_2d", AttentionMask.causal().with_segment_ids(segment_2d)),
        ("sliding_segment_2d", AttentionMask.causal(sliding_window=16).with_segment_ids(segment_2d)),
    ]
    cases = []
    for head_name, q_heads, kv_heads in [("no_gqa", 2, 2), ("gqa", 4, 2)]:
        for mask_name, mask in masks:
            cases.append(CudnnMatrixCase(f"{head_name}:{mask_name}", mask, q_heads, kv_heads))
    return cases


def _block_until_ready(value):
    return jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)


def _time_call(name: str, fn: Callable[[], jax.Array], *, warmup: int, iters: int) -> None:
    compile_start = time.perf_counter()
    out = fn()
    _block_until_ready(out)
    compile_elapsed = time.perf_counter() - compile_start

    for _ in range(warmup):
        out = fn()
        _block_until_ready(out)

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        _block_until_ready(out)
    elapsed = time.perf_counter() - start

    print(f"{name}: compile_plus_first={compile_elapsed:.6f}s steady={elapsed / iters:.6f}s iters={iters}")


def _gpu_attention_fn(name: str):
    match name:
        case "gpu_cudnn":
            return gpu_cudnn_attention
        case "gpu_xla":
            return gpu_xla_attention
        case "gpu_flex_pallas":
            return gpu_flex_pallas_attention
        case "gpu_flex_pallas_reference_vjp":
            return lambda q, k, v, mask: gpu_flex_pallas_attention(
                q, k, v, mask, use_reference_vjp=True, prefer_flash=False
            )
        case "gpu_flash_triton":
            return lambda q, k, v, mask: tokamax_flash_attention(
                q, k, v, mask, implementation="triton", assume_packed_segment_ids=True
            )
        case "gpu_flash_mosaic":
            return lambda q, k, v, mask: tokamax_flash_attention(
                q, k, v, mask, implementation="mosaic", assume_packed_segment_ids=True
            )
        case _:
            raise ValueError(f"Unsupported GPU implementation: {name}")


def _check_correctness(dtype: jnp.dtype, implementation: str, *, dynamic_segment_ids: bool) -> None:
    q, k, v = _make_qkv(batch=2, seq=64, q_heads=4, kv_heads=2, head_dim=32, dtype=dtype)
    segment_ids = _packed_segment_ids(batch=2, seq=64, segments=4)
    mask = AttentionMask.causal(sliding_window=16).with_segment_ids(segment_ids)
    gpu_attention = _gpu_attention_fn(implementation)

    ref = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    if dynamic_segment_ids:
        actual = jax.jit(
            lambda q_arg, k_arg, v_arg, segment_ids_arg: gpu_attention(
                q_arg,
                k_arg,
                v_arg,
                AttentionMask.causal(sliding_window=16).with_segment_ids(segment_ids_arg),
            )
        )(q, k, v, segment_ids)
    else:
        actual = jax.jit(gpu_attention)(q, k, v, mask)
    jax.block_until_ready(actual)

    diff = (actual.astype(jnp.float32) - ref.astype(jnp.float32)).reshape(-1)
    max_abs = float(jnp.max(jnp.abs(diff)))
    mean_abs = float(jnp.mean(jnp.abs(diff)))
    print(
        f"correctness_small: implementation={implementation} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} dtype={dtype}"
    )
    np.testing.assert_allclose(actual, ref, atol=4e-2 if dtype != jnp.float32 else 5e-4, rtol=4e-2)

    cotangent = jax.random.normal(jax.random.PRNGKey(1), actual.shape, dtype=dtype)

    def ref_loss(q_arg, k_arg, v_arg):
        out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def actual_loss(q_arg, k_arg, v_arg):
        out = gpu_attention(q_arg, k_arg, v_arg, mask)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    if dynamic_segment_ids:

        def actual_loss_dynamic(q_arg, k_arg, v_arg, segment_ids_arg):
            dynamic_mask = AttentionMask.causal(sliding_window=16).with_segment_ids(segment_ids_arg)
            out = gpu_attention(q_arg, k_arg, v_arg, dynamic_mask)
            return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

        actual_grads = jax.jit(jax.grad(actual_loss_dynamic, argnums=(0, 1, 2)))(q, k, v, segment_ids)
    else:
        actual_grads = jax.jit(jax.grad(actual_loss, argnums=(0, 1, 2)))(q, k, v)
    ref_grads = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))(q, k, v)
    grad_diffs = [
        jnp.abs(actual_grad.astype(jnp.float32) - ref_grad.astype(jnp.float32)).reshape(-1)
        for actual_grad, ref_grad in zip(actual_grads, ref_grads, strict=True)
    ]
    max_grad_abs = max(float(jnp.max(diff)) for diff in grad_diffs)
    mean_grad_abs = sum(float(jnp.mean(diff)) for diff in grad_diffs) / len(grad_diffs)
    print(
        f"grad_correctness_small: implementation={implementation} "
        f"max_abs={max_grad_abs:.6g} mean_abs={mean_grad_abs:.6g} dtype={dtype}"
    )
    for actual_grad, ref_grad in zip(actual_grads, ref_grads, strict=True):
        np.testing.assert_allclose(actual_grad, ref_grad, atol=5e-2 if dtype != jnp.float32 else 5e-4, rtol=5e-2)


def _run_cudnn_matrix(dtype: jnp.dtype, *, batch: int, seq: int, head_dim: int) -> None:
    print(f"cudnn_matrix: batch={batch} seq={seq} head_dim={head_dim} dtype={dtype}")
    for case in _cudnn_matrix_cases(batch=batch, seq=seq):
        q, k, v = _make_qkv(batch, seq, case.q_heads, case.kv_heads, head_dim, dtype)

        def run(q_arg, k_arg, v_arg, mask=case.mask):
            return gpu_cudnn_attention(q_arg, k_arg, v_arg, mask)

        try:
            out = jax.jit(run)(q, k, v)
            jax.block_until_ready(out)
        except Exception as exc:
            message = str(exc).splitlines()[0]
            print(f"MATRIX {case.name}: FAIL {type(exc).__name__}: {message}")
            continue

        ref = reference_attention(q, k, v, case.mask, logits_dtype=jnp.float32)
        diff = (out.astype(jnp.float32) - ref.astype(jnp.float32)).reshape(-1)
        max_abs = float(jnp.max(jnp.abs(diff)))
        mean_abs = float(jnp.mean(jnp.abs(diff)))
        print(f"MATRIX {case.name}: PASS max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")


def _run_cudnn_direct_sanity() -> None:
    print("cudnn_direct_sanity: direct jax.nn.dot_product_attention no-mask calls")
    cases = [
        ("fp16_B1_S128_H1_D64", jnp.float16, 1, 128, 1, 64),
        ("bf16_B1_S128_H1_D64", jnp.bfloat16, 1, 128, 1, 64),
        ("fp16_B2_S128_H2_D64", jnp.float16, 2, 128, 2, 64),
        ("bf16_B2_S128_H2_D64", jnp.bfloat16, 2, 128, 2, 64),
        ("fp16_B2_S256_H2_D128", jnp.float16, 2, 256, 2, 128),
        ("bf16_B2_S256_H2_D128", jnp.bfloat16, 2, 256, 2, 128),
    ]
    for name, dtype, batch, seq, heads, head_dim in cases:
        q, k, v = _make_qkv(batch, seq, heads, heads, head_dim, dtype)

        def run(q_arg, k_arg, v_arg):
            return jax.nn.dot_product_attention(q_arg, k_arg, v_arg, implementation="cudnn")

        try:
            out = jax.jit(run)(q, k, v)
            jax.block_until_ready(out)
        except Exception as exc:
            message = str(exc).splitlines()[0]
            print(f"DIRECT {name}: FAIL {type(exc).__name__}: {message}")
            continue

        ref = jax.nn.dot_product_attention(q, k, v, implementation="xla")
        diff = (out.astype(jnp.float32) - ref.astype(jnp.float32)).reshape(-1)
        max_abs = float(jnp.max(jnp.abs(diff)))
        mean_abs = float(jnp.mean(jnp.abs(diff)))
        print(f"DIRECT {name}: PASS max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--sliding-window", type=int, default=256)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--implementation",
        choices=(
            "gpu_xla",
            "gpu_cudnn",
            "gpu_flex_pallas",
            "gpu_flex_pallas_reference_vjp",
            "gpu_flash_triton",
            "gpu_flash_mosaic",
            "all",
        ),
        default="gpu_cudnn",
    )
    parser.add_argument("--mode", choices=("benchmark", "cudnn-matrix", "cudnn-direct"), default="benchmark")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--skip-reference-timing", action="store_true")
    parser.add_argument("--dynamic-segment-ids", action="store_true")
    args = parser.parse_args()

    print(f"backend={jax.default_backend()} devices={[str(d) for d in jax.devices()]}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("This benchmark requires a JAX GPU backend.")

    dtype = _dtype_from_name(args.dtype)
    if args.mode == "cudnn-matrix":
        _run_cudnn_matrix(dtype, batch=2, seq=64, head_dim=64)
        return
    if args.mode == "cudnn-direct":
        _run_cudnn_direct_sanity()
        return

    q, k, v = _make_qkv(args.batch, args.seq, args.q_heads, args.kv_heads, args.head_dim, dtype)
    segment_ids = _packed_segment_ids(args.batch, args.seq, args.segments)
    mask = AttentionMask.causal(sliding_window=args.sliding_window).with_segment_ids(segment_ids)
    cotangent = jax.random.normal(jax.random.PRNGKey(2), q.shape, dtype=dtype)

    def batch_mask(segment_ids_arg):
        return AttentionMask.causal(sliding_window=args.sliding_window).with_segment_ids(segment_ids_arg)

    def ref_attention_with_segment_ids(q_arg, k_arg, v_arg, segment_ids_arg):
        ref_mask = batch_mask(segment_ids_arg) if args.dynamic_segment_ids else mask
        return reference_attention(q_arg, k_arg, v_arg, ref_mask, logits_dtype=jnp.float32)

    ref_jit = jax.jit(ref_attention_with_segment_ids)

    def ref_loss(q_arg, k_arg, v_arg, segment_ids_arg):
        out = ref_attention_with_segment_ids(q_arg, k_arg, v_arg, segment_ids_arg)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    ref_fwd_bwd_jit = jax.jit(jax.value_and_grad(ref_loss, argnums=(0, 1, 2)))

    print(
        "shape="
        f"B={args.batch} S={args.seq} Hq={args.q_heads} Hkv={args.kv_heads} D={args.head_dim} "
        f"dtype={args.dtype} segments={args.segments} sliding_window={args.sliding_window} "
        f"dynamic_segment_ids={args.dynamic_segment_ids}"
    )
    if not args.skip_reference_timing:
        _time_call("reference_xla", lambda: ref_jit(q, k, v, segment_ids), warmup=args.warmup, iters=args.iters)
        _time_call(
            "reference_xla_fwd_bwd",
            lambda: ref_fwd_bwd_jit(q, k, v, segment_ids),
            warmup=args.warmup,
            iters=args.iters,
        )

    implementations = (
        ["gpu_flex_pallas", "gpu_flex_pallas_reference_vjp", "gpu_flash_triton", "gpu_flash_mosaic"]
        if args.implementation == "all"
        else [args.implementation]
    )
    for implementation in implementations:
        try:
            _check_correctness(dtype, implementation, dynamic_segment_ids=args.dynamic_segment_ids)
            gpu_attention = _gpu_attention_fn(implementation)

            def gpu_attention_with_segment_ids(q_arg, k_arg, v_arg, segment_ids_arg, attention_fn=gpu_attention):
                gpu_mask = batch_mask(segment_ids_arg) if args.dynamic_segment_ids else mask
                return attention_fn(q_arg, k_arg, v_arg, gpu_mask)

            gpu_jit = jax.jit(gpu_attention_with_segment_ids)

            def gpu_loss(q_arg, k_arg, v_arg, segment_ids_arg, attention_fn=gpu_attention):
                out = gpu_attention_with_segment_ids(q_arg, k_arg, v_arg, segment_ids_arg, attention_fn)
                return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

            gpu_fwd_bwd_jit = jax.jit(jax.value_and_grad(gpu_loss, argnums=(0, 1, 2)))
            _time_call(
                implementation,
                lambda jit_fn=gpu_jit: jit_fn(q, k, v, segment_ids),
                warmup=args.warmup,
                iters=args.iters,
            )
            _time_call(
                f"{implementation}_fwd_bwd",
                lambda jit_fn=gpu_fwd_bwd_jit: jit_fn(q, k, v, segment_ids),
                warmup=args.warmup,
                iters=args.iters,
            )
        except Exception as exc:
            message = str(exc).splitlines()[0]
            print(f"{implementation}: FAIL {type(exc).__name__}: {message}")


if __name__ == "__main__":
    main()
