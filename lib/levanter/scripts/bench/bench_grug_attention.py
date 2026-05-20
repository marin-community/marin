# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug.attention import AttentionMask, GrugAttentionImplementation, attention, reference_attention

Mode = Literal["forward", "forward_backward"]


@dataclass(frozen=True, slots=True)
class Shape:
    batch: int
    seq_len: int
    q_heads: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    implementation: str
    mode: Mode
    batch: int
    seq_len: int
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    sliding_window: int | None
    max_segments_per_seq: int
    compile_s: float
    steady_s: float
    tokens_per_s: float
    max_abs_vs_reference: float | None = None
    grad_max_abs_vs_reference: float | None = None
    comparison_implementation: str | None = None
    max_abs_vs_comparison: float | None = None
    grad_max_abs_vs_comparison: float | None = None


def _dtype_from_name(name: str) -> jnp.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if normalized in {"fp16", "float16"}:
        return jnp.float16
    if normalized in {"fp32", "float32"}:
        return jnp.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _make_segment_ids(
    *,
    batch: int,
    seq_len: int,
    target_segment_len: int,
    padding_tokens: int,
) -> tuple[jax.Array, int]:
    if target_segment_len <= 0:
        raise ValueError(f"target_segment_len must be positive, got {target_segment_len}")
    if padding_tokens < 0 or padding_tokens >= seq_len:
        raise ValueError(f"padding_tokens must be in [0, {seq_len}), got {padding_tokens}")

    ids = np.full((batch, seq_len), -1, dtype=np.int32)
    # Non-block-aligned lengths exercise the dynamic segment path that the data loader produces.
    length_pattern = np.array(
        [
            max(1, target_segment_len - 37),
            target_segment_len + 19,
            max(1, target_segment_len // 2 + 11),
            target_segment_len + 73,
        ],
        dtype=np.int32,
    )
    max_segments = 0
    usable_seq_len = seq_len - padding_tokens
    for batch_idx in range(batch):
        pos = 0
        segment_idx = 0
        while pos < usable_seq_len:
            length = int(length_pattern[(segment_idx + batch_idx) % len(length_pattern)])
            end = min(usable_seq_len, pos + length)
            ids[batch_idx, pos:end] = batch_idx * 1_000_000 + segment_idx
            pos = end
            segment_idx += 1
        max_segments = max(max_segments, segment_idx)
    return jnp.asarray(ids), max_segments


def _segment_slices(segment_ids: jax.Array) -> tuple[tuple[int, int, int], ...]:
    segment_ids_host = np.asarray(jax.device_get(segment_ids))
    slices = []
    for batch_idx, row in enumerate(segment_ids_host):
        start = None
        current_id = None
        for pos, segment_id in enumerate(row):
            if segment_id < 0:
                if start is not None:
                    slices.append((batch_idx, start, pos))
                    start = None
                    current_id = None
                continue
            if start is None:
                start = pos
                current_id = segment_id
                continue
            if segment_id != current_id:
                slices.append((batch_idx, start, pos))
                start = pos
                current_id = segment_id
        if start is not None:
            slices.append((batch_idx, start, row.shape[0]))
    return tuple(slices)


def _make_inputs(shape: Shape, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (shape.batch, shape.seq_len, shape.q_heads, shape.head_dim), dtype=dtype)
    k = jax.random.normal(k_key, (shape.batch, shape.seq_len, shape.kv_heads, shape.head_dim), dtype=dtype)
    v = jax.random.normal(v_key, (shape.batch, shape.seq_len, shape.kv_heads, shape.head_dim), dtype=dtype)
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=dtype)
    return q, k, v, cotangent


def _block_until_ready(value):
    leaves = jax.tree.leaves(value)
    for leaf in leaves:
        leaf.block_until_ready()
    return value


def _time_jitted(fn, *args, steps: int, warmup: int) -> tuple[float, float]:
    start = time.perf_counter()
    out = fn(*args)
    _block_until_ready(out)
    compile_s = time.perf_counter() - start

    for _ in range(warmup):
        out = fn(*args)
        _block_until_ready(out)

    start = time.perf_counter()
    for _ in range(steps):
        out = fn(*args)
        _block_until_ready(out)
    steady_s = (time.perf_counter() - start) / steps
    return compile_s, steady_s


def _mask(segment_ids: jax.Array, *, sliding_window: int | None, max_segments_per_seq: int) -> AttentionMask:
    return AttentionMask.causal(
        sliding_window=sliding_window, max_segments_per_seq=max_segments_per_seq
    ).with_segment_ids(segment_ids)


def _forward_fn(implementation: GrugAttentionImplementation, *, sliding_window: int | None, max_segments_per_seq: int):
    def forward(q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: jax.Array) -> jax.Array:
        return attention(
            q,
            k,
            v,
            _mask(segment_ids, sliding_window=sliding_window, max_segments_per_seq=max_segments_per_seq),
            implementation=implementation,
        )

    return jax.jit(forward)


def _forward_backward_fn(
    implementation: GrugAttentionImplementation,
    *,
    sliding_window: int | None,
    max_segments_per_seq: int,
):
    def loss(q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: jax.Array, cotangent: jax.Array) -> jax.Array:
        out = attention(
            q,
            k,
            v,
            _mask(segment_ids, sliding_window=sliding_window, max_segments_per_seq=max_segments_per_seq),
            implementation=implementation,
        )
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))


def _segmented_reference_fn(segment_ids: jax.Array, *, sliding_window: int | None):
    slices = _segment_slices(segment_ids)
    segment_mask = AttentionMask.causal(sliding_window=sliding_window)

    def ref(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        out = jnp.zeros((*q.shape[:3], v.shape[-1]), dtype=v.dtype)
        for batch_idx, start, end in slices:
            segment_out = reference_attention(
                q[batch_idx : batch_idx + 1, start:end],
                k[batch_idx : batch_idx + 1, start:end],
                v[batch_idx : batch_idx + 1, start:end],
                segment_mask,
                logits_dtype=jnp.float32,
            )
            out = out.at[batch_idx : batch_idx + 1, start:end].set(segment_out)
        return out

    return ref


def _max_abs(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def _reference_diffs(
    implementation: GrugAttentionImplementation,
    shape: Shape,
    *,
    dtype: jnp.dtype,
    segment_len: int,
    padding_tokens: int,
    sliding_window: int | None,
    check_grad: bool,
) -> tuple[float, float | None]:
    q, k, v, cotangent = _make_inputs(shape, dtype)
    segment_ids, max_segments = _make_segment_ids(
        batch=shape.batch,
        seq_len=shape.seq_len,
        target_segment_len=segment_len,
        padding_tokens=padding_tokens,
    )
    valid = segment_ids >= 0
    cotangent = cotangent * valid[..., None, None].astype(dtype)
    mask = _mask(segment_ids, sliding_window=sliding_window, max_segments_per_seq=max_segments)
    ref = _segmented_reference_fn(segment_ids, sliding_window=sliding_window)

    def actual(q_arg: jax.Array, k_arg: jax.Array, v_arg: jax.Array) -> jax.Array:
        return attention(q_arg, k_arg, v_arg, mask, implementation=implementation)

    ref_out = jax.jit(ref)(q, k, v)
    actual_out = jax.jit(actual)(q, k, v)
    ref_out, actual_out = _block_until_ready((ref_out, actual_out))
    value_max_abs = _max_abs(
        jnp.where(valid[..., None, None], actual_out, ref_out),
        ref_out,
    )

    grad_max_abs = None
    if check_grad:

        def ref_loss(q_arg: jax.Array, k_arg: jax.Array, v_arg: jax.Array) -> jax.Array:
            return jnp.sum(ref(q_arg, k_arg, v_arg).astype(jnp.float32) * cotangent.astype(jnp.float32))

        def actual_loss(q_arg: jax.Array, k_arg: jax.Array, v_arg: jax.Array) -> jax.Array:
            return jnp.sum(actual(q_arg, k_arg, v_arg).astype(jnp.float32) * cotangent.astype(jnp.float32))

        ref_grads = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))(q, k, v)
        actual_grads = jax.jit(jax.grad(actual_loss, argnums=(0, 1, 2)))(q, k, v)
        ref_grads, actual_grads = _block_until_ready((ref_grads, actual_grads))
        grad_max_abs = max(
            _max_abs(actual_grad, ref_grad) for actual_grad, ref_grad in zip(actual_grads, ref_grads, strict=True)
        )
    return value_max_abs, grad_max_abs


def _implementation_diffs(
    implementation: GrugAttentionImplementation,
    comparison_implementation: GrugAttentionImplementation,
    shape: Shape,
    *,
    dtype: jnp.dtype,
    segment_len: int,
    padding_tokens: int,
    sliding_window: int | None,
    check_grad: bool,
) -> tuple[float, float | None]:
    q, k, v, cotangent = _make_inputs(shape, dtype)
    segment_ids, max_segments = _make_segment_ids(
        batch=shape.batch,
        seq_len=shape.seq_len,
        target_segment_len=segment_len,
        padding_tokens=padding_tokens,
    )
    valid = segment_ids >= 0
    cotangent = cotangent * valid[..., None, None].astype(dtype)
    mask = _mask(segment_ids, sliding_window=sliding_window, max_segments_per_seq=max_segments)
    segmented_ref = _segmented_reference_fn(segment_ids, sliding_window=sliding_window)

    def call(
        impl: GrugAttentionImplementation,
        q_arg: jax.Array,
        k_arg: jax.Array,
        v_arg: jax.Array,
    ) -> jax.Array:
        if impl == "reference":
            return segmented_ref(q_arg, k_arg, v_arg)
        return attention(q_arg, k_arg, v_arg, mask, implementation=impl)

    def baseline(q_arg: jax.Array, k_arg: jax.Array, v_arg: jax.Array) -> jax.Array:
        return call(comparison_implementation, q_arg, k_arg, v_arg)

    def actual(q_arg: jax.Array, k_arg: jax.Array, v_arg: jax.Array) -> jax.Array:
        return call(implementation, q_arg, k_arg, v_arg)

    baseline_out = jax.jit(baseline)(q, k, v)
    actual_out = jax.jit(actual)(q, k, v)
    baseline_out, actual_out = _block_until_ready((baseline_out, actual_out))
    value_max_abs = _max_abs(
        jnp.where(valid[..., None, None], actual_out, baseline_out),
        baseline_out,
    )

    grad_max_abs = None
    if check_grad:

        def baseline_loss(q_arg: jax.Array, k_arg: jax.Array, v_arg: jax.Array) -> jax.Array:
            return jnp.sum(baseline(q_arg, k_arg, v_arg).astype(jnp.float32) * cotangent.astype(jnp.float32))

        def actual_loss(q_arg: jax.Array, k_arg: jax.Array, v_arg: jax.Array) -> jax.Array:
            return jnp.sum(actual(q_arg, k_arg, v_arg).astype(jnp.float32) * cotangent.astype(jnp.float32))

        baseline_grads = jax.jit(jax.grad(baseline_loss, argnums=(0, 1, 2)))(q, k, v)
        actual_grads = jax.jit(jax.grad(actual_loss, argnums=(0, 1, 2)))(q, k, v)
        baseline_grads, actual_grads = _block_until_ready((baseline_grads, actual_grads))
        grad_max_abs = max(
            _max_abs(actual_grad, baseline_grad)
            for actual_grad, baseline_grad in zip(actual_grads, baseline_grads, strict=True)
        )
    return value_max_abs, grad_max_abs


def _check_tolerance(
    *,
    implementation: str,
    label: str,
    value_max_abs: float | None,
    grad_max_abs: float | None,
    atol: float,
) -> None:
    if value_max_abs is not None and value_max_abs > atol:
        raise RuntimeError(f"{implementation} exceeded {label} tolerance {atol}: value max abs {value_max_abs}")
    if grad_max_abs is None or grad_max_abs <= atol:
        return
    raise RuntimeError(
        f"{implementation} exceeded {label} tolerance {atol}: "
        f"value max abs {value_max_abs}, grad max abs {grad_max_abs}"
    )


def _benchmark_implementation(
    implementation: GrugAttentionImplementation,
    shape: Shape,
    *,
    dtype: jnp.dtype,
    dtype_name: str,
    segment_len: int,
    padding_tokens: int,
    sliding_window: int | None,
    steps: int,
    warmup: int,
    reference_check_shape: Shape | None,
    reference_atol: float,
    comparison_implementation: GrugAttentionImplementation | None,
    comparison_atol: float,
    modes: tuple[Mode, ...],
) -> list[BenchmarkResult]:
    q, k, v, cotangent = _make_inputs(shape, dtype)
    segment_ids, max_segments = _make_segment_ids(
        batch=shape.batch,
        seq_len=shape.seq_len,
        target_segment_len=segment_len,
        padding_tokens=padding_tokens,
    )
    valid = segment_ids >= 0
    cotangent = cotangent * valid[..., None, None].astype(dtype)
    tokens = int(jnp.sum(valid))
    check_grad = "forward_backward" in modes

    value_max_abs = None
    grad_max_abs = None
    if reference_check_shape is not None:
        value_max_abs, grad_max_abs = _reference_diffs(
            implementation,
            reference_check_shape,
            dtype=dtype,
            segment_len=min(segment_len, max(1, reference_check_shape.seq_len // 4)),
            padding_tokens=min(padding_tokens, max(0, reference_check_shape.seq_len // 8)),
            sliding_window=sliding_window,
            check_grad=check_grad,
        )
        _check_tolerance(
            implementation=implementation,
            label="reference",
            value_max_abs=value_max_abs,
            grad_max_abs=grad_max_abs,
            atol=reference_atol,
        )

    comparison_value_max_abs = None
    comparison_grad_max_abs = None
    if comparison_implementation == implementation:
        comparison_value_max_abs = 0.0
        comparison_grad_max_abs = 0.0 if check_grad else None
    elif comparison_implementation is not None:
        comparison_value_max_abs, comparison_grad_max_abs = _implementation_diffs(
            implementation,
            comparison_implementation,
            shape,
            dtype=dtype,
            segment_len=segment_len,
            padding_tokens=padding_tokens,
            sliding_window=sliding_window,
            check_grad=check_grad,
        )
        _check_tolerance(
            implementation=implementation,
            label=f"{comparison_implementation} comparison",
            value_max_abs=comparison_value_max_abs,
            grad_max_abs=comparison_grad_max_abs,
            atol=comparison_atol,
        )

    results = []
    if "forward" in modes:
        forward = _forward_fn(implementation, sliding_window=sliding_window, max_segments_per_seq=max_segments)
        forward_compile, forward_steady = _time_jitted(forward, q, k, v, segment_ids, steps=steps, warmup=warmup)
        results.append(
            BenchmarkResult(
                implementation=implementation,
                mode="forward",
                batch=shape.batch,
                seq_len=shape.seq_len,
                q_heads=shape.q_heads,
                kv_heads=shape.kv_heads,
                head_dim=shape.head_dim,
                dtype=dtype_name,
                sliding_window=sliding_window,
                max_segments_per_seq=max_segments,
                compile_s=forward_compile,
                steady_s=forward_steady,
                tokens_per_s=tokens / forward_steady,
                max_abs_vs_reference=value_max_abs,
                grad_max_abs_vs_reference=grad_max_abs,
                comparison_implementation=comparison_implementation,
                max_abs_vs_comparison=comparison_value_max_abs,
                grad_max_abs_vs_comparison=comparison_grad_max_abs,
            )
        )

    if "forward_backward" in modes:
        forward_backward = _forward_backward_fn(
            implementation,
            sliding_window=sliding_window,
            max_segments_per_seq=max_segments,
        )
        fwd_bwd_compile, fwd_bwd_steady = _time_jitted(
            forward_backward,
            q,
            k,
            v,
            segment_ids,
            cotangent,
            steps=steps,
            warmup=warmup,
        )
        results.append(
            BenchmarkResult(
                implementation=implementation,
                mode="forward_backward",
                batch=shape.batch,
                seq_len=shape.seq_len,
                q_heads=shape.q_heads,
                kv_heads=shape.kv_heads,
                head_dim=shape.head_dim,
                dtype=dtype_name,
                sliding_window=sliding_window,
                max_segments_per_seq=max_segments,
                compile_s=fwd_bwd_compile,
                steady_s=fwd_bwd_steady,
                tokens_per_s=tokens / fwd_bwd_steady,
                max_abs_vs_reference=value_max_abs,
                grad_max_abs_vs_reference=grad_max_abs,
                comparison_implementation=comparison_implementation,
                max_abs_vs_comparison=comparison_value_max_abs,
                grad_max_abs_vs_comparison=comparison_grad_max_abs,
            )
        )

    return results


def _parse_modes(value: str) -> tuple[Mode, ...]:
    modes = tuple(mode.strip() for mode in value.split(",") if mode.strip())
    allowed = {"forward", "forward_backward"}
    unknown = set(modes) - allowed
    if unknown:
        raise ValueError(f"Unknown benchmark mode(s): {sorted(unknown)}")
    if not modes:
        raise ValueError("At least one benchmark mode is required.")
    return modes  # type: ignore[return-value]


def _default_implementations() -> list[str]:
    if jax.default_backend() != "gpu":
        return ["reference"]
    return ["gpu_te"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Grug packed-segment attention implementations.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--segment-len", type=int, default=256)
    parser.add_argument("--padding-tokens", type=int, default=0)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--implementations", default=None)
    parser.add_argument(
        "--modes",
        default="forward,forward_backward",
        help="Comma-separated benchmark modes: forward,forward_backward. Use forward for FA4/CuTe until backward lands.",
    )
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument(
        "--reference-seq-len",
        type=int,
        default=None,
        help=(
            "Sequence length for the reference check. Defaults to the benchmark sequence length; "
            "the reference is computed segment-by-segment, so 8192 checks avoid full [S,S] masks."
        ),
    )
    parser.add_argument("--reference-atol", type=float, default=1e-3)
    parser.add_argument("--comparison-implementation", default=None)
    parser.add_argument("--comparison-atol", type=float, default=1e-3)
    args = parser.parse_args()

    dtype = _dtype_from_name(args.dtype)
    shape = Shape(
        batch=args.batch,
        seq_len=args.seq_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )
    if args.implementations is None:
        implementations = _default_implementations()
    else:
        implementations = [name.strip() for name in args.implementations.split(",") if name.strip()]
    comparison_implementation = None
    if args.comparison_implementation is not None:
        comparison_implementation = args.comparison_implementation.strip()
    modes = _parse_modes(args.modes)

    reference_shape = None
    if args.check_reference:
        reference_seq_len = (
            args.seq_len if args.reference_seq_len is None else min(args.seq_len, args.reference_seq_len)
        )
        reference_shape = Shape(
            batch=args.batch,
            seq_len=reference_seq_len,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
        )

    print(
        json.dumps(
            {
                "devices": [str(device) for device in jax.devices()],
                "default_backend": jax.default_backend(),
                "shape": asdict(shape),
                "implementations": implementations,
                "modes": modes,
                "comparison_implementation": comparison_implementation,
            }
        )
    )
    for implementation in implementations:
        results = _benchmark_implementation(
            implementation,  # type: ignore[arg-type]
            shape,
            dtype=dtype,
            dtype_name=args.dtype,
            segment_len=args.segment_len,
            padding_tokens=args.padding_tokens,
            sliding_window=args.sliding_window,
            steps=args.steps,
            warmup=args.warmup,
            reference_check_shape=reference_shape,
            reference_atol=args.reference_atol,
            comparison_implementation=comparison_implementation,  # type: ignore[arg-type]
            comparison_atol=args.comparison_atol,
            modes=modes,
        )
        for result in results:
            print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
