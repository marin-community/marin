# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the GB200 hero's RMSNorm-GatedNorm boundary.

The default ``[16, 4096, 6144]`` input is the exact device-local shape for the
one-rack FSDP hero: global batch 1024 divided over 64 GPUs, at sequence length
4096. The benchmark compares the current XLA algebra with the CODA-style QuACK
candidate for both forward and forward-plus-backward execution.

Run on a single reserved GB200 GPU::

    uv run python -m experiments.grug.moe_hero_fsdp.bench_rms_gated_norm

Stdout is JSONL. Human-readable progress and the comparison summary go to
stderr, so stdout can be redirected directly to an artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from levanter.grug._moe.rms_gated_norm import (
    exact_gated_norm_up_reverse,
    exact_rms_backward_consumer_reference,
)

_KERNEL = "grug_moe_rms_gated_norm"
_HERO_BATCH_PER_DEVICE = 16
_HERO_SEQUENCE_LENGTH = 4096
_HERO_HIDDEN_DIM = 6144
_GATED_NORM_RANK = 128
_LAYER_NORM_EPS = 1e-5
_BACKEND_ENV_PREFIXES = ("CUDA_", "CUTLASS_", "JAX_", "NCCL_", "QUACK_", "XLA_")

ArrayCallable = Callable[..., jax.Array]


def _xla_current(
    x: jax.Array,
    norm_weight: jax.Array,
    w_down: jax.Array,
    w_up: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    """Current model algebra: GatedNorm(RMSNorm(x))."""
    x_float = x.astype(jnp.float32)
    inverse_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_float), axis=-1, keepdims=True) + eps)
    normalized = (x_float * inverse_rms * norm_weight).astype(x.dtype)
    gate_hidden = jax.nn.silu(jnp.einsum("...d,dr->...r", normalized, w_down))
    gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, w_up))
    return normalized * gate.astype(x.dtype)


def _coda_reference(
    x: jax.Array,
    norm_weight: jax.Array,
    w_down: jax.Array,
    w_up: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    """Vanilla JAX oracle for the delayed inverse-RMS CODA algebra."""
    inverse_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1) + eps)
    scaled_w_down = (norm_weight[:, None] * w_down).astype(x.dtype)
    gate_hidden_acc = jnp.einsum(
        "...d,dr->...r",
        x,
        scaled_w_down,
        preferred_element_type=jnp.float32,
    )
    gate_hidden = jax.nn.silu(gate_hidden_acc * inverse_rms[..., None]).astype(x.dtype)
    gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, w_up))
    normalized = (x.astype(jnp.float32) * norm_weight * inverse_rms[..., None]).astype(x.dtype)
    return normalized * gate.astype(x.dtype)


def _quack_forward(
    x: jax.Array,
    norm_weight: jax.Array,
    w_down: jax.Array,
    w_up: jax.Array,
    *,
    eps: float,
    tile_mn: tuple[int, int],
    cluster_mnk: tuple[int, int, int],
    max_swizzle: int,
) -> jax.Array:
    del tile_mn, cluster_mnk, max_swizzle
    return _xla_current(x, norm_weight, w_down, w_up, eps=eps)


def _quack_candidate(
    *,
    eps: float,
    tile_mn: tuple[int, int],
    cluster_mnk: tuple[int, int, int],
    backward_tile_mn: tuple[int, int],
    backward_cluster_mnk: tuple[int, int, int],
    max_swizzle: int,
) -> ArrayCallable:
    """Build the candidate with the same analytic-VJP contract as the model path."""

    @jax.custom_vjp
    def candidate(x, norm_weight, w_down, w_up):
        return _quack_forward(
            x,
            norm_weight,
            w_down,
            w_up,
            eps=eps,
            tile_mn=tile_mn,
            cluster_mnk=cluster_mnk,
            max_swizzle=max_swizzle,
        )

    def candidate_fwd(x, norm_weight, w_down, w_up):
        original_shape = x.shape
        x_flat = x.reshape((-1, x.shape[-1]))
        inverse_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_flat.astype(jnp.float32)), axis=-1) + eps)
        normalized = (x_flat.astype(jnp.float32) * inverse_rms[:, None] * norm_weight).astype(x.dtype)
        gate_preactivation = jnp.einsum("td,dr->tr", normalized, w_down)
        silu_sigmoid = jax.nn.sigmoid(gate_preactivation)
        gate_hidden = gate_preactivation * silu_sigmoid
        gate = jax.nn.sigmoid(jnp.einsum("tr,rd->td", gate_hidden, w_up))
        output = (normalized * gate).reshape(original_shape)
        residuals = (
            x,
            norm_weight,
            w_down,
            w_up,
            inverse_rms,
            normalized,
            gate_preactivation,
            silu_sigmoid,
            gate_hidden,
            gate,
        )
        return output, residuals

    def candidate_bwd(residuals, output_cotangent):
        from levanter.grug._moe.quack_rms_cute import (  # noqa: PLC0415
            quack_coda_rms_backward_producer,
            quack_silu_backward_gemm,
        )

        x, norm_weight, w_down, w_up, inverse_rms, normalized, gate_preactivation, _, _, _ = residuals
        x_flat = x.reshape((-1, x.shape[-1]))
        direct_cotangent, gate_accumulator_cotangent, w_up_cotangent = exact_gated_norm_up_reverse(
            output_cotangent, residuals
        )
        gate_preactivation_cotangent, _ = quack_silu_backward_gemm(
            gate_accumulator_cotangent,
            w_up,
            gate_preactivation,
            tile_mn=tile_mn,
            cluster_mnk=cluster_mnk,
            max_swizzle=max_swizzle,
        )
        w_down_cotangent = jnp.einsum("td,tr->dr", normalized, gate_preactivation_cotangent)
        unweighted_cotangent, row_dot_partial = quack_coda_rms_backward_producer(
            gate_preactivation_cotangent,
            w_down,
            direct_cotangent,
            x_flat,
            norm_weight,
            inverse_rms,
            tile_mn=backward_tile_mn,
            cluster_mnk=backward_cluster_mnk,
            max_swizzle=max_swizzle,
        )
        normalized_x = x_flat.astype(jnp.float32) * inverse_rms[:, None]
        row_dot = jnp.sum(row_dot_partial, axis=-1)
        norm_weight_cotangent = jnp.sum(unweighted_cotangent.astype(jnp.float32) * normalized_x, axis=0).astype(
            norm_weight.dtype
        )
        x_cotangent = exact_rms_backward_consumer_reference(
            unweighted_cotangent,
            row_dot,
            x_flat,
            norm_weight,
            inverse_rms,
        ).reshape(x.shape)
        return x_cotangent, norm_weight_cotangent, w_down_cotangent, w_up_cotangent

    candidate.defvjp(candidate_fwd, candidate_bwd)
    return candidate


def _forward_backward(
    fn: ArrayCallable,
    x: jax.Array,
    norm_weight: jax.Array,
    w_down: jax.Array,
    w_up: jax.Array,
    output_cotangent: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    output, pullback = jax.vjp(fn, x, norm_weight, w_down, w_up)
    return output, pullback(output_cotangent)


def _time_jitted(
    fn: Callable[..., Any],
    *args: jax.Array,
    warmup: int,
    steps: int,
) -> tuple[float, float, Any]:
    start = time.perf_counter()
    output = fn(*args)
    jax.block_until_ready(output)
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        output = fn(*args)
        jax.block_until_ready(output)

    start = time.perf_counter()
    for _ in range(steps):
        output = fn(*args)
        jax.block_until_ready(output)
    steady_state_time = (time.perf_counter() - start) / steps
    return compile_time, steady_state_time, output


@jax.jit
def _array_deviation(candidate: jax.Array, baseline: jax.Array) -> tuple[jax.Array, ...]:
    absolute = jnp.abs(candidate.astype(jnp.float32) - baseline.astype(jnp.float32))
    candidate_norm = jnp.linalg.norm(candidate.astype(jnp.float32))
    baseline_norm = jnp.linalg.norm(baseline.astype(jnp.float32))
    relative_l2 = jnp.linalg.norm(absolute) / baseline_norm
    return jnp.max(absolute), jnp.mean(absolute), relative_l2, candidate_norm, baseline_norm


def _deviation(candidate: jax.Array, baseline: jax.Array) -> dict[str, float]:
    maximum, mean, relative_l2, candidate_norm, baseline_norm = jax.device_get(_array_deviation(candidate, baseline))
    return {
        "max_abs": float(maximum),
        "mean_abs": float(mean),
        "relative_l2": float(relative_l2),
        "candidate_l2": float(candidate_norm),
        "baseline_l2": float(baseline_norm),
    }


def _gradient_deviations(
    candidate: tuple[jax.Array, ...], baseline: tuple[jax.Array, ...]
) -> tuple[dict[str, dict[str, float]], float, float]:
    names = ("x", "norm_weight", "w_down", "w_up")
    per_gradient = {
        name: _deviation(candidate_gradient, baseline_gradient)
        for name, candidate_gradient, baseline_gradient in zip(names, candidate, baseline, strict=True)
    }
    maximum = max(stats["max_abs"] for stats in per_gradient.values())
    total_elements = sum(gradient.size for gradient in baseline)
    mean = (
        sum(stats["mean_abs"] * gradient.size for stats, gradient in zip(per_gradient.values(), baseline, strict=True))
        / total_elements
    )
    return per_gradient, maximum, mean


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _backend_env() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(_BACKEND_ENV_PREFIXES) and key != "XLA_FLAGS"
    }


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:2000]


def _base_row(
    *,
    implementation: str,
    implementation_family: str,
    pass_mode: str,
    shape: dict[str, int],
    dtype: jnp.dtype,
    backend: str,
    device_type: str,
    device_count: int,
    block_sizes: dict[str, Any],
    warmup: int,
    steps: int,
    git_sha: str,
    backend_env: dict[str, str],
) -> dict[str, Any]:
    return {
        "kernel": _KERNEL,
        "implementation": implementation,
        "implementation_family": implementation_family,
        "pass_mode": pass_mode,
        "shape": shape,
        "dtype": jnp.dtype(dtype).name,
        "backend": backend,
        "device_type": device_type,
        "device_count": device_count,
        "block_sizes": block_sizes,
        "compile_time": None,
        "steady_state_time": None,
        "time_unit": "seconds",
        "warmup": warmup,
        "steps": steps,
        "tokens_per_second": None,
        "speedup_vs_xla": None,
        "output_max_abs_deviation": None,
        "output_mean_abs_deviation": None,
        "gradient_max_abs_deviation": None,
        "gradient_mean_abs_deviation": None,
        "gradient_deviations": None,
        "error": None,
        "git_sha": git_sha,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": backend_env,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-per-device", type=int, default=_HERO_BATCH_PER_DEVICE)
    parser.add_argument("--sequence-length", type=int, default=_HERO_SEQUENCE_LENGTH)
    parser.add_argument("--hidden-dim", type=int, default=_HERO_HIDDEN_DIM)
    parser.add_argument("--rank", type=int, default=_GATED_NORM_RANK)
    parser.add_argument("--eps", type=float, default=_LAYER_NORM_EPS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--cluster-m", type=int, default=2)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--cluster-k", type=int, default=1)
    parser.add_argument("--backward-tile-m", type=int, default=256)
    parser.add_argument("--backward-tile-n", type=int, default=128)
    parser.add_argument("--backward-cluster-m", type=int, default=2)
    parser.add_argument("--backward-cluster-n", type=int, default=1)
    parser.add_argument("--backward-cluster-k", type=int, default=1)
    parser.add_argument("--max-swizzle", type=int, default=8)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument(
        "--allow-other-device",
        action="store_true",
        help="Allow smoke tests on a non-GB200 backend; the QuACK arm may still be unsupported.",
    )
    args = parser.parse_args(argv)
    for name in (
        "batch_per_device",
        "sequence_length",
        "hidden_dim",
        "rank",
        "warmup",
        "steps",
        "tile_m",
        "tile_n",
        "cluster_m",
        "cluster_n",
        "cluster_k",
        "backward_tile_m",
        "backward_tile_n",
        "backward_cluster_m",
        "backward_cluster_n",
        "backward_cluster_k",
        "max_swizzle",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    devices = jax.local_devices()
    if len(devices) != 1:
        raise RuntimeError(
            f"benchmark requires exactly one visible local device, found {len(devices)}; "
            "set CUDA_VISIBLE_DEVICES to one GB200"
        )
    backend = jax.default_backend()
    device_type = devices[0].device_kind
    if not args.allow_other_device and (backend != "gpu" or "gb200" not in device_type.lower()):
        raise RuntimeError(
            f"expected one GB200 GPU, found backend={backend!r}, device_type={device_type!r}; "
            "pass --allow-other-device only for smoke testing"
        )

    dtype = jnp.bfloat16
    tokens = args.batch_per_device * args.sequence_length
    shape = {
        "batch_per_device": args.batch_per_device,
        "sequence_length": args.sequence_length,
        "tokens": tokens,
        "hidden_dim": args.hidden_dim,
        "rank": args.rank,
    }
    tile_mn = (args.tile_m, args.tile_n)
    cluster_mnk = (args.cluster_m, args.cluster_n, args.cluster_k)
    backward_tile_mn = (args.backward_tile_m, args.backward_tile_n)
    backward_cluster_mnk = (args.backward_cluster_m, args.backward_cluster_n, args.backward_cluster_k)
    initializer_std = 0.5 / math.sqrt(args.hidden_dim)
    quack_block_sizes = {
        "tile_mn": list(tile_mn),
        "cluster_mnk": list(cluster_mnk),
        "backward_tile_mn": list(backward_tile_mn),
        "backward_cluster_mnk": list(backward_cluster_mnk),
        "max_swizzle": args.max_swizzle,
    }

    print(f"device: {device_type}; shape: {shape}; dtype: {jnp.dtype(dtype).name}", file=sys.stderr)
    print("initializing inputs", file=sys.stderr)
    key_x, key_norm, key_down, key_up, key_cotangent = jax.random.split(jax.random.key(args.seed), 5)
    x = jax.random.normal(
        key_x,
        (args.batch_per_device, args.sequence_length, args.hidden_dim),
        dtype=dtype,
    )
    norm_weight = 1.0 + 0.01 * jax.random.normal(key_norm, (args.hidden_dim,), dtype=jnp.float32)
    w_down = (jax.random.normal(key_down, (args.hidden_dim, args.rank), dtype=jnp.float32) * initializer_std).astype(
        dtype
    )
    w_up = (jax.random.normal(key_up, (args.rank, args.hidden_dim), dtype=jnp.float32) * initializer_std).astype(dtype)
    output_cotangent = jax.random.normal(
        key_cotangent,
        (args.batch_per_device, args.sequence_length, args.hidden_dim),
        dtype=dtype,
    )
    jax.block_until_ready((x, norm_weight, w_down, w_up, output_cotangent))

    def xla_fn(x, norm_weight, w_down, w_up):
        return _xla_current(x, norm_weight, w_down, w_up, eps=args.eps)

    quack_fn = _quack_candidate(
        eps=args.eps,
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        backward_tile_mn=backward_tile_mn,
        backward_cluster_mnk=backward_cluster_mnk,
        max_swizzle=args.max_swizzle,
    )
    variants = (("xla_current", xla_fn, {}), ("quack_coda", quack_fn, quack_block_sizes))

    git_sha = _git_sha()
    backend_env = _backend_env()
    rows: list[dict[str, Any]] = []
    outputs: dict[str, jax.Array] = {}
    gradients: dict[str, tuple[jax.Array, ...]] = {}

    for family, fn, block_sizes in variants:
        for pass_mode in ("forward", "forward_backward"):
            implementation = f"{family}_{pass_mode}"
            row = _base_row(
                implementation=implementation,
                implementation_family=family,
                pass_mode=pass_mode,
                shape=shape,
                dtype=dtype,
                backend=backend,
                device_type=device_type,
                device_count=len(devices),
                block_sizes=block_sizes,
                warmup=args.warmup,
                steps=args.steps,
                git_sha=git_sha,
                backend_env=backend_env,
            )
            print(f"benchmarking {implementation}", file=sys.stderr)
            try:
                jax.clear_caches()
                if pass_mode == "forward":
                    timed_fn = jax.jit(fn)
                    compile_time, steady_state_time, output = _time_jitted(
                        timed_fn,
                        x,
                        norm_weight,
                        w_down,
                        w_up,
                        warmup=args.warmup,
                        steps=args.steps,
                    )
                    outputs[family] = output
                else:

                    def forward_backward_fn(x, norm_weight, w_down, w_up, cotangent, *, _fn=fn):
                        return _forward_backward(_fn, x, norm_weight, w_down, w_up, cotangent)

                    timed_fn = jax.jit(forward_backward_fn)
                    compile_time, steady_state_time, forward_backward_output = _time_jitted(
                        timed_fn,
                        x,
                        norm_weight,
                        w_down,
                        w_up,
                        output_cotangent,
                        warmup=args.warmup,
                        steps=args.steps,
                    )
                    _, gradients[family] = forward_backward_output
                row.update(
                    {
                        "compile_time": compile_time,
                        "steady_state_time": steady_state_time,
                        "tokens_per_second": tokens / steady_state_time,
                    }
                )
            except Exception as error:  # pragma: no cover - accelerator/runtime dependent
                row["error"] = _error_text(error)
                print(f"{implementation} failed: {row['error']}", file=sys.stderr)
            rows.append(row)

    output_stats = None
    if "xla_current" in outputs and "quack_coda" in outputs:
        output_stats = _deviation(outputs["quack_coda"], outputs["xla_current"])
    gradient_stats = None
    gradient_maximum = None
    gradient_mean = None
    if "xla_current" in gradients and "quack_coda" in gradients:
        gradient_stats, gradient_maximum, gradient_mean = _gradient_deviations(
            gradients["quack_coda"], gradients["xla_current"]
        )

    xla_times = {
        row["pass_mode"]: row["steady_state_time"]
        for row in rows
        if row["implementation_family"] == "xla_current" and row["steady_state_time"] is not None
    }
    for row in rows:
        baseline_time = xla_times.get(row["pass_mode"])
        if baseline_time is not None and row["steady_state_time"] is not None:
            row["speedup_vs_xla"] = baseline_time / row["steady_state_time"]
        if row["implementation_family"] == "xla_current":
            row.update(
                {
                    "output_max_abs_deviation": 0.0,
                    "output_mean_abs_deviation": 0.0,
                    "gradient_max_abs_deviation": 0.0,
                    "gradient_mean_abs_deviation": 0.0,
                }
            )
        else:
            if output_stats is not None:
                row["output_max_abs_deviation"] = output_stats["max_abs"]
                row["output_mean_abs_deviation"] = output_stats["mean_abs"]
            if gradient_stats is not None:
                row["gradient_max_abs_deviation"] = gradient_maximum
                row["gradient_mean_abs_deviation"] = gradient_mean
                row["gradient_deviations"] = gradient_stats

    serialized_rows = [json.dumps(row, sort_keys=True) for row in rows]
    for serialized in serialized_rows:
        print(serialized)
    if args.output_jsonl is not None:
        args.output_jsonl.write_text("\n".join(serialized_rows) + "\n", encoding="utf-8")

    for row in rows:
        if row["steady_state_time"] is not None:
            print(
                f"{row['implementation']}: {row['steady_state_time'] * 1e3:.3f} ms, "
                f"{row['tokens_per_second']:.0f} tokens/s, speedup {row['speedup_vs_xla']:.3f}x",
                file=sys.stderr,
            )
    if output_stats is not None:
        print(
            f"output deviation: max={output_stats['max_abs']:.6g}, mean={output_stats['mean_abs']:.6g}",
            file=sys.stderr,
        )
    if gradient_stats is not None:
        print(
            f"gradient deviation: max={gradient_maximum:.6g}, mean={gradient_mean:.6g}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
