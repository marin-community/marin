# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and compare native SM100 attention schedules on identical inputs."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from levanter.cutlass_kernel_cache import gpu_compute_capability
from levanter.grug.attention import AttentionMask, GrugAttentionImplementation, attention, reference_attention
from levanter.grug.attention._fa4_cute import _packed_segment_causal_lower_bounds
from levanter.grug.attention._fa4_cute_backend import (
    fa4_cute_attention_forward,
    segmented_flash_attention_backward,
    segmented_flash_attention_forward,
)
from levanter.grug.attention._fa4_cute_config import (
    Flash4CuteSm100ForwardConfig,
    flash4_cute_kernel_config,
)

VARIANTS = ("baseline", "forward")


def emit(record):
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    output = os.environ.get("IRIS_OUTPUT_DIR")
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "native-sm100.jsonl").open("a") as stream:
            stream.write(line + "\n")


def kernel_config(variant, q_stage):
    arch = gpu_compute_capability()
    if variant != "baseline" and arch != 100:
        raise ValueError(f"{variant} requires SM100, got SM{arch}.")
    config = flash4_cute_kernel_config(128, arch=arch)
    if variant == "forward":
        config = dataclasses.replace(config, sm100_forward=Flash4CuteSm100ForwardConfig((128, 128), q_stage))
    return config


def inputs(batch, sequence, kv_heads, iteration):
    shapes = ((batch, sequence, 48, 128), (batch, sequence, kv_heads, 128))
    keys = jax.random.split(jax.random.key(1700 + iteration), 4)
    return tuple(
        jax.random.normal(key, shape, dtype=jnp.bfloat16)
        for key, shape in zip(keys, (shapes[0], shapes[1], shapes[1], shapes[0]), strict=True)
    )


def bounds_for(ids, window):
    return _packed_segment_causal_lower_bounds(ids, batch_size=ids.shape[0], seq_len=ids.shape[1], sliding_window=window)


def check(variant, sequence, window, kv_heads, q_stage, implementation: GrugAttentionImplementation | None = None):
    config = kernel_config(variant, q_stage)

    def actual_loss(q, k, v, cotangent, ids):
        lower, valid = bounds_for(ids, window)
        if implementation is None:
            out = fa4_cute_attention_forward(q, k, v, lower, valid, kernel_config=config)
        else:
            out = attention(
                q, k, v, AttentionMask.causal(sliding_window=window).with_segment_ids(ids), implementation=implementation
            )
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), out

    def reference_loss(q, k, v, cotangent, ids):
        mask = AttentionMask.causal(sliding_window=window).with_segment_ids(ids)
        out = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
        out = jnp.where((ids >= 0)[..., None, None], out, 0)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), out

    actual_call = jax.jit(jax.value_and_grad(actual_loss, (0, 1, 2), has_aux=True))
    reference_call = jax.jit(jax.value_and_grad(reference_loss, (0, 1, 2), has_aux=True))
    batch = 2 if sequence < 1024 else 1
    for iteration in range(3):
        boundaries = np.array([101] if sequence > 2048 else [31, 129, 193])
        ids = np.stack(
            [np.searchsorted(boundaries + iteration + row * 7, np.arange(sequence)) for row in range(batch)]
        ).astype(np.int32)
        ids[:, : 19 + iteration] = -1
        ids[:, -17:] = -1
        if batch == 2 and iteration == 2:
            ids[1] = -1
        args = (*inputs(batch, sequence, kv_heads, iteration), jnp.asarray(ids))
        (_, expected), expected_grad = reference_call(*args)
        start = time.perf_counter()
        (_, actual), actual_grad = actual_call(*args)
        jax.block_until_ready((actual, actual_grad))
        metrics = {}
        for name, got, want in zip(
            ("out", "dq", "dk", "dv"), (actual, *actual_grad), (expected, *expected_grad), strict=True
        ):
            got, want = np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32)
            diff = np.abs(got - want)
            metrics[name] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
            np.testing.assert_allclose(got, want, atol=7e-2, rtol=7e-2, err_msg=name)
            np.testing.assert_array_equal(got[ids < 0], 0, err_msg=name)
        emit(
            {
                "kind": "correctness",
                "variant": variant,
                "sequence": sequence,
                "window": window,
                "kv_heads": kv_heads,
                "iteration": iteration,
                "metrics": metrics,
                "compile_and_execute_seconds": time.perf_counter() - start,
            }
        )


def measure(call, args, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        jax.block_until_ready(call(*args))
    return (time.perf_counter() - start) / iterations


def benchmark(variants, batch, sequence, window, kv_heads, documents, q_stage, iterations, rounds):
    q, k, v, cotangent = inputs(batch, sequence, kv_heads, 42)
    boundaries = np.linspace(0, sequence, documents + 1, dtype=np.int32)[1:-1]
    ids = jnp.asarray(
        np.stack([np.searchsorted(boundaries + row % 64, np.arange(sequence), side="right") for row in range(batch)]),
        dtype=jnp.int32,
    )
    lower, valid = bounds_for(ids, window)
    scale = 128**-0.5
    calls = {}
    for variant in variants:
        config = kernel_config(variant, q_stage)
        forward = jax.jit(
            lambda q, k, v, lower, valid, cfg=config: segmented_flash_attention_forward(
                q, k, v, lower, valid, softmax_scale=scale, kernel_config=cfg
            )
        )
        backward = jax.jit(
            lambda q, k, v, out, dout, lse, lower, valid, cfg=config: segmented_flash_attention_backward(
                q, k, v, out, dout, lse, lower, valid, softmax_scale=scale, kernel_config=cfg
            )
        )

        def loss(q, k, v, cotangent, lower, valid, cfg=config):
            out = fa4_cute_attention_forward(q, k, v, lower, valid, kernel_config=cfg)
            return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

        total = jax.jit(jax.value_and_grad(loss, (0, 1, 2)))
        out, lse = forward(q, k, v, lower, valid)
        forward_args = (q, k, v, lower, valid)
        backward_args = (q, k, v, out, cotangent, lse, lower, valid)
        total_args = (q, k, v, cotangent, lower, valid)
        calls[variant] = ((forward, forward_args), (backward, backward_args), (total, total_args))
        for call, args in calls[variant]:
            for _ in range(3):
                jax.block_until_ready(call(*args))
        compiled = total.lower(*total_args).compile()
        memory = compiled.memory_analysis()
        emit(
            {
                "kind": "compiled",
                "variant": variant,
                "batch": batch,
                "sequence": sequence,
                "window": window,
                "kv_heads": kv_heads,
                "documents": documents,
                "temporary_bytes": memory.temp_size_in_bytes,
                "argument_bytes": memory.argument_size_in_bytes,
                "output_bytes": memory.output_size_in_bytes,
            }
        )
    order = list(variants)
    rng = random.Random(71)
    for round_index in range(rounds):
        rng.shuffle(order)
        for variant in order:
            metrics = {
                name: measure(call, args, iterations)
                for name, (call, args) in zip(("forward", "backward", "total"), calls[variant], strict=True)
            }
            emit(
                {
                    "kind": "timing",
                    "variant": variant,
                    "batch": batch,
                    "sequence": sequence,
                    "window": window,
                    "kv_heads": kv_heads,
                    "documents": documents,
                    "round": round_index,
                    "seconds": metrics,
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "check", "frontend", "bench"), required=True)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--q-stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=6)
    parser.add_argument("--documents", type=int, default=5)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if gpu_compute_capability() != 100 and args.variant != "baseline":
        raise ValueError("Select --variant baseline to validate the existing backend on other architectures.")
    source = Path(__file__).resolve().parents[3] / "lib/levanter/src/levanter/grug/attention"
    emit(
        {
            "kind": "environment",
            "devices": [{"device": str(d), "kind": d.device_kind} for d in jax.devices()],
            "compute_capability": gpu_compute_capability(),
            "benchmark_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "packages": {
                p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "flash-attn-4", "nvidia-cutlass-dsl")
            },
            "source_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("_fa4*.py"))
            },
            "arguments": vars(args),
        }
    )
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    if args.mode == "frontend":
        for sequence, window in ((257, None), (257, 31), (2305, 2048)):
            for kv_heads in (6, 12):
                check("forward", sequence, window, kv_heads, 2, implementation="gpu_fa4_cute_sm100")
    elif args.mode in ("smoke", "check"):
        cases = (
            ((257, 31, 12),)
            if args.mode == "smoke"
            else tuple(
                (sequence, window, kv) for sequence, window in ((257, None), (257, 31), (2305, 2048)) for kv in (6, 12)
            )
        )
        for variant in variants:
            for sequence, window, kv in cases:
                check(variant, sequence, window, kv, args.q_stage)
    else:
        benchmark(
            variants,
            args.batch,
            args.sequence,
            args.window or None,
            args.kv_heads,
            args.documents,
            args.q_stage,
            args.iterations,
            args.rounds,
        )
    emit({"kind": "complete", "mode": args.mode})


if __name__ == "__main__":
    main()
