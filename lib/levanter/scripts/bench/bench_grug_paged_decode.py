# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure one paged decode attention layer, excluding cache updates and model work."""

import argparse
import json
import importlib
import os
import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
from rigging.provenance import launch_provenance

from levanter.grug.attention import ragged_paged_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["reference", "tpu", "gpu_pallas"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--window", type=int)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--kv-splits", type=int, choices=[8, 16], default=8)
    parser.add_argument("--baseline", choices=["flashinfer_xqa", "tpu_vllm_rpa", "tpu_vllm_rpa_fp32"])
    args = parser.parse_args()
    if args.baseline in ("tpu_vllm_rpa", "tpu_vllm_rpa_fp32"):
        # The optional fork sets its environment before initializing JAX devices.
        importlib.import_module("tpu_inference.kernels.ragged_paged_attention.v3.kernel")
    dtype = jnp.dtype(args.dtype)
    pages_per_sequence = (args.context + args.page_size - 1) // args.page_size
    page_count = args.batch_size * pages_per_sequence
    q_key, kv_key = jax.random.split(jax.random.key(42))
    q = jax.random.normal(q_key, (args.batch_size, args.kv_heads, args.groups, args.head_dim), dtype)
    cache = jax.random.normal(kv_key, (page_count, args.page_size, 2 * args.kv_heads, args.head_dim), dtype)
    inputs = (
        q,
        cache,
        jnp.full((args.batch_size,), args.context, jnp.int32),
        jnp.arange(page_count, dtype=jnp.int32).reshape(args.batch_size, pages_per_sequence),
        jnp.arange(args.batch_size + 1, dtype=jnp.int32),
        jnp.array(args.batch_size, jnp.int32),
    )
    jax.block_until_ready(inputs)
    fn = jax.jit(
        partial(
            ragged_paged_attention,
            sm_scale=args.head_dim**-0.5,
            sliding_window=args.window,
            implementation=args.implementation,
            gpu_kv_splits=args.kv_splits,
        )
    )
    start = time.perf_counter()
    compiled = fn.lower(*inputs).compile()
    compile_time = time.perf_counter() - start
    start = time.perf_counter()
    actual = compiled(*inputs).block_until_ready()
    first_run_time = time.perf_counter() - start
    for _ in range(3):
        compiled(*inputs).block_until_ready()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        compiled(*inputs).block_until_ready()
        times.append(time.perf_counter() - start)
    elapsed = statistics.median(times)
    visible = args.context if args.window is None else min(args.context, args.window)
    kv_bytes = 2 * args.batch_size * visible * args.kv_heads * args.head_dim * dtype.itemsize
    flops = 4 * args.batch_size * visible * args.kv_heads * args.groups * args.head_dim
    print(
        json.dumps(
            {
                "kernel": "grug_paged_decode",
                "timing_boundary": "host_submit_and_synchronize",
                "implementation": args.implementation,
                "shape": vars(args),
                "dtype": args.dtype,
                "backend": jax.default_backend(),
                "device_type": jax.devices()[0].device_kind,
                "device_count": 1,
                "block_sizes": (
                    {"page_size": args.page_size, "kv_splits": min(args.kv_splits, pages_per_sequence)}
                    if args.implementation == "gpu_pallas"
                    else None
                ),
                "compile_time": compile_time,
                "first_run_time": first_run_time,
                "steady_state_time": elapsed,
                "error": None,
                "git_sha": launch_provenance().base_commit,
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
                "logical_kv_bytes": kv_bytes,
                "logical_kv_bandwidth_gb": kv_bytes / elapsed / 1e9,
                "qk_av_flops": flops,
                "qk_av_arithmetic_intensity": flops / kv_bytes,
            }
        ),
        flush=True,
    )

    if args.baseline is not None:
        baseline = (
            _flashinfer_baseline(inputs, actual, args)
            if args.baseline == "flashinfer_xqa"
            else _tpu_vllm_baseline(inputs, actual, args)
        )
        print(json.dumps(baseline), flush=True)


def _flashinfer_baseline(inputs, expected, args):
    # FlashInfer is optional and installed only in the isolated GPU benchmark environment.
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    if jax.default_backend() != "gpu" or args.dtype != "bfloat16":
        raise ValueError("The FlashInfer XQA comparison requires a GPU and BF16 inputs")
    q, cache, lengths, table, _, _ = inputs
    query = torch.from_dlpack(q).reshape(args.batch_size, args.kv_heads * args.groups, args.head_dim)
    cache_torch = torch.from_dlpack(cache)
    keys, values = cache_torch[:, :, 0::2, :], cache_torch[:, :, 1::2, :]
    block_tables, seq_lens = torch.from_dlpack(table), torch.from_dlpack(lengths)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=query.device)
    output = torch.empty_like(query)
    fn = partial(
        flashinfer.decode.xqa_batch_decode_with_kv_cache,
        query,
        (keys, values),
        workspace,
        block_tables,
        seq_lens,
        args.context,
        bmm1_scale=args.head_dim**-0.5,
        window_left=-1 if args.window is None else args.window - 1,
        kv_layout="NHD",
        out=output,
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first_run = time.perf_counter() - start
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    expected_torch = torch.from_dlpack(expected).reshape_as(output)
    difference = (output.float() - expected_torch.float()).abs()
    return {
        "kernel": "flashinfer_xqa",
        "implementation": "flashinfer_xqa",
        "shape": vars(args),
        "dtype": args.dtype,
        "device_type": torch.cuda.get_device_name(),
        "device_count": 1,
        "backend": "cuda",
        "block_sizes": {"page_size": args.page_size},
        "compile_time": None,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {"CUDA_HOME": os.environ.get("CUDA_HOME", "")},
        "timing_boundary": "host_submit_and_synchronize",
        "steady_state_time": statistics.median(times),
        "first_run_including_jit_time": first_run,
        "error": {"max_abs_vs_grug": difference.max().item(), "mean_abs_vs_grug": difference.mean().item()},
        "flashinfer_version": flashinfer.__version__,
        "torch_version": torch.__version__,
        "git_sha": launch_provenance().base_commit,
        "cache_layout": "interleaved_strided_views_no_copy",
    }


def _tpu_vllm_baseline(inputs, expected, args):
    rpa = importlib.import_module("tpu_inference.kernels.ragged_paged_attention.v3.kernel")
    if jax.default_backend() != "tpu":
        raise ValueError("The vLLM RPA comparison requires TPU")
    accumulator_dtype = jnp.float32 if args.baseline == "tpu_vllm_rpa_fp32" else jnp.bfloat16

    def attend(q, cache, lengths, table, offsets, num_seqs):
        query = q.reshape(args.batch_size, args.kv_heads * args.groups, args.head_dim)
        current_kv = jnp.zeros((args.batch_size, args.kv_heads, args.head_dim), cache.dtype)
        shape = rpa.get_kv_cache_shape(cache.shape[0], args.page_size, args.kv_heads, args.head_dim, cache.dtype)
        output, _ = rpa.ragged_paged_attention(
            query,
            current_kv,
            current_kv,
            cache.reshape(shape),
            lengths,
            table.reshape(-1),
            offsets,
            jnp.repeat(num_seqs[None], 3),
            update_kv_cache=False,
            sm_scale=args.head_dim**-0.5,
            sliding_window=args.window,
            out_dtype=accumulator_dtype,
        )
        return output.reshape(q.shape).astype(q.dtype)

    # The outer wrapper owns no donated inputs; repeated timings reuse immutable cache/query arrays.
    start = time.perf_counter()
    compiled = jax.jit(attend).lower(*inputs).compile()
    compile_time = time.perf_counter() - start
    start = time.perf_counter()
    actual = compiled(*inputs).block_until_ready()
    first_run = time.perf_counter() - start
    for _ in range(3):
        compiled(*inputs).block_until_ready()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        compiled(*inputs).block_until_ready()
        times.append(time.perf_counter() - start)
    difference = jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))
    return {
        "kernel": "tpu_vllm_rpa_v3",
        "implementation": args.baseline,
        "shape": vars(args),
        "dtype": args.dtype,
        "accumulator_dtype": str(accumulator_dtype),
        "backend": "tpu",
        "device_type": jax.devices()[0].device_kind,
        "device_count": 1,
        "block_sizes": "fork_default",
        "timing_boundary": "host_submit_and_synchronize",
        "compile_time": compile_time,
        "first_run_time": first_run,
        "steady_state_time": statistics.median(times),
        "error": {"max_abs_vs_grug": float(difference.max()), "mean_abs_vs_grug": float(difference.mean())},
        "git_sha": launch_provenance().base_commit,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
        "cache_update": False,
    }


if __name__ == "__main__":
    main()
