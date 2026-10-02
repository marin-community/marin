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
from importlib import metadata
from typing import NamedTuple
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np
from rigging.provenance import launch_provenance

from levanter.grug.attention import ragged_paged_attention
from levanter.testing.precision import round_to_bfloat16

WARMUP_STEPS = 3
HOST_TIMING_BOUNDARY = "host_submit_and_synchronize"
MAX_ERROR_EXAMPLES = 8
ERROR_ATOL = 1e-4
ERROR_RTOL = 1e-4
TPU_RPA_MODULE = "tpu_inference.kernels.ragged_paged_attention.v3.kernel"


class _JaxMeasurements(NamedTuple):
    output: jax.Array
    compile_time: float
    first_run_time: float
    steady_state_time: float


def _jax_runtime_versions():
    packages = ("jax", "jaxlib", "numpy")
    if jax.default_backend() == "tpu":
        packages += ("libtpu",)
    return {name: metadata.version(name) for name in packages}


def _measure_jax(fn, inputs, repeats):
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*inputs).compile()
    compile_time = time.perf_counter() - start
    start = time.perf_counter()
    output = compiled(*inputs).block_until_ready()
    first_run_time = time.perf_counter() - start
    for _ in range(WARMUP_STEPS):
        compiled(*inputs).block_until_ready()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        compiled(*inputs).block_until_ready()
        times.append(time.perf_counter() - start)
    return _JaxMeasurements(output, compile_time, first_run_time, statistics.median(times))


def _profile_tpu(fn, inputs, repeats):
    # Use the XPlane reader also used by bench_ce_hero_shape. Module events
    # measure device execution without summing nested per-operation events.
    with TemporaryDirectory(prefix="grug-tpu-profile-") as directory:
        with jax.profiler.trace(directory):
            for step in range(repeats):
                with jax.profiler.StepTraceAnnotation("paged_decode", step_num=step):
                    jax.block_until_ready(fn(*inputs))
        paths = sorted(Path(directory).rglob("*.xplane.pb"))
        if not paths:
            raise RuntimeError("TPU profiler did not produce an XPlane trace")
        lines = []
        module_times = []
        for path in paths:
            data = jax.profiler.ProfileData.from_file(str(path))
            for plane in data.planes:
                if "tpu" not in plane.name.lower():
                    continue
                for line in plane.lines:
                    durations = [event.duration_ns / 1e9 for event in line.events]
                    if not durations:
                        continue
                    median = statistics.median(durations)
                    lines.append(
                        {
                            "plane": plane.name,
                            "line": line.name,
                            "events": len(durations),
                            "median_time": median,
                            "total_time": sum(durations),
                        }
                    )
                    if line.name == "XLA Modules" and len(durations) == repeats:
                        module_times.append(median)
        return {
            "timing_boundary": "tpu_xplane_xla_module_duration",
            "median_time": max(module_times) if module_times else None,
            "availability": "measured" if module_times else "no_matching_module_line",
            "device_lines": lines,
        }


def _error_metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    difference = np.abs(actual - expected)
    outside = difference > ERROR_ATOL + ERROR_RTOL * np.abs(expected)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "elements_outside_atol_rtol_1e4": int(outside.sum()),
        "elements": actual.size,
        "mismatch_examples": [
            {"index": index.tolist(), "actual": float(actual[tuple(index)]), "expected": float(expected[tuple(index)])}
            for index in np.argwhere(outside)[:MAX_ERROR_EXAMPLES]
        ],
    }


def _mismatch_oracle(inputs, args, comparisons, actual, reference):
    q, cache, _, table, _, _ = inputs
    indices = sorted({tuple(row["index"]) for comparison in comparisons for row in comparison["mismatch_examples"]})
    rows = []
    begin = 0 if args.window is None else max(0, args.context - args.window)
    scale = np.float64(np.float32(args.head_dim**-0.5))
    for batch, head, group, dim in indices:
        query = np.asarray(q[batch, head, group], np.float64)
        keys = cache[table[batch], :, 2 * head, :].reshape(-1, args.head_dim)[begin : args.context]
        values = cache[table[batch], :, 2 * head + 1, dim].reshape(-1)[begin : args.context]
        scores = np.asarray(keys, np.float64) @ query * scale
        probabilities = np.exp(scores - scores.max())
        expected = probabilities @ np.asarray(values, np.float64) / probabilities.sum()
        rounded = (
            float(round_to_bfloat16(np.asarray(expected)))
            if q.dtype == jnp.bfloat16
            else float(np.asarray(expected, dtype=q.dtype))
        )
        index = (batch, head, group, dim)
        tolerance = ERROR_ATOL + ERROR_RTOL * abs(rounded)
        rows.append(
            {
                "index": list(index),
                "float64_expected": float(expected),
                "rounded_expected": rounded,
                "actual": float(actual[index]),
                "reference": float(reference[index]),
                "actual_outside_tolerance": abs(float(actual[index]) - rounded) > tolerance,
                "reference_outside_tolerance": abs(float(reference[index]) - rounded) > tolerance,
            }
        )
    return {
        "coverage": f"up to {MAX_ERROR_EXAMPLES} mismatches from each pairwise comparison",
        "checked_coordinates": len(rows),
        "actual_failures": sum(row["actual_outside_tolerance"] for row in rows),
        "reference_failures": sum(row["reference_outside_tolerance"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        choices=["reference", "tpu", "tpu_fp32_tiles", "gpu_pallas", "gpu_pallas_bf16_3x"],
        required=True,
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--window", type=int)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--av-precision", choices=["ieee", "bf16_3x"], default="ieee")
    parser.add_argument("--profile-device", action="store_true")
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument("--kv-splits", type=int, choices=[8, 16], default=8)
    parser.add_argument("--baseline", choices=["flashinfer_xqa", "tpu_vllm_rpa", "tpu_vllm_rpa_fp32"])
    args = parser.parse_args()
    if args.implementation == "gpu_pallas_bf16_3x":
        args.av_precision = "bf16_3x"
    if args.baseline in ("tpu_vllm_rpa", "tpu_vllm_rpa_fp32"):
        # The optional fork sets its environment before initializing JAX devices.
        importlib.import_module(TPU_RPA_MODULE)
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
            gpu_av_precision=args.av_precision,
        )
    )
    measurements = _measure_jax(fn, inputs, args.repeats)
    actual = measurements.output
    elapsed = measurements.steady_state_time
    error = None
    if args.check_reference:
        reference = jax.jit(
            partial(
                ragged_paged_attention,
                sm_scale=args.head_dim**-0.5,
                sliding_window=args.window,
                implementation="reference",
            )
        )(*inputs)
        error = {"vs_reference": _error_metrics(actual, reference)}
        if args.implementation in ("gpu_pallas", "gpu_pallas_bf16_3x") and args.av_precision == "bf16_3x":
            ieee = jax.jit(
                partial(
                    ragged_paged_attention,
                    sm_scale=args.head_dim**-0.5,
                    sliding_window=args.window,
                    implementation="gpu_pallas",
                    gpu_kv_splits=args.kv_splits,
                    gpu_av_precision="ieee",
                )
            )(*inputs)
            error["ieee_vs_reference"] = _error_metrics(ieee, reference)
            error["vs_ieee"] = _error_metrics(actual, ieee)
        error["mismatch_float64_oracle"] = _mismatch_oracle(inputs, args, list(error.values()), actual, reference)
    device_profile = None
    if args.profile_device and jax.default_backend() == "tpu":
        device_profile = _profile_tpu(fn, inputs, args.repeats)
    elif args.profile_device:
        profiler = importlib.import_module("jax.experimental.mosaic.gpu.profiler")

        _, kernel_runs = profiler.measure(fn, aggregate=False, iterations=args.repeats)(*inputs)
        if kernel_runs is None:
            raise RuntimeError("CUPTI did not capture GPU kernels")
        if args.repeats == 1:
            kernel_runs = [kernel_runs]
        device_profile = {
            "timing_boundary": "cupti_sum_of_kernel_durations",
            "median_time": statistics.median(sum(duration for _, duration in run) for run in kernel_runs) / 1000,
            "kernel_runs_ms": kernel_runs,
        }
    visible = args.context if args.window is None else min(args.context, args.window)
    kv_bytes = 2 * args.batch_size * visible * args.kv_heads * args.head_dim * dtype.itemsize
    flops = 4 * args.batch_size * visible * args.kv_heads * args.groups * args.head_dim
    print(
        json.dumps(
            {
                "kernel": "grug_paged_decode",
                "timing_boundary": HOST_TIMING_BOUNDARY,
                "implementation": args.implementation,
                "shape": vars(args),
                "dtype": args.dtype,
                "backend": jax.default_backend(),
                "device_type": jax.devices()[0].device_kind,
                "device_count": 1,
                "block_sizes": (
                    {
                        "page_size": args.page_size,
                        "kv_splits": min(args.kv_splits, 1 << (pages_per_sequence - 1).bit_length()),
                    }
                    if args.implementation in ("gpu_pallas", "gpu_pallas_bf16_3x")
                    else None
                ),
                "compile_time": measurements.compile_time,
                "first_run_time": measurements.first_run_time,
                "steady_state_time": elapsed,
                "error": error,
                "device_profile": device_profile,
                "git_sha": launch_provenance().base_commit,
                "jax_runtime_versions": _jax_runtime_versions(),
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
    for _ in range(WARMUP_STEPS):
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
        "timing_boundary": HOST_TIMING_BOUNDARY,
        "steady_state_time": statistics.median(times),
        "first_run_including_jit_time": first_run,
        "error": {"max_abs_vs_grug": difference.max().item(), "mean_abs_vs_grug": difference.mean().item()},
        "flashinfer_version": flashinfer.__version__,
        "torch_version": torch.__version__,
        "git_sha": launch_provenance().base_commit,
        "cache_layout": "interleaved_strided_views_no_copy",
    }


def _tpu_vllm_baseline(inputs, expected, args):
    rpa = importlib.import_module(TPU_RPA_MODULE)
    if jax.default_backend() != "tpu":
        raise ValueError("The vLLM RPA comparison requires TPU")
    package = metadata.distribution("tpu-inference")
    source = json.loads(package.read_text("direct_url.json") or "{}")
    accumulator_dtype = jnp.float32 if args.baseline == "tpu_vllm_rpa_fp32" else jnp.bfloat16

    def attend(q, cache, lengths, table, offsets, num_seqs):
        # RPA reuses its query DMA buffers for output, so their packing must match.
        query = q.astype(accumulator_dtype).reshape(args.batch_size, args.kv_heads * args.groups, args.head_dim)
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
    measurements = _measure_jax(attend, inputs, args.repeats)
    actual = measurements.output
    difference = jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))
    return {
        "kernel": "tpu_vllm_rpa_v3",
        "tpu_inference_version": package.version,
        "tpu_inference_commit": source.get("vcs_info", {}).get("commit_id"),
        "implementation": args.baseline,
        "shape": vars(args),
        "dtype": args.dtype,
        "accumulator_dtype": str(jnp.dtype(accumulator_dtype)),
        "query_dtype": str(jnp.dtype(accumulator_dtype)),
        "cache_dtype": args.dtype,
        "kernel_output_dtype": str(jnp.dtype(accumulator_dtype)),
        "compared_output_dtype": args.dtype,
        "backend": "tpu",
        "device_type": jax.devices()[0].device_kind,
        "device_count": 1,
        "block_sizes": "fork_default",
        "timing_boundary": HOST_TIMING_BOUNDARY,
        "compile_time": measurements.compile_time,
        "first_run_time": measurements.first_run_time,
        "steady_state_time": measurements.steady_state_time,
        "error": {"max_abs_vs_grug": float(difference.max()), "mean_abs_vs_grug": float(difference.mean())},
        "git_sha": launch_provenance().base_commit,
        "jax_runtime_versions": _jax_runtime_versions(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": {"LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", "")},
        "cache_update": False,
    }


if __name__ == "__main__":
    main()
