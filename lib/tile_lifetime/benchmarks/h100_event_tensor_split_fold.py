# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark generated CTA EventTensorPlan readiness against two controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from benchmark_metadata import command_record, nvidia_smi_snapshot, toolchain_snapshot
from torch.utils.cpp_extension import load

from tile_lifetime.cuda_event_dataflow_codegen import (
    CudaEventCounterLowering,
    generate_cuda_event_counter_lowering,
)
from tile_lifetime.event_dataflow import EventMemoryScope, derive_event_tensor_plan
from tile_lifetime.event_dataflow_examples import split_fold_dependence


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--partitions", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--source-directory", type=Path, default=Path("/tmp/shuttle-event-tensor"))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--nvcc", default="nvcc")
    return parser.parse_args()


def _load_generated(
    generated: CudaEventCounterLowering,
    source_directory: Path,
) -> tuple[ModuleType, Path]:
    source_directory.mkdir(parents=True, exist_ok=True)
    module_name = f"shuttle_event_tensor_{generated.source_sha256[:16]}"
    source_path = source_directory / f"{module_name}.cu"
    build_directory = source_directory / f"build_{module_name}"
    build_directory.mkdir(parents=True, exist_ok=True)
    source_path.write_text(generated.source + "\n")
    module = load(
        name=module_name,
        sources=[str(source_path)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--threads=4", "-lineinfo", "--ptxas-options=-v"],
        build_directory=str(build_directory),
        with_cuda=True,
        verbose=True,
    )
    return module, source_path


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual - expected).abs()
    return {
        "maximum_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
    }


def _source_ordered_reference(input_tensor: torch.Tensor, *, rows: int, partitions: int) -> torch.Tensor:
    input_array = input_tensor.detach().cpu().numpy().reshape(rows, partitions)
    output = np.zeros((rows,), dtype=np.float32)
    for partition in range(partitions):
        output = np.asarray(output + input_array[:, partition], dtype=np.float32)
    return torch.from_numpy(output).to(device=input_tensor.device)


def _benchmark(
    variants: tuple[tuple[str, Callable[[], None]], ...],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        for _, function in variants:
            function()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name, _ in variants}
    execution_order: list[list[str]] = []
    for repeat in range(repeats):
        ordered = variants if repeat % 2 == 0 else tuple(reversed(variants))
        execution_order.append([name for name, _ in ordered])
        for name, function in ordered:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / iterations)
    records = {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "minimum_ms": min(values),
            "maximum_ms": max(values),
        }
        for name, values in samples.items()
    }
    event_median = float(records["counted_event"]["median_ms"])
    return {
        "variants": records,
        "execution_order": execution_order,
        "counted_event_to_block_barrier": event_median / float(records["block_barrier"]["median_ms"]),
        "counted_event_to_kernel_boundary": event_median / float(records["kernel_boundary"]["median_ms"]),
    }


def _coprime_strides(partitions: int) -> tuple[int, ...]:
    candidates = (1, 3, 5, 7, 11, 17, 31)
    return tuple(value for value in candidates if math.gcd(value, partitions) == 1)


def _correctness(
    module: ModuleType,
    input_tensor: torch.Tensor,
    partials: torch.Tensor,
    output: torch.Tensor,
    *,
    rows: int,
    partitions: int,
) -> dict[str, Any]:
    source_ordered = _source_ordered_reference(input_tensor, rows=rows, partitions=partitions)
    tree_reduced = input_tensor.reshape(rows, partitions).sum(dim=1)
    records: list[dict[str, Any]] = []
    first_hash: str | None = None
    for order_offset, order_stride in ((0, _coprime_strides(partitions)[0]), (13, _coprime_strides(partitions)[-1])):
        partials.fill_(float("nan"))
        output.fill_(float("nan"))
        module.run_counted_event_out(input_tensor, partials, output, order_offset, order_stride, 997)
        torch.cuda.synchronize()
        output_hash = _tensor_sha256(output)
        if first_hash is None:
            first_hash = output_hash
        finite = bool(torch.isfinite(output).all().item())
        partials_match = torch.equal(partials, input_tensor)
        source_order_match = torch.equal(output, source_ordered)
        if not finite:
            raise RuntimeError("counted-event output contains a non-finite value")
        if not partials_match:
            raise RuntimeError("counted-event producers did not write every required partial exactly once")
        if not source_order_match:
            error = _error(output, source_ordered)
            raise RuntimeError(f"counted-event output does not match source-ordered FP32 reference: {error}")
        if first_hash != output_hash:
            raise RuntimeError("producer-order perturbation changed the source-ordered output")
        records.append(
            {
                "order_offset": order_offset,
                "order_stride": order_stride,
                "delay_cycles": 997,
                "source_ordered_error": _error(output, source_ordered),
                "torch_tree_reduction_error": _error(output, tree_reduced),
                "output_sha256": output_hash,
                "matches_first_order_bitwise": output_hash == first_hash,
                "partials_match_input_bitwise": partials_match,
                "finite": finite,
            }
        )
    module.run_counted_event_out(input_tensor, partials, output, 5, _coprime_strides(partitions)[0], 0)
    torch.cuda.synchronize()
    repeat_hash = _tensor_sha256(output)
    module.run_counted_event_out(input_tensor, partials, output, 5, _coprime_strides(partitions)[0], 0)
    torch.cuda.synchronize()
    repeat_bitwise = repeat_hash == _tensor_sha256(output)
    if not repeat_bitwise:
        raise RuntimeError("fresh per-invocation event storage produced a nondeterministic output")
    return {
        "perturbed_orders": records,
        "source_ordered_reference_sha256": _tensor_sha256(source_ordered),
        "torch_tree_reduction_sha256": _tensor_sha256(tree_reduced),
        "fresh_event_storage_each_invocation": True,
        "repeat_bitwise": repeat_bitwise,
    }


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("the CUDA Event Tensor benchmark requires a CUDA device")
    if args.rows <= 0 or args.partitions <= 0:
        raise ValueError("rows and partitions must be positive")
    if args.partitions > 1024:
        raise ValueError("the first CUDA Event Tensor skeleton supports at most 1024 partitions")
    if args.repeats <= 0 or args.repeats % 2:
        raise ValueError("counterbalanced repeats must be a positive even number")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    strides = _coprime_strides(args.partitions)
    if not strides:
        raise ValueError("no configured producer-order stride is coprime with this partition count")

    dependence = split_fold_dependence(
        row_count=args.rows,
        partition_count=args.partitions,
        visibility_scope=EventMemoryScope.CTA,
    )
    plan = derive_event_tensor_plan(dependence, name="generated_gpu_readiness")
    generated = generate_cuda_event_counter_lowering(plan)
    module, source_path = _load_generated(generated, args.source_directory)

    generator = torch.Generator(device="cuda").manual_seed(20260809)
    input_tensor = torch.randn(
        (generated.source_count,),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    partials = torch.empty_like(input_tensor)
    output = torch.empty((generated.consumer_count,), dtype=torch.float32, device="cuda")
    correctness = _correctness(
        module,
        input_tensor,
        partials,
        output,
        rows=args.rows,
        partitions=args.partitions,
    )

    order_stride = strides[-1]

    def counted_event() -> None:
        module.run_counted_event_out(input_tensor, partials, output, 13, order_stride, 0)

    def block_barrier() -> None:
        module.run_block_barrier_control_out(input_tensor, partials, output, 13, order_stride, 0)

    def kernel_boundary() -> None:
        module.run_kernel_boundary_control_out(input_tensor, partials, output, 13, order_stride, 0)

    telemetry_before = nvidia_smi_snapshot()
    timing = _benchmark(
        (
            ("counted_event", counted_event),
            ("block_barrier", block_barrier),
            ("kernel_boundary", kernel_boundary),
        ),
        warmups=args.warmups,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    telemetry_after = nvidia_smi_snapshot()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "kind": "generated_event_tensor_split_fold_gpu",
        "command": command_record(),
        "requested_shuttle_revision": args.shuttle_revision,
        "observed_shuttle_revision": revision,
        "shape": {"rows": args.rows, "partitions": args.partitions},
        "plan": {
            "fingerprint": generated.plan_fingerprint,
            "event_count": generated.event_count,
            "source_count": generated.source_count,
            "consumer_count": generated.consumer_count,
            "threads_per_block": generated.threads_per_block,
            "initial_counts": generated.event_initial_counts,
            "memory_scope": generated.memory_scope.value,
            "generation_policy": generated.generation_policy.value,
            "visibility": {
                "release_on_notify": plan.visibility.release_on_notify,
                "acquire_before_consumer": plan.visibility.acquire_before_consumer,
            },
        },
        "generated_source": {
            "path": str(source_path),
            "sha256": generated.source_sha256,
        },
        "kernel_resources": {
            "field_order": ["registers_per_thread", "static_shared_bytes", "local_bytes", "max_threads_per_block"],
            "counted_event": module.counted_event_attributes(),
            "block_barrier": module.block_barrier_attributes(),
        },
        "correctness": correctness,
        "timing": timing,
        "toolchain": toolchain_snapshot(args.nvcc),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "before": telemetry_before,
            "after": telemetry_after,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
