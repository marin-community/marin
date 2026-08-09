# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate runtime and phased EventTensorPlan lowerings on one CUDA GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections.abc import Callable, MutableMapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from benchmark_metadata import command_record, nvidia_smi_snapshot, toolchain_snapshot  # pyrefly: ignore[missing-import]
from torch.utils.cpp_extension import load

from tile_lifetime.cuda_dynamic_event_dataflow_codegen import (
    CudaDynamicEventLowering,
    CudaPhasedPipelineLowering,
    generate_cuda_phased_pipeline_lowering,
    generate_cuda_runtime_event_lowering,
)
from tile_lifetime.event_dataflow import (
    EventDataflowProgram,
    EventMemoryScope,
    EventSchedulingMode,
    derive_event_tensor_plan,
    event_tensor_runtime_inputs,
    execute_event_dataflow,
)
from tile_lifetime.event_dataflow_examples import (
    pipelined_contract_fold_program,
    relation_segment_dependence,
    single_dependence_event_program,
)
from tile_lifetime.relation import RelationPlan, build_relation_plan


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-items", type=int, default=2048)
    parser.add_argument("--route-slots", type=int, default=2)
    parser.add_argument("--destination-count", type=int, default=64)
    parser.add_argument("--active-destinations", type=int, default=48)
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--pipeline-depth", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--source-directory", type=Path, default=Path("/tmp/shuttle-dynamic-event-tensor"))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--holder-revision", required=True)
    parser.add_argument("--nvcc", default="nvcc")
    return parser.parse_args()


def _load_source(source: str, source_sha256: str, source_directory: Path, prefix: str) -> tuple[ModuleType, Path]:
    source_directory.mkdir(parents=True, exist_ok=True)
    module_name = f"{prefix}_{source_sha256[:16]}"
    source_path = source_directory / f"{module_name}.cu"
    build_directory = source_directory / f"build_{module_name}"
    build_directory.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source + "\n")
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


def _benchmark(functions: tuple[tuple[str, Callable[[], None]], ...], args: argparse.Namespace) -> dict[str, Any]:
    for _ in range(args.warmups):
        for _, function in functions:
            function()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {name: [] for name, _ in functions}
    execution_order: list[list[str]] = []
    for repeat in range(args.repeats):
        ordered = functions if repeat % 2 == 0 else tuple(reversed(functions))
        execution_order.append([name for name, _ in ordered])
        for name, function in ordered:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / args.iterations)
    return {
        "execution_order": execution_order,
        "variants": {
            name: {
                "samples_ms": values,
                "median_ms": statistics.median(values),
                "mean_ms": statistics.mean(values),
                "minimum_ms": min(values),
                "maximum_ms": max(values),
            }
            for name, values in samples.items()
        },
    }


def _trace_record(program: EventDataflowProgram) -> dict[str, Any]:
    def execute_task(_coordinate: tuple[int, ...], _state: MutableMapping[str, object]) -> None:
        return

    result = execute_event_dataflow(
        program,
        actions={family.name: execute_task for family in program.task_families},
        state={},
        scheduling_mode=EventSchedulingMode.DYNAMIC,
    )
    entries = [
        {
            "step": entry.step,
            "kind": entry.kind.value,
            "subject": entry.subject,
            "coordinate": list(entry.coordinate),
            "generation": entry.generation,
            "remaining": entry.remaining,
        }
        for entry in result.trace
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "entry_count": len(entries),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "head": entries[:16],
        "tail": entries[-16:],
    }


def _relation_plan(args: argparse.Namespace, *, seed: int) -> RelationPlan:
    rng = np.random.default_rng(seed)
    destination_indices = rng.integers(
        0,
        args.active_destinations,
        size=(args.source_items, args.route_slots),
        dtype=np.int32,
    )
    destination_order = rng.permutation(args.destination_count)
    destination_rank = np.zeros(args.destination_count, dtype=np.int32)
    destination_local = np.empty(args.destination_count, dtype=np.int32)
    destination_local[destination_order] = np.arange(args.destination_count, dtype=np.int32)
    return build_relation_plan(
        destination_indices,
        np.ones(destination_indices.shape, dtype=np.float32),
        destination_rank_by_item=destination_rank,
        destination_local_item_by_item=destination_local,
        padding_quantum=1,
    )


def _runtime_tensors(plan: RelationPlan) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, tuple[int, ...]]:
    dependence = relation_segment_dependence(plan, visibility_scope=EventMemoryScope.CTA)
    event_plan = derive_event_tensor_plan(dependence, name="runtime_segment_readiness")
    runtime = event_tensor_runtime_inputs(event_plan)
    counts = torch.tensor(runtime.event_initial_counts, dtype=torch.int32, device="cuda")
    offsets = torch.tensor(runtime.event_source_offsets, dtype=torch.int32, device="cuda")
    sources = torch.tensor(runtime.event_sources, dtype=torch.int32, device="cuda")
    return counts, offsets, sources, max(runtime.event_initial_counts), runtime.initially_ready_events


def _runtime_reference(
    input_tensor: torch.Tensor,
    offsets: torch.Tensor,
    sources: torch.Tensor,
) -> torch.Tensor:
    input_array = input_tensor.detach().cpu().numpy()
    offset_array = offsets.cpu().numpy()
    source_array = sources.cpu().numpy()
    output = np.zeros(offset_array.shape[0] - 1, dtype=np.float32)
    for event in range(output.shape[0]):
        accumulator = np.float32(0.0)
        for index in range(int(offset_array[event]), int(offset_array[event + 1])):
            accumulator = np.float32(accumulator + input_array[source_array[index]])
        output[event] = accumulator
    return torch.from_numpy(output).to(device="cuda")


def _run_runtime_case(
    module: ModuleType,
    plan: RelationPlan,
    input_tensor: torch.Tensor,
) -> dict[str, Any]:
    counts, offsets, sources, maximum_count, initially_ready = _runtime_tensors(plan)
    partials = torch.full_like(input_tensor, float("nan"))
    output = torch.full((counts.numel(),), float("nan"), dtype=torch.float32, device="cuda")
    module.run_runtime_counted_event_out(
        input_tensor,
        partials,
        output,
        counts,
        offsets,
        sources,
        maximum_count,
    )
    torch.cuda.synchronize()
    expected = _runtime_reference(input_tensor, offsets, sources)
    if not torch.equal(partials, input_tensor):
        raise RuntimeError("runtime relation producers did not write every partial exactly once")
    if not torch.equal(output, expected):
        raise RuntimeError(f"runtime relation output mismatch: maximum error {(output - expected).abs().max().item()}")
    if not torch.isfinite(output).all():
        raise RuntimeError("runtime relation output contains non-finite values")
    return {
        "event_counts": counts.cpu().tolist(),
        "event_source_offsets": offsets.cpu().tolist(),
        "initially_ready_events": list(initially_ready),
        "maximum_count": maximum_count,
        "output_sha256": _tensor_sha256(output),
        "bitwise_reference": True,
    }


def _phased_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    score = torch.einsum("gd,gkd->gk", query, key)
    probability = torch.softmax(score, dim=1)
    return torch.sum(probability * value, dim=1)


def _run_phased_case(
    module: ModuleType,
    *,
    generations: int,
    depth: int,
    dimension: int,
    seed: int,
) -> tuple[dict[str, Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn((generations, dimension), dtype=torch.float32, device="cuda", generator=generator) * 0.1
    key = torch.randn((generations, depth, dimension), dtype=torch.float32, device="cuda", generator=generator) * 0.1
    value = torch.randn((generations, depth), dtype=torch.float32, device="cuda", generator=generator)
    output = torch.full((generations,), float("nan"), dtype=torch.float32, device="cuda")
    module.run_phased_contract_fold_pipeline_out(query, key, value, output)
    torch.cuda.synchronize()
    expected = _phased_reference(query, key, value)
    maximum_error = (output - expected).abs().max().item()
    if not torch.allclose(output, expected, rtol=2e-5, atol=2e-5):
        raise RuntimeError(f"phased Contract/Fold output mismatch: maximum error {maximum_error}")
    first_hash = _tensor_sha256(output)
    module.run_phased_contract_fold_pipeline_out(query, key, value, output)
    torch.cuda.synchronize()
    if _tensor_sha256(output) != first_hash:
        raise RuntimeError("phased slot reuse is not deterministic across fresh invocations")
    return (
        {
            "generations": generations,
            "pipeline_depth": depth,
            "dimension": dimension,
            "maximum_absolute_error": maximum_error,
            "output_sha256": first_hash,
            "repeat_bitwise": True,
        },
        (query, key, value, output),
    )


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("dynamic Event Tensor validation requires a CUDA GPU")
    if args.active_destinations <= 0 or args.active_destinations >= args.destination_count:
        raise ValueError("active destinations must leave at least one empty runtime segment")
    if args.repeats <= 0 or args.repeats % 2:
        raise ValueError("counterbalanced repeats must be a positive even number")

    first_relation = _relation_plan(args, seed=20260809)
    second_relation = _relation_plan(args, seed=20260810)
    first_event_plan = derive_event_tensor_plan(
        relation_segment_dependence(first_relation, visibility_scope=EventMemoryScope.CTA),
        name="runtime_segment_readiness",
    )
    runtime_trace = _trace_record(
        single_dependence_event_program(
            relation_segment_dependence(first_relation, visibility_scope=EventMemoryScope.CTA),
            name="runtime_segment_readiness",
            scheduling_mode=EventSchedulingMode.DYNAMIC,
        )
    )
    generated_runtime: CudaDynamicEventLowering = generate_cuda_runtime_event_lowering(first_event_plan)
    runtime_module, runtime_source_path = _load_source(
        generated_runtime.source,
        generated_runtime.source_sha256,
        args.source_directory,
        "shuttle_runtime_event",
    )
    input_generator = torch.Generator(device="cuda").manual_seed(20260811)
    input_tensor = torch.randn(
        (generated_runtime.source_count,),
        dtype=torch.float32,
        device="cuda",
        generator=input_generator,
    )
    first_runtime = _run_runtime_case(runtime_module, first_relation, input_tensor)
    second_runtime = _run_runtime_case(runtime_module, second_relation, input_tensor)

    phased_program = pipelined_contract_fold_program(
        generation_count=args.generations,
        pipeline_depth=args.pipeline_depth,
    )
    generated_phased: CudaPhasedPipelineLowering = generate_cuda_phased_pipeline_lowering(phased_program)
    phased_trace = _trace_record(phased_program)
    phased_module, phased_source_path = _load_source(
        generated_phased.source,
        generated_phased.source_sha256,
        args.source_directory,
        "shuttle_phased_event",
    )
    phased_correctness, phased_tensors = _run_phased_case(
        phased_module,
        generations=args.generations,
        depth=args.pipeline_depth,
        dimension=args.dimension,
        seed=20260812,
    )
    mutation_depth = max(1, args.pipeline_depth // 2)
    phased_mutation, _ = _run_phased_case(
        phased_module,
        generations=args.generations + 1,
        depth=mutation_depth,
        dimension=args.dimension,
        seed=20260813,
    )

    first_counts, first_offsets, first_sources, first_maximum, _ = _runtime_tensors(first_relation)
    runtime_partials = torch.empty_like(input_tensor)
    runtime_output = torch.empty((first_counts.numel(),), dtype=torch.float32, device="cuda")

    def runtime_event() -> None:
        runtime_module.run_runtime_counted_event_out(
            input_tensor,
            runtime_partials,
            runtime_output,
            first_counts,
            first_offsets,
            first_sources,
            first_maximum,
        )

    query, key, value, phased_output = phased_tensors

    def phased_event() -> None:
        phased_module.run_phased_contract_fold_pipeline_out(query, key, value, phased_output)

    telemetry_before = nvidia_smi_snapshot()
    timing = _benchmark((("runtime_segment", runtime_event), ("phased_pipeline", phased_event)), args)
    telemetry_after = nvidia_smi_snapshot()
    observed_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "kind": "generated_dynamic_event_tensor_gpu_validation",
        "command": command_record(),
        "requested_shuttle_revision": args.shuttle_revision,
        "observed_shuttle_revision": observed_revision,
        "holder_revision": args.holder_revision,
        "runtime_relation": {
            "generated_source": {
                "path": str(runtime_source_path),
                "sha256": generated_runtime.source_sha256,
            },
            "first": first_runtime,
            "mutation": second_runtime,
            "same_compiled_module": True,
            "logical_execution_trace": runtime_trace,
            "visibility_assertion": {
                "scope": first_event_plan.memory_scope.value,
                "release_on_notify": first_event_plan.visibility.release_on_notify,
                "acquire_before_consumer": first_event_plan.visibility.acquire_before_consumer,
                "physical_realization": "cuda block barrier wait after producer arrivals",
                "zero_count_policy": "initially ready identity; no barrier initialized",
            },
        },
        "phased_pipeline": {
            "generated_source": {
                "path": str(phased_source_path),
                "sha256": generated_phased.source_sha256,
            },
            "primary": phased_correctness,
            "mutation": phased_mutation,
            "same_compiled_module": True,
            "logical_execution_trace": phased_trace,
            "generation_assertion": {
                "policy": "phased",
                "identity": "physical slot plus monotonically increasing generation",
                "release": "threadfence_block then atomic generation publish",
                "acquire": "atomic generation poll before shared-buffer read",
            },
        },
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
