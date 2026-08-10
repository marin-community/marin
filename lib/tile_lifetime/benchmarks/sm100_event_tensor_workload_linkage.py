# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay Event Tensor-linked payload kernels through Torch-free JAX FFI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from shuttle.ir import DType
from tile_lifetime.cuda_event_workload_codegen import (
    EventLinkedCudaFfi,
    evaluate_segmented_contract_event,
    evaluate_streaming_contract_fold_event,
    generate_segmented_contract_event_ffi,
    generate_streaming_contract_fold_event_ffi,
)
from tile_lifetime.event_dataflow_adapters import (
    relation_segmented_contract_task_dataflow,
    streaming_fold_task_dataflow,
)
from tile_lifetime.jax_event_dataflow_ffi import (
    call_cuda_segmented_contract_ffi,
    call_cuda_streaming_contract_fold_ffi,
    compile_cuda_event_ffi,
    register_cuda_event_ffi,
)
from tile_lifetime.relation import RelationPlan, build_relation_plan
from tile_lifetime.streaming_attention import (
    StreamingTileSchedule,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)

_GPU_QUERY_FIELDS = (
    "name",
    "uuid",
    "compute_cap",
    "driver_version",
    "power.limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.max.sm",
    "clocks.max.memory",
    "pstate",
    "persistence_mode",
)


def _gpu_record() -> dict[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = output.splitlines()
    if len(rows) != 1:
        raise ValueError(f"expected exactly one visible GPU, got {len(rows)}")
    values = tuple(value.strip() for value in rows[0].split(","))
    if len(values) != len(_GPU_QUERY_FIELDS):
        raise ValueError(f"expected {len(_GPU_QUERY_FIELDS)} GPU fields, got {len(values)}")
    return dict(zip(_GPU_QUERY_FIELDS, values, strict=True))


def _toolchain_versions() -> dict[str, str]:
    selected: dict[str, str] = {}
    jax_packages = {"jax", "jaxlib", "jax-cuda13-pjrt", "jax-cuda13-plugin"}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata["Name"])
        normalized = name.lower().replace("_", "-")
        if normalized in jax_packages or normalized.startswith("nvidia-"):
            selected[name] = distribution.version
    return dict(sorted(selected.items(), key=lambda item: item[0].lower()))


def _source_record() -> dict[str, object]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return {"revision": revision, "dirty": dirty}


def _relation(*, mutation: bool) -> RelationPlan:
    source_count = 64
    route_slots = 2
    destination_count = 8
    source = np.arange(source_count, dtype=np.int32)[:, None]
    slots = np.arange(route_slots, dtype=np.int32)[None, :]
    destination = (source * 3 + slots * 5 + (1 if mutation else 0)) % destination_count
    weights = np.ones((source_count, route_slots), dtype=np.float32)
    items = np.arange(destination_count, dtype=np.int32)
    return build_relation_plan(
        destination,
        weights,
        destination_rank_by_item=np.zeros(destination_count, dtype=np.int32),
        destination_local_item_by_item=items,
        padding_quantum=1,
    )


def _streaming_dataflow(*, key_length: int, pipeline_depth: int):
    semantic = build_attention_tensor_program(
        batch_size=1,
        query_length=8,
        key_length=key_length,
        query_heads=1,
        key_value_heads=1,
        key_dimension=16,
        value_dimension=8,
        score_map=scaled_score_map(0.25),
        input_dtype=DType.FP32,
    )
    streaming = derive_streaming_attention(
        semantic,
        schedule=StreamingTileSchedule(query_tile_size=4, key_value_tile_size=4, pipeline_depth=pipeline_depth),
    )
    return streaming_fold_task_dataflow(streaming)


def _measure(operation, *, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        operation().block_until_ready()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation().block_until_ready()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples


def _bind_operation(operation, arguments):
    def bound():
        return operation(*arguments)

    return bound


def _sample_record(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "mean_ms": statistics.mean(samples),
    }


def _audit_record(generated: EventLinkedCudaFfi) -> dict[str, object]:
    return {
        "plan_fingerprint": generated.ffi.plan_fingerprint,
        "source_sha256": generated.ffi.source_sha256,
        "device_source_sha256": generated.ffi.device_source_sha256,
        "typed_ffi_inputs": [
            {"name": value.name, "dtype": value.dtype.value, "shape": value.shape} for value in generated.ffi.inputs
        ],
        "typed_ffi_outputs": [
            {"name": value.name, "dtype": value.dtype.value, "shape": value.shape} for value in generated.ffi.outputs
        ],
        "event_realizations": [
            {
                "plan": entry.plan_name,
                "kind": entry.kind.value,
                "mechanism": entry.mechanism,
                "reason": entry.reason,
            }
            for entry in generated.event_audit.entries
        ],
        "physical_schedule": generated.physical_schedule,
    }


def _hlo_record(compiled, *arguments) -> dict[str, object]:
    text = compiled.lower(*arguments).compiler_ir(dialect="hlo").as_hlo_text()
    lines = text.splitlines()
    return {
        "custom_call_target_lines": [line.strip() for line in lines if "custom-call" in line],
        "constant_lines": sum(" constant(" in line for line in lines),
        "copy_lines": sum(" copy(" in line for line in lines),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _compile_segmented(
    *,
    relation: RelationPlan,
    mutation: bool,
    build_directory: Path,
    nvcc: Path,
    architecture: str,
):
    dataflow = relation_segmented_contract_task_dataflow(relation, output_tile_count=1)
    generated = generate_segmented_contract_event_ffi(
        dataflow,
        relation,
        reduction_dimension=32,
        output_dimension=16,
        target_name=f"shuttle.event_segmented_{'mutation' if mutation else 'primary'}",
    )
    library = compile_cuda_event_ffi(
        generated.ffi,
        directory=build_directory,
        nvcc=nvcc,
        architecture=architecture,
    )
    register_cuda_event_ffi(generated.ffi, library)
    counts = jnp.asarray(relation.group_count, dtype=jnp.int32)
    offsets = jnp.asarray(relation.destination_edge_offsets, dtype=jnp.int32)
    edge_sources = jnp.asarray(relation.grouped_source_item, dtype=jnp.int32)

    @jax.jit
    def operation(source, weight, runtime_counts, runtime_offsets, runtime_edge_sources):
        return call_cuda_segmented_contract_ffi(
            generated.ffi,
            source=source,
            weight=weight,
            event_counts=runtime_counts,
            event_offsets=runtime_offsets,
            edge_sources=runtime_edge_sources,
        )

    return generated, library, operation, counts, offsets, edge_sources


def _compile_streaming(
    *,
    key_length: int,
    pipeline_depth: int,
    mutation: bool,
    build_directory: Path,
    nvcc: Path,
    architecture: str,
):
    dataflow = _streaming_dataflow(key_length=key_length, pipeline_depth=pipeline_depth)
    generated = generate_streaming_contract_fold_event_ffi(
        dataflow,
        query_tile_size=4,
        key_value_tile_size=4,
        reduction_dimension=16,
        value_dimension=8,
        score_scale=0.25,
        target_name=f"shuttle.event_streaming_{'mutation' if mutation else 'primary'}",
    )
    library = compile_cuda_event_ffi(
        generated.ffi,
        directory=build_directory,
        nvcc=nvcc,
        architecture=architecture,
    )
    register_cuda_event_ffi(generated.ffi, library)

    @jax.jit
    def operation(query, key, value, domain_valid):
        return call_cuda_streaming_contract_fold_ffi(
            generated.ffi,
            query=query,
            key=key,
            value=value,
            domain_valid=domain_valid,
        )

    return generated, library, operation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_100a")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--requested-gpu-model", required=True)
    parser.add_argument("--requested-gpu-count", type=int, required=True)
    parser.add_argument("--requested-cpu", type=int, required=True)
    parser.add_argument("--requested-host-memory-gb", type=int, required=True)
    parser.add_argument("--requested-disk-gb", type=int, required=True)
    parser.add_argument("--requested-priority", required=True)
    args = parser.parse_args()

    device = jax.devices("gpu")[0]
    hardware = _gpu_record()
    if args.requested_gpu_model not in hardware["name"]:
        raise ValueError(f"requested {args.requested_gpu_model}, got {hardware['name']}")
    rng = np.random.default_rng(20260809)
    records = {}
    libraries = []
    for mutation in (False, True):
        label = "mutation" if mutation else "primary"
        relation = _relation(mutation=mutation)
        generated, library, operation, counts, offsets, edge_sources = _compile_segmented(
            relation=relation,
            mutation=mutation,
            build_directory=args.build_directory / f"segmented_{label}",
            nvcc=args.nvcc,
            architecture=args.architecture,
        )
        libraries.append(library)
        source_np = rng.normal(size=(relation.source_item_count, 32)).astype(np.float32)
        weight_np = rng.normal(size=(relation.destination_count, 32, 16)).astype(np.float32)
        source = jax.device_put(source_np, device)
        weight = jax.device_put(weight_np, device)
        arguments = (source, weight, counts, offsets, edge_sources)
        first = np.asarray(operation(*arguments))
        second = np.asarray(operation(*arguments))
        reference = evaluate_segmented_contract_event(relation, source_np, weight_np)
        error = np.abs(first - reference)
        records[f"segmented_{label}"] = {
            **_audit_record(generated),
            **_sample_record(
                _measure(_bind_operation(operation, arguments), warmups=args.warmups, repeats=args.repeats)
            ),
            "max_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()),
            "bitwise_deterministic": np.array_equal(first, second),
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "hlo": _hlo_record(operation, *arguments),
            "relation_counts": relation.group_count.tolist(),
        }

    for mutation, key_length, pipeline_depth in ((False, 16, 2), (True, 20, 3)):
        label = "mutation" if mutation else "primary"
        generated, library, operation = _compile_streaming(
            key_length=key_length,
            pipeline_depth=pipeline_depth,
            mutation=mutation,
            build_directory=args.build_directory / f"streaming_{label}",
            nvcc=args.nvcc,
            architecture=args.architecture,
        )
        libraries.append(library)
        row_count, query_tile, reduction = generated.ffi.inputs[0].shape
        _, partition_count, key_tile, _ = generated.ffi.inputs[1].shape
        value_dimension = generated.ffi.inputs[2].shape[-1]
        query_np = rng.normal(size=(row_count, query_tile, reduction)).astype(np.float32)
        key_np = rng.normal(size=(row_count, partition_count, key_tile, reduction)).astype(np.float32)
        value_np = rng.normal(size=(row_count, partition_count, key_tile, value_dimension)).astype(np.float32)
        valid_np = np.ones((row_count, query_tile, partition_count, key_tile), dtype=np.int32)
        valid_np[0, 0, -1] = 0
        arguments = tuple(jax.device_put(value, device) for value in (query_np, key_np, value_np, valid_np))
        first = np.asarray(operation(*arguments))
        second = np.asarray(operation(*arguments))
        reference = evaluate_streaming_contract_fold_event(
            query_np,
            key_np,
            value_np,
            valid_np,
            score_scale=0.25,
        )
        error = np.abs(first - reference)
        records[f"streaming_{label}"] = {
            **_audit_record(generated),
            **_sample_record(
                _measure(_bind_operation(operation, arguments), warmups=args.warmups, repeats=args.repeats)
            ),
            "max_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()),
            "bitwise_deterministic": np.array_equal(first, second),
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "hlo": _hlo_record(operation, *arguments),
        }

    result = {
        "benchmark": "shuttle_event_tensor_workload_linkage_sm100",
        "hardware": hardware,
        "resource_request": {
            "gpu_model": args.requested_gpu_model,
            "gpu_count": args.requested_gpu_count,
            "cpu": args.requested_cpu,
            "host_memory_gb": args.requested_host_memory_gb,
            "disk_gb": args.requested_disk_gb,
            "priority": args.requested_priority,
        },
        "shuttle_source": _source_record(),
        "toolchain_packages": _toolchain_versions(),
        "device": str(device),
        "nvcc": subprocess.check_output([str(args.nvcc), "--version"], text=True).strip(),
        "architecture": args.architecture,
        "timing_boundary": "host dispatch through JAX typed FFI and output completion",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
