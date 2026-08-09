#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay generated Event Tensor CUDA through a Torch-free JAX typed FFI."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_metadata import command_record, nvidia_smi_snapshot, toolchain_snapshot  # pyrefly: ignore[missing-import]

from tile_lifetime.cuda_dynamic_event_dataflow_codegen import (
    CudaEventFfiLowering,
    generate_cuda_phased_pipeline_ffi_lowering,
    generate_cuda_runtime_event_ffi_lowering,
)
from tile_lifetime.event_dataflow import EventMemoryScope, derive_event_tensor_plan
from tile_lifetime.event_dataflow_examples import pipelined_contract_fold_program, relation_segment_dependence
from tile_lifetime.jax_event_dataflow_ffi import (
    RuntimeEventFfiArguments,
    call_cuda_phased_pipeline_ffi,
    call_cuda_runtime_event_ffi,
    compile_cuda_event_ffi,
    register_cuda_event_ffi,
    runtime_event_ffi_arguments,
)
from tile_lifetime.relation import RelationPlan, build_relation_plan

_RUNTIME_TARGET = "shuttle.event_tensor.runtime_h100_replay_v1"
_PHASED_TARGET = "shuttle.event_tensor.phased_h100_replay_v1"


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
    parser.add_argument("--determinism-repeats", type=int, default=5)
    parser.add_argument("--build-directory", type=Path, default=Path("/tmp/shuttle-jax-event-tensor"))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--holder-revision", required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_90a")
    parser.add_argument("--allocation-gpus", type=int, required=True)
    parser.add_argument("--allocation-cpu", type=float, required=True)
    parser.add_argument("--allocation-memory", required=True)
    parser.add_argument("--allocation-disk", required=True)
    parser.add_argument("--allocation-priority", required=True)
    return parser.parse_args()


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


def _event_plan(relation: RelationPlan):
    return derive_event_tensor_plan(
        relation_segment_dependence(relation, visibility_scope=EventMemoryScope.CTA),
        name="runtime_segment_readiness",
    )


def _hash(value: jax.Array) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _runtime_reference(arguments: RuntimeEventFfiArguments) -> np.ndarray:
    payload = np.asarray(arguments.input, dtype=np.float32)
    offsets = np.asarray(arguments.event_source_offsets, dtype=np.int32)
    sources = np.asarray(arguments.event_sources, dtype=np.int32)
    output = np.zeros(len(offsets) - 1, dtype=np.float32)
    for event in range(len(output)):
        accumulator = np.float32(0.0)
        for index in range(int(offsets[event]), int(offsets[event + 1])):
            accumulator = np.float32(accumulator + payload[sources[index]])
        output[event] = accumulator
    return output


def _runtime_callable(
    generated: CudaEventFfiLowering,
    maximum_count: int,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    @jax.jit
    def call(
        payload: jax.Array,
        counts: jax.Array,
        offsets: jax.Array,
        sources: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return call_cuda_runtime_event_ffi(
            generated,
            RuntimeEventFfiArguments(payload, counts, offsets, sources, maximum_count),
        )

    return call


def _phased_callable(
    generated: CudaEventFfiLowering,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def call(query: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
        return call_cuda_phased_pipeline_ffi(generated, query=query, key=key, value=value)

    return call


def _phased_reference(query: jax.Array, key: jax.Array, value: jax.Array) -> np.ndarray:
    query_array = np.asarray(query, dtype=np.float32)
    key_array = np.asarray(key, dtype=np.float32)
    value_array = np.asarray(value, dtype=np.float32)
    output = np.empty(query_array.shape[0], dtype=np.float32)
    for generation in range(query_array.shape[0]):
        scores = np.empty(key_array.shape[1], dtype=np.float32)
        for slot in range(key_array.shape[1]):
            accumulator = np.float32(0.0)
            for index in range(query_array.shape[1]):
                accumulator = np.float32(
                    accumulator + np.float32(query_array[generation, index] * key_array[generation, slot, index])
                )
            scores[slot] = accumulator
        maximum = np.max(scores)
        probabilities = np.exp(scores - maximum, dtype=np.float32)
        denominator = np.sum(probabilities, dtype=np.float32)
        output[generation] = np.sum(probabilities * value_array[generation], dtype=np.float32) / denominator
    return output


def _determinism(function: Callable[[], Any], *, repeats: int) -> dict[str, Any]:
    hashes: list[list[str]] = []
    for _ in range(repeats):
        result = function()
        jax.block_until_ready(result)
        values = result if isinstance(result, tuple) else (result,)
        hashes.append([_hash(value) for value in values])
    if any(value != hashes[0] for value in hashes[1:]):
        raise RuntimeError("generated Event Tensor output is not bitwise deterministic")
    return {"repeats": repeats, "hashes": hashes, "bitwise": True}


def _measure(
    variants: tuple[tuple[str, Callable[[], Any]], ...],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        for _, function in variants:
            jax.block_until_ready(function())
    samples: dict[str, list[float]] = {name: [] for name, _ in variants}
    execution_order: list[list[str]] = []
    for repeat in range(repeats):
        ordered = variants if repeat % 2 == 0 else tuple(reversed(variants))
        execution_order.append([name for name, _ in ordered])
        for name, function in ordered:
            started = time.perf_counter()
            result = None
            for _ in range(iterations):
                result = function()
            jax.block_until_ready(result)
            samples[name].append((time.perf_counter() - started) * 1e3 / iterations)
    return {
        "method": "host enqueue interval followed by jax.block_until_ready",
        "warmups": warmups,
        "repeats": repeats,
        "iterations_per_repeat": iterations,
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


def _write_source(generated: CudaEventFfiLowering, path: Path) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated.source + "\n")
    return {
        "path": str(path),
        "source_sha256": generated.source_sha256,
        "device_source_sha256": generated.device_source_sha256,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Compile, execute, mutate, and benchmark both generic event families."""
    devices = jax.devices("gpu")
    if not devices or "H100" not in devices[0].device_kind:
        raise RuntimeError(f"H100 replay requires an H100 device, found {devices}")
    if args.repeats <= 0 or args.repeats % 2:
        raise ValueError("counterbalanced repeats must be a positive even number")
    if args.active_destinations <= 0 or args.active_destinations >= args.destination_count:
        raise ValueError("active destinations must leave at least one empty runtime segment")

    first_relation = _relation_plan(args, seed=20260809)
    second_relation = _relation_plan(args, seed=20260810)
    first_plan = _event_plan(first_relation)
    second_plan = _event_plan(second_relation)
    generated_runtime = generate_cuda_runtime_event_ffi_lowering(first_plan, target_name=_RUNTIME_TARGET)
    mutated_runtime = generate_cuda_runtime_event_ffi_lowering(second_plan, target_name=_RUNTIME_TARGET)
    if generated_runtime.source_sha256 != mutated_runtime.source_sha256:
        raise RuntimeError("runtime RelationPlan mutation changed the physical FFI source")
    runtime_library = compile_cuda_event_ffi(
        generated_runtime,
        directory=args.build_directory / "runtime",
        nvcc=args.nvcc,
        architecture=args.architecture,
    )
    register_cuda_event_ffi(generated_runtime, runtime_library)
    payload = jax.random.normal(jax.random.key(20260811), generated_runtime.inputs[0].shape, dtype=jnp.float32)
    first_arguments = runtime_event_ffi_arguments(first_plan, payload)
    second_arguments = runtime_event_ffi_arguments(second_plan, payload)
    first_runtime_call = _runtime_callable(generated_runtime, first_arguments.maximum_count)
    second_runtime_call = _runtime_callable(mutated_runtime, second_arguments.maximum_count)

    first_runtime = first_runtime_call(
        first_arguments.input,
        first_arguments.event_counts,
        first_arguments.event_source_offsets,
        first_arguments.event_sources,
    )
    second_runtime = second_runtime_call(
        second_arguments.input,
        second_arguments.event_counts,
        second_arguments.event_source_offsets,
        second_arguments.event_sources,
    )
    jax.block_until_ready((first_runtime, second_runtime))
    if not np.array_equal(np.asarray(first_runtime[0]), np.asarray(payload)):
        raise RuntimeError("runtime relation did not write every source partial exactly once")
    first_reference = _runtime_reference(first_arguments)
    second_reference = _runtime_reference(second_arguments)
    if not np.array_equal(np.asarray(first_runtime[1]), first_reference):
        raise RuntimeError("primary runtime relation differs from source-ordered reference")
    if not np.array_equal(np.asarray(second_runtime[1]), second_reference):
        raise RuntimeError("mutated runtime relation differs from source-ordered reference")

    primary_program = pipelined_contract_fold_program(
        generation_count=args.generations,
        pipeline_depth=args.pipeline_depth,
    )
    mutation_depth = max(1, args.pipeline_depth // 2)
    mutation_program = pipelined_contract_fold_program(
        generation_count=args.generations + 1,
        pipeline_depth=mutation_depth,
    )
    generated_phased = generate_cuda_phased_pipeline_ffi_lowering(
        primary_program,
        dimension=args.dimension,
        target_name=_PHASED_TARGET,
    )
    mutated_phased = generate_cuda_phased_pipeline_ffi_lowering(
        mutation_program,
        dimension=args.dimension,
        target_name=_PHASED_TARGET,
    )
    if generated_phased.source_sha256 != mutated_phased.source_sha256:
        raise RuntimeError("phased schedule mutation changed the physical FFI source")
    phased_library = compile_cuda_event_ffi(
        generated_phased,
        directory=args.build_directory / "phased",
        nvcc=args.nvcc,
        architecture=args.architecture,
    )
    register_cuda_event_ffi(generated_phased, phased_library)
    primary_key, mutation_key = jax.random.split(jax.random.key(20260812))
    query_key, key_key, value_key = jax.random.split(primary_key, 3)
    query = jax.random.normal(query_key, generated_phased.inputs[0].shape, dtype=jnp.float32) * 0.1
    key = jax.random.normal(key_key, generated_phased.inputs[1].shape, dtype=jnp.float32) * 0.1
    value = jax.random.normal(value_key, generated_phased.inputs[2].shape, dtype=jnp.float32)
    mutation_query_key, mutation_key_key, mutation_value_key = jax.random.split(mutation_key, 3)
    mutation_query = jax.random.normal(mutation_query_key, mutated_phased.inputs[0].shape, dtype=jnp.float32) * 0.1
    mutation_key_tensor = (
        jax.random.normal(
            mutation_key_key,
            mutated_phased.inputs[1].shape,
            dtype=jnp.float32,
        )
        * 0.1
    )
    mutation_value = jax.random.normal(
        mutation_value_key,
        mutated_phased.inputs[2].shape,
        dtype=jnp.float32,
    )
    primary_phased_call = _phased_callable(generated_phased)
    mutation_phased_call = _phased_callable(mutated_phased)
    primary_output = primary_phased_call(query, key, value)
    mutation_output = mutation_phased_call(mutation_query, mutation_key_tensor, mutation_value)
    jax.block_until_ready((primary_output, mutation_output))
    primary_reference = _phased_reference(query, key, value)
    mutation_reference = _phased_reference(mutation_query, mutation_key_tensor, mutation_value)
    primary_error = np.abs(np.asarray(primary_output) - primary_reference)
    mutation_error = np.abs(np.asarray(mutation_output) - mutation_reference)
    if not np.allclose(np.asarray(primary_output), primary_reference, rtol=2e-5, atol=2e-5):
        raise RuntimeError(f"primary phased result has maximum error {primary_error.max()}")
    if not np.allclose(np.asarray(mutation_output), mutation_reference, rtol=2e-5, atol=2e-5):
        raise RuntimeError(f"mutated phased result has maximum error {mutation_error.max()}")

    def runtime_primary():
        return first_runtime_call(
            first_arguments.input,
            first_arguments.event_counts,
            first_arguments.event_source_offsets,
            first_arguments.event_sources,
        )

    def runtime_mutation():
        return second_runtime_call(
            second_arguments.input,
            second_arguments.event_counts,
            second_arguments.event_source_offsets,
            second_arguments.event_sources,
        )

    def phased_primary():
        return primary_phased_call(query, key, value)

    def phased_mutation():
        return mutation_phased_call(mutation_query, mutation_key_tensor, mutation_value)

    determinism = {
        "runtime_primary": _determinism(runtime_primary, repeats=args.determinism_repeats),
        "runtime_mutation": _determinism(runtime_mutation, repeats=args.determinism_repeats),
        "phased_primary": _determinism(phased_primary, repeats=args.determinism_repeats),
        "phased_mutation": _determinism(phased_mutation, repeats=args.determinism_repeats),
    }
    telemetry_before = nvidia_smi_snapshot()
    timing = _measure(
        (
            ("runtime_primary", runtime_primary),
            ("runtime_mutation", runtime_mutation),
            ("phased_primary", phased_primary),
            ("phased_mutation", phased_mutation),
        ),
        warmups=args.warmups,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    telemetry_after = nvidia_smi_snapshot()
    runtime_count_symbol = getattr(runtime_library, f"{generated_runtime.handler_symbol}_call_count")
    runtime_count_symbol.restype = ctypes.c_int
    phased_count_symbol = getattr(phased_library, f"{generated_phased.handler_symbol}_call_count")
    phased_count_symbol.restype = ctypes.c_int

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "kind": "jax_typed_ffi_event_tensor_h100_replay",
        "command": command_record(),
        "requested_shuttle_revision": args.shuttle_revision,
        "observed_shuttle_revision": (
            subprocess.run(("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()
        ),
        "holder_revision": args.holder_revision,
        "allocation": {
            "gpu_variant": "H100",
            "gpu_count": args.allocation_gpus,
            "cpu": args.allocation_cpu,
            "memory": args.allocation_memory,
            "disk": args.allocation_disk,
            "priority": args.allocation_priority,
            "gb200_used": False,
        },
        "runtime_relation": {
            "generated_source": _write_source(
                generated_runtime,
                args.json_output.parent / "generated_runtime_event_ffi.cu",
            ),
            "same_source_after_relation_mutation": True,
            "primary_plan_fingerprint": generated_runtime.plan_fingerprint,
            "mutation_plan_fingerprint": mutated_runtime.plan_fingerprint,
            "primary_counts": np.asarray(first_arguments.event_counts).tolist(),
            "mutation_counts": np.asarray(second_arguments.event_counts).tolist(),
            "primary_empty_events": int(np.count_nonzero(np.asarray(first_arguments.event_counts) == 0)),
            "mutation_empty_events": int(np.count_nonzero(np.asarray(second_arguments.event_counts) == 0)),
            "primary_bitwise_reference": True,
            "mutation_bitwise_reference": True,
            "ffi_handler_call_count": runtime_count_symbol(),
        },
        "phased_pipeline": {
            "generated_source": _write_source(
                generated_phased,
                args.json_output.parent / "generated_phased_event_ffi.cu",
            ),
            "same_source_after_schedule_mutation": True,
            "primary_plan_fingerprint": generated_phased.plan_fingerprint,
            "mutation_plan_fingerprint": mutated_phased.plan_fingerprint,
            "primary": {
                "generations": args.generations,
                "pipeline_depth": args.pipeline_depth,
                "dimension": args.dimension,
                "maximum_absolute_error": float(primary_error.max(initial=0.0)),
                "mean_absolute_error": float(primary_error.mean()),
            },
            "mutation": {
                "generations": args.generations + 1,
                "pipeline_depth": mutation_depth,
                "dimension": args.dimension,
                "maximum_absolute_error": float(mutation_error.max(initial=0.0)),
                "mean_absolute_error": float(mutation_error.mean()),
            },
            "ffi_handler_call_count": phased_count_symbol(),
        },
        "determinism": determinism,
        "timing": timing,
        "toolchain": toolchain_snapshot(str(args.nvcc)),
        "jax": {
            "version": jax.__version__,
            "device_kind": devices[0].device_kind,
            "platform_version": devices[0].client.platform_version,
        },
        "gpu": {
            "family": "H100",
            "gb200_or_b200": False,
            "before": telemetry_before,
            "after": telemetry_after,
        },
    }
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    print(json.dumps(run(_arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
