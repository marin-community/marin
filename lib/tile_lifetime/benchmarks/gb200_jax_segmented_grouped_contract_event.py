#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay RelationPlan readiness into a generic SM100 Contract through JAX FFI."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from tile_lifetime.cuda_toolchain import cuda_toolkit_library_directories
from tile_lifetime.relation import RelationPlan, build_relation_plan
from tile_lifetime.segmented_grouped_contract_event_schedule import (
    derive_same_stream_segmented_grouped_contract_schedule,
)
from tile_lifetime.sm100_grouped_contract_event_codegen import (
    sm100_bf16_grouped_contract_descriptor,
    sm100_bf16_grouped_contract_event_schedule,
    verify_sm100_grouped_contract_event_include,
)

MOK_COMMIT = "3e1cf43ab93ad040afed52a45ab03cb490ffe4be"
THUNDERKITTENS_COMMIT = "1c3920d993404dd49a6d4c7267ea11d583bd5c68"
PACK_TARGET = "shuttle.relation_segment_pack.sm100.v1"
CONTRACT_TARGET = "shuttle.segmented_contract.sm100.v1"
GROUP_COUNT = 4
CAPACITY = 256
REDUCTION = 256
COLUMNS = 256
SOURCE_ROWS = 192


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mok-root", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--requested-cpu", type=float, required=True)
    parser.add_argument("--requested-priority", required=True)
    return parser.parse_args()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _validate_sources(mok_root: Path) -> dict[str, Any]:
    thunderkittens = mok_root / "third_party" / "ThunderKittens"
    observed_mok = _git(mok_root, "rev-parse", "HEAD")
    observed_tk = _git(thunderkittens, "rev-parse", "HEAD")
    if observed_mok != MOK_COMMIT or observed_tk != THUNDERKITTENS_COMMIT:
        raise ValueError(f"source pin mismatch: MoK={observed_mok}, ThunderKittens={observed_tk}")
    return {
        "mok_commit": observed_mok,
        "mok_dirty": bool(_git(mok_root, "status", "--porcelain")),
        "thunderkittens_commit": observed_tk,
        "thunderkittens_dirty": bool(_git(thunderkittens, "status", "--porcelain")),
    }


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1] / "backends" / "sm100" / "grouped_contract_ffi"


def _compile(args: argparse.Namespace) -> tuple[ctypes.CDLL, dict[str, Any]]:
    source = _backend_root() / "grouped_contract_ffi.cu"
    event_include = _backend_root().parent / "mok_gmm_probe" / "generated_event_schedule.inc"
    descriptor = sm100_bf16_grouped_contract_descriptor()
    inner = derive_same_stream_segmented_grouped_contract_schedule(
        _relation(False),
        output_tile_count=1,
        descriptor=descriptor,
        reduction_partition_count=REDUCTION // 64,
    ).contract_pipeline
    # The primitive's checked-in ABI instantiates enough partitions to expose
    # all bounded-buffer generations, so verify it against its canonical plan.
    verify_sm100_grouped_contract_event_include(event_include, sm100_bf16_grouped_contract_event_schedule())
    include = Path(jaxlib.__file__).resolve().parent / "include"
    python_include = Path(sysconfig.get_path("include"))
    thunderkittens = args.mok_root / "third_party" / "ThunderKittens"
    args.build_directory.mkdir(parents=True, exist_ok=True)
    output = args.build_directory / "libshuttle_segmented_contract_ffi.so"
    library_directories = cuda_toolkit_library_directories(args.nvcc)
    command = [
        str(args.nvcc),
        str(source),
        "-std=c++20",
        "-O3",
        "--use_fast_math",
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
        "-forward-unknown-to-host-compiler",
        "-ftemplate-backtrace-limit=0",
        "-shared",
        "-Xcompiler=-fPIC",
        "-Xcompiler=-Wno-psabi",
        "-Xcompiler=-fno-strict-aliasing",
        "-DKITTENS_SM100",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-gencode",
        "arch=compute_100a,code=sm_100a",
        f"-I{include}",
        f"-I{python_include}",
        f"-I{args.torch_root / 'include'}",
        f"-I{args.torch_root / 'include' / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-I{args.mok_root / 'csrc'}",
        f"-I{thunderkittens / 'include'}",
        f"-I{event_include.parent}",
        *[item for directory in library_directories for item in ("-L", str(directory))],
        "-lcuda",
        "-lcudadevrt",
        "-lcudart_static",
        "-lrt",
        "-lpthread",
        "-ldl",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    dependencies = subprocess.check_output(["ldd", str(output)], text=True)
    if "torch" in dependencies.lower() or "c10" in dependencies.lower():
        raise RuntimeError(f"runtime library retained a Torch dependency:\n{dependencies}")
    library = ctypes.CDLL(str(output))
    for symbol, target in (
        ("shuttle_relation_segment_pack_ffi", PACK_TARGET),
        ("shuttle_grouped_contract_ffi", CONTRACT_TARGET),
    ):
        handler = getattr(library, symbol)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(handler),
            platform="CUDA",
            api_version=1,
        )
    return library, {
        "command": command,
        "library": str(output),
        "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "dynamic_dependencies": dependencies.splitlines(),
        "bounded_shape_inner_event_fingerprint": inner.fingerprint,
    }


def _relation(mutation: bool) -> RelationPlan:
    counts = (64, 80, 48, 0) if not mutation else (72, 56, 64, 0)
    destinations = np.concatenate([np.full(count, group, dtype=np.int32) for group, count in enumerate(counts)])
    if destinations.size != SOURCE_ROWS:
        raise AssertionError(f"relation fixture has {destinations.size} edges")
    if mutation:
        destinations = destinations[np.random.default_rng(20260810).permutation(SOURCE_ROWS)]
    return build_relation_plan(
        destinations[:, None],
        np.ones((SOURCE_ROWS, 1), dtype=np.float32),
        destination_rank_by_item=np.zeros(GROUP_COUNT, dtype=np.int32),
        destination_local_item_by_item=np.arange(GROUP_COUNT, dtype=np.int32),
        padding_quantum=1,
    )


def _runtime_tables(relation: RelationPlan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    schedule = derive_same_stream_segmented_grouped_contract_schedule(
        relation,
        output_tile_count=1,
        descriptor=sm100_bf16_grouped_contract_descriptor(),
        reduction_partition_count=REDUCTION // 64,
    )
    runtime = schedule.segment_runtime_inputs
    return (
        np.asarray(runtime.event_initial_counts, dtype=np.int32),
        np.asarray(runtime.event_source_offsets, dtype=np.int32),
        np.asarray(runtime.event_sources, dtype=np.int32),
    )


def _operation(active_groups: int):
    packed_shape = jax.ShapeDtypeStruct((active_groups * CAPACITY, REDUCTION), jnp.bfloat16)
    count_shape = jax.ShapeDtypeStruct((GROUP_COUNT,), jnp.int32)
    output_shape = jax.ShapeDtypeStruct((active_groups * CAPACITY, COLUMNS), jnp.bfloat16)

    @jax.jit
    def execute(source, weights, counts, offsets, sources):
        packed, padded_counts = jax.ffi.ffi_call(
            PACK_TARGET,
            (packed_shape, count_shape),
            vmap_method="broadcast_all",
        )(
            source,
            counts,
            offsets,
            sources,
            group_count=np.int64(GROUP_COUNT),
            capacity=np.int64(CAPACITY),
            reduction=np.int64(REDUCTION),
        )
        output = jax.ffi.ffi_call(
            CONTRACT_TARGET,
            output_shape,
            vmap_method="broadcast_all",
        )(
            packed,
            weights,
            padded_counts,
            groups=np.int64(GROUP_COUNT),
            rows=np.int64(active_groups * CAPACITY),
            reduction=np.int64(REDUCTION),
            columns=np.int64(COLUMNS),
        )
        return output, packed, padded_counts

    return execute


def _call_count(library: ctypes.CDLL, symbol: str) -> int:
    function = getattr(library, symbol)
    function.restype = ctypes.c_int
    return int(function())


def _reference(
    source: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    edge_sources: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed_groups: list[np.ndarray] = []
    padded_counts = np.zeros(GROUP_COUNT, dtype=np.int32)
    for group, count in enumerate(counts):
        if count == 0:
            continue
        rows = source[edge_sources[offsets[group] : offsets[group + 1]]]
        padding = np.zeros((CAPACITY - int(count), REDUCTION), dtype=np.float32)
        packed_groups.append(np.concatenate((rows, padding), axis=0))
        padded_counts[group] = CAPACITY
    packed = np.concatenate(packed_groups, axis=0)
    outputs: list[np.ndarray] = []
    row = 0
    for group, count in enumerate(padded_counts):
        if count == 0:
            continue
        outputs.append(packed[row : row + count] @ weights[group].T)
        row += int(count)
    return np.concatenate(outputs, axis=0), packed, padded_counts


def _error(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "maximum_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "finite": bool(np.isfinite(actual).all()),
    }


def main() -> None:
    args = _arguments()
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported before the Torch-free Shuttle runtime")
    devices = jax.devices("gpu")
    if not devices or "GB200" not in devices[0].device_kind:
        raise RuntimeError(f"one physical GB200 is required, found {devices}")
    if args.samples <= 0 or args.warmups < 0:
        raise ValueError("samples must be positive and warmups nonnegative")
    source_record = _validate_sources(args.mok_root)
    library, build = _compile(args)
    if "torch" in sys.modules:
        raise RuntimeError("Torch entered sys.modules during compile or FFI registration")

    rng = np.random.default_rng(20260810)
    source_f32 = rng.normal(size=(SOURCE_ROWS, REDUCTION)).astype(np.float32)
    weights_f32 = (rng.normal(size=(GROUP_COUNT, COLUMNS, REDUCTION)) / math.sqrt(REDUCTION)).astype(np.float32)
    source = jnp.asarray(source_f32, dtype=jnp.bfloat16)
    weights = jnp.asarray(weights_f32, dtype=jnp.bfloat16)
    source_rounded = np.asarray(source.astype(jnp.float32))
    weights_rounded = np.asarray(weights.astype(jnp.float32))
    operation = _operation(active_groups=3)
    records: dict[str, Any] = {}
    expected_executions = 0
    program_fingerprint = None
    runtime_fingerprints: list[str] = []

    for mutation in (False, True):
        name = "mutation" if mutation else "primary"
        relation = _relation(mutation)
        schedule = derive_same_stream_segmented_grouped_contract_schedule(
            relation,
            output_tile_count=1,
            descriptor=sm100_bf16_grouped_contract_descriptor(),
            reduction_partition_count=REDUCTION // 64,
        )
        if program_fingerprint is None:
            program_fingerprint = schedule.program_fingerprint
        elif schedule.program_fingerprint != program_fingerprint:
            raise RuntimeError("relation mutation unexpectedly changed the schedule program fingerprint")
        runtime_fingerprints.append(schedule.runtime_fingerprint)
        counts_np, offsets_np, sources_np = _runtime_tables(relation)
        arguments = (
            source,
            weights,
            jnp.asarray(counts_np),
            jnp.asarray(offsets_np),
            jnp.asarray(sources_np),
        )
        first = operation(*arguments)
        jax.block_until_ready(first)
        second = operation(*arguments)
        jax.block_until_ready(second)
        expected_executions += 2
        actual, packed, padded_counts = (np.asarray(value) for value in first)
        repeated = tuple(np.asarray(value) for value in second)
        expected, expected_packed, expected_padded_counts = _reference(
            source_rounded, weights_rounded, counts_np, offsets_np, sources_np
        )
        if not np.array_equal(packed.astype(np.float32), expected_packed):
            raise RuntimeError(f"{name} grouping/padding did not match the RelationPlan reference")
        if not np.array_equal(padded_counts, expected_padded_counts):
            raise RuntimeError(f"{name} padded counts did not match the RelationPlan reference")
        if any(
            not np.array_equal(value, repeat)
            for value, repeat in zip((actual, packed, padded_counts), repeated, strict=True)
        ):
            raise RuntimeError(f"{name} output is not bitwise deterministic")
        metrics = _error(actual, expected)
        if not np.allclose(actual.astype(np.float32), expected, atol=0.2, rtol=0.1):
            raise RuntimeError(f"{name} grouped Contract failed correctness: {metrics}")
        for _ in range(args.warmups):
            jax.block_until_ready(operation(*arguments))
            expected_executions += 1
        samples = []
        for _ in range(args.samples):
            started = time.perf_counter()
            jax.block_until_ready(operation(*arguments))
            samples.append((time.perf_counter() - started) * 1e3)
            expected_executions += 1
        hlo = operation.lower(*arguments).compile().as_text()
        records[name] = {
            "relation_counts": counts_np.tolist(),
            "empty_segments": [index for index, count in enumerate(counts_np) if count == 0],
            "program_fingerprint": schedule.program_fingerprint,
            "runtime_fingerprint": schedule.runtime_fingerprint,
            "inner_event_fingerprint": schedule.contract_pipeline.fingerprint,
            "outer_realization": [item.kind.value for item in schedule.segment_realization.entries],
            "correctness": metrics,
            "bitwise_deterministic": True,
            "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "padded_counts": padded_counts.tolist(),
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "hlo": {
                "pack_target_occurrences": hlo.count(PACK_TARGET),
                "contract_target_occurrences": hlo.count(CONTRACT_TARGET),
                "sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            },
        }

    if runtime_fingerprints[0] == runtime_fingerprints[1]:
        raise RuntimeError("relation mutation did not change runtime fingerprint")
    observed_counts = {
        "relation_plan_pack": _call_count(library, "shuttle_relation_segment_pack_ffi_call_count"),
        "grouped_contract": _call_count(library, "shuttle_grouped_contract_ffi_call_count"),
    }
    if set(observed_counts.values()) != {expected_executions}:
        raise RuntimeError(f"handler counts {observed_counts} do not match {expected_executions} executions")
    fingerprint_function = library.shuttle_grouped_contract_event_fingerprint
    fingerprint_function.restype = ctypes.c_char_p
    runtime_inner_fingerprint = fingerprint_function().decode()
    canonical_inner_fingerprint = records["primary"]["inner_event_fingerprint"]
    # The generated include spans more K generations than this bounded runtime
    # shape, so preserve both fingerprints instead of claiming identity.
    result = {
        "benchmark": "shuttle_jax_segmented_grouped_contract_event_gb200_v0",
        "status": "ok",
        "scope": "one bounded linkage replay; no overlap or tuning claim",
        "hardware": {
            "device": str(devices[0]),
            "device_kind": devices[0].device_kind,
            "physical_target": "GB200",
        },
        "resource_request": {
            "cpu": args.requested_cpu,
            "priority": args.requested_priority,
        },
        "shuttle_revision": args.shuttle_revision,
        "source": source_record,
        "build": build,
        "torch_free_runtime": {
            "torch_in_sys_modules": "torch" in sys.modules,
            "runtime_library_has_torch_dependency": False,
        },
        "handler_counts": observed_counts,
        "expected_handler_count": expected_executions,
        "records": records,
        "runtime_primitive_event_fingerprint": runtime_inner_fingerprint,
        "bounded_shape_inner_event_fingerprint": canonical_inner_fingerprint,
        "ownership": {
            "outer_relation_readiness": "Shuttle EventTensorPlan, erased by verified same JAX stream order",
            "grouping_and_padding": "generic Shuttle typed-FFI handler",
            "grouped_contract_wrapper_abi": "Shuttle generated synchronization ABI",
            "internal_mbarrier_sites": "primitive-owned; not generated by Shuttle",
            "ad": "JAX-owned; this forward-only component replay does not define AD",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
