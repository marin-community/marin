# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-only Linux preflight for the JAX right-resource runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from jax_right_resource_runtime import (  # noqa: E402
    compile_and_register_partial_merge_ffi,
    compile_right_resource_physical_call,
    prepare_jax_right_resource_runtime,
)

from tile_lifetime import DType, StreamingTileSchedule, build_attention_tensor_program  # noqa: E402
from tile_lifetime.relation import build_relation_plan  # noqa: E402
from tile_lifetime.sm100_routed_lowering import (  # noqa: E402
    default_sm100_routed_schedules,
    lower_sm100_routed_streaming_program,
)
from tile_lifetime.streaming_attention import derive_streaming_attention, scaled_score_map  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msa-root", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_100a")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _lowering(*, destination_shift: int):
    query_length = 128
    key_length = 1024
    query_heads = 16
    key_value_heads = 2
    selected_count = 4
    right_count = key_length // 128
    selected = np.empty((query_length, key_value_heads, selected_count), dtype=np.int32)
    for left_item, partition in np.ndindex(query_length, key_value_heads):
        base = left_item * 5 + partition * 3
        selected[left_item, partition] = np.asarray(
            [
                (base + destination_shift + 1) % right_count,
                (base + destination_shift + 3) % right_count,
                (base + destination_shift + 4) % right_count,
                (base + destination_shift + 6) % right_count,
            ],
            dtype=np.int32,
        )
    destinations = selected.reshape(query_length, key_value_heads * selected_count)
    relation = build_relation_plan(
        destinations,
        np.ones(destinations.shape, dtype=np.float32),
        destination_rank_by_item=np.zeros(right_count, dtype=np.int32),
        destination_local_item_by_item=np.arange(right_count, dtype=np.int32),
        padding_quantum=1,
    )
    score_map = scaled_score_map(128**-0.5)
    tensor_program = build_attention_tensor_program(
        batch_size=1,
        query_length=query_length,
        key_length=key_length,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        key_dimension=128,
        value_dimension=128,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    program = derive_streaming_attention(
        tensor_program,
        schedule=StreamingTileSchedule(
            query_tile_size=128,
            key_value_tile_size=128,
            pipeline_depth=2,
        ),
    )
    schedule = replace(
        default_sm100_routed_schedules()[1],
        right_edges_per_task=128,
    )
    return lower_sm100_routed_streaming_program(program, relation, schedule)


def main() -> None:
    arguments = _arguments()
    if platform.system() != "Linux":
        raise RuntimeError("the CUTLASS/CUDA dependency preflight must run on Linux")
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported before the JAX runtime preflight")

    baseline = prepare_jax_right_resource_runtime(arguments.msa_root, _lowering(destination_shift=0))
    mutation = prepare_jax_right_resource_runtime(
        arguments.msa_root,
        _lowering(destination_shift=1),
        merge_target="shuttle.partial_state_fold_finalize_mutated",
    )
    if baseline.sources.emitter_plan.event_schedule.program_fingerprint != (
        mutation.sources.emitter_plan.event_schedule.program_fingerprint
    ):
        raise RuntimeError("a relation-only mutation changed the EventTensor program fingerprint")
    if baseline.sources.emitter_plan.event_schedule.runtime_fingerprint == (
        mutation.sources.emitter_plan.event_schedule.runtime_fingerprint
    ):
        raise RuntimeError("a relation-only mutation did not change the EventTensor runtime fingerprint")

    dependency_modules = {
        name: importlib.import_module(name)
        for name in ("cutlass", "cutlass.cute", "cutlass.jax", "cuda.bindings.driver")
    }
    merge_library = compile_and_register_partial_merge_ffi(
        baseline.merge_ffi,
        directory=arguments.build_directory / "partial_merge",
        nvcc=arguments.nvcc,
        architecture=arguments.architecture,
    )
    physical_call = compile_right_resource_physical_call(
        baseline,
        msa_root=arguments.msa_root,
        source_directory=arguments.build_directory / "extracted_python",
    )
    if "torch" in sys.modules:
        raise RuntimeError("the JAX/CuTe preflight imported Torch")

    tables = baseline.tables
    record = {
        "status": "linux_compile_import_preflight_passed",
        "python": platform.python_version(),
        "jax": jax.__version__,
        "architecture": arguments.architecture,
        "event_program_fingerprint": baseline.sources.emitter_plan.event_schedule.program_fingerprint,
        "event_runtime_fingerprint": baseline.sources.emitter_plan.event_schedule.runtime_fingerprint,
        "mutation_runtime_fingerprint": mutation.sources.emitter_plan.event_schedule.runtime_fingerprint,
        "physical_source_sha256": baseline.sources.lineage["physical_source_sha256"],
        "partial_merge_source_sha256": baseline.merge_ffi.source_sha256,
        "partial_merge_handler": baseline.merge_ffi.handler_symbol,
        "partial_merge_library": str(Path(merge_library._name).resolve()),
        "physical_call_type": type(physical_call).__qualname__,
        "dependency_modules": {name: str(module.__file__) for name, module in dependency_modules.items()},
        "torch_loaded": "torch" in sys.modules,
        "external_semantic_kernels": list(baseline.sources.emitter_plan.external_semantic_kernels),
        "work_capacity": tables.work_capacity,
        "work_count": np.asarray(tables.work_count).tolist(),
        "right_to_left_offsets_shape": list(tables.right_to_left_offsets.shape),
        "right_to_left_sources_shape": list(tables.right_to_left_sources.shape),
        "partial_slot_sources_shape": list(tables.partial_slot_sources.shape),
        "split_counts_shape": list(tables.split_counts.shape),
        "scheduler_metadata_shape": list(tables.scheduler_metadata.shape),
    }
    rendered = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
