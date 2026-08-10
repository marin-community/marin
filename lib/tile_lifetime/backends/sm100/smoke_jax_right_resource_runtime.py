# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded GB200 smoke for the Torch-free JAX right-resource runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from jax_right_resource_runtime import (  # noqa: E402
    JaxRightResourceInputs,
    call_partial_merge_ffi,
    call_right_resource_physical,
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

RIGHT_ITEM_WIDTH = 128


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msa-root", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--query-length", type=int, default=128)
    parser.add_argument("--key-length", type=int, default=1024)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--key-value-heads", type=int, default=2)
    parser.add_argument("--selected-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--architecture", default="sm_100a")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _selected_right_items(
    *,
    left_count: int,
    partition_count: int,
    right_count: int,
    selected_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Construct a nonmonotone relation and an occupancy-preserving mutation."""
    if right_count < 2:
        raise ValueError("the linkage smoke requires one active and one empty right resource")
    active = np.arange(right_count - 1, dtype=np.int32)
    if selected_count > active.size:
        raise ValueError("selected count exceeds the active right-resource domain")
    generator = np.random.default_rng(seed)
    baseline = np.empty((left_count, partition_count, selected_count), dtype=np.int32)
    for left, partition in np.ndindex(left_count, partition_count):
        baseline[left, partition] = generator.choice(active, size=selected_count, replace=False)
    permutation = np.roll(active, 1)
    mutation = permutation[baseline]
    empty = right_count - 1
    if np.any(baseline == empty) or np.any(mutation == empty):
        raise AssertionError("the designated empty resource received an edge")
    return baseline, mutation, empty


def _lowering(arguments: argparse.Namespace, selected: np.ndarray):
    score_map = scaled_score_map(RIGHT_ITEM_WIDTH**-0.5)
    tensor_program = build_attention_tensor_program(
        batch_size=1,
        query_length=arguments.query_length,
        key_length=arguments.key_length,
        query_heads=arguments.query_heads,
        key_value_heads=arguments.key_value_heads,
        key_dimension=RIGHT_ITEM_WIDTH,
        value_dimension=RIGHT_ITEM_WIDTH,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    program = derive_streaming_attention(
        tensor_program,
        schedule=StreamingTileSchedule(
            query_tile_size=RIGHT_ITEM_WIDTH,
            key_value_tile_size=RIGHT_ITEM_WIDTH,
            pipeline_depth=2,
        ),
    )
    destinations = selected.reshape(arguments.query_length, -1)
    right_count = arguments.key_length // RIGHT_ITEM_WIDTH
    relation = build_relation_plan(
        destinations,
        np.ones(destinations.shape, dtype=np.float32),
        destination_rank_by_item=np.zeros(right_count, dtype=np.int32),
        destination_local_item_by_item=np.arange(right_count, dtype=np.int32),
        padding_quantum=1,
    )
    schedule = replace(default_sm100_routed_schedules()[1], right_edges_per_task=128)
    return lower_sm100_routed_streaming_program(program, relation, schedule)


def _reference(
    selected: np.ndarray,
    resident: jax.Array,
    first_streamed: jax.Array,
    second_streamed: jax.Array,
) -> jax.Array:
    left_count, query_heads, feature = resident.shape
    partition_count = first_streamed.shape[1]
    heads_per_partition = query_heads // partition_count
    resident_fp32 = resident.astype(jnp.float32).reshape(
        left_count,
        partition_count,
        heads_per_partition,
        feature,
    )
    offsets = jnp.arange(RIGHT_ITEM_WIDTH, dtype=jnp.int32)
    outputs = []
    for partition in range(partition_count):
        token_indices = jnp.asarray(selected[:, partition, :, None]) * RIGHT_ITEM_WIDTH + offsets
        token_indices = token_indices.reshape(left_count, -1)
        first = first_streamed[token_indices, partition].astype(jnp.float32)
        second = second_streamed[token_indices, partition].astype(jnp.float32)
        scores = jnp.einsum("lgd,ltd->lgt", resident_fp32[:, partition], first)
        weights = jax.nn.softmax(scores * (feature**-0.5), axis=-1)
        outputs.append(jnp.einsum("lgt,ltd->lgd", weights, second))
    return jnp.stack(outputs, axis=1).reshape(left_count, query_heads, feature)


def _output_hash(value: jax.Array) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _execute(
    compiled,
    plan,
    inputs: JaxRightResourceInputs,
) -> jax.Array:
    value_state, scalar_state = call_right_resource_physical(compiled, plan, inputs)
    return call_partial_merge_ffi(plan.merge_ffi, scalar_state, value_state, plan.tables.split_counts)


def main() -> None:
    arguments = _arguments()
    if importlib.util.find_spec("torch") is not None or "torch" in sys.modules:
        raise RuntimeError("the JAX linkage smoke must run without Torch")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected a JAX GPU backend, found {jax.default_backend()}")
    if arguments.key_length % RIGHT_ITEM_WIDTH:
        raise ValueError("key length must be divisible by the right-item width")
    if arguments.query_heads % arguments.key_value_heads:
        raise ValueError("query heads must be divisible by key/value heads")

    selected, mutation, empty_right = _selected_right_items(
        left_count=arguments.query_length,
        partition_count=arguments.key_value_heads,
        right_count=arguments.key_length // RIGHT_ITEM_WIDTH,
        selected_count=arguments.selected_count,
        seed=arguments.seed,
    )
    baseline = prepare_jax_right_resource_runtime(
        arguments.msa_root,
        _lowering(arguments, selected),
    )
    mutated = prepare_jax_right_resource_runtime(
        arguments.msa_root,
        _lowering(arguments, mutation),
        merge_target="shuttle.partial_state_fold_finalize_mutated_relation",
    )
    baseline_schedule = baseline.sources.emitter_plan.event_schedule
    mutated_schedule = mutated.sources.emitter_plan.event_schedule
    if baseline_schedule.program_fingerprint != mutated_schedule.program_fingerprint:
        raise RuntimeError("the relation mutation changed the Event Tensor program")
    if baseline_schedule.runtime_fingerprint == mutated_schedule.runtime_fingerprint:
        raise RuntimeError("the relation mutation did not change Event Tensor runtime state")

    compile_and_register_partial_merge_ffi(
        baseline.merge_ffi,
        directory=arguments.build_directory / "partial_merge",
        nvcc=arguments.nvcc,
        architecture=arguments.architecture,
    )
    compile_and_register_partial_merge_ffi(
        mutated.merge_ffi,
        directory=arguments.build_directory / "partial_merge_mutation",
        nvcc=arguments.nvcc,
        architecture=arguments.architecture,
    )
    compiled = compile_right_resource_physical_call(
        baseline,
        msa_root=arguments.msa_root,
        source_directory=arguments.build_directory / "extracted_python",
    )

    first_key, second_key, resident_key = jax.random.split(jax.random.key(arguments.seed), 3)
    inputs = JaxRightResourceInputs(
        resident=jax.random.normal(
            resident_key,
            (arguments.query_length, arguments.query_heads, RIGHT_ITEM_WIDTH),
            dtype=jnp.bfloat16,
        ),
        first_streamed=jax.random.normal(
            first_key,
            (arguments.key_length, arguments.key_value_heads, RIGHT_ITEM_WIDTH),
            dtype=jnp.bfloat16,
        ),
        second_streamed=jax.random.normal(
            second_key,
            (arguments.key_length, arguments.key_value_heads, RIGHT_ITEM_WIDTH),
            dtype=jnp.bfloat16,
        ),
    )

    cases = (("baseline", baseline, selected), ("relation_mutation", mutated, mutation))
    case_records = {}
    for name, plan, selected_items in cases:
        output = _execute(compiled, plan, inputs)
        output.block_until_ready()
        expected = _reference(selected_items, inputs.resident, inputs.first_streamed, inputs.second_streamed)
        difference = jnp.abs(output.astype(jnp.float32) - expected)
        for _ in range(arguments.warmups):
            _execute(compiled, plan, inputs).block_until_ready()
        samples = []
        hashes = []
        for _ in range(arguments.repeats):
            start = time.perf_counter()
            repeated = _execute(compiled, plan, inputs)
            repeated.block_until_ready()
            samples.append((time.perf_counter() - start) * 1000.0)
            hashes.append(_output_hash(repeated))
        case_records[name] = {
            "maximum_absolute_error": float(jnp.max(difference)),
            "mean_absolute_error": float(jnp.mean(difference)),
            "output_hash": _output_hash(output),
            "repeated_output_hashes": hashes,
            "deterministic": len(set(hashes)) == 1,
            "samples_ms": samples,
            "median_ms": float(np.median(samples)),
            "event_runtime_fingerprint": plan.sources.emitter_plan.event_schedule.runtime_fingerprint,
            "work_count": int(np.asarray(plan.tables.work_count)[0]),
        }

    if "torch" in sys.modules:
        raise RuntimeError("the JAX linkage smoke imported Torch")
    event = baseline.sources.emitter_plan.event_schedule
    record = {
        "status": "gb200_jax_right_resource_linkage_executed",
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "query_length": arguments.query_length,
        "key_length": arguments.key_length,
        "query_heads": arguments.query_heads,
        "key_value_heads": arguments.key_value_heads,
        "selected_count": arguments.selected_count,
        "empty_right_resource": empty_right,
        "empty_right_resource_edge_count": int(np.count_nonzero(selected == empty_right)),
        "event_program_fingerprint": event.program_fingerprint,
        "resource_task_count": event.grouping.task_count,
        "resource_buffer_capacity": event.resource_buffer.capacity,
        "event_realizations": [
            [entry.plan_name, entry.kind.value, entry.mechanism] for entry in event.realization.entries
        ],
        "physical_call_type": type(compiled.call).__qualname__,
        "physical_source_sha256": baseline.sources.generated_source_sha256["physical"],
        "partial_merge_source_sha256": baseline.merge_ffi.source_sha256,
        "partial_merge_handler": baseline.merge_ffi.handler_symbol,
        "external_semantic_kernels": list(baseline.sources.emitter_plan.external_semantic_kernels),
        "internal_synchronization_boundary": "primitive-owned mbarrier sites",
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "torch_loaded": "torch" in sys.modules,
        "cases": case_records,
    }
    rendered = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
