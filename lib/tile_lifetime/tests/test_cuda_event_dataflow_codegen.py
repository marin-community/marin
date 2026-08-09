# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from tile_lifetime.cuda_event_dataflow_codegen import (
    CudaEventLoweringError,
    generate_cuda_event_counter_lowering,
)
from tile_lifetime.event_dataflow import (
    EventDomain,
    EventGenerationPolicy,
    EventMemoryScope,
    TaskAxis,
    coarsen_event_tensor_plan,
    derive_event_tensor_plan,
)
from tile_lifetime.event_dataflow_examples import split_fold_dependence


def _gpu_plan(*, rows: int, partitions: int):
    dependence = split_fold_dependence(
        row_count=rows,
        partition_count=partitions,
        visibility_scope=EventMemoryScope.CTA,
    )
    return derive_event_tensor_plan(dependence, name="gpu_readiness")


def test_cuda_event_lowering_encodes_mechanical_counts_and_relations() -> None:
    plan = _gpu_plan(rows=3, partitions=5)
    generated = generate_cuda_event_counter_lowering(plan)

    assert generated.event_initial_counts == (5, 5, 5)
    assert generated.event_source_offsets == (0, 5, 10, 15)
    assert generated.event_sources == tuple(range(15))
    assert generated.event_consumers == (0, 1, 2)
    assert generated.consumer_source_offsets == (0, 5, 10, 15)
    assert generated.consumer_sources == tuple(range(15))
    assert generated.threads_per_block == 32
    assert generated.memory_scope is EventMemoryScope.CTA
    assert generated.generation_policy is EventGenerationPolicy.PER_INVOCATION


def test_cuda_event_lowering_exposes_real_barrier_and_workload_independent_body() -> None:
    generated = generate_cuda_event_counter_lowering(_gpu_plan(rows=2, partitions=64))

    assert "cuda::barrier<cuda::thread_scope_block>" in generated.source
    assert "event.arrive()" in generated.source
    assert "event.wait(cuda::std::move(token))" in generated.source
    assert "__syncthreads()" in generated.source
    assert "shuttle_kernel_boundary_producer" in generated.source
    assert not {"moe", "attention", "fold"} & set(generated.source.lower().split())


def test_partition_mutation_regenerates_same_physical_family() -> None:
    first = generate_cuda_event_counter_lowering(_gpu_plan(rows=4, partitions=32))
    second = generate_cuda_event_counter_lowering(_gpu_plan(rows=4, partitions=96))

    assert first.plan_fingerprint != second.plan_fingerprint
    assert first.source_sha256 != second.source_sha256
    assert first.threads_per_block == 32
    assert second.threads_per_block == 128
    assert "shuttle_counted_event_kernel" in first.source
    assert "shuttle_counted_event_kernel" in second.source


def test_cuda_event_lowering_rejects_scope_generation_and_coarsening_outside_first_skeleton() -> None:
    device_scope = derive_event_tensor_plan(
        split_fold_dependence(row_count=2, partition_count=4),
        name="device_scope",
    )
    with pytest.raises(CudaEventLoweringError, match="CTA-scope"):
        generate_cuda_event_counter_lowering(device_scope)

    phased = replace(_gpu_plan(rows=2, partitions=4), generation_policy=EventGenerationPolicy.PHASED)
    with pytest.raises(CudaEventLoweringError, match="fresh event storage"):
        generate_cuda_event_counter_lowering(phased)

    fine = _gpu_plan(rows=2, partitions=4)
    coarse = coarsen_event_tensor_plan(
        fine,
        domain=EventDomain("coarse", (TaskAxis("all_rows", 1),)),
        project=lambda _coordinate: (0,),
        name="coarse",
    )
    with pytest.raises(CudaEventLoweringError, match="one consumer per event"):
        generate_cuda_event_counter_lowering(coarse)
