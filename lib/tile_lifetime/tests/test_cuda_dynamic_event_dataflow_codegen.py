# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from shuttle.ir import DType
from tile_lifetime.cuda_dynamic_event_dataflow_codegen import (
    CudaDynamicEventLoweringError,
    CudaEventFfiKind,
    generate_cuda_phased_pipeline_ffi_lowering,
    generate_cuda_phased_pipeline_lowering,
    generate_cuda_runtime_event_ffi_lowering,
    generate_cuda_runtime_event_lowering,
)
from tile_lifetime.event_dataflow import EventMemoryScope, derive_event_tensor_plan
from tile_lifetime.event_dataflow_examples import (
    pipelined_contract_fold_program,
    relation_segment_dependence,
)
from tile_lifetime.relation import build_relation_plan


def _relation_plan(destination_indices: np.ndarray):
    return build_relation_plan(
        destination_indices,
        np.ones(destination_indices.shape, dtype=np.float32),
        destination_rank_by_item=np.asarray([1, 0, 1, 0], dtype=np.int32),
        destination_local_item_by_item=np.asarray([1, 0, 0, 1], dtype=np.int32),
        padding_quantum=1,
    )


def test_runtime_segment_lowering_keeps_counts_and_offsets_as_device_inputs() -> None:
    relation = _relation_plan(np.asarray([[0, 1], [1, 3], [3, 1]], dtype=np.int32))
    assert np.any(relation.group_count == 0)
    dependence = relation_segment_dependence(relation, visibility_scope=EventMemoryScope.CTA)
    plan = derive_event_tensor_plan(dependence, name="runtime_segments")

    generated = generate_cuda_runtime_event_lowering(plan)

    assert "const int* event_counts" in generated.source
    assert "const int* event_source_offsets" in generated.source
    assert "const int* event_sources" in generated.source
    assert "if (producer_count == 0)" in generated.source
    assert "producer_count != source_end - source_begin" in generated.source
    assert "remaining = producer_count" in generated.source
    assert "__threadfence_block()" in generated.source
    assert "atomicSub(&remaining, 1)" in generated.source
    assert "prior_remaining == 1" in generated.source
    assert "__device__ __constant__" not in generated.source
    assert all(name not in generated.source.lower() for name in ("moe", "expert", "attention", "flash"))


def test_runtime_relation_mutation_changes_plan_not_generated_kernel_family() -> None:
    first_relation = _relation_plan(np.asarray([[0, 1], [1, 3], [3, 1]], dtype=np.int32))
    second_relation = _relation_plan(np.asarray([[2, 1], [1, 3], [3, 1]], dtype=np.int32))
    first = generate_cuda_runtime_event_lowering(
        derive_event_tensor_plan(
            relation_segment_dependence(first_relation, visibility_scope=EventMemoryScope.CTA),
            name="runtime_segments",
        )
    )
    second = generate_cuda_runtime_event_lowering(
        derive_event_tensor_plan(
            relation_segment_dependence(second_relation, visibility_scope=EventMemoryScope.CTA),
            name="runtime_segments",
        )
    )

    assert first.source_sha256 == second.source_sha256
    assert first.plan_fingerprint != second.plan_fingerprint


def test_phased_pipeline_lowering_emits_real_generation_wait_and_release() -> None:
    program = pipelined_contract_fold_program(generation_count=3, pipeline_depth=4)

    generated = generate_cuda_phased_pipeline_lowering(program)

    assert generated.generation_count == 3
    assert generated.pipeline_depth == 4
    assert "wait_for_generation(&slot_reusable[slot], generation)" in generated.source
    assert "wait_for_generation(&first_ready[slot], generation)" in generated.source
    assert "wait_for_generation(&state_ready[slot], generation)" in generated.source
    assert "__threadfence_block()" in generated.source
    assert "atomicExch(&slot_reusable[slot], generation + 1)" in generated.source
    assert "running_weighted / running_sum" in generated.source
    assert all(name not in generated.source.lower() for name in ("attention", "flash", "gdn", "moe"))


def test_phased_schedule_shape_is_runtime_not_a_kernel_identity_switch() -> None:
    shallow = generate_cuda_phased_pipeline_lowering(
        pipelined_contract_fold_program(generation_count=2, pipeline_depth=2)
    )
    deeper = generate_cuda_phased_pipeline_lowering(
        pipelined_contract_fold_program(generation_count=4, pipeline_depth=8)
    )

    assert shallow.source_sha256 == deeper.source_sha256
    assert shallow.plan_fingerprint != deeper.plan_fingerprint


def test_phased_pipeline_rejects_more_slots_than_the_bounded_physical_template() -> None:
    with pytest.raises(CudaDynamicEventLoweringError, match="at most 32 slots"):
        generate_cuda_phased_pipeline_lowering(pipelined_contract_fold_program(generation_count=2, pipeline_depth=33))


def test_runtime_event_ffi_uses_same_device_body_without_torch_boundary() -> None:
    relation = _relation_plan(np.asarray([[0, 1], [1, 3], [3, 1]], dtype=np.int32))
    plan = derive_event_tensor_plan(
        relation_segment_dependence(relation, visibility_scope=EventMemoryScope.CTA),
        name="runtime_segments",
    )

    torch_lowering = generate_cuda_runtime_event_lowering(plan)
    ffi_lowering = generate_cuda_runtime_event_ffi_lowering(
        plan,
        target_name="shuttle.event_tensor.runtime_test_v1",
    )

    assert ffi_lowering.kind is CudaEventFfiKind.RUNTIME_RELATION
    assert ffi_lowering.device_source_sha256 == torch_lowering.device_source_sha256
    assert [(value.name, value.dtype, value.shape) for value in ffi_lowering.inputs] == [
        ("input", DType.FP32, (6,)),
        ("event_counts", DType.INT32, (4,)),
        ("event_source_offsets", DType.INT32, (5,)),
        ("event_sources", DType.INT32, (6,)),
    ]
    assert [(value.name, value.dtype, value.shape) for value in ffi_lowering.outputs] == [
        ("partials", DType.FP32, (6,)),
        ("output", DType.FP32, (4,)),
    ]
    assert "XLA_FFI_DEFINE_HANDLER_SYMBOL" in ffi_lowering.source
    assert "torch" not in ffi_lowering.source.lower()


def test_phased_event_ffi_uses_same_device_body_and_mutates_by_attributes() -> None:
    program = pipelined_contract_fold_program(generation_count=3, pipeline_depth=4)

    torch_lowering = generate_cuda_phased_pipeline_lowering(program)
    ffi_lowering = generate_cuda_phased_pipeline_ffi_lowering(
        program,
        dimension=128,
        target_name="shuttle.event_tensor.phased_test_v1",
    )

    assert ffi_lowering.kind is CudaEventFfiKind.PHASED_PIPELINE
    assert ffi_lowering.device_source_sha256 == torch_lowering.device_source_sha256
    assert [(value.name, value.dtype, value.shape) for value in ffi_lowering.inputs] == [
        ("query", DType.FP32, (3, 128)),
        ("key", DType.FP32, (3, 4, 128)),
        ("value", DType.FP32, (3, 4)),
    ]
    assert [(value.name, value.dtype, value.shape) for value in ffi_lowering.outputs] == [
        ("output", DType.FP32, (3,)),
    ]
    assert 'Attr<std::int64_t>("generation_count")' in ffi_lowering.source
    assert 'Attr<std::int64_t>("pipeline_depth")' in ffi_lowering.source
    assert 'Attr<std::int64_t>("dimension")' in ffi_lowering.source
    assert "torch" not in ffi_lowering.source.lower()
