# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.cuda_dynamic_event_dataflow_codegen import (
    generate_cuda_phased_pipeline_ffi_lowering,
    generate_cuda_runtime_event_ffi_lowering,
)
from tile_lifetime.event_dataflow import EventMemoryScope, derive_event_tensor_plan
from tile_lifetime.event_dataflow_examples import pipelined_contract_fold_program, relation_segment_dependence
from tile_lifetime.jax_event_dataflow_ffi import (
    call_cuda_phased_pipeline_ffi,
    cuda_event_ffi_compile_plan,
    runtime_event_ffi_arguments,
)
from tile_lifetime.relation import build_relation_plan


def _runtime_plan(destination_indices: np.ndarray):
    relation = build_relation_plan(
        destination_indices,
        np.ones(destination_indices.shape, dtype=np.float32),
        destination_rank_by_item=np.zeros(4, dtype=np.int32),
        destination_local_item_by_item=np.arange(4, dtype=np.int32),
        padding_quantum=1,
    )
    return derive_event_tensor_plan(
        relation_segment_dependence(relation, visibility_scope=EventMemoryScope.CTA),
        name="runtime_segments",
    )


def test_runtime_event_arguments_are_jax_owned_relation_tables() -> None:
    plan = _runtime_plan(np.asarray([[0, 1], [1, 3], [3, 1]], dtype=np.int32))

    arguments = runtime_event_ffi_arguments(plan, jnp.arange(6, dtype=jnp.float32))

    np.testing.assert_array_equal(np.asarray(arguments.event_counts), np.asarray([1, 3, 0, 2], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(arguments.event_source_offsets),
        np.asarray([0, 1, 4, 4, 6], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.sort(np.asarray(arguments.event_sources)), np.arange(6, dtype=np.int32))
    assert arguments.maximum_count == 3


def test_runtime_event_arguments_follow_relation_mutation() -> None:
    first = _runtime_plan(np.asarray([[0, 1], [1, 3], [3, 1]], dtype=np.int32))
    second = _runtime_plan(np.asarray([[2, 1], [1, 3], [3, 1]], dtype=np.int32))
    payload = jnp.arange(6, dtype=jnp.float32)

    first_arguments = runtime_event_ffi_arguments(first, payload)
    second_arguments = runtime_event_ffi_arguments(second, payload)

    assert not np.array_equal(np.asarray(first_arguments.event_counts), np.asarray(second_arguments.event_counts))
    assert first_arguments.input is payload
    assert second_arguments.input is payload


def test_phased_call_rejects_shape_mismatch_before_ffi_dispatch() -> None:
    generated = generate_cuda_phased_pipeline_ffi_lowering(
        pipelined_contract_fold_program(generation_count=3, pipeline_depth=4),
        dimension=8,
        target_name="shuttle.event_tensor.phased_shape_test_v1",
    )

    with pytest.raises(ValueError, match=r"query.*shape"):
        call_cuda_phased_pipeline_ffi(
            generated,
            query=jnp.zeros((2, 8), dtype=jnp.float32),
            key=jnp.zeros((3, 4, 8), dtype=jnp.float32),
            value=jnp.zeros((3, 4), dtype=jnp.float32),
        )


def test_compile_plan_uses_jax_headers_and_selected_cuda_toolkit(tmp_path: Path) -> None:
    toolkit = tmp_path / "cuda"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    library_directory = toolkit / "lib64"
    library_directory.mkdir()
    cudart = library_directory / "libcudart.so.13"
    cudart.touch()
    include = tmp_path / "jaxlib" / "include"
    include.mkdir(parents=True)
    generated = generate_cuda_runtime_event_ffi_lowering(
        _runtime_plan(np.asarray([[0, 1], [1, 3], [3, 1]], dtype=np.int32)),
        target_name="shuttle.event_tensor.runtime_compile_test_v1",
    )

    plan = cuda_event_ffi_compile_plan(
        generated,
        directory=tmp_path / "build",
        nvcc=nvcc,
        architecture="sm_90a",
        jaxlib_include=include,
    )

    assert plan.source_path.name == "shuttle_event_tensor_runtime_compile_test_v1.cu"
    assert plan.library_path.name == "shuttle_event_tensor_runtime_compile_test_v1.so"
    assert plan.argv[0] == str(nvcc)
    assert "-arch=sm_90a" in plan.argv
    assert str(include) in plan.argv
    assert str(toolkit / "lib64") in plan.argv
    assert "-cudart=none" in plan.argv
    assert str(cudart) in plan.argv
    assert all("torch" not in argument.lower() for argument in plan.argv)
