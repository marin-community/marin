# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Torch-free JAX runtime for generic right-resource Contract/Fold plans."""

from __future__ import annotations

import ctypes
import importlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from clean_routed_streaming_emitter import (
    ExtractedSM100Sources,
    GeneratedPartialMergeFfi,
    PartialMergeScheduleKind,
    PartialValueDType,
    extract_clean_sm100_sources,
    import_extracted_python_sources,
    render_partial_merge_ffi_cuda,
)

from tile_lifetime.cuda_toolchain import cuda_toolkit_link_flags, cuda_toolkit_shared_library_link_flags
from tile_lifetime.right_resource_jax_tables import (
    JaxRightResourceWorkTables,
    derive_right_resource_work_tables,
    right_resource_work_tables_as_jax,
)
from tile_lifetime.sm100_routed_lowering import SM100RoutedStreamingLowering


@dataclass(frozen=True)
class JaxRightResourceRuntimePlan:
    """Compiler-owned JAX operands, physical source, and Fold finalizer."""

    lowering: SM100RoutedStreamingLowering
    sources: ExtractedSM100Sources
    tables: JaxRightResourceWorkTables
    merge_ffi: GeneratedPartialMergeFfi
    source_audit: dict[str, object]


@dataclass(frozen=True)
class JaxRightResourceInputs:
    """JAX-owned payload operands for one grouped Contract/Fold execution."""

    resident: jax.Array
    first_streamed: jax.Array
    second_streamed: jax.Array


@dataclass(frozen=True)
class CompiledRightResourcePhysicalCall:
    """CUTLASS JAX callable with its host-specialized launch capacity."""

    call: Any
    work_capacity: int

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


def prepare_jax_right_resource_runtime(
    msa_root: Path,
    lowering: SM100RoutedStreamingLowering,
    *,
    merge_target: str = "shuttle.partial_state_fold_finalize",
) -> JaxRightResourceRuntimePlan:
    """Prepare Torch-free source and runtime tables without importing CuTe."""
    sources = extract_clean_sm100_sources(
        msa_root,
        lowering,
        paged_key_value=False,
        partial_value_dtype=PartialValueDType.BF16,
        partial_merge_schedule=PartialMergeScheduleKind.ROW_BLOCK,
    )
    work_tables = derive_right_resource_work_tables(lowering.relation, sources.emitter_plan.event_schedule)
    jax_tables = right_resource_work_tables_as_jax(work_tables)
    merge_ffi = render_partial_merge_ffi_cuda(
        sources.emitter_plan.partial_merge,
        target=merge_target,
        partial_count=lowering.selected_count,
        query_count=lowering.query_length,
        query_heads=lowering.key_value_heads * lowering.head_group_size,
        key_value_heads=lowering.key_value_heads,
        value_width=128,
    )
    source_audit = {
        "physical_torch_reference": "torch" in sources.physical_source.lower(),
        "semantic_torch_reference": "torch" in sources.semantic_source.lower(),
        "merge_torch_reference": "torch" in merge_ffi.source.lower(),
        "event_program_fingerprint": sources.emitter_plan.event_schedule.program_fingerprint,
        "event_runtime_fingerprint": sources.emitter_plan.event_schedule.runtime_fingerprint,
        "external_semantic_kernels": sources.emitter_plan.external_semantic_kernels,
    }
    if any(
        source_audit[name]
        for name in (
            "physical_torch_reference",
            "semantic_torch_reference",
            "merge_torch_reference",
            "external_semantic_kernels",
        )
    ):
        raise ValueError(f"JAX right-resource runtime retains a forbidden dependency: {source_audit}")
    return JaxRightResourceRuntimePlan(lowering, sources, jax_tables, merge_ffi, source_audit)


def compile_and_register_partial_merge_ffi(
    generated: GeneratedPartialMergeFfi,
    *,
    directory: Path,
    nvcc: Path,
    architecture: str,
) -> ctypes.CDLL:
    """Compile and register one generated Fold finalizer with JAX."""
    include_directory = Path(jaxlib.__file__).resolve().parent / "include"
    if not include_directory.is_dir():
        raise ValueError(f"jaxlib include directory does not exist: {include_directory}")
    if not nvcc.is_file():
        raise ValueError(f"CUDA compiler does not exist: {nvcc}")
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / f"{generated.handler_symbol}.cu"
    library_path = directory / f"{generated.handler_symbol}.so"
    if not source_path.exists() or source_path.read_text() != generated.source + "\n":
        source_path.write_text(generated.source + "\n")
    argv = (
        str(nvcc),
        "-std=c++17",
        "-O3",
        f"-arch={architecture}",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-I",
        str(include_directory),
        str(source_path),
        "-o",
        str(library_path),
        "-cudart=none",
        *cuda_toolkit_link_flags(nvcc, runtime_search_path=True),
        *cuda_toolkit_shared_library_link_flags(nvcc, ("cudart",)),
    )
    subprocess.run(argv, check=True)
    library = ctypes.CDLL(library_path)
    handler = getattr(library, generated.handler_symbol)
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        generated.target,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )
    return library


def call_partial_merge_ffi(
    generated: GeneratedPartialMergeFfi,
    scalar_state: jax.Array,
    value_state: jax.Array,
    partial_counts: jax.Array,
) -> jax.Array:
    """Finalize generated partial Fold state through JAX typed FFI."""
    expected = (
        ((generated.partial_count, generated.query_count, generated.query_heads), jnp.float32),
        (
            (generated.partial_count, generated.query_count, generated.query_heads, generated.value_width),
            jnp.bfloat16 if generated.value_dtype is PartialValueDType.BF16 else jnp.float32,
        ),
        ((generated.query_count, generated.key_value_heads), jnp.int32),
    )
    for name, operand, (shape, dtype) in zip(
        ("scalar_state", "value_state", "partial_counts"),
        (scalar_state, value_state, partial_counts),
        expected,
        strict=True,
    ):
        if operand.shape != shape or np.dtype(operand.dtype) != np.dtype(dtype):
            raise ValueError(f"{name} must be {np.dtype(dtype)}{shape}, found {operand.dtype}{operand.shape}")
    output = jax.ShapeDtypeStruct(
        (generated.query_count, generated.query_heads, generated.value_width),
        jnp.bfloat16,
    )
    return jax.ffi.ffi_call(generated.target, output, vmap_method="broadcast_all")(
        scalar_state,
        value_state,
        partial_counts,
    )


def compile_right_resource_physical_call(
    plan: JaxRightResourceRuntimePlan,
    *,
    msa_root: Path,
    source_directory: Path | None = None,
) -> CompiledRightResourcePhysicalCall:
    """Compile the extracted grouped body through CUTLASS JAX TVM-FFI."""
    validate_runtime_work_capacity(plan.tables)
    if plan.lowering.head_group_size in (1, 2, 4):
        raise ValueError("the first JAX binding excludes the optional gather-descriptor operand")
    cutlass = importlib.import_module("cutlass")
    cute = importlib.import_module("cutlass.cute")
    cjax = importlib.import_module("cutlass.jax")
    cuda = importlib.import_module("cuda.bindings.driver")
    physical_module = import_extracted_python_sources(
        plan.sources,
        msa_root=msa_root,
        source_directory=source_directory,
    )
    constructor = dict(plan.sources.emitter_plan.physical_constructor)
    constructor["qk_dtype"] = cutlass.BFloat16
    constructor["pv_dtype"] = cutlass.BFloat16
    physical_class = getattr(physical_module, plan.sources.emitter_plan.physical_class)
    physical = physical_class(**constructor)
    lowering = plan.lowering
    work_capacity = plan.tables.work_capacity

    @cute.jit
    def launch(
        stream: cuda.CUstream,
        first_streamed: cute.Tensor,
        second_streamed: cute.Tensor,
        grouped_sources: cute.Tensor,
        partial_slot_sources: cute.Tensor,
        grouped_offsets: cute.Tensor,
        scheduler_metadata: cute.Tensor,
        work_count: cute.Tensor,
        resident: cute.Tensor,
        left_offsets: cute.Tensor,
        right_payload_offsets: cute.Tensor,
        value_state: cute.Tensor,
        scalar_state: cute.Tensor,
    ):
        physical(
            first_streamed,
            second_streamed,
            grouped_sources,
            partial_slot_sources,
            grouped_offsets,
            scheduler_metadata,
            work_count,
            value_state,
            scalar_state,
            None,
            resident,
            None,
            None,
            None,
            left_offsets,
            right_payload_offsets,
            cutlass.Float32(lowering.score_map.scale),
            cutlass.Float32(1.0),
            cutlass.Int32(lowering.right_block_count),
            cutlass.Int32(lowering.key_value_heads),
            cutlass.Int32(lowering.query_length),
            work_capacity,
            stream,
        )

    tensor_spec = cjax.TensorSpec
    matrix_spec = tensor_spec(static=True)
    vector_spec = tensor_spec(static=True)
    input_spec = (
        matrix_spec,
        matrix_spec,
        matrix_spec,
        matrix_spec,
        matrix_spec,
        matrix_spec,
        vector_spec,
        matrix_spec,
        vector_spec,
        vector_spec,
    )
    output_spec = (matrix_spec, matrix_spec)
    partial_count = lowering.selected_count
    query_heads = lowering.key_value_heads * lowering.head_group_size
    value_shape = jax.ShapeDtypeStruct((partial_count * lowering.query_length * query_heads, 128), jnp.bfloat16)
    scalar_shape = jax.ShapeDtypeStruct((partial_count, lowering.query_length, query_heads), jnp.float32)
    return CompiledRightResourcePhysicalCall(
        call=cjax.cutlass_call(
            launch,
            output_shape_dtype=(value_shape, scalar_shape),
            input_spec=input_spec,
            output_spec=output_spec,
            use_static_tensors=True,
        ),
        work_capacity=work_capacity,
    )


def call_right_resource_physical(
    compiled: CompiledRightResourcePhysicalCall,
    plan: JaxRightResourceRuntimePlan,
    inputs: JaxRightResourceInputs,
) -> tuple[jax.Array, jax.Array]:
    """Execute the generic grouped body and return exposed Fold partials."""
    lowering = plan.lowering
    query_heads = lowering.key_value_heads * lowering.head_group_size
    resident = inputs.resident.reshape(lowering.query_length * query_heads, 128)
    tables = plan.tables
    validate_runtime_work_capacity(tables)
    if tables.work_capacity != compiled.work_capacity:
        raise ValueError(
            "right-resource plan capacity does not match the host-specialized physical launch: "
            f"plan={tables.work_capacity}, compiled={compiled.work_capacity}"
        )
    value_state, scalar_state = compiled(
        inputs.first_streamed,
        inputs.second_streamed,
        tables.right_to_left_sources,
        tables.partial_slot_sources,
        tables.right_to_left_offsets,
        tables.scheduler_metadata,
        tables.work_count,
        resident,
        tables.left_offsets,
        tables.right_payload_offsets,
    )
    return value_state.reshape(lowering.selected_count, lowering.query_length, query_heads, 128), scalar_state


def validate_runtime_work_capacity(tables: JaxRightResourceWorkTables) -> int:
    """Reject runtime work metadata that exceeds the specialized launch."""
    work_count = np.asarray(jax.device_get(tables.work_count))
    if work_count.shape != (1,) or not np.issubdtype(work_count.dtype, np.integer):
        raise ValueError(f"runtime work count must be one integer scalar, found {work_count.dtype}{work_count.shape}")
    count = int(work_count[0])
    if count < 0 or count > tables.work_capacity:
        raise ValueError(
            "runtime work count exceeds the host-specialized launch capacity: "
            f"count={count}, capacity={tables.work_capacity}"
        )
    return count
