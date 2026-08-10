# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate CUDA for rank-two Map/Fold programs over one logical axis.

The physical skeleton is intentionally small: one block owns one item on the
unreduced axis, evaluates one or more generated contribution ASTs, combines
them with a fixed reduction tree, and either emits the reduced value or applies
a generated final Map over the full row/column.  It has no normalization or
model-specific semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.ir import DType
from tile_lifetime.tensor_program import (
    ScalarExpression,
    ScalarExpressionKind,
    scalar_expression_inputs,
    serialize_scalar_expression,
)


class AxisFoldDirection(StrEnum):
    """Logical rank-two axis reduced by a generated kernel."""

    ROWS = "rows"
    COLUMNS = "columns"


class AxisFoldInputLayout(StrEnum):
    """Broadcast/index relation for one scalar input."""

    ELEMENT = "element"
    ROW = "row"
    COLUMN = "column"
    SCALAR = "scalar"


class AxisFoldOutputKind(StrEnum):
    """Whether finalization emits one reduced value or a full matrix Map."""

    REDUCED = "reduced"
    ELEMENT = "element"


class AxisFoldReassociation(StrEnum):
    """Floating-point ordering contract for the reduction tree."""

    DETERMINISTIC_TREE = "deterministic_tree"
    SOURCE_ORDERED = "source_ordered"


class AxisFoldPipelineSchedule(StrEnum):
    """Physical kernelization policy for a semantic axis-Fold pipeline."""

    SEPARATE_STAGES = "separate_stages"
    COALESCE_COMPATIBLE_ROW_STAGES = "coalesce_compatible_row_stages"


@dataclass(frozen=True)
class AxisFoldInput:
    """One typed tensor input and its logical broadcast relation."""

    name: str
    dtype: DType
    layout: AxisFoldInputLayout

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"axis-Fold input name must be an identifier: {self.name!r}")
        if self.dtype not in {DType.BF16, DType.FP32}:
            raise ValueError("axis-Fold inputs must be BF16 or FP32")


@dataclass(frozen=True)
class AxisFoldReduction:
    """One scalar contribution accumulated over the selected axis."""

    name: str
    contribution: ScalarExpression

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"axis-Fold reduction name must be an identifier: {self.name!r}")


@dataclass(frozen=True)
class AxisFoldProgram:
    """Generic rank-two Map/Fold semantics plus a bounded CUDA schedule."""

    rows: int
    columns: int
    inputs: tuple[AxisFoldInput, ...]
    reductions: tuple[AxisFoldReduction, ...]
    reduction_axis: AxisFoldDirection
    output_kind: AxisFoldOutputKind
    output_expression: ScalarExpression
    output_dtype: DType
    threads: int = 256
    groups_per_block: int = 1
    outputs_per_group: int = 1
    reassociation: AxisFoldReassociation = AxisFoldReassociation.DETERMINISTIC_TREE

    def __post_init__(self) -> None:
        if min(self.rows, self.columns, self.threads, self.groups_per_block, self.outputs_per_group) <= 0:
            raise ValueError("axis-Fold dimensions and thread count must be positive")
        if self.threads & (self.threads - 1):
            raise ValueError("axis-Fold thread count must be a power of two")
        if self.groups_per_block & (self.groups_per_block - 1):
            raise ValueError("axis-Fold groups per block must be a power of two")
        if self.threads % self.groups_per_block:
            raise ValueError("axis-Fold groups per block must divide the thread count")
        if self.outputs_per_group & (self.outputs_per_group - 1):
            raise ValueError("axis-Fold outputs per group must be a power of two")
        if self.outputs_per_group > 1 and self.groups_per_block == 1:
            raise ValueError("multiple axis-Fold outputs per group require tiled row-axis scheduling")
        if self.groups_per_block > 1 and (
            self.reduction_axis is not AxisFoldDirection.ROWS or self.output_kind is not AxisFoldOutputKind.REDUCED
        ):
            raise ValueError("tiled axis-Fold groups currently require a reduced row-axis Fold")
        if self.output_dtype not in {DType.BF16, DType.FP32}:
            raise ValueError("axis-Fold output must be BF16 or FP32")
        input_names = tuple(value.name for value in self.inputs)
        reduction_names = tuple(value.name for value in self.reductions)
        if not input_names or len(set(input_names)) != len(input_names):
            raise ValueError("axis-Fold input names must be nonempty and unique")
        if not reduction_names or len(set(reduction_names)) != len(reduction_names):
            raise ValueError("axis-Fold reduction names must be nonempty and unique")
        if set(input_names) & set(reduction_names):
            raise ValueError("axis-Fold input and reduction names must not overlap")
        for reduction in self.reductions:
            unknown = scalar_expression_inputs(reduction.contribution) - set(input_names)
            if unknown:
                raise ValueError(f"axis-Fold contribution {reduction.name!r} has unknown inputs {sorted(unknown)}")
        allowed_output = set(input_names) | set(reduction_names)
        unknown_output = scalar_expression_inputs(self.output_expression) - allowed_output
        if unknown_output:
            raise ValueError(f"axis-Fold output expression has unknown inputs {sorted(unknown_output)}")
        if self.output_kind is AxisFoldOutputKind.REDUCED:
            invalid_inputs = {
                value.name
                for value in self.inputs
                if value.layout is AxisFoldInputLayout.ELEMENT
                or (self.reduction_axis is AxisFoldDirection.ROWS and value.layout is AxisFoldInputLayout.ROW)
                or (self.reduction_axis is AxisFoldDirection.COLUMNS and value.layout is AxisFoldInputLayout.COLUMN)
            }
            if scalar_expression_inputs(self.output_expression) & invalid_inputs:
                raise ValueError("reduced axis-Fold output cannot reference an input varying over the folded axis")
        if self.reassociation is AxisFoldReassociation.SOURCE_ORDERED and self.threads != 1:
            raise ValueError("source-ordered axis Fold requires one worker")

    @property
    def semantic_fingerprint(self) -> str:
        """Return a stable digest excluding physical thread count."""
        payload = {
            "rows": self.rows,
            "columns": self.columns,
            "inputs": [
                {"name": value.name, "dtype": value.dtype.value, "layout": value.layout.value} for value in self.inputs
            ],
            "reductions": [
                {"name": value.name, "contribution": json.loads(serialize_scalar_expression(value.contribution))}
                for value in self.reductions
            ],
            "reduction_axis": self.reduction_axis.value,
            "output_kind": self.output_kind.value,
            "output_expression": json.loads(serialize_scalar_expression(self.output_expression)),
            "output_dtype": self.output_dtype.value,
            "reassociation": self.reassociation.value,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(frozen=True)
class GeneratedCudaAxisFold:
    """Self-contained Torch CUDA extension source for one axis-Fold program."""

    source: str
    semantic_fingerprint: str
    source_sha256: str


@dataclass(frozen=True)
class CudaAxisFoldFfiBuffer:
    """One fixed-rank buffer in a generated XLA typed-FFI signature."""

    name: str
    dtype: DType
    shape: tuple[int, ...]

    @property
    def rank(self) -> int:
        """Static rank encoded in the typed-FFI signature."""
        return len(self.shape)


@dataclass(frozen=True)
class GeneratedCudaAxisFoldFfi:
    """Torch-free typed-FFI source for a sequence of axis-Fold programs."""

    source: str
    target_name: str
    handler_symbol: str
    inputs: tuple[CudaAxisFoldFfiBuffer, ...]
    outputs: tuple[CudaAxisFoldFfiBuffer, ...]
    semantic_fingerprints: tuple[str, ...]
    pipeline_schedule: AxisFoldPipelineSchedule
    source_sha256: str


@dataclass(frozen=True)
class AxisFoldPipelineStage:
    """One topologically ordered Fold whose result may feed later Folds."""

    output_name: str
    program: AxisFoldProgram
    expose_output: bool

    def __post_init__(self) -> None:
        if not self.output_name.isidentifier():
            raise ValueError(f"axis-Fold pipeline output must be an identifier: {self.output_name!r}")


@dataclass(frozen=True)
class AxisFoldPipeline:
    """A bounded dataflow composition of generic axis-Fold programs."""

    stages: tuple[AxisFoldPipelineStage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("axis-Fold pipeline requires at least one stage")
        _axis_fold_pipeline_buffers(self)


def generate_cuda_axis_fold(program: AxisFoldProgram) -> GeneratedCudaAxisFold:
    """Render a generic deterministic Map/Fold CUDA extension."""
    if program.groups_per_block > 1:
        return _generate_tiled_row_axis_fold(program)
    argument_declarations = ",\n    ".join(f"torch::Tensor {value.name}" for value in program.inputs)
    pointer_declarations = "\n".join(_pointer_declaration(value) for value in program.inputs)
    wrapper_checks = "\n".join(_wrapper_check(value, program) for value in program.inputs)
    local_reductions = "\n".join(f"  float local_{value.name} = 0.0f;" for value in program.reductions)
    contribution_updates = "\n".join(
        f"    local_{value.name} = __fadd_rn(local_{value.name}, "
        f"{_cuda_expression(value.contribution, _input_aliases(program.inputs, 'row', 'column'))});"
        for value in program.reductions
    )
    shared_declarations = "\n".join(f"  __shared__ float shared_{value.name}[kThreads];" for value in program.reductions)
    shared_initialization = "\n".join(
        f"  shared_{value.name}[threadIdx.x] = local_{value.name};" for value in program.reductions
    )
    shared_updates = "\n".join(
        f"      shared_{value.name}[threadIdx.x] = __fadd_rn("
        f"shared_{value.name}[threadIdx.x], shared_{value.name}[threadIdx.x + stride]);"
        for value in program.reductions
    )
    reduction_aliases = {value.name: f"shared_{value.name}[0]" for value in program.reductions}
    if program.output_kind is AxisFoldOutputKind.ELEMENT:
        output_body = _element_output_body(program, reduction_aliases)
        output_shape = "{kRows, kColumns}"
        output_check = (
            "  TORCH_CHECK(output.dim() == 2 && output.size(0) == kRows && output.size(1) == kColumns,\n"
            '              "axis-Fold element output shape mismatch");'
        )
    else:
        output_body = _reduced_output_body(program, reduction_aliases)
        output_extent = "kColumns" if program.reduction_axis is AxisFoldDirection.ROWS else "kRows"
        output_shape = f"{{{output_extent}}}"
        output_check = (
            f"  TORCH_CHECK(output.dim() == 1 && output.size(0) == {output_extent},\n"
            '              "axis-Fold reduced output shape mismatch");'
        )
    output_torch_dtype = "torch::kBFloat16" if program.output_dtype is DType.BF16 else "torch::kFloat32"
    output_pointer_type = "__nv_bfloat16" if program.output_dtype is DType.BF16 else "float"
    reduction_extent = "kRows" if program.reduction_axis is AxisFoldDirection.ROWS else "kColumns"
    group_extent = "kColumns" if program.reduction_axis is AxisFoldDirection.ROWS else "kRows"
    coordinate_setup = (
        "    const int row = reduction_index;\n    const int column = group;"
        if program.reduction_axis is AxisFoldDirection.ROWS
        else "    const int row = group;\n    const int column = reduction_index;"
    )
    source = f"""
// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic rank-two Map/Fold semantics; do not edit.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {{

constexpr int kRows = {program.rows};
constexpr int kColumns = {program.columns};
constexpr int kThreads = {program.threads};

__global__ __launch_bounds__(kThreads) void shuttle_axis_fold_kernel(
    {', '.join(_kernel_parameter(value) for value in program.inputs)},
    {output_pointer_type}* output) {{
{shared_declarations}
  const int group = blockIdx.x;
  if (group >= {group_extent}) return;
{local_reductions}
  for (int reduction_index = threadIdx.x; reduction_index < {reduction_extent};
       reduction_index += kThreads) {{
{coordinate_setup}
{contribution_updates}
  }}
{shared_initialization}
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {{
    if (threadIdx.x < stride) {{
{shared_updates}
    }}
    __syncthreads();
  }}
{output_body}
}}

}}  // namespace

torch::Tensor shuttle_axis_fold_out(
    {argument_declarations},
    torch::Tensor output) {{
{wrapper_checks}
  TORCH_CHECK(output.is_cuda(), "axis-Fold output must be CUDA");
  TORCH_CHECK(output.scalar_type() == {output_torch_dtype}, "axis-Fold output dtype mismatch");
  TORCH_CHECK(output.is_contiguous(), "axis-Fold output must be contiguous");
{output_check}
  const c10::cuda::CUDAGuard device_guard(output.device());
{pointer_declarations}
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuttle_axis_fold_kernel<<<{group_extent}, kThreads, 0, stream>>>(
      {', '.join(f'{value.name}_pointer' for value in program.inputs)},
      reinterpret_cast<{output_pointer_type}*>(output.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}}

torch::Tensor shuttle_axis_fold(
    {argument_declarations}) {{
  auto output = torch::empty({output_shape}, {program.inputs[0].name}.options().dtype({output_torch_dtype}));
  return shuttle_axis_fold_out({', '.join(value.name for value in program.inputs)}, output);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
  module.def("run", &shuttle_axis_fold);
  module.def("run_out", &shuttle_axis_fold_out);
}}
""".strip()
    return GeneratedCudaAxisFold(
        source=source,
        semantic_fingerprint=program.semantic_fingerprint,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def generate_cuda_axis_fold_ffi(
    programs: tuple[AxisFoldProgram, ...],
    *,
    target_name: str,
) -> GeneratedCudaAxisFoldFfi:
    """Render fixed-shape axis Folds behind one CUDA XLA typed-FFI call.

    The handler is only a runtime boundary. Each device kernel is still
    generated from the program's scalar contribution, reducer, finalization,
    broadcast relations, and physical Fold schedule.
    """
    if not programs:
        raise ValueError("axis-Fold FFI generation requires at least one program")
    pipeline = AxisFoldPipeline(
        tuple(
            AxisFoldPipelineStage(output_name=f"output{index}", program=program, expose_output=True)
            for index, program in enumerate(programs)
        )
    )
    return generate_cuda_axis_fold_pipeline_ffi(pipeline, target_name=target_name)


def generate_cuda_axis_fold_pipeline_ffi(
    pipeline: AxisFoldPipeline,
    *,
    target_name: str,
    schedule: AxisFoldPipelineSchedule = AxisFoldPipelineSchedule.SEPARATE_STAGES,
) -> GeneratedCudaAxisFoldFfi:
    """Render a topological Fold pipeline with internal scratch values."""
    handler_symbol = target_name.replace(".", "_")
    if not handler_symbol.isidentifier():
        raise ValueError(f"FFI target does not map to a C identifier: {target_name!r}")
    inputs, stage_outputs, outputs = _axis_fold_pipeline_buffers(pipeline)
    input_arguments = ",\n    ".join(
        f"ffi::Buffer<{_ffi_dtype(value.dtype)}, {value.rank}> {value.name}_buffer" for value in inputs
    )
    output_arguments = ",\n    ".join(
        f"ffi::Result<ffi::Buffer<{_ffi_dtype(value.dtype)}, {value.rank}>> {value.name}" for value in outputs
    )
    input_pointers = "\n".join(_ffi_input_pointer(value) for value in inputs)
    output_pointers = "\n".join(_ffi_output_pointer(value) for value in outputs)
    kernel_groups = _axis_fold_kernel_groups(pipeline, schedule)
    kernels = "\n\n".join(_ffi_axis_fold_kernel_group(pipeline, group) for group in kernel_groups)
    scratch_declarations = "\n".join(
        _ffi_pipeline_scratch(stage_output)
        for stage, stage_output in zip(pipeline.stages, stage_outputs, strict=True)
        if not stage.expose_output
    )
    exposed_data = {value.name: f"{value.name}_data" for value in outputs}
    launches = "\n".join(
        _ffi_axis_fold_group_launch(pipeline, group, exposed_data=exposed_data) for group in kernel_groups
    )
    input_bindings = "\n".join(f"      .Arg<ffi::Buffer<{_ffi_dtype(value.dtype)}, {value.rank}>>()" for value in inputs)
    output_bindings = "\n".join(
        f"      .Ret<ffi::Buffer<{_ffi_dtype(value.dtype)}, {value.rank}>>()" for value in outputs
    )
    source = f"""
// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic rank-two Map/Fold semantics; do not edit.
#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {{
std::atomic<int> call_count{{0}};

{kernels}

ffi::Error ShuttleAxisFoldRegion(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    {input_arguments},
    {output_arguments}) {{
{input_pointers}
{output_pointers}
{scratch_declarations}
{launches}
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttleAxisFoldRegionBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
{input_bindings}
{output_bindings};
}}
}}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {handler_symbol},
    ShuttleAxisFoldRegion,
    ShuttleAxisFoldRegionBinding());

extern "C" int shuttle_axis_fold_ffi_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
""".strip()
    return GeneratedCudaAxisFoldFfi(
        source=source,
        target_name=target_name,
        handler_symbol=handler_symbol,
        inputs=inputs,
        outputs=outputs,
        semantic_fingerprints=tuple(stage.program.semantic_fingerprint for stage in pipeline.stages),
        pipeline_schedule=schedule,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def _generate_tiled_row_axis_fold(program: AxisFoldProgram) -> GeneratedCudaAxisFold:
    """Render a coalesced multi-group schedule for a row-axis reduction."""
    argument_declarations = ",\n    ".join(f"torch::Tensor {value.name}" for value in program.inputs)
    pointer_declarations = "\n".join(_pointer_declaration(value) for value in program.inputs)
    wrapper_checks = "\n".join(_wrapper_check(value, program) for value in program.inputs)
    local_reductions = "\n".join(f"  float local_{value.name}[kOutputsPerGroup] = {{}};" for value in program.reductions)
    contribution_updates = "\n".join(
        f"        local_{value.name}[output_lane] = __fadd_rn(local_{value.name}[output_lane], "
        f"{_cuda_expression(value.contribution, _input_aliases(program.inputs, 'row', 'column'))});"
        for value in program.reductions
    )
    shared_declarations = "\n".join(
        f"  __shared__ float shared_{value.name}[kThreads * kOutputsPerGroup];" for value in program.reductions
    )
    shared_initialization = "\n".join(
        f"    shared_{value.name}[shared_index] = local_{value.name}[output_lane];" for value in program.reductions
    )
    shared_updates = "\n".join(
        f"        shared_{value.name}[shared_index] = __fadd_rn("
        f"shared_{value.name}[shared_index], shared_{value.name}[shared_index + stride * kGroupsPerBlock]);"
        for value in program.reductions
    )
    reduction_aliases = {value.name: f"shared_{value.name}[shared_index]" for value in program.reductions}
    output_aliases = _reduced_output_aliases(program.inputs, program.reduction_axis, "group") | reduction_aliases
    output_expression = _cuda_expression(program.output_expression, output_aliases)
    output_store = _output_store(program.output_dtype, "group", output_expression)
    output_torch_dtype = "torch::kBFloat16" if program.output_dtype is DType.BF16 else "torch::kFloat32"
    output_pointer_type = "__nv_bfloat16" if program.output_dtype is DType.BF16 else "float"
    source = f"""
// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic rank-two Map/Fold semantics; do not edit.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {{

constexpr int kRows = {program.rows};
constexpr int kColumns = {program.columns};
constexpr int kThreads = {program.threads};
constexpr int kGroupsPerBlock = {program.groups_per_block};
constexpr int kOutputsPerGroup = {program.outputs_per_group};
constexpr int kReductionLanes = kThreads / kGroupsPerBlock;

__global__ __launch_bounds__(kThreads) void shuttle_axis_fold_kernel(
    {', '.join(_kernel_parameter(value) for value in program.inputs)},
    {output_pointer_type}* output) {{
{shared_declarations}
  const int group_lane = threadIdx.x % kGroupsPerBlock;
  const int reduction_lane = threadIdx.x / kGroupsPerBlock;
{local_reductions}
  #pragma unroll
  for (int output_lane = 0; output_lane < kOutputsPerGroup; ++output_lane) {{
    const int group = blockIdx.x * kGroupsPerBlock * kOutputsPerGroup
        + output_lane * kGroupsPerBlock + group_lane;
    if (group < kColumns) {{
      for (int row = reduction_lane; row < kRows; row += kReductionLanes) {{
        const int column = group;
{contribution_updates}
      }}
    }}
  }}
  #pragma unroll
  for (int output_lane = 0; output_lane < kOutputsPerGroup; ++output_lane) {{
    const int shared_index = output_lane * kThreads + threadIdx.x;
{shared_initialization}
  }}
  __syncthreads();
  for (int stride = kReductionLanes / 2; stride > 0; stride /= 2) {{
    if (reduction_lane < stride) {{
      #pragma unroll
      for (int output_lane = 0; output_lane < kOutputsPerGroup; ++output_lane) {{
        const int shared_index = output_lane * kThreads + threadIdx.x;
{shared_updates}
      }}
    }}
    __syncthreads();
  }}
  if (reduction_lane == 0) {{
    #pragma unroll
    for (int output_lane = 0; output_lane < kOutputsPerGroup; ++output_lane) {{
      const int group = blockIdx.x * kGroupsPerBlock * kOutputsPerGroup
          + output_lane * kGroupsPerBlock + group_lane;
      const int shared_index = output_lane * kThreads + threadIdx.x;
      if (group < kColumns) {{
        {output_store}
      }}
    }}
  }}
}}

}}  // namespace

torch::Tensor shuttle_axis_fold_out(
    {argument_declarations},
    torch::Tensor output) {{
{wrapper_checks}
  TORCH_CHECK(output.is_cuda(), "axis-Fold output must be CUDA");
  TORCH_CHECK(output.scalar_type() == {output_torch_dtype}, "axis-Fold output dtype mismatch");
  TORCH_CHECK(output.is_contiguous(), "axis-Fold output must be contiguous");
  TORCH_CHECK(output.dim() == 1 && output.size(0) == kColumns,
              "axis-Fold reduced output shape mismatch");
  const c10::cuda::CUDAGuard device_guard(output.device());
{pointer_declarations}
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kBlocks =
      (kColumns + kGroupsPerBlock * kOutputsPerGroup - 1) / (kGroupsPerBlock * kOutputsPerGroup);
  shuttle_axis_fold_kernel<<<kBlocks, kThreads, 0, stream>>>(
      {', '.join(f'{value.name}_pointer' for value in program.inputs)},
      reinterpret_cast<{output_pointer_type}*>(output.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}}

torch::Tensor shuttle_axis_fold(
    {argument_declarations}) {{
  auto output = torch::empty({{kColumns}}, {program.inputs[0].name}.options().dtype({output_torch_dtype}));
  return shuttle_axis_fold_out({', '.join(value.name for value in program.inputs)}, output);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
  module.def("run", &shuttle_axis_fold);
  module.def("run_out", &shuttle_axis_fold_out);
}}
""".strip()
    return GeneratedCudaAxisFold(
        source=source,
        semantic_fingerprint=program.semantic_fingerprint,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def _ffi_input_buffers(programs: tuple[AxisFoldProgram, ...]) -> tuple[CudaAxisFoldFfiBuffer, ...]:
    ordered: list[CudaAxisFoldFfiBuffer] = []
    by_name: dict[str, CudaAxisFoldFfiBuffer] = {}
    for program in programs:
        for value in program.inputs:
            candidate = CudaAxisFoldFfiBuffer(
                name=value.name,
                dtype=value.dtype,
                shape={
                    AxisFoldInputLayout.ELEMENT: (program.rows, program.columns),
                    AxisFoldInputLayout.ROW: (program.rows,),
                    AxisFoldInputLayout.COLUMN: (program.columns,),
                    AxisFoldInputLayout.SCALAR: (),
                }[value.layout],
            )
            previous = by_name.get(value.name)
            if previous is not None and previous != candidate:
                raise ValueError(f"axis-Fold FFI input {value.name!r} has incompatible uses: {previous} and {candidate}")
            if previous is None:
                by_name[value.name] = candidate
                ordered.append(candidate)
    return tuple(ordered)


def _axis_fold_input_buffer(value: AxisFoldInput, program: AxisFoldProgram) -> CudaAxisFoldFfiBuffer:
    return CudaAxisFoldFfiBuffer(
        name=value.name,
        dtype=value.dtype,
        shape={
            AxisFoldInputLayout.ELEMENT: (program.rows, program.columns),
            AxisFoldInputLayout.ROW: (program.rows,),
            AxisFoldInputLayout.COLUMN: (program.columns,),
            AxisFoldInputLayout.SCALAR: (),
        }[value.layout],
    )


def _axis_fold_output_buffer(stage: AxisFoldPipelineStage) -> CudaAxisFoldFfiBuffer:
    program = stage.program
    shape = (
        (program.rows, program.columns)
        if program.output_kind is AxisFoldOutputKind.ELEMENT
        else ((program.columns,) if program.reduction_axis is AxisFoldDirection.ROWS else (program.rows,))
    )
    return CudaAxisFoldFfiBuffer(name=stage.output_name, dtype=program.output_dtype, shape=shape)


def _axis_fold_pipeline_buffers(
    pipeline: AxisFoldPipeline,
) -> tuple[
    tuple[CudaAxisFoldFfiBuffer, ...],
    tuple[CudaAxisFoldFfiBuffer, ...],
    tuple[CudaAxisFoldFfiBuffer, ...],
]:
    external_inputs: list[CudaAxisFoldFfiBuffer] = []
    external_by_name: dict[str, CudaAxisFoldFfiBuffer] = {}
    stage_outputs: list[CudaAxisFoldFfiBuffer] = []
    produced_by_name: dict[str, CudaAxisFoldFfiBuffer] = {}
    exposed_outputs: list[CudaAxisFoldFfiBuffer] = []
    for stage in pipeline.stages:
        for value in stage.program.inputs:
            candidate = _axis_fold_input_buffer(value, stage.program)
            produced = produced_by_name.get(value.name)
            if produced is not None:
                if produced != candidate:
                    raise ValueError(
                        f"axis-Fold pipeline value {value.name!r} has incompatible producer and consumer uses: "
                        f"{produced} and {candidate}"
                    )
                continue
            previous = external_by_name.get(value.name)
            if previous is not None and previous != candidate:
                raise ValueError(
                    f"axis-Fold pipeline input {value.name!r} has incompatible uses: {previous} and {candidate}"
                )
            if previous is None:
                external_by_name[value.name] = candidate
                external_inputs.append(candidate)
        if stage.output_name in produced_by_name or stage.output_name in external_by_name:
            raise ValueError(f"axis-Fold pipeline output {stage.output_name!r} is not single-assignment")
        output = _axis_fold_output_buffer(stage)
        produced_by_name[stage.output_name] = output
        stage_outputs.append(output)
        if stage.expose_output:
            exposed_outputs.append(output)
    if not exposed_outputs:
        raise ValueError("axis-Fold pipeline must expose at least one output")
    return tuple(external_inputs), tuple(stage_outputs), tuple(exposed_outputs)


def _ffi_dtype(dtype: DType) -> str:
    return {DType.BF16: "ffi::BF16", DType.FP32: "ffi::F32"}[dtype]


def _ffi_input_pointer(value: CudaAxisFoldFfiBuffer) -> str:
    if value.dtype is DType.BF16:
        return (
            f"  const auto* {value.name} = reinterpret_cast<const __nv_bfloat16*>(" f"{value.name}_buffer.typed_data());"
        )
    return f"  const auto* {value.name} = {value.name}_buffer.typed_data();"


def _ffi_output_pointer(value: CudaAxisFoldFfiBuffer) -> str:
    if value.dtype is DType.BF16:
        return f"  auto* {value.name}_data = reinterpret_cast<__nv_bfloat16*>(" f"{value.name}->typed_data());"
    return f"  auto* {value.name}_data = {value.name}->typed_data();"


def _ffi_pipeline_scratch(value: CudaAxisFoldFfiBuffer) -> str:
    cpp_type = "__nv_bfloat16" if value.dtype is DType.BF16 else "float"
    element_count = math.prod(value.shape)
    return f"""
  auto {value.name}_storage = scratch.Allocate(
      sizeof({cpp_type}) * {element_count}, alignof({cpp_type}));
  if (!{value.name}_storage) {{
    return ffi::Error::Internal("failed to allocate axis-Fold pipeline value {value.name}");
  }}
  auto* {value.name} = static_cast<{cpp_type}*>(*{value.name}_storage);
""".rstrip()


@dataclass(frozen=True)
class _AxisFoldKernelGroup:
    stage_indices: tuple[int, ...]


def _axis_fold_kernel_groups(
    pipeline: AxisFoldPipeline,
    schedule: AxisFoldPipelineSchedule,
) -> tuple[_AxisFoldKernelGroup, ...]:
    if schedule is AxisFoldPipelineSchedule.SEPARATE_STAGES:
        return tuple(_AxisFoldKernelGroup((index,)) for index in range(len(pipeline.stages)))
    groups: list[_AxisFoldKernelGroup] = []
    index = 0
    while index < len(pipeline.stages):
        if index + 1 < len(pipeline.stages) and _can_coalesce_row_stages(
            pipeline.stages[index], pipeline.stages[index + 1]
        ):
            groups.append(_AxisFoldKernelGroup((index, index + 1)))
            index += 2
            continue
        groups.append(_AxisFoldKernelGroup((index,)))
        index += 1
    if all(len(group.stage_indices) == 1 for group in groups):
        raise ValueError("axis-Fold pipeline has no compatible adjacent row stages to coalesce")
    return tuple(groups)


def _can_coalesce_row_stages(first: AxisFoldPipelineStage, second: AxisFoldPipelineStage) -> bool:
    first_program = first.program
    second_program = second.program
    if (
        first_program.rows != second_program.rows
        or first_program.columns != second_program.columns
        or first_program.threads != second_program.threads
        or first_program.groups_per_block != 1
        or second_program.groups_per_block != 1
        or first_program.reassociation is not second_program.reassociation
        or first_program.reduction_axis is not AxisFoldDirection.COLUMNS
        or second_program.reduction_axis is not AxisFoldDirection.COLUMNS
        or first_program.output_kind is not AxisFoldOutputKind.REDUCED
        or second_program.output_kind is not AxisFoldOutputKind.ELEMENT
    ):
        return False
    first_input = next((value for value in second_program.inputs if value.name == first.output_name), None)
    if first_input is None or first_input.layout is not AxisFoldInputLayout.ROW:
        return False
    if any(
        first.output_name in scalar_expression_inputs(reduction.contribution) for reduction in second_program.reductions
    ):
        return False
    return first.output_name in scalar_expression_inputs(second_program.output_expression)


def _ffi_axis_fold_kernel_group(pipeline: AxisFoldPipeline, group: _AxisFoldKernelGroup) -> str:
    if len(group.stage_indices) == 1:
        index = group.stage_indices[0]
        return _ffi_axis_fold_kernel(
            pipeline.stages[index].program,
            symbol=f"ShuttleAxisFoldKernel{index}",
            prefix=f"Program{index}",
        )
    first_index, second_index = group.stage_indices
    return _ffi_coalesced_row_axis_fold_kernel(
        pipeline.stages[first_index],
        pipeline.stages[second_index],
        first_index=first_index,
        second_index=second_index,
    )


def _ffi_axis_fold_group_launch(
    pipeline: AxisFoldPipeline,
    group: _AxisFoldKernelGroup,
    *,
    exposed_data: dict[str, str],
) -> str:
    if len(group.stage_indices) == 1:
        index = group.stage_indices[0]
        stage = pipeline.stages[index]
        return _ffi_axis_fold_launch_to_pointer(
            stage.program,
            index=index,
            output_pointer=exposed_data.get(stage.output_name, stage.output_name),
        )
    first_index, second_index = group.stage_indices
    first = pipeline.stages[first_index]
    second = pipeline.stages[second_index]
    inputs = _coalesced_stage_inputs(first, second)
    arguments = ",\n      ".join(
        (
            *(value.name for value in inputs),
            exposed_data.get(first.output_name, first.output_name),
            exposed_data.get(second.output_name, second.output_name),
        )
    )
    symbol = f"ShuttleAxisFoldKernel{first_index}And{second_index}"
    return f"""
  {symbol}<<<kProgram{first_index}And{second_index}Rows, kProgram{first_index}And{second_index}Threads, 0, stream>>>(
      {arguments});
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {{
    return ffi::Error::Internal(
        std::string("{symbol}: ") + cudaGetErrorString(status));
  }}
""".rstrip()


def _coalesced_stage_inputs(
    first: AxisFoldPipelineStage,
    second: AxisFoldPipelineStage,
) -> tuple[AxisFoldInput, ...]:
    ordered: list[AxisFoldInput] = []
    by_name: dict[str, AxisFoldInput] = {}
    for value in (*first.program.inputs, *second.program.inputs):
        if value.name == first.output_name:
            continue
        previous = by_name.get(value.name)
        if previous is not None and previous != value:
            raise ValueError(f"coalesced axis-Fold input {value.name!r} has incompatible uses")
        if previous is None:
            ordered.append(value)
            by_name[value.name] = value
    return tuple(ordered)


def _ffi_coalesced_row_axis_fold_kernel(
    first: AxisFoldPipelineStage,
    second: AxisFoldPipelineStage,
    *,
    first_index: int,
    second_index: int,
) -> str:
    if not _can_coalesce_row_stages(first, second):
        raise ValueError("axis-Fold stages are not compatible for one-pass row coalescing")
    first_program = first.program
    second_program = second.program
    prefix = f"Program{first_index}And{second_index}"
    rows = f"k{prefix}Rows"
    columns = f"k{prefix}Columns"
    threads = f"k{prefix}Threads"
    symbol = f"ShuttleAxisFoldKernel{first_index}And{second_index}"
    inputs = _coalesced_stage_inputs(first, second)
    parameters = ", ".join(
        (
            *(_kernel_parameter(value) for value in inputs),
            f"{_ffi_output_type(first_program)} {first.output_name}",
            f"{_ffi_output_type(second_program)} {second.output_name}",
        )
    )
    first_locals = "\n".join(
        f"  float local_stage{first_index}_{value.name} = 0.0f;" for value in first_program.reductions
    )
    second_locals = "\n".join(
        f"  float local_stage{second_index}_{value.name} = 0.0f;" for value in second_program.reductions
    )
    first_shared = "\n".join(
        f"  __shared__ float shared_stage{first_index}_{value.name}[{threads}];" for value in first_program.reductions
    )
    second_shared = "\n".join(
        f"  __shared__ float shared_stage{second_index}_{value.name}[{threads}];" for value in second_program.reductions
    )
    first_aliases = _ffi_input_aliases(first_program.inputs, "row", "column", columns)
    second_aliases = _ffi_input_aliases(second_program.inputs, "row", "column", columns)
    second_aliases.pop(first.output_name, None)
    first_updates = "\n".join(
        f"    local_stage{first_index}_{value.name} = __fadd_rn(local_stage{first_index}_{value.name}, "
        f"{_cuda_expression(value.contribution, first_aliases)});"
        for value in first_program.reductions
    )
    second_updates = "\n".join(
        f"    local_stage{second_index}_{value.name} = __fadd_rn(local_stage{second_index}_{value.name}, "
        f"{_cuda_expression(value.contribution, second_aliases)});"
        for value in second_program.reductions
    )
    first_initialization = "\n".join(
        f"  shared_stage{first_index}_{value.name}[threadIdx.x] = local_stage{first_index}_{value.name};"
        for value in first_program.reductions
    )
    second_initialization = "\n".join(
        f"  shared_stage{second_index}_{value.name}[threadIdx.x] = local_stage{second_index}_{value.name};"
        for value in second_program.reductions
    )
    first_reduction_updates = "\n".join(
        f"      shared_stage{first_index}_{value.name}[threadIdx.x] = __fadd_rn("
        f"shared_stage{first_index}_{value.name}[threadIdx.x], "
        f"shared_stage{first_index}_{value.name}[threadIdx.x + stride]);"
        for value in first_program.reductions
    )
    second_reduction_updates = "\n".join(
        f"      shared_stage{second_index}_{value.name}[threadIdx.x] = __fadd_rn("
        f"shared_stage{second_index}_{value.name}[threadIdx.x], "
        f"shared_stage{second_index}_{value.name}[threadIdx.x + stride]);"
        for value in second_program.reductions
    )
    first_reduction_aliases = {
        value.name: f"shared_stage{first_index}_{value.name}[0]" for value in first_program.reductions
    }
    first_output_aliases = (
        _ffi_reduced_output_aliases(first_program.inputs, first_program.reduction_axis, "group")
        | first_reduction_aliases
    )
    first_expression = _cuda_expression(first_program.output_expression, first_output_aliases)
    first_value = (
        first_expression
        if first_program.output_dtype is DType.FP32
        else f"__bfloat162float(__float2bfloat16_rn({first_expression}))"
    )
    first_store = _output_store_to(first.output_name, first_program.output_dtype, "group", "first_value")
    second_reduction_aliases = {
        value.name: f"shared_stage{second_index}_{value.name}[0]" for value in second_program.reductions
    }
    second_output_aliases = _ffi_input_aliases(second_program.inputs, "row", "column", columns)
    second_output_aliases[first.output_name] = "first_value"
    second_expression = _cuda_expression(
        second_program.output_expression,
        second_output_aliases | second_reduction_aliases,
    )
    second_store = _output_store_to(
        second.output_name,
        second_program.output_dtype,
        f"row * {columns} + column",
        second_expression,
    )
    return f"""
constexpr int {rows} = {first_program.rows};
constexpr int {columns} = {first_program.columns};
constexpr int {threads} = {first_program.threads};

__global__ __launch_bounds__({threads}) void {symbol}({parameters}) {{
{first_shared}
{second_shared}
  const int group = blockIdx.x;
  if (group >= {rows}) return;
{first_locals}
{second_locals}
  for (int reduction_index = threadIdx.x; reduction_index < {columns};
       reduction_index += {threads}) {{
    const int row = group;
    const int column = reduction_index;
{first_updates}
{second_updates}
  }}
{first_initialization}
{second_initialization}
  __syncthreads();
  for (int stride = {threads} / 2; stride > 0; stride /= 2) {{
    if (threadIdx.x < stride) {{
{first_reduction_updates}
{second_reduction_updates}
    }}
    __syncthreads();
  }}
  const float first_value = {first_value};
  if (threadIdx.x == 0) {{
    {first_store}
  }}
  for (int element = threadIdx.x; element < {columns}; element += blockDim.x) {{
    const int row = group;
    const int column = element;
    {second_store}
  }}
}}
""".strip()


def _ffi_axis_fold_kernel(program: AxisFoldProgram, *, symbol: str, prefix: str) -> str:
    rows = f"k{prefix}Rows"
    columns = f"k{prefix}Columns"
    threads = f"k{prefix}Threads"
    parameters = ", ".join(
        (*(_kernel_parameter(value) for value in program.inputs), f"{_ffi_output_type(program)} output")
    )
    local_reductions = "\n".join(f"  float local_{value.name} = 0.0f;" for value in program.reductions)
    shared_declarations = "\n".join(
        f"  __shared__ float shared_{value.name}[{threads}];" for value in program.reductions
    )
    shared_initialization = "\n".join(
        f"  shared_{value.name}[threadIdx.x] = local_{value.name};" for value in program.reductions
    )
    if program.groups_per_block > 1:
        return _ffi_tiled_row_axis_fold_kernel(
            program,
            symbol=symbol,
            prefix=prefix,
            parameters=parameters,
            local_reductions=local_reductions,
            shared_declarations=shared_declarations,
            shared_initialization=shared_initialization,
        )
    reduction_extent = rows if program.reduction_axis is AxisFoldDirection.ROWS else columns
    group_extent = columns if program.reduction_axis is AxisFoldDirection.ROWS else rows
    coordinate_setup = (
        "    const int row = reduction_index;\n    const int column = group;"
        if program.reduction_axis is AxisFoldDirection.ROWS
        else "    const int row = group;\n    const int column = reduction_index;"
    )
    aliases = _ffi_input_aliases(program.inputs, "row", "column", columns)
    contribution_updates = "\n".join(
        f"    local_{value.name} = __fadd_rn(local_{value.name}, {_cuda_expression(value.contribution, aliases)});"
        for value in program.reductions
    )
    shared_updates = "\n".join(
        f"      shared_{value.name}[threadIdx.x] = __fadd_rn("
        f"shared_{value.name}[threadIdx.x], shared_{value.name}[threadIdx.x + stride]);"
        for value in program.reductions
    )
    reduction_aliases = {value.name: f"shared_{value.name}[0]" for value in program.reductions}
    output_body = _ffi_output_body(program, reduction_aliases, rows=rows, columns=columns)
    return f"""
constexpr int {rows} = {program.rows};
constexpr int {columns} = {program.columns};
constexpr int {threads} = {program.threads};

__global__ __launch_bounds__({threads}) void {symbol}({parameters}) {{
{shared_declarations}
  const int group = blockIdx.x;
  if (group >= {group_extent}) return;
{local_reductions}
  for (int reduction_index = threadIdx.x; reduction_index < {reduction_extent};
       reduction_index += {threads}) {{
{coordinate_setup}
{contribution_updates}
  }}
{shared_initialization}
  __syncthreads();
  for (int stride = {threads} / 2; stride > 0; stride /= 2) {{
    if (threadIdx.x < stride) {{
{shared_updates}
    }}
    __syncthreads();
  }}
{output_body}
}}
""".strip()


def _ffi_tiled_row_axis_fold_kernel(
    program: AxisFoldProgram,
    *,
    symbol: str,
    prefix: str,
    parameters: str,
    local_reductions: str,
    shared_declarations: str,
    shared_initialization: str,
) -> str:
    rows = f"k{prefix}Rows"
    columns = f"k{prefix}Columns"
    threads = f"k{prefix}Threads"
    groups = f"k{prefix}GroupsPerBlock"
    outputs = f"k{prefix}OutputsPerGroup"
    lanes = f"k{prefix}ReductionLanes"
    aliases = _ffi_input_aliases(program.inputs, "row", "column", columns)
    contribution_updates = "\n".join(
        f"        local_{value.name}[output_lane] = __fadd_rn(local_{value.name}[output_lane], "
        f"{_cuda_expression(value.contribution, aliases)});"
        for value in program.reductions
    )
    shared_updates = "\n".join(
        f"        shared_{value.name}[shared_index] = __fadd_rn("
        f"shared_{value.name}[shared_index], shared_{value.name}[shared_index + stride * {groups}]);"
        for value in program.reductions
    )
    reduction_aliases = {value.name: f"shared_{value.name}[shared_index]" for value in program.reductions}
    output_aliases = _ffi_reduced_output_aliases(program.inputs, program.reduction_axis, "group") | reduction_aliases
    output_expression = _cuda_expression(program.output_expression, output_aliases)
    output_store = _output_store(program.output_dtype, "group", output_expression)
    local_reductions = "\n".join(f"  float local_{value.name}[{outputs}] = {{}};" for value in program.reductions)
    shared_declarations = "\n".join(
        f"  __shared__ float shared_{value.name}[{threads} * {outputs}];" for value in program.reductions
    )
    shared_initialization = "\n".join(
        f"    shared_{value.name}[shared_index] = local_{value.name}[output_lane];" for value in program.reductions
    )
    return f"""
constexpr int {rows} = {program.rows};
constexpr int {columns} = {program.columns};
constexpr int {threads} = {program.threads};
constexpr int {groups} = {program.groups_per_block};
constexpr int {outputs} = {program.outputs_per_group};
constexpr int {lanes} = {threads} / {groups};

__global__ __launch_bounds__({threads}) void {symbol}({parameters}) {{
{shared_declarations}
  const int group_lane = threadIdx.x % {groups};
  const int reduction_lane = threadIdx.x / {groups};
{local_reductions}
  #pragma unroll
  for (int output_lane = 0; output_lane < {outputs}; ++output_lane) {{
    const int group = blockIdx.x * {groups} * {outputs} + output_lane * {groups} + group_lane;
    if (group < {columns}) {{
      for (int row = reduction_lane; row < {rows}; row += {lanes}) {{
        const int column = group;
{contribution_updates}
      }}
    }}
  }}
  #pragma unroll
  for (int output_lane = 0; output_lane < {outputs}; ++output_lane) {{
    const int shared_index = output_lane * {threads} + threadIdx.x;
{shared_initialization}
  }}
  __syncthreads();
  for (int stride = {lanes} / 2; stride > 0; stride /= 2) {{
    if (reduction_lane < stride) {{
      #pragma unroll
      for (int output_lane = 0; output_lane < {outputs}; ++output_lane) {{
        const int shared_index = output_lane * {threads} + threadIdx.x;
{shared_updates}
      }}
    }}
    __syncthreads();
  }}
  if (reduction_lane == 0) {{
    #pragma unroll
    for (int output_lane = 0; output_lane < {outputs}; ++output_lane) {{
      const int group = blockIdx.x * {groups} * {outputs} + output_lane * {groups} + group_lane;
      const int shared_index = output_lane * {threads} + threadIdx.x;
      if (group < {columns}) {{
        {output_store}
      }}
    }}
  }}
}}
""".strip()


def _ffi_output_type(program: AxisFoldProgram) -> str:
    return "__nv_bfloat16*" if program.output_dtype is DType.BF16 else "float*"


def _ffi_input_aliases(
    inputs: tuple[AxisFoldInput, ...],
    row: str,
    column: str,
    columns: str,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in inputs:
        index = {
            AxisFoldInputLayout.ELEMENT: f"{row} * {columns} + {column}",
            AxisFoldInputLayout.ROW: row,
            AxisFoldInputLayout.COLUMN: column,
            AxisFoldInputLayout.SCALAR: "0",
        }[value.layout]
        load = f"{value.name}[{index}]"
        aliases[value.name] = f"__bfloat162float({load})" if value.dtype is DType.BF16 else load
    return aliases


def _ffi_reduced_output_aliases(
    inputs: tuple[AxisFoldInput, ...],
    reduction_axis: AxisFoldDirection,
    group: str,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in inputs:
        index = {
            AxisFoldInputLayout.ROW: group if reduction_axis is AxisFoldDirection.COLUMNS else None,
            AxisFoldInputLayout.COLUMN: group if reduction_axis is AxisFoldDirection.ROWS else None,
            AxisFoldInputLayout.SCALAR: "0",
            AxisFoldInputLayout.ELEMENT: None,
        }[value.layout]
        if index is None:
            continue
        load = f"{value.name}[{index}]"
        aliases[value.name] = f"__bfloat162float({load})" if value.dtype is DType.BF16 else load
    return aliases


def _ffi_output_body(
    program: AxisFoldProgram,
    reduction_aliases: dict[str, str],
    *,
    rows: str,
    columns: str,
) -> str:
    if program.output_kind is AxisFoldOutputKind.REDUCED:
        aliases = _ffi_reduced_output_aliases(program.inputs, program.reduction_axis, "group") | reduction_aliases
        expression = _cuda_expression(program.output_expression, aliases)
        return f"  if (threadIdx.x == 0) {{\n    {_output_store(program.output_dtype, 'group', expression)}\n  }}"
    element_extent = rows if program.reduction_axis is AxisFoldDirection.ROWS else columns
    row = "element" if program.reduction_axis is AxisFoldDirection.ROWS else "group"
    column = "group" if program.reduction_axis is AxisFoldDirection.ROWS else "element"
    aliases = _ffi_input_aliases(program.inputs, row, column, columns) | reduction_aliases
    expression = _cuda_expression(program.output_expression, aliases)
    return (
        f"  for (int element = threadIdx.x; element < {element_extent}; element += blockDim.x) {{\n"
        f"    const int row = {row};\n"
        f"    const int column = {column};\n"
        f"    {_output_store(program.output_dtype, 'row * ' + columns + ' + column', expression)}\n"
        "  }"
    )


def _ffi_axis_fold_launch(program: AxisFoldProgram, *, index: int, output: CudaAxisFoldFfiBuffer) -> str:
    return _ffi_axis_fold_launch_to_pointer(program, index=index, output_pointer=f"{output.name}_data")


def _ffi_axis_fold_launch_to_pointer(program: AxisFoldProgram, *, index: int, output_pointer: str) -> str:
    prefix = f"Program{index}"
    blocks = (
        f"(k{prefix}Columns + k{prefix}GroupsPerBlock * k{prefix}OutputsPerGroup - 1) / "
        f"(k{prefix}GroupsPerBlock * k{prefix}OutputsPerGroup)"
        if program.groups_per_block > 1
        else (f"k{prefix}Columns" if program.reduction_axis is AxisFoldDirection.ROWS else f"k{prefix}Rows")
    )
    arguments = ",\n      ".join((*tuple(value.name for value in program.inputs), output_pointer))
    return f"""
  ShuttleAxisFoldKernel{index}<<<{blocks}, k{prefix}Threads, 0, stream>>>(
      {arguments});
  if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {{
    return ffi::Error::Internal(
        std::string("ShuttleAxisFoldKernel{index}: ") + cudaGetErrorString(status));
  }}
""".rstrip()


def evaluate_axis_fold_program(
    program: AxisFoldProgram,
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate logical axis-Fold semantics independently of CUDA scheduling."""
    expected_names = {value.name for value in program.inputs}
    if set(inputs) != expected_names:
        raise ValueError(f"axis-Fold inputs must be {sorted(expected_names)}, found {sorted(inputs)}")
    arrays = {name: np.asarray(value) for name, value in inputs.items()}
    for value in program.inputs:
        expected_shape = {
            AxisFoldInputLayout.ELEMENT: (program.rows, program.columns),
            AxisFoldInputLayout.ROW: (program.rows,),
            AxisFoldInputLayout.COLUMN: (program.columns,),
            AxisFoldInputLayout.SCALAR: (),
        }[value.layout]
        if arrays[value.name].shape != expected_shape:
            raise ValueError(
                f"axis-Fold input {value.name!r} must have shape {expected_shape}, " f"found {arrays[value.name].shape}"
            )
    group_extent = program.columns if program.reduction_axis is AxisFoldDirection.ROWS else program.rows
    reduction_extent = program.rows if program.reduction_axis is AxisFoldDirection.ROWS else program.columns
    reduced = np.empty((group_extent, len(program.reductions)), dtype=np.float32)
    for group in range(group_extent):
        states = np.zeros(len(program.reductions), dtype=np.float32)
        for reduction_index in range(reduction_extent):
            row, column = (
                (reduction_index, group)
                if program.reduction_axis is AxisFoldDirection.ROWS
                else (group, reduction_index)
            )
            aliases = _numpy_input_aliases(program.inputs, arrays, row, column)
            for index, reduction in enumerate(program.reductions):
                contribution = np.float32(_evaluate_expression(reduction.contribution, aliases))
                states[index] = np.float32(states[index] + contribution)
        reduced[group] = states
    if program.output_kind is AxisFoldOutputKind.REDUCED:
        output = np.empty((group_extent,), dtype=np.float32)
        for group in range(group_extent):
            aliases = _numpy_reduced_output_aliases(
                program.inputs,
                arrays,
                program.reduction_axis,
                group,
            )
            aliases.update(
                {reduction.name: float(reduced[group, index]) for index, reduction in enumerate(program.reductions)}
            )
            output[group] = np.float32(_evaluate_expression(program.output_expression, aliases))
        return output
    output = np.empty((program.rows, program.columns), dtype=np.float32)
    for row in range(program.rows):
        for column in range(program.columns):
            group = column if program.reduction_axis is AxisFoldDirection.ROWS else row
            aliases = _numpy_input_aliases(program.inputs, arrays, row, column)
            aliases.update(
                {reduction.name: float(reduced[group, index]) for index, reduction in enumerate(program.reductions)}
            )
            output[row, column] = np.float32(_evaluate_expression(program.output_expression, aliases))
    return output


def evaluate_axis_fold_pipeline(
    pipeline: AxisFoldPipeline,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Evaluate a topological Fold pipeline, including internal values."""
    external_inputs, _, exposed_outputs = _axis_fold_pipeline_buffers(pipeline)
    expected_names = {value.name for value in external_inputs}
    if set(inputs) != expected_names:
        raise ValueError(f"axis-Fold pipeline inputs must be {sorted(expected_names)}, found {sorted(inputs)}")
    values = {name: np.asarray(value) for name, value in inputs.items()}
    exposed_names = {value.name for value in exposed_outputs}
    exposed: list[np.ndarray] = []
    for stage in pipeline.stages:
        stage_inputs = {value.name: values[value.name] for value in stage.program.inputs}
        output = evaluate_axis_fold_program(stage.program, stage_inputs)
        values[stage.output_name] = output
        if stage.output_name in exposed_names:
            exposed.append(output)
    return tuple(exposed)


def _input_aliases(inputs: tuple[AxisFoldInput, ...], row: str, column: str) -> dict[str, str]:
    return {value.name: _input_load(value, row, column) for value in inputs}


def _input_load(value: AxisFoldInput, row: str, column: str) -> str:
    index = {
        AxisFoldInputLayout.ELEMENT: f"{row} * kColumns + {column}",
        AxisFoldInputLayout.ROW: row,
        AxisFoldInputLayout.COLUMN: column,
        AxisFoldInputLayout.SCALAR: "0",
    }[value.layout]
    load = f"{value.name}[{index}]"
    return f"__bfloat162float({load})" if value.dtype is DType.BF16 else load


def _kernel_parameter(value: AxisFoldInput) -> str:
    dtype = "const __nv_bfloat16*" if value.dtype is DType.BF16 else "const float*"
    return f"{dtype} {value.name}"


def _pointer_declaration(value: AxisFoldInput) -> str:
    dtype = "__nv_bfloat16" if value.dtype is DType.BF16 else "float"
    return f"  const auto* {value.name}_pointer = reinterpret_cast<const {dtype}*>({value.name}.data_ptr());"


def _wrapper_check(value: AxisFoldInput, program: AxisFoldProgram) -> str:
    torch_dtype = "torch::kBFloat16" if value.dtype is DType.BF16 else "torch::kFloat32"
    expected_shape = {
        AxisFoldInputLayout.ELEMENT: (program.rows, program.columns),
        AxisFoldInputLayout.ROW: (program.rows,),
        AxisFoldInputLayout.COLUMN: (program.columns,),
        AxisFoldInputLayout.SCALAR: (),
    }[value.layout]
    if not expected_shape:
        shape_check = f"{value.name}.dim() == 0"
    else:
        dimensions = " && ".join(
            f"{value.name}.size({index}) == {extent}" for index, extent in enumerate(expected_shape)
        )
        shape_check = f"{value.name}.dim() == {len(expected_shape)} && {dimensions}"
    return (
        f'  TORCH_CHECK({value.name}.is_cuda(), "axis-Fold input {value.name} must be CUDA");\n'
        f'  TORCH_CHECK({value.name}.scalar_type() == {torch_dtype}, "axis-Fold input {value.name} dtype mismatch");\n'
        f'  TORCH_CHECK({value.name}.is_contiguous(), "axis-Fold input {value.name} must be contiguous");\n'
        f'  TORCH_CHECK({value.name}.device() == output.device(), "axis-Fold input {value.name} device mismatch");\n'
        f'  TORCH_CHECK({shape_check}, "axis-Fold input {value.name} shape mismatch");'
    )


def _element_output_body(program: AxisFoldProgram, reduction_aliases: dict[str, str]) -> str:
    aliases = _input_aliases(program.inputs, "row", "column") | reduction_aliases
    expression = _cuda_expression(program.output_expression, aliases)
    store = _output_store(program.output_dtype, "row * kColumns + column", expression)
    element_extent = "kRows" if program.reduction_axis is AxisFoldDirection.ROWS else "kColumns"
    row = "element" if program.reduction_axis is AxisFoldDirection.ROWS else "group"
    column = "group" if program.reduction_axis is AxisFoldDirection.ROWS else "element"
    return f"""
  for (int element = threadIdx.x; element < {element_extent};
       element += kThreads) {{
    const int row = {row};
    const int column = {column};
    {store}
  }}
""".rstrip()


def _reduced_output_body(program: AxisFoldProgram, reduction_aliases: dict[str, str]) -> str:
    aliases = _reduced_output_aliases(program.inputs, program.reduction_axis, "group") | reduction_aliases
    expression = _cuda_expression(program.output_expression, aliases)
    store = _output_store(program.output_dtype, "group", expression)
    return f"""
  if (threadIdx.x == 0) {{
    {store}
  }}
""".rstrip()


def _output_store(dtype: DType, index: str, expression: str) -> str:
    return _output_store_to("output", dtype, index, expression)


def _output_store_to(pointer: str, dtype: DType, index: str, expression: str) -> str:
    if dtype is DType.BF16:
        return f"{pointer}[{index}] = __float2bfloat16_rn({expression});"
    return f"{pointer}[{index}] = {expression};"


def _cuda_expression(expression: ScalarExpression, aliases: dict[str, str]) -> str:
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        try:
            return aliases[expression.input_name]
        except KeyError as error:
            raise ValueError(f"unbound axis-Fold scalar input {expression.input_name!r}") from error
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        if isinstance(expression.constant, bool):
            return "true" if expression.constant else "false"
        return _cuda_float(float(expression.constant))
    operands = tuple(_cuda_expression(operand, aliases) for operand in expression.operands)
    if kind is ScalarExpressionKind.ADD:
        return f"__fadd_rn({operands[0]}, {operands[1]})"
    if kind is ScalarExpressionKind.SUBTRACT:
        return f"__fsub_rn({operands[0]}, {operands[1]})"
    if kind is ScalarExpressionKind.MULTIPLY:
        return f"__fmul_rn({operands[0]}, {operands[1]})"
    if kind is ScalarExpressionKind.DIVIDE:
        return f"({operands[0]} / {operands[1]})"
    if kind is ScalarExpressionKind.EXP:
        return f"expf({operands[0]})"
    if kind is ScalarExpressionKind.LOG:
        return f"logf({operands[0]})"
    if kind is ScalarExpressionKind.RSQRT:
        return f"rsqrtf({operands[0]})"
    if kind is ScalarExpressionKind.TANH:
        return f"tanhf({operands[0]})"
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return f"({operands[0]} <= {operands[1]})"
    if kind is ScalarExpressionKind.SELECT:
        return f"({operands[0]} ? {operands[1]} : {operands[2]})"
    raise ValueError(f"unsupported axis-Fold scalar operation {kind.value}")


def _cuda_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("axis-Fold CUDA literals must be finite")
    rendered = repr(value)
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return f"{rendered}f"


def _numpy_input_aliases(
    inputs: tuple[AxisFoldInput, ...],
    arrays: dict[str, np.ndarray],
    row: int,
    column: int,
) -> dict[str, float]:
    indices: dict[AxisFoldInputLayout, tuple[int, ...]] = {
        AxisFoldInputLayout.ELEMENT: (row, column),
        AxisFoldInputLayout.ROW: (row,),
        AxisFoldInputLayout.COLUMN: (column,),
        AxisFoldInputLayout.SCALAR: (),
    }
    return {value.name: float(arrays[value.name][indices[value.layout]]) for value in inputs}


def _reduced_output_aliases(
    inputs: tuple[AxisFoldInput, ...],
    reduction_axis: AxisFoldDirection,
    group: str,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in inputs:
        index = {
            AxisFoldInputLayout.ROW: group if reduction_axis is AxisFoldDirection.COLUMNS else None,
            AxisFoldInputLayout.COLUMN: group if reduction_axis is AxisFoldDirection.ROWS else None,
            AxisFoldInputLayout.SCALAR: "0",
            AxisFoldInputLayout.ELEMENT: None,
        }[value.layout]
        if index is not None:
            load = f"{value.name}[{index}]"
            aliases[value.name] = f"__bfloat162float({load})" if value.dtype is DType.BF16 else load
    return aliases


def _numpy_reduced_output_aliases(
    inputs: tuple[AxisFoldInput, ...],
    arrays: dict[str, np.ndarray],
    reduction_axis: AxisFoldDirection,
    group: int,
) -> dict[str, float]:
    aliases: dict[str, float] = {}
    for value in inputs:
        index = {
            AxisFoldInputLayout.ROW: (group,) if reduction_axis is AxisFoldDirection.COLUMNS else None,
            AxisFoldInputLayout.COLUMN: (group,) if reduction_axis is AxisFoldDirection.ROWS else None,
            AxisFoldInputLayout.SCALAR: (),
            AxisFoldInputLayout.ELEMENT: None,
        }[value.layout]
        if index is not None:
            aliases[value.name] = float(arrays[value.name][index])
    return aliases


def _evaluate_expression(expression: ScalarExpression, aliases: dict[str, float]) -> float | bool:
    kind = expression.kind
    if kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return aliases[expression.input_name]
    if kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return expression.constant
    operands = tuple(_evaluate_expression(operand, aliases) for operand in expression.operands)
    if kind is ScalarExpressionKind.ADD:
        return float(operands[0]) + float(operands[1])
    if kind is ScalarExpressionKind.SUBTRACT:
        return float(operands[0]) - float(operands[1])
    if kind is ScalarExpressionKind.MULTIPLY:
        return float(operands[0]) * float(operands[1])
    if kind is ScalarExpressionKind.DIVIDE:
        return float(operands[0]) / float(operands[1])
    if kind is ScalarExpressionKind.EXP:
        return math.exp(float(operands[0]))
    if kind is ScalarExpressionKind.LOG:
        return math.log(float(operands[0]))
    if kind is ScalarExpressionKind.RSQRT:
        return 1.0 / math.sqrt(float(operands[0]))
    if kind is ScalarExpressionKind.TANH:
        return math.tanh(float(operands[0]))
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return float(operands[0]) <= float(operands[1])
    if kind is ScalarExpressionKind.SELECT:
        return operands[1] if bool(operands[0]) else operands[2]
    raise AssertionError(f"unhandled axis-Fold scalar operation {kind}")
