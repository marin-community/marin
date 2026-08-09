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
    reassociation: AxisFoldReassociation = AxisFoldReassociation.DETERMINISTIC_TREE

    def __post_init__(self) -> None:
        if min(self.rows, self.columns, self.threads) <= 0:
            raise ValueError("axis-Fold dimensions and thread count must be positive")
        if self.threads & (self.threads - 1):
            raise ValueError("axis-Fold thread count must be a power of two")
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


def generate_cuda_axis_fold(program: AxisFoldProgram) -> GeneratedCudaAxisFold:
    """Render a generic deterministic Map/Fold CUDA extension."""
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
    if dtype is DType.BF16:
        return f"output[{index}] = __float2bfloat16_rn({expression});"
    return f"output[{index}] = {expression};"


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
    if kind is ScalarExpressionKind.RSQRT:
        return 1.0 / math.sqrt(float(operands[0]))
    if kind is ScalarExpressionKind.TANH:
        return math.tanh(float(operands[0]))
    if kind is ScalarExpressionKind.LESS_EQUAL:
        return float(operands[0]) <= float(operands[1])
    if kind is ScalarExpressionKind.SELECT:
        return operands[1] if bool(operands[0]) else operands[2]
    raise AssertionError(f"unhandled axis-Fold scalar operation {kind}")
