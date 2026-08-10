# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a bounded CUDA Contract with an exposed preparation program."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from tile_lifetime.gemm_program import GemmProgram
from tile_lifetime.tile_program import TileOp, TilePrimitive, TileProgramError


class PreparedContractOperandDelivery(StrEnum):
    """Broadcast domain of one external preparation value."""

    ROW = "row"
    FEATURE = "feature"


@dataclass(frozen=True)
class PreparedContractOperand:
    """One external value consumed by a generated Contract preparation."""

    parameter: str
    source: str
    delivery: PreparedContractOperandDelivery


@dataclass(frozen=True)
class GeneratedCudaPreparedContract:
    """Bounded scalar CUDA source for a generic prepared Contract."""

    source: str
    semantic_digest: str
    source_digest: str
    operands: tuple[PreparedContractOperand, ...]


@dataclass(frozen=True)
class PreparedContractSourceAudit:
    """Static evidence for the generated physical boundary."""

    kernel_count: int
    row_operand_count: int
    feature_operand_count: int
    has_explicit_fp32_preparation: bool
    has_bf16_rne_mainloop_boundary: bool
    has_ordered_fp32_accumulation: bool
    has_atomics: bool
    opaque_semantic_dependencies: tuple[str, ...]


def generate_cuda_prepared_contract(program: GemmProgram) -> GeneratedCudaPreparedContract:
    """Generate a correctness-oriented scalar Contract from preparation ops.

    This deliberately owns only a generic loop nest. It is an executable
    physical specification for backends that have not yet learned to stage
    multiple preparation operands through a tensor-core mainloop.
    """
    operations = tuple(
        operation
        for operation in program.preparation
        if operation.primitive not in {TilePrimitive.CONVERT, TilePrimitive.VIEW}
    )
    if not operations:
        raise TileProgramError("prepared Contract code generation requires a nonempty preparation")
    operands, expression_lines, result = _preparation_source(program, operations)
    m, n, k = program.shape
    operand_arguments = tuple(f"const float* __restrict__ {operand.parameter}" for operand in operands)
    source = f"""// Generated from a generic Contract preparation; do not edit.
#include <cuda_bf16.h>

namespace {{
constexpr int kRows = {m};
constexpr int kOutputFeatures = {n};
constexpr int kReduction = {k};

__global__ void ShuttlePreparedContractKernel(
    const __nv_bfloat16* __restrict__ activation,
    const __nv_bfloat16* __restrict__ weight,
    {",\n    ".join(operand_arguments)},
    __nv_bfloat16* __restrict__ output) {{
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= kRows * kOutputFeatures) {{
    return;
  }}
  const int row = linear / kOutputFeatures;
  const int output_feature = linear - row * kOutputFeatures;
  float accumulator = 0.0f;
  for (int reduction = 0; reduction < kReduction; ++reduction) {{
    const float activation_fp32 = __bfloat162float(activation[row * kReduction + reduction]);
{chr(10).join(expression_lines)}
    const __nv_bfloat16 mainloop_input_bf16 = __float2bfloat16_rn({result});
    const float rhs = __bfloat162float(weight[output_feature * kReduction + reduction]);
    accumulator = __fadd_rn(
        accumulator,
        __fmul_rn(__bfloat162float(mainloop_input_bf16), rhs));
  }}
  output[linear] = __float2bfloat16_rn(accumulator);
}}
}}
"""
    semantic_record = {
        "shape": program.shape,
        "operations": [
            {
                "primitive": operation.primitive.value,
                "inputs": operation.inputs,
                "outputs": operation.outputs,
                "attributes": operation.attributes,
            }
            for operation in operations
        ],
        "operands": [{"source": operand.source, "delivery": operand.delivery.value} for operand in operands],
        "mainloop_boundary": "fp32_preparation_to_bf16_rne",
        "accumulation": "ordered_fp32_fma_decomposed",
    }
    semantic_digest = hashlib.sha256(
        json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return GeneratedCudaPreparedContract(
        source=source,
        semantic_digest=semantic_digest,
        source_digest=hashlib.sha256(source.encode()).hexdigest(),
        operands=operands,
    )


def execute_preparation_reference(
    program: GemmProgram,
    activation: np.ndarray,
    external_values: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Execute preparation in ordered FP32 and return the BF16-rounded boundary."""
    m, _, k = program.shape
    if activation.shape != (m, k):
        raise ValueError(f"activation shape {activation.shape} must be {(m, k)}")
    values: dict[str, np.ndarray] = {program.input: _round_bf16(activation)}
    for operation in program.preparation:
        if operation.primitive is TilePrimitive.CONVERT:
            values[operation.outputs[0]] = _round_bf16(values[operation.inputs[0]])
            continue
        if operation.primitive is TilePrimitive.VIEW:
            values[operation.outputs[0]] = values[operation.inputs[0]]
            continue
        left = values[operation.inputs[0]]
        if operation.primitive is TilePrimitive.SCALE_ROW:
            right = _external_array(
                external_values,
                operation.inputs[1],
                PreparedContractOperandDelivery.ROW,
                m,
                k,
            )
            result = np.multiply(left, right, dtype=np.float32)
        elif operation.primitive in {TilePrimitive.ADD, TilePrimitive.SUBTRACT, TilePrimitive.MULTIPLY}:
            delivery = _delivery(operation.attributes)
            if delivery == "tile":
                right = values[operation.inputs[1]]
            else:
                right = _external_array(
                    external_values,
                    operation.inputs[1],
                    PreparedContractOperandDelivery(delivery),
                    m,
                    k,
                )
            if operation.primitive is TilePrimitive.ADD:
                result = np.add(left, right, dtype=np.float32)
            elif operation.primitive is TilePrimitive.SUBTRACT:
                result = np.subtract(left, right, dtype=np.float32)
            else:
                result = np.multiply(left, right, dtype=np.float32)
        else:
            raise TileProgramError(f"bounded preparation reference does not support {operation.primitive.value}")
        values[operation.outputs[0]] = result
    return values[program.mainloop_input]


def execute_prepared_contract_reference(
    program: GemmProgram,
    activation: np.ndarray,
    weight_nk: np.ndarray,
    external_values: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Execute the bounded CUDA family's ordered scalar semantics on CPU."""
    prepared = execute_preparation_reference(program, activation, external_values)
    m, n, k = program.shape
    if weight_nk.shape != (n, k):
        raise ValueError(f"weight shape {weight_nk.shape} must be {(n, k)}")
    weight = _round_bf16(weight_nk)
    output = np.empty((m, n), dtype=np.float32)
    for row in range(m):
        for feature in range(n):
            accumulator = np.float32(0.0)
            for reduction in range(k):
                product = np.float32(prepared[row, reduction] * weight[feature, reduction])
                accumulator = np.float32(accumulator + product)
            output[row, feature] = accumulator
    return _round_bf16(output)


def audit_cuda_prepared_contract_source(
    generated: GeneratedCudaPreparedContract,
) -> PreparedContractSourceAudit:
    """Audit arithmetic ownership without relying on a workload name."""
    lowered = generated.source.lower()
    opaque_tokens = (
        "layernorm",
        "rmsnorm",
        "transformer",
        "flash_attention",
        "mok_forward",
        "cublas",
        "quack",
    )
    return PreparedContractSourceAudit(
        kernel_count=generated.source.count("__global__ void "),
        row_operand_count=sum(operand.delivery is PreparedContractOperandDelivery.ROW for operand in generated.operands),
        feature_operand_count=sum(
            operand.delivery is PreparedContractOperandDelivery.FEATURE for operand in generated.operands
        ),
        has_explicit_fp32_preparation="const float activation_fp32" in generated.source,
        has_bf16_rne_mainloop_boundary=("mainloop_input_bf16 = __float2bfloat16_rn" in generated.source),
        has_ordered_fp32_accumulation=(
            "accumulator = __fadd_rn(" in generated.source and "__fmul_rn(" in generated.source
        ),
        has_atomics="atomic" in lowered,
        opaque_semantic_dependencies=tuple(token for token in opaque_tokens if token in lowered),
    )


def _preparation_source(
    program: GemmProgram, operations: tuple[TileOp, ...]
) -> tuple[tuple[PreparedContractOperand, ...], tuple[str, ...], str]:
    expressions = {program.input: "activation_fp32"}
    operands: list[PreparedContractOperand] = []
    operand_by_source: dict[tuple[str, PreparedContractOperandDelivery], str] = {}
    lines: list[str] = []

    def external(source: str, delivery: PreparedContractOperandDelivery) -> str:
        key = (source, delivery)
        parameter = operand_by_source.get(key)
        if parameter is None:
            parameter = f"operand_{len(operands)}"
            operand_by_source[key] = parameter
            operands.append(PreparedContractOperand(parameter, source, delivery))
        index = "row" if delivery is PreparedContractOperandDelivery.ROW else "reduction"
        return f"{parameter}[{index}]"

    for index, operation in enumerate(operations):
        left = expressions.get(operation.inputs[0])
        if left is None:
            raise TileProgramError(f"unbound preparation value {operation.inputs[0]!r}")
        if operation.primitive is TilePrimitive.SCALE_ROW:
            right = external(operation.inputs[1], PreparedContractOperandDelivery.ROW)
            function = "__fmul_rn"
        elif operation.primitive in {TilePrimitive.ADD, TilePrimitive.SUBTRACT, TilePrimitive.MULTIPLY}:
            delivery = _delivery(operation.attributes)
            if delivery == "tile":
                right = expressions.get(operation.inputs[1])
                if right is None:
                    raise TileProgramError(f"unbound preparation tile value {operation.inputs[1]!r}")
            else:
                right = external(operation.inputs[1], PreparedContractOperandDelivery(delivery))
            function = {
                TilePrimitive.ADD: "__fadd_rn",
                TilePrimitive.SUBTRACT: "__fsub_rn",
                TilePrimitive.MULTIPLY: "__fmul_rn",
            }[operation.primitive]
        else:
            raise TileProgramError(f"bounded prepared Contract does not support {operation.primitive.value}")
        variable = f"prepared_{index}"
        lines.append(f"    const float {variable} = {function}({left}, {right});")
        for output in operation.outputs:
            expressions[output] = variable
    result = expressions.get(program.mainloop_input.removesuffix(".mainloop_bf16"))
    if result is None:
        raise TileProgramError("prepared Contract does not expose a pre-conversion mainloop value")
    return tuple(operands), tuple(lines), result


def _delivery(attributes: tuple[tuple[str, str], ...]) -> str:
    delivery = dict(attributes).get("input.1_delivery", "tile")
    if delivery not in {"tile", "row", "feature"}:
        raise TileProgramError(f"unsupported preparation operand delivery {delivery!r}")
    return delivery


def _external_array(
    values: Mapping[str, np.ndarray],
    source: str,
    delivery: PreparedContractOperandDelivery,
    rows: int,
    features: int,
) -> np.ndarray:
    value = np.asarray(values[source], dtype=np.float32)
    expected = (rows,) if delivery is PreparedContractOperandDelivery.ROW else (features,)
    if value.shape != expected:
        raise ValueError(f"{delivery.value} operand {source!r} shape {value.shape} must be {expected}")
    return value[:, None] if delivery is PreparedContractOperandDelivery.ROW else value[None, :]


def _round_bf16(value: np.ndarray) -> np.ndarray:
    fp32 = np.asarray(value, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((rounded >> np.uint32(16)) << np.uint32(16)).view(np.float32)
