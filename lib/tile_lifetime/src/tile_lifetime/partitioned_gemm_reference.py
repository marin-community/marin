# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent reference execution for partitioned Contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import ml_dtypes
import numpy as np

from tile_lifetime.cast_scalar_program import evaluate_cast_scalar_program
from tile_lifetime.partitioned_gemm_program import PartitionedGemmProgram

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\](?:\{[^}]+\})?")


def evaluate_partitioned_gemm_reference(
    program: PartitionedGemmProgram,
    operands: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Execute ordered FP32 products, BF16 boundaries, and scalar ASTs on CPU."""
    if program.partitioned_operand != 1:
        raise ValueError("partitioned Contract reference currently supports static RHS partitions")
    if len(operands) != len(program.operand_shapes):
        raise ValueError(f"partitioned Contract expected {len(program.operand_shapes)} operands, found {len(operands)}")
    expected_shapes = tuple(_shape_dimensions(shape) for shape in program.operand_shapes)
    arrays = tuple(np.asarray(value, dtype=ml_dtypes.bfloat16) for value in operands)
    for index, (array, shape) in enumerate(zip(arrays, expected_shapes, strict=True)):
        if array.shape != shape:
            raise ValueError(f"partitioned Contract operand {index} must have shape {shape}, found {array.shape}")

    m, n, k = program.shape
    lhs = arrays[0].reshape(m, k)
    boundary = np.empty((m, n), dtype=ml_dtypes.bfloat16)
    for partition, rhs in zip(program.partitions, arrays[1:], strict=True):
        rhs_matrix = rhs.reshape(partition.extent, k)
        for row in range(m):
            for local_feature in range(partition.extent):
                accumulator = np.float32(0.0)
                for reduction in range(k):
                    product = np.float32(
                        np.float32(lhs[row, reduction]) * np.float32(rhs_matrix[local_feature, reduction])
                    )
                    accumulator = np.float32(accumulator + product)
                boundary[row, partition.start + local_feature] = ml_dtypes.bfloat16(accumulator)

    outputs: list[np.ndarray] = []
    for finalization in program.scalar_finalizations:
        extent = program.partitions[finalization.source_partitions[0]].extent
        output = np.empty((m, extent), dtype=ml_dtypes.bfloat16)
        for row in range(m):
            for feature in range(extent):
                inputs: dict[str, float] = {}
                for partition_index, scalar_input in zip(
                    finalization.source_partitions, finalization.program.inputs, strict=True
                ):
                    assert scalar_input.input_name is not None and scalar_input.input_index is not None
                    partition = program.partitions[partition_index]
                    source_row = row + scalar_input.input_index.row_offset
                    source_feature = feature + scalar_input.input_index.feature_offset
                    if source_row < 0 or source_row >= m or source_feature < 0 or source_feature >= extent:
                        raise ValueError("scalar partition Map index relation leaves its source domain")
                    inputs[scalar_input.input_name] = float(boundary[source_row, partition.start + source_feature])
                value = evaluate_cast_scalar_program(finalization.program, inputs)
                if isinstance(value, bool):
                    raise ValueError("partitioned Contract BF16 output cannot store a predicate scalar Map")
                output[row, feature] = ml_dtypes.bfloat16(value)
        outputs.append(output.reshape(_shape_dimensions(finalization.output_shape)))
    for finalization in program.passthrough_finalizations:
        partition = program.partitions[finalization.source_partition]
        output = boundary[:, partition.start : partition.limit]
        outputs.append(output.reshape(_shape_dimensions(finalization.output_shape)).copy())
    return tuple(outputs)


def partitioned_gemm_error_metrics(
    actual: Sequence[np.ndarray],
    expected: Sequence[np.ndarray],
) -> tuple[Mapping[str, float], ...]:
    """Return pointwise absolute-error metrics for corresponding outputs."""
    if len(actual) != len(expected):
        raise ValueError(f"output count differs: {len(actual)} != {len(expected)}")
    metrics: list[Mapping[str, float]] = []
    for actual_output, expected_output in zip(actual, expected, strict=True):
        if actual_output.shape != expected_output.shape:
            raise ValueError(f"output shapes differ: {actual_output.shape} != {expected_output.shape}")
        absolute = np.abs(actual_output.astype(np.float32) - expected_output.astype(np.float32))
        metrics.append(
            {
                "maximum_absolute_error": float(np.max(absolute, initial=0.0)),
                "mean_absolute_error": float(np.mean(absolute)) if absolute.size else 0.0,
            }
        )
    return tuple(metrics)


def _shape_dimensions(shape: str) -> tuple[int, ...]:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        raise ValueError(f"unsupported partitioned Contract shape {shape!r}")
    dimensions = match.group("dims")
    return tuple(int(value) for value in dimensions.split(",")) if dimensions else ()
