# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral GEMM programs with logical accumulator partitions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.cast_scalar_program import (
    CastScalarDType,
    CastScalarProgram,
    GeneratedCudaScalarBody,
    generate_cuda_scalar_body,
)


@dataclass(frozen=True)
class AccumulatorPartition:
    """One contiguous logical view of a GEMM accumulator's N dimension."""

    start: int
    limit: int
    result_shape: str

    @property
    def extent(self) -> int:
        """Return the logical feature width of this partition."""
        return self.limit - self.start


@dataclass(frozen=True)
class ScalarPartitionFinalization:
    """A generated scalar Map over equally shaped accumulator partitions."""

    source_partitions: tuple[int, ...]
    program: CastScalarProgram
    output_shape: str


@dataclass(frozen=True)
class PassthroughPartitionFinalization:
    """A partition rounded and stored without a further scalar Map."""

    source_partition: int
    output_shape: str


class PartitionFoldReassociation(StrEnum):
    """Permitted reassociation of one auxiliary partition Fold."""

    SOURCE_ORDERED = "source_ordered"
    ALLOW_ROUNDING_REORDER = "allow_rounding_reorder"


@dataclass(frozen=True)
class AuxiliaryPartitionFold:
    """A Map/Fold emitted alongside a retained raw accumulator partition.

    ``input_shape`` is a logical view of the rounded partition. Its final axis
    is folded, while all leading axes form the output domain. This permits a
    physical Contract to retain its BF16 output for other consumers and emit a
    derived statistic without materializing the pointwise contribution.
    """

    source_partition: int
    input_shape: str
    contribution: CastScalarProgram
    reducer: CastScalarProgram
    initializer: float
    output_shape: str
    accumulator_dtype: str
    output_dtype: str
    reassociation: PartitionFoldReassociation


@dataclass(frozen=True)
class PartitionedGemmProgram:
    """One shared GEMM mainloop with generated logical-partition outputs.

    Accumulators are FP32. Each source partition is rounded to BF16 before it
    enters a scalar finalization, matching an HLO dot/slice/Map boundary rather
    than silently moving the Map ahead of the exported BF16 result.
    """

    shape: tuple[int, int, int]
    partitioned_operand: int
    operand_shapes: tuple[str, ...]
    partitions: tuple[AccumulatorPartition, ...]
    scalar_finalizations: tuple[ScalarPartitionFinalization, ...]
    passthrough_finalizations: tuple[PassthroughPartitionFinalization, ...]
    input_dtype: str
    accumulation_dtype: str
    partition_dtype: str
    output_dtype: str
    output_rounding: str
    auxiliary_folds: tuple[AuxiliaryPartitionFold, ...] = ()

    def __post_init__(self) -> None:
        m, n, k = self.shape
        if min(m, n, k) <= 0:
            raise ValueError("partitioned GEMM dimensions must be positive")
        if self.partitioned_operand not in {0, 1}:
            raise ValueError("partitioned GEMM operand must be lhs or rhs")
        if not self.operand_shapes:
            raise ValueError("partitioned GEMM requires physical operands")
        if not self.partitions:
            raise ValueError("partitioned GEMM requires logical accumulator partitions")
        if self.partitions[0].start != 0 or self.partitions[-1].limit != n:
            raise ValueError("accumulator partitions must cover the complete N dimension")
        if any(left.limit != right.start for left, right in zip(self.partitions, self.partitions[1:], strict=False)):
            raise ValueError("accumulator partitions must be contiguous and nonoverlapping")
        if any(partition.extent <= 0 for partition in self.partitions):
            raise ValueError("accumulator partitions must have positive width")
        if (self.input_dtype, self.accumulation_dtype, self.partition_dtype, self.output_dtype) != (
            "bf16",
            "f32",
            "bf16",
            "bf16",
        ):
            raise ValueError("the first partitioned GEMM program requires BF16 inputs/outputs and FP32 accumulation")
        if self.output_rounding != "round_to_nearest_even":
            raise ValueError("the first partitioned GEMM program requires round-to-nearest-even BF16 boundaries")
        consumed: set[int] = set()
        for finalization in self.scalar_finalizations:
            if not finalization.source_partitions:
                raise ValueError("scalar partition finalization requires at least one source")
            if any(index < 0 or index >= len(self.partitions) for index in finalization.source_partitions):
                raise ValueError("scalar partition finalization references a missing partition")
            extents = {self.partitions[index].extent for index in finalization.source_partitions}
            if len(extents) != 1:
                raise ValueError("one scalar partition Map requires equally wide source partitions")
            if len(finalization.program.inputs) != len(finalization.source_partitions):
                raise ValueError("scalar partition Map input count does not match its source partitions")
            if consumed.intersection(finalization.source_partitions):
                raise ValueError("one accumulator partition cannot feed two stored finalizations")
            consumed.update(finalization.source_partitions)
        for finalization in self.passthrough_finalizations:
            if finalization.source_partition < 0 or finalization.source_partition >= len(self.partitions):
                raise ValueError("passthrough finalization references a missing partition")
            if finalization.source_partition in consumed:
                raise ValueError("one accumulator partition cannot be both mapped and passed through")
            consumed.add(finalization.source_partition)
        if consumed != set(range(len(self.partitions))):
            raise ValueError("every accumulator partition must reach exactly one stored output")
        for fold in self.auxiliary_folds:
            if fold.source_partition < 0 or fold.source_partition >= len(self.partitions):
                raise ValueError("auxiliary Fold references a missing partition")
            if len(fold.contribution.inputs) != 1:
                raise ValueError("auxiliary Fold contribution requires one partition scalar")
            if len(fold.reducer.inputs) != 2:
                raise ValueError("auxiliary Fold reducer requires accumulator and contribution scalars")
            if fold.accumulator_dtype != "f32" or fold.output_dtype != "f32":
                raise ValueError("the bounded auxiliary Fold requires FP32 accumulation and output")
            if fold.contribution.inputs[0].dtype is not CastScalarDType.BF16:
                raise ValueError("auxiliary Fold contribution must read the rounded BF16 partition boundary")
            if fold.contribution.expression.dtype is not CastScalarDType.F32:
                raise ValueError("auxiliary Fold contribution must produce FP32")
            if any(value.dtype is not CastScalarDType.F32 for value in fold.reducer.inputs):
                raise ValueError("auxiliary Fold reducer inputs must be FP32")
            if fold.reducer.expression.dtype is not CastScalarDType.F32:
                raise ValueError("auxiliary Fold reducer must produce FP32")
            if not math.isfinite(fold.initializer):
                raise ValueError("auxiliary Fold initializer must be finite")

    @property
    def output_shapes(self) -> tuple[str, ...]:
        """Return stored partition outputs followed by auxiliary Fold outputs."""
        return (
            *(finalization.output_shape for finalization in self.scalar_finalizations),
            *(finalization.output_shape for finalization in self.passthrough_finalizations),
            *(fold.output_shape for fold in self.auxiliary_folds),
        )

    @property
    def semantic_digest(self) -> str:
        """Return a source-name-independent physical-program digest."""
        record = {
            "template": _program_template(self),
            "shape": self.shape,
            "partitioned_operand": self.partitioned_operand,
            "operand_shapes": self.operand_shapes,
            "partitions": [
                {"start": partition.start, "limit": partition.limit, "shape": partition.result_shape}
                for partition in self.partitions
            ],
            "scalar_finalizations": [
                {
                    "sources": finalization.source_partitions,
                    "program": finalization.program.serialized,
                    "output_shape": finalization.output_shape,
                }
                for finalization in self.scalar_finalizations
            ],
            "passthrough_finalizations": [
                {"source": finalization.source_partition, "output_shape": finalization.output_shape}
                for finalization in self.passthrough_finalizations
            ],
            "auxiliary_folds": [
                {
                    "source": fold.source_partition,
                    "input_shape": fold.input_shape,
                    "contribution": fold.contribution.serialized,
                    "reducer": fold.reducer.serialized,
                    "initializer": fold.initializer,
                    "output_shape": fold.output_shape,
                    "accumulator_dtype": fold.accumulator_dtype,
                    "output_dtype": fold.output_dtype,
                    "reassociation": fold.reassociation.value,
                }
                for fold in self.auxiliary_folds
            ],
            "numerical": {
                "input_dtype": self.input_dtype,
                "accumulation_dtype": self.accumulation_dtype,
                "partition_dtype": self.partition_dtype,
                "output_dtype": self.output_dtype,
                "output_rounding": self.output_rounding,
            },
        }
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class GeneratedPartitionedGemmFinalization:
    """Generated scalar bodies and the generic epilogue adapter they require."""

    template: str
    semantic_digest: str
    scalar_bodies: tuple[GeneratedCudaScalarBody, ...]
    auxiliary_contribution_bodies: tuple[GeneratedCudaScalarBody, ...]
    auxiliary_reducer_bodies: tuple[GeneratedCudaScalarBody, ...]
    backend_requirement: str


def generate_partitioned_gemm_finalization(program: PartitionedGemmProgram) -> GeneratedPartitionedGemmFinalization:
    """Generate scalar bodies without selecting a workload-specific epilogue."""
    bodies = tuple(
        generate_cuda_scalar_body(finalization.program, symbol=f"generated_partition_map_{index}")
        for index, finalization in enumerate(program.scalar_finalizations)
    )
    contributions = tuple(
        generate_cuda_scalar_body(fold.contribution, symbol=f"generated_partition_fold_contribution_{index}")
        for index, fold in enumerate(program.auxiliary_folds)
    )
    reducers = tuple(
        generate_cuda_scalar_body(fold.reducer, symbol=f"generated_partition_fold_reducer_{index}")
        for index, fold in enumerate(program.auxiliary_folds)
    )
    return GeneratedPartitionedGemmFinalization(
        template=_program_template(program),
        semantic_digest=program.semantic_digest,
        scalar_bodies=bodies,
        auxiliary_contribution_bodies=contributions,
        auxiliary_reducer_bodies=reducers,
        backend_requirement=(
            "retain one shared-reduction GEMM mainloop; expose contiguous accumulator partitions; round each Map "
            "input partition from FP32 to BF16 RNE before invoking its generated scalar body; store scalar outputs "
            "and passthrough partitions directly without a concatenated result; evaluate auxiliary Fold "
            "contributions from the retained BF16 boundary and reduce them under their declared reassociation policy"
        ),
    )


def _program_template(program: PartitionedGemmProgram) -> str:
    return (
        "partitioned_gemm_auxiliary_fold_finalization"
        if program.auxiliary_folds
        else "partitioned_gemm_scalar_finalization"
    )
