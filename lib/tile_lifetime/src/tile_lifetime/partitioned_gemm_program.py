# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral GEMM programs with logical accumulator partitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from tile_lifetime.cast_scalar_program import CastScalarProgram, GeneratedCudaScalarBody, generate_cuda_scalar_body


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

    @property
    def output_shapes(self) -> tuple[str, ...]:
        """Return scalar-Map outputs followed by passthrough outputs."""
        return (
            *(finalization.output_shape for finalization in self.scalar_finalizations),
            *(finalization.output_shape for finalization in self.passthrough_finalizations),
        )

    @property
    def semantic_digest(self) -> str:
        """Return a source-name-independent physical-program digest."""
        record = {
            "template": "partitioned_gemm_scalar_finalization",
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
    backend_requirement: str


def generate_partitioned_gemm_finalization(program: PartitionedGemmProgram) -> GeneratedPartitionedGemmFinalization:
    """Generate scalar bodies without selecting a workload-specific epilogue."""
    bodies = tuple(
        generate_cuda_scalar_body(finalization.program, symbol=f"generated_partition_map_{index}")
        for index, finalization in enumerate(program.scalar_finalizations)
    )
    return GeneratedPartitionedGemmFinalization(
        template="partitioned_gemm_scalar_finalization",
        semantic_digest=program.semantic_digest,
        scalar_bodies=bodies,
        backend_requirement=(
            "retain one shared-reduction GEMM mainloop; expose contiguous accumulator partitions; round each Map "
            "input partition from FP32 to BF16 RNE before invoking its generated scalar body; store scalar outputs "
            "and passthrough partitions directly without a concatenated result"
        ),
    )
