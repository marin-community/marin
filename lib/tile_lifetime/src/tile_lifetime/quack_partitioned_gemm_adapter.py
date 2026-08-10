# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower logical accumulator partitions to a bounded QuACK adapter contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.partitioned_gemm_program import PartitionedGemmProgram, generate_partitioned_gemm_finalization

QUACK_PARTITION_ADAPTER_BASE_REVISION = "84ef91df9bec87c7e4938517234fafb07ef844dd"

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\](?:\{[^}]+\})?")


class QuackPartitionFinalizationKind(StrEnum):
    """How one logical accumulator partition group reaches global memory."""

    SCALAR_MAP = "scalar_map"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class QuackSegmentedRhsSource:
    """One independently allocated RHS tensor in a shared logical N domain."""

    operand_index: int
    n_start: int
    n_limit: int
    shape: str


@dataclass(frozen=True)
class QuackAccumulatorPartitionView:
    """A logical N interval exposed from each FP32 accumulator subtile."""

    partition_index: int
    n_start: int
    n_limit: int
    boundary_dtype: str
    boundary_rounding: str


@dataclass(frozen=True)
class QuackPartitionStore:
    """One direct output store generated from logical accumulator views."""

    output_index: int
    kind: QuackPartitionFinalizationKind
    source_partitions: tuple[int, ...]
    output_shape: str
    scalar_body: str | None


@dataclass(frozen=True)
class QuackPartitionedGemmAdapterPlan:
    """Exact reusable extension required around QuACK's existing GEMM driver."""

    base_revision: str
    semantic_digest: str
    unpartitioned_operand_index: int
    segmented_rhs_sources: tuple[QuackSegmentedRhsSource, ...]
    accumulator_views: tuple[QuackAccumulatorPartitionView, ...]
    stores: tuple[QuackPartitionStore, ...]
    implementation_sites: tuple[str, ...]
    requires_composed_rhs_tma: bool
    requires_physical_proof: bool


def plan_quack_partitioned_gemm_adapter(program: PartitionedGemmProgram) -> QuackPartitionedGemmAdapterPlan:
    """Describe the smallest generic QuACK extension for a partitioned GEMM.

    This is an inspectable backend contract, not a registered kernel. The first
    bounded adapter supports independently allocated, contiguous RHS N
    partitions. It deliberately rejects split-LHS and packed-buffer assumptions
    rather than silently introducing copies or separate GEMM launches.
    """
    if program.partitioned_operand != 1:
        raise ValueError("the bounded QuACK adapter currently supports only RHS N partitions")
    if len(program.operand_shapes) != len(program.partitions) + 1:
        raise ValueError("partitioned GEMM ABI must contain one lhs and one rhs tensor per N partition")

    m, _, k = program.shape
    lhs_dims = _shape_dims(program.operand_shapes[0])
    if lhs_dims[-1] != k or _product(lhs_dims[:-1]) != m:
        raise ValueError("unpartitioned lhs shape does not match the flattened GEMM M/K dimensions")

    rhs_sources: list[QuackSegmentedRhsSource] = []
    views: list[QuackAccumulatorPartitionView] = []
    for partition_index, (partition, shape) in enumerate(
        zip(program.partitions, program.operand_shapes[1:], strict=True)
    ):
        rhs_dims = _shape_dims(shape)
        if rhs_dims != (partition.extent, k):
            raise ValueError("each RHS source must have physical shape [partition_N, K]")
        rhs_sources.append(
            QuackSegmentedRhsSource(
                operand_index=partition_index + 1,
                n_start=partition.start,
                n_limit=partition.limit,
                shape=shape,
            )
        )
        views.append(
            QuackAccumulatorPartitionView(
                partition_index=partition_index,
                n_start=partition.start,
                n_limit=partition.limit,
                boundary_dtype=program.partition_dtype,
                boundary_rounding=program.output_rounding,
            )
        )

    generated = program.scalar_finalizations
    generated_bodies = tuple(
        finalization.source for finalization in generate_partitioned_gemm_finalization(program).scalar_bodies
    )
    stores = tuple(
        QuackPartitionStore(
            output_index=output_index,
            kind=QuackPartitionFinalizationKind.SCALAR_MAP,
            source_partitions=finalization.source_partitions,
            output_shape=finalization.output_shape,
            scalar_body=generated_bodies[output_index],
        )
        for output_index, finalization in enumerate(generated)
    ) + tuple(
        QuackPartitionStore(
            output_index=len(generated) + output_index,
            kind=QuackPartitionFinalizationKind.PASSTHROUGH,
            source_partitions=(finalization.source_partition,),
            output_shape=finalization.output_shape,
            scalar_body=None,
        )
        for output_index, finalization in enumerate(program.passthrough_finalizations)
    )
    for store in stores:
        extent = program.partitions[store.source_partitions[0]].extent
        if any(program.partitions[index].extent != extent for index in store.source_partitions):
            raise ValueError("one direct store requires aligned source partitions")
        output_dims = _shape_dims(store.output_shape)
        if output_dims[-1] != extent or _product(output_dims[:-1]) != m:
            raise ValueError("direct output shape does not match its accumulator partition domain")

    return QuackPartitionedGemmAdapterPlan(
        base_revision=QUACK_PARTITION_ADAPTER_BASE_REVISION,
        semantic_digest=program.semantic_digest,
        unpartitioned_operand_index=0,
        segmented_rhs_sources=tuple(rhs_sources),
        accumulator_views=tuple(views),
        stores=stores,
        implementation_sites=(
            "quack/gemm_sm90.py: staged B producer accepts static segmented RHS tensors",
            "quack/epilogue/frontend.py: EpiMod cache key carries accumulator partition descriptors",
            "quack/epilogue/visit.py: epi_visit_subtile forms coordinate-aligned partition views",
            "quack/epilogue/ops.py: TileStore maps an input N interval to output-local coordinates",
            "quack/gemm_base.py: existing multi-output register/shared/TMA store driver remains unchanged",
        ),
        requires_composed_rhs_tma=True,
        requires_physical_proof=True,
    )


def _shape_dims(shape: str) -> tuple[int, ...]:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        raise ValueError(f"unsupported physical array shape {shape!r}")
    dims = match.group("dims")
    return tuple(int(dim) for dim in dims.split(",")) if dims else ()


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
