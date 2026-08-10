# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan one SM90 tiled mainloop over independently allocated RHS segments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from tile_lifetime.cast_scalar_program import CastScalarDType, CastScalarExpression, CastScalarKind
from tile_lifetime.partitioned_gemm_program import PartitionedGemmProgram, generate_partitioned_gemm_finalization
from tile_lifetime.quack_partitioned_gemm_adapter import (
    QUACK_PARTITION_ADAPTER_BASE_REVISION,
    QuackPartitionFinalizationKind,
    plan_quack_partitioned_gemm_adapter,
)

QUACK_0_5_0_WHEEL_SHA256 = "08821ebfb8e638cc20308d5c59410c6dbb3b637ccc7b07bd57c7a9261a06af74"
QUACK_PARTITIONED_SM90_PATCH_SHA256 = "0bbb2354cff80b2fdf475fce12cef277f961591623b0c078160c27f09e5658db"
_SM90_MMA_N_QUANTUM = 8
_SM90_MMA_N_MAX = 256
_REQUIRED_EXTENSION_SYMBOLS = (
    "validate_rhs_segments",
    "partition_accumulator_groups",
    "gemm_groups_w_idx",
    "round_group_to_bf16_rne",
    "PartitionedGemmSm90",
)
_FORBIDDEN_EXTENSION_TOKENS = (
    "swiglu",
    "router",
    "mixture_of_kittens",
    "mok_forward",
    "flash_attention",
)


@dataclass(frozen=True)
class QuackRhsMmaGroup:
    """One RHS allocation and its independently staged WGMMA N group."""

    operand_index: int
    logical_n_start: int
    logical_n_limit: int
    mma_n: int
    valid_n: int


@dataclass(frozen=True)
class QuackAccumulatorGroup:
    """One coordinate-aligned FP32 accumulator group."""

    partition_index: int
    logical_n_start: int
    logical_n_limit: int
    mma_n: int
    valid_n: int
    boundary_dtype: str
    boundary_rounding: str


@dataclass(frozen=True)
class QuackMainloopStore:
    """A direct output from one or more aligned accumulator groups."""

    output_index: int
    kind: QuackPartitionFinalizationKind
    source_groups: tuple[int, ...]
    valid_n: int
    output_shape: str
    scalar_body: str | None


@dataclass(frozen=True)
class QuackPartitionedMainloopPlan:
    """Bounded physical plan for a segmented SM90 Contract.

    The plan deliberately uses one CTA program and one ordered K loop. The A
    stage is shared by all RHS groups. Each independently allocated RHS segment
    has its own B stage and WGMMA accumulator group. Equal-width groups have
    congruent register coordinates, so generated scalar Maps can combine them
    elementwise without reconstructing a dense concatenated accumulator.
    """

    base_revision: str
    inspected_wheel_sha256: str
    semantic_digest: str
    tile_m: int
    tile_k: int
    rhs_groups: tuple[QuackRhsMmaGroup, ...]
    accumulator_groups: tuple[QuackAccumulatorGroup, ...]
    stores: tuple[QuackMainloopStore, ...]
    one_kernel: bool
    one_k_loop: bool
    shared_a_stage: bool
    requires_external_quack_extension: bool
    extension_sites: tuple[str, ...]

    @property
    def physical_digest(self) -> str:
        """Return a digest that includes generated scalar semantics."""
        record = {
            "base_revision": self.base_revision,
            "semantic_digest": self.semantic_digest,
            "tile_m": self.tile_m,
            "tile_k": self.tile_k,
            "rhs_groups": [asdict(group) for group in self.rhs_groups],
            "accumulator_groups": [asdict(group) for group in self.accumulator_groups],
            "stores": [asdict(store) for store in self.stores],
            "one_kernel": self.one_kernel,
            "one_k_loop": self.one_k_loop,
            "shared_a_stage": self.shared_a_stage,
        }
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class QuackPartitionedExtensionAudit:
    """Static audit of the isolated QuACK extension patch."""

    sha256: str
    required_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    forbidden_tokens: tuple[str, ...]
    creates_one_module: bool
    syntax_compiles: bool

    @property
    def clean(self) -> bool:
        """Return whether the reusable helper patch passes the static boundary."""
        return (
            self.sha256 == QUACK_PARTITIONED_SM90_PATCH_SHA256
            and not self.missing_symbols
            and not self.forbidden_tokens
            and self.creates_one_module
            and self.syntax_compiles
        )


@dataclass(frozen=True)
class GeneratedQuackPartitionedMainloop:
    """CuTe authoring module generated from a partitioned Contract."""

    module_name: str
    source: str
    semantic_digest: str
    source_digest: str
    rhs_mma_ns: tuple[int, ...]
    output_count: int


def audit_quack_partitioned_extension_patch(patch_path: Path) -> QuackPartitionedExtensionAudit:
    """Check the pinned extension artifact without importing CUDA packages."""
    source = patch_path.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    added_source = "\n".join(
        line[1:] for line in source.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    syntax_compiles = True
    try:
        compile(added_source, "quack/partitioned_sm90.py", "exec")
    except SyntaxError:
        syntax_compiles = False
    lowered = added_source.lower()
    missing = tuple(
        symbol
        for symbol in _REQUIRED_EXTENSION_SYMBOLS
        if f"def {symbol}(" not in added_source and f"class {symbol}(" not in added_source
    )
    forbidden = tuple(token for token in _FORBIDDEN_EXTENSION_TOKENS if token in lowered)
    return QuackPartitionedExtensionAudit(
        sha256=digest,
        required_symbols=_REQUIRED_EXTENSION_SYMBOLS,
        missing_symbols=missing,
        forbidden_tokens=forbidden,
        creates_one_module=(
            source.count("diff --git ") == 1
            and "new file mode 100644" in source
            and "+++ b/quack/partitioned_sm90.py" in source
        ),
        syntax_compiles=syntax_compiles,
    )


def plan_quack_partitioned_mainloop(
    program: PartitionedGemmProgram,
    *,
    tile_m: int = 64,
    tile_k: int = 64,
) -> QuackPartitionedMainloopPlan:
    """Lower a generic partitioned Contract to coordinate-aligned WGMMA groups.

    This is the physical candidate immediately below ``PartitionedGemmProgram``.
    It does not claim the stock QuACK package can execute the plan: both the
    pinned revision and QuACK 0.5.0 expose only one B tensor and one accumulator
    in ``GemmSm90``. ``requires_external_quack_extension`` remains true until
    that reusable primitive exists and is compiled on SM90.
    """
    if tile_m != 64:
        raise ValueError("the first SM90 partitioned mainloop executor requires tile M 64")
    if tile_k <= 0 or tile_k % 16:
        raise ValueError("SM90 partitioned mainloop tile K must be a positive multiple of 16")
    adapter = plan_quack_partitioned_gemm_adapter(program)
    generated = generate_partitioned_gemm_finalization(program)

    rhs_groups = tuple(
        QuackRhsMmaGroup(
            operand_index=source.operand_index,
            logical_n_start=source.n_start,
            logical_n_limit=source.n_limit,
            mma_n=_mma_n(source.n_limit - source.n_start),
            valid_n=source.n_limit - source.n_start,
        )
        for source in adapter.segmented_rhs_sources
    )
    accumulator_groups = tuple(
        QuackAccumulatorGroup(
            partition_index=view.partition_index,
            logical_n_start=view.n_start,
            logical_n_limit=view.n_limit,
            mma_n=rhs_groups[view.partition_index].mma_n,
            valid_n=view.n_limit - view.n_start,
            boundary_dtype=view.boundary_dtype,
            boundary_rounding=view.boundary_rounding,
        )
        for view in adapter.accumulator_views
    )
    scalar_bodies = iter(body.source for body in generated.scalar_bodies)
    stores = tuple(
        QuackMainloopStore(
            output_index=store.output_index,
            kind=store.kind,
            source_groups=store.source_partitions,
            valid_n=program.partitions[store.source_partitions[0]].extent,
            output_shape=store.output_shape,
            scalar_body=next(scalar_bodies) if store.kind is QuackPartitionFinalizationKind.SCALAR_MAP else None,
        )
        for store in adapter.stores
    )
    return QuackPartitionedMainloopPlan(
        base_revision=QUACK_PARTITION_ADAPTER_BASE_REVISION,
        inspected_wheel_sha256=QUACK_0_5_0_WHEEL_SHA256,
        semantic_digest=program.semantic_digest,
        tile_m=tile_m,
        tile_k=tile_k,
        rhs_groups=rhs_groups,
        accumulator_groups=accumulator_groups,
        stores=stores,
        one_kernel=True,
        one_k_loop=True,
        shared_a_stage=True,
        requires_external_quack_extension=True,
        extension_sites=(
            "quack/partitioned_sm90.py: accept a static RHS tensor tuple in __call__ and kernel",
            "quack/partitioned_sm90.py: stage one A tile and one B tile per RHS group under one AB barrier",
            "quack/partitioned_sm90.py: issue all group WGMMA operations inside the same ordered K loop",
            "quack/partitioned_sm90.py: carry multiple direct output tensors through the generated finalizer",
            "quack/partitioned_sm90.py: invoke generated scalar Maps on congruent accumulator-group coordinates",
        ),
    )


def generate_quack_partitioned_mainloop(
    program: PartitionedGemmProgram,
    *,
    tile_m: int = 64,
    tile_k: int = 64,
) -> GeneratedQuackPartitionedMainloop:
    """Render the static finalizer and generic QuACK executor instantiation."""
    plan = plan_quack_partitioned_mainloop(program, tile_m=tile_m, tile_k=tile_k)
    scalar_methods = "\n\n".join(
        _render_cute_scalar_method(index, finalization.program.expression)
        for index, finalization in enumerate(program.scalar_finalizations)
    )
    store_body = _render_cute_store_body(program, tile_m=tile_m)
    rhs_names = tuple(f"rhs{index}" for index in range(len(plan.rhs_groups)))
    output_names = tuple(f"output{index}" for index in range(len(plan.stores)))
    parameters = ", ".join(("lhs", *rhs_names, *output_names, "stream"))
    source = f"""# Generated from PartitionedGemmProgram {program.semantic_digest}; do not edit.
from typing import Tuple

import cutlass
import cutlass.cute as cute

from quack.partitioned_sm90 import PartitionedGemmSm90


class GeneratedPartitionFinalizer:
    output_count = {len(plan.stores)}

{scalar_methods}

    @cute.jit
    def store(self, boundaries, outputs, m_start, m_extent, batch_idx, tidx, threads):
{store_body}


_executor = PartitionedGemmSm90(
    {tuple(group.mma_n for group in plan.rhs_groups)!r},
    tile_m={tile_m},
    tile_k={tile_k},
    finalizer=GeneratedPartitionFinalizer(),
)


@cute.jit
def run({parameters}):
    _executor(lhs, {rhs_names!r}, {output_names!r}, stream)
"""
    for name in (*rhs_names, *output_names):
        source = source.replace(repr(name), name)
    digest = hashlib.sha256(source.encode()).hexdigest()
    return GeneratedQuackPartitionedMainloop(
        module_name=f"shuttle_partitioned_sm90_{digest[:16]}",
        source=source,
        semantic_digest=program.semantic_digest,
        source_digest=digest,
        rhs_mma_ns=tuple(group.mma_n for group in plan.rhs_groups),
        output_count=len(plan.stores),
    )


def _mma_n(valid_n: int) -> int:
    padded = math.ceil(valid_n / _SM90_MMA_N_QUANTUM) * _SM90_MMA_N_QUANTUM
    if padded > _SM90_MMA_N_MAX:
        raise ValueError(f"one SM90 RHS group exceeds the supported MMA N extent: {valid_n}")
    return padded


def _render_cute_scalar_method(index: int, expression: CastScalarExpression) -> str:
    inputs = _scalar_inputs(expression)
    arguments = ", ".join(value.input_name for value in inputs)
    body = _render_cute_expression(expression)
    return f"    @cute.jit\n    def scalar_{index}(self, {arguments}):\n        return {body}\n"


def _render_cute_store_body(program: PartitionedGemmProgram, *, tile_m: int) -> str:
    lines: list[str] = []
    for output_index, finalization in enumerate(program.scalar_finalizations):
        extent = program.partitions[finalization.source_partitions[0]].extent
        lines.extend(
            (
                f"        for linear in cutlass.range(tidx, {tile_m} * {extent}, threads):",
                f"            local_m = linear // {extent}",
                f"            feature = linear - local_m * {extent}",
                "            global_m = m_start + local_m",
                "            if global_m < m_extent:",
            )
        )
        arguments: list[str] = []
        for partition_index, scalar_input in zip(
            finalization.source_partitions, finalization.program.inputs, strict=True
        ):
            assert scalar_input.input_index is not None
            arguments.append(
                "cutlass.Float32(boundaries["
                f"{partition_index}][local_m + {scalar_input.input_index.row_offset}, "
                f"feature + {scalar_input.input_index.feature_offset}])"
            )
        lines.append(f"                value = self.scalar_{output_index}({', '.join(arguments)})")
        lines.append(f"                outputs[{output_index}][global_m, feature, batch_idx] = cutlass.BFloat16(value)")
    output_index = len(program.scalar_finalizations)
    for finalization in program.passthrough_finalizations:
        extent = program.partitions[finalization.source_partition].extent
        lines.extend(
            (
                f"        for linear in cutlass.range(tidx, {tile_m} * {extent}, threads):",
                f"            local_m = linear // {extent}",
                f"            feature = linear - local_m * {extent}",
                "            global_m = m_start + local_m",
                "            if global_m < m_extent:",
                f"                outputs[{output_index}][global_m, feature, batch_idx] = "
                f"boundaries[{finalization.source_partition}][local_m, feature]",
            )
        )
        output_index += 1
    return "\n".join(lines)


def _scalar_inputs(expression: CastScalarExpression) -> tuple[CastScalarExpression, ...]:
    leaves: dict[str, CastScalarExpression] = {}

    def visit(value: CastScalarExpression) -> None:
        if value.kind is CastScalarKind.INPUT:
            assert value.input_name is not None
            leaves[value.input_name] = value
        for operand in value.operands:
            visit(operand)

    visit(expression)
    return tuple(leaves[name] for name in sorted(leaves))


def _render_cute_expression(expression: CastScalarExpression) -> str:
    if expression.kind is CastScalarKind.INPUT:
        assert expression.input_name is not None
        rendered = expression.input_name
    elif expression.kind is CastScalarKind.CONSTANT:
        assert expression.constant is not None
        rendered = repr(expression.constant)
    else:
        operands = tuple(_render_cute_expression(operand) for operand in expression.operands)
        if expression.kind is CastScalarKind.CONVERT:
            rendered = operands[0]
        elif expression.kind is CastScalarKind.NEGATE:
            rendered = f"(-{operands[0]})"
        elif expression.kind is CastScalarKind.ADD:
            rendered = f"({operands[0]} + {operands[1]})"
        elif expression.kind is CastScalarKind.SUBTRACT:
            rendered = f"({operands[0]} - {operands[1]})"
        elif expression.kind is CastScalarKind.MULTIPLY:
            rendered = f"({operands[0]} * {operands[1]})"
        elif expression.kind is CastScalarKind.DIVIDE:
            rendered = f"({operands[0]} / {operands[1]})"
        elif expression.kind is CastScalarKind.EXP:
            rendered = f"cute.math.exp({operands[0]}, fastmath=False)"
        elif expression.kind is CastScalarKind.TANH:
            rendered = f"cute.math.tanh({operands[0]}, fastmath=False)"
        else:
            assert expression.kind is CastScalarKind.SELECT
            rendered = f"({operands[1]} if {operands[0]} else {operands[2]})"
    if expression.dtype is CastScalarDType.BF16:
        return f"cutlass.Float32(cutlass.BFloat16({rendered}))"
    if expression.dtype is CastScalarDType.F32:
        return f"cutlass.Float32({rendered})"
    return f"cutlass.Boolean({rendered})"
