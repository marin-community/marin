# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent programs executed during one tile lifetime."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class TileProgramStage(StrEnum):
    """Phase of an expert skeleton in which a tile operation executes."""

    PREPARATION = "preparation"
    FINALIZATION = "finalization"


class TilePrimitive(StrEnum):
    """Small semantic vocabulary shared by tiled skeleton backends."""

    LOAD_TILE = "load_tile"
    LOAD_ROW = "load_row"
    LOAD_EDGE_WEIGHT = "load_edge_weight"
    LOAD_STATE = "load_state"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    SCALE_ROW = "scale_row"
    RESIDUAL_ADD = "residual_add"
    MULTIPLY_GAMMA = "multiply_gamma"
    PARTIAL_SUM_SQUARE = "partial_sum_square"
    PARTIAL_SUM = "partial_sum"
    PAIRWISE_MAP = "pairwise_map"
    PAIRWISE_LINEAR_MAP = "pairwise_linear_map"
    PAIRWISE_SWIGLU = "pairwise_swiglu"
    PAIRWISE_ROPE = "pairwise_rope"
    VIEW = "view"
    PARTITION = "partition"
    CONVERT = "convert"
    STORE = "store"


@dataclass(frozen=True)
class TileOp:
    """One semantic operation over tile-resident values."""

    primitive: TilePrimitive
    stage: TileProgramStage
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ValueLifetime:
    """Inclusive operation-index interval in which a value remains live."""

    value: str
    first_operation: int
    last_operation: int
    layout: str | None
    external: bool


@dataclass(frozen=True)
class TileProgram:
    """Optimized tile operations plus inspectable aliases and resource facts."""

    operations: tuple[TileOp, ...]
    aliases: tuple[tuple[str, str], ...]
    lifetimes: tuple[ValueLifetime, ...]
    peak_live_values: int
    value_layouts: tuple[tuple[str, str], ...]

    def operations_at(self, stage: TileProgramStage) -> tuple[TileOp, ...]:
        """Return operations assigned to one skeleton phase."""
        return tuple(operation for operation in self.operations if operation.stage is stage)

    def primitives_at(self, stage: TileProgramStage) -> tuple[TilePrimitive, ...]:
        """Return the primitive signature assigned to one skeleton phase."""
        return tuple(operation.primitive for operation in self.operations_at(stage))


class TileProgramError(ValueError):
    """A tile program is unsupported or violates an explicit layout contract."""


_EFFECTFUL_PRIMITIVES = frozenset({TilePrimitive.STORE})


def optimize_tile_program(
    operations: tuple[TileOp, ...],
    *,
    required_outputs: tuple[str, ...],
    value_layouts: Mapping[str, str] | None = None,
) -> TileProgram:
    """Apply CSE and dead-code elimination, then estimate value liveness."""
    layouts = dict(value_layouts or {})
    deduplicated, aliases = _eliminate_common_subexpressions(operations)
    required = tuple(_resolve_alias(value, aliases) for value in required_outputs)
    retained = _eliminate_dead_operations(deduplicated, frozenset(required))
    _validate_stage_order(retained)
    lifetimes = _value_lifetimes(retained, layouts)
    return TileProgram(
        operations=retained,
        aliases=tuple(sorted(aliases.items())),
        lifetimes=lifetimes,
        peak_live_values=_peak_live_values(lifetimes),
        value_layouts=tuple(sorted(layouts.items())),
    )


def _eliminate_common_subexpressions(
    operations: tuple[TileOp, ...],
) -> tuple[tuple[TileOp, ...], dict[str, str]]:
    aliases: dict[str, str] = {}
    available: dict[tuple[object, ...], tuple[str, ...]] = {}
    retained: list[TileOp] = []
    for operation in operations:
        inputs = tuple(_resolve_alias(value, aliases) for value in operation.inputs)
        rewritten = TileOp(
            primitive=operation.primitive,
            stage=operation.stage,
            inputs=inputs,
            outputs=operation.outputs,
            attributes=operation.attributes,
        )
        if operation.primitive in _EFFECTFUL_PRIMITIVES:
            retained.append(rewritten)
            continue
        key = (operation.stage, operation.primitive, inputs, operation.attributes)
        existing = available.get(key)
        if existing is not None and len(existing) == len(operation.outputs):
            for duplicate, canonical in zip(operation.outputs, existing, strict=True):
                if duplicate != canonical:
                    aliases[duplicate] = canonical
            continue
        available[key] = operation.outputs
        retained.append(rewritten)
    rewritten = tuple(
        TileOp(
            primitive=operation.primitive,
            stage=operation.stage,
            inputs=tuple(_resolve_alias(value, aliases) for value in operation.inputs),
            outputs=operation.outputs,
            attributes=operation.attributes,
        )
        for operation in retained
    )
    return rewritten, aliases


def _eliminate_dead_operations(
    operations: tuple[TileOp, ...],
    required_outputs: frozenset[str],
) -> tuple[TileOp, ...]:
    needed = set(required_outputs)
    retained: list[TileOp] = []
    for operation in reversed(operations):
        effectful = operation.primitive in _EFFECTFUL_PRIMITIVES
        if not effectful and needed.isdisjoint(operation.outputs):
            continue
        retained.append(operation)
        needed.difference_update(operation.outputs)
        needed.update(operation.inputs)
    retained.reverse()
    return tuple(retained)


def _resolve_alias(value: str, aliases: Mapping[str, str]) -> str:
    seen: set[str] = set()
    while value in aliases:
        if value in seen:
            raise TileProgramError(f"cyclic tile-value alias involving {value!r}")
        seen.add(value)
        value = aliases[value]
    return value


def _validate_stage_order(operations: tuple[TileOp, ...]) -> None:
    reached_finalization = False
    for operation in operations:
        if operation.stage is TileProgramStage.FINALIZATION:
            reached_finalization = True
        elif reached_finalization:
            raise TileProgramError("preparation operation follows a finalization operation")


def _value_lifetimes(
    operations: tuple[TileOp, ...],
    layouts: Mapping[str, str],
) -> tuple[ValueLifetime, ...]:
    producers: dict[str, int] = {}
    uses: dict[str, list[int]] = {}
    for index, operation in enumerate(operations):
        for value in operation.outputs:
            producers.setdefault(value, index)
        for value in operation.inputs:
            uses.setdefault(value, []).append(index)
    values = set(producers) | set(uses)
    lifetimes: list[ValueLifetime] = []
    for value in sorted(values):
        producer = producers.get(value)
        value_uses = uses.get(value, ())
        first_operation = producer if producer is not None else min(value_uses)
        last_operation = max(value_uses) if value_uses else first_operation
        lifetimes.append(
            ValueLifetime(
                value=value,
                first_operation=first_operation,
                last_operation=last_operation,
                layout=layouts.get(value),
                external=producer is None,
            )
        )
    return tuple(lifetimes)


def _peak_live_values(lifetimes: tuple[ValueLifetime, ...]) -> int:
    internal = tuple(lifetime for lifetime in lifetimes if not lifetime.external)
    if not internal:
        return 0
    last_operation = max(lifetime.last_operation for lifetime in internal)
    return max(
        sum(lifetime.first_operation <= index <= lifetime.last_operation for lifetime in internal)
        for index in range(last_operation + 1)
    )
