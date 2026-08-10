# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover generic partial-value completion from post-SPMD HLO collectives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from shuttle.ir import DType
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import HloInstruction, HloModuleGraph, parse_hlo_module_text

_TO_APPLY = re.compile(r"(?:^|[, ])to_apply=%([A-Za-z0-9_.-]+)")
_CHANNEL_ID = re.compile(r"(?:^|[, ])channel_id=([0-9]+)")
_REPLICA_GROUPS = re.compile(r"(?:^|[, ])replica_groups=\{\{(.*?)\}\}")
_GLOBAL_DEVICE_IDS = re.compile(r"(?:^|[, ])use_global_device_ids=(true|false)")
_HLO_DTYPES = {
    "bf16": DType.BF16,
    "f32": DType.FP32,
    "f64": DType.FP64,
    "s32": DType.INT32,
}
_DTYPE_HLO_NAMES = {dtype: hlo_name for hlo_name, dtype in _HLO_DTYPES.items()}


class ValueCompleteness(StrEnum):
    """Whether a value still has contributions pending over a placement domain."""

    PARTIAL = "partial"
    COMPLETE = "complete"


class CollectiveReduction(StrEnum):
    """Generic merge operator used to complete a partial value."""

    SUM = "sum"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    PRODUCT = "product"


@dataclass(frozen=True)
class ReplicaGroupDomain:
    """Logical participants whose contributions complete one value."""

    groups: tuple[tuple[int, ...], ...]
    use_global_device_ids: bool

    def __post_init__(self) -> None:
        if not self.groups or any(not group for group in self.groups):
            raise ValueError("replica groups must contain at least one non-empty group")
        flattened = tuple(device for group in self.groups for device in group)
        if len(set(flattened)) != len(flattened):
            raise ValueError("a device may appear in only one replica group")
        if any(device < 0 for device in flattened):
            raise ValueError("replica-group device IDs must be non-negative")


@dataclass(frozen=True)
class CollectiveFoldPlan:
    """Semantic Fold that turns placement-partial contributions into a complete value."""

    reduction: CollectiveReduction
    dtype: DType
    numerical_policy: NumericalPolicy
    input_completeness: ValueCompleteness = ValueCompleteness.PARTIAL
    output_completeness: ValueCompleteness = ValueCompleteness.COMPLETE


@dataclass(frozen=True)
class PlacementTransitionPlan:
    """Transport domain for a value completion, independent of its physical mechanism."""

    source_value: str
    destination_value: str
    replica_domain: ReplicaGroupDomain
    channel_id: int | None


@dataclass(frozen=True)
class CollectiveCompletionPlan:
    """One generic Fold plus the placement transition carrying its contributions."""

    shape: str
    fold: CollectiveFoldPlan
    transport: PlacementTransitionPlan


def recover_collective_completion_plans(
    hlo_text: str,
    *,
    producer_values: tuple[str, ...] | None = None,
) -> tuple[CollectiveCompletionPlan, ...]:
    """Recover direct all-reduce consumers as generic Fold/Transport plans.

    The result deliberately records no NCCL or XLA implementation choice. The
    HLO collective is evidence that a placement-partial producer requires a
    merge over a replica domain before its value is complete.
    """
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    producers = None if producer_values is None else frozenset(producer_values)
    plans: list[CollectiveCompletionPlan] = []
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    for instruction in entry.instructions:
        if instruction.opcode != "all-reduce":
            continue
        if len(instruction.operands) != 1:
            raise ValueError(f"all-reduce %{instruction.name} must have exactly one operand")
        source_name = instruction.operands[0]
        if producers is not None and source_name not in producers:
            continue
        source = instructions[source_name]
        if source.shape != instruction.shape:
            raise ValueError(
                f"all-reduce %{instruction.name} changes shape from {source.shape!r} to {instruction.shape!r}"
            )
        dtype = _dtype(instruction)
        reduction = _reduction(module, instruction, dtype)
        plans.append(
            CollectiveCompletionPlan(
                shape=instruction.shape,
                fold=CollectiveFoldPlan(
                    reduction=reduction,
                    dtype=dtype,
                    numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
                ),
                transport=PlacementTransitionPlan(
                    source_value=source_name,
                    destination_value=instruction.name,
                    replica_domain=ReplicaGroupDomain(
                        groups=_replica_groups(instruction),
                        use_global_device_ids=_global_device_ids(instruction),
                    ),
                    channel_id=_channel_id(instruction),
                ),
            )
        )
    return tuple(plans)


def _dtype(instruction: HloInstruction) -> DType:
    try:
        return _HLO_DTYPES[instruction.dtype]
    except KeyError as error:
        raise ValueError(f"unsupported collective dtype {instruction.dtype!r}") from error


def _reduction(module: HloModuleGraph, instruction: HloInstruction, dtype: DType) -> CollectiveReduction:
    match = _TO_APPLY.search(instruction.attributes)
    if match is None:
        raise ValueError(f"all-reduce %{instruction.name} has no reduction computation")
    computation = module.computation(match.group(1))
    parameters = tuple(value for value in computation.instructions if value.opcode == "parameter")
    root = computation.root
    parameter_names = {value.name for value in parameters}
    if len(parameters) != 2 or len(root.operands) != 2 or set(root.operands) != parameter_names:
        raise ValueError(f"all-reduce %{instruction.name} reduction must be a binary scalar Fold")
    if root.dtype != _DTYPE_HLO_NAMES[dtype]:
        raise ValueError(
            f"all-reduce %{instruction.name} reducer dtype {root.dtype!r} disagrees with value dtype {dtype.value!r}"
        )
    reductions = {
        "add": CollectiveReduction.SUM,
        "maximum": CollectiveReduction.MAXIMUM,
        "minimum": CollectiveReduction.MINIMUM,
        "multiply": CollectiveReduction.PRODUCT,
    }
    try:
        return reductions[root.opcode]
    except KeyError as error:
        raise ValueError(f"unsupported all-reduce Fold operator {root.opcode!r}") from error


def _replica_groups(instruction: HloInstruction) -> tuple[tuple[int, ...], ...]:
    match = _REPLICA_GROUPS.search(instruction.attributes)
    if match is None:
        raise ValueError(f"all-reduce %{instruction.name} has no explicit replica groups")
    return tuple(tuple(int(device) for device in group.split(",")) for group in match.group(1).split("},{"))


def _global_device_ids(instruction: HloInstruction) -> bool:
    match = _GLOBAL_DEVICE_IDS.search(instruction.attributes)
    if match is None:
        raise ValueError(f"all-reduce %{instruction.name} does not declare device-ID semantics")
    return match.group(1) == "true"


def _channel_id(instruction: HloInstruction) -> int | None:
    match = _CHANNEL_ID.search(instruction.attributes)
    return None if match is None else int(match.group(1))
