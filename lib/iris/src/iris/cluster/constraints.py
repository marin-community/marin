# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Constraint types and helpers for Iris resource scheduling.

This module is the canonical home for all constraint-related types:

- WellKnownAttribute: canonical string keys for worker metadata
- AttributeValue, ConstraintOp, Constraint: core constraint dataclasses
- DeviceType and device-config helpers (get_device_type, get_device_variant, etc.)
- PlacementRequirements and extraction functions for demand routing
- Constraint factory functions (preemptible_constraint, region_constraint, etc.)
- constraints_from_resources: auto-generates device constraints from ResourceSpecProto

All production code should reference WellKnownAttribute enum members instead of
raw string literals so that typos are caught at import time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum
from typing import Any, ClassVar

from iris.rpc import config_pb2
from iris.rpc import job_pb2


class WellKnownAttribute(StrEnum):
    """Canonical attribute keys for constraint-based scheduling."""

    DEVICE_TYPE = "device-type"
    DEVICE_VARIANT = "device-variant"
    PREEMPTIBLE = "preemptible"
    REGION = "region"
    ZONE = "zone"
    TPU_NAME = "tpu-name"
    TPU_WORKER_ID = "tpu-worker-id"
    TPU_TOPOLOGY = "tpu-topology"
    TPU_VM_COUNT = "tpu-vm-count"
    GPU_VARIANT = "gpu-variant"
    GPU_COUNT = "gpu-count"


# ---------------------------------------------------------------------------
# Step 1 types: core constraint primitives (depend only on job_pb2)
# ---------------------------------------------------------------------------


class DeviceType(Enum):
    """Device type for demand routing."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


def get_device_type_enum(device: job_pb2.DeviceConfig) -> DeviceType:
    """Extract device type as enum from DeviceConfig."""
    if device.HasField("gpu"):
        return DeviceType.GPU
    if device.HasField("tpu"):
        return DeviceType.TPU
    return DeviceType.CPU


def get_device_type(device: job_pb2.DeviceConfig) -> str:
    """Extract device type string from DeviceConfig.

    Delegates to get_device_type_enum() to avoid duplicating the dispatch logic.
    """
    return get_device_type_enum(device).value


def get_device_variant(device: job_pb2.DeviceConfig) -> str | None:
    """Extract device variant (e.g., GPU model) from DeviceConfig."""
    if device.HasField("gpu"):
        return device.gpu.variant if device.gpu.variant else None
    if device.HasField("tpu"):
        return device.tpu.variant if device.tpu.variant else None
    return None


@dataclass(frozen=True)
class AttributeValue:
    """Typed attribute value for worker attributes and constraint matching.

    Used for coscheduling and constraint-based worker filtering.
    Values can be strings, integers, or floats.

    String values are stripped and lowercased at construction so that
    constraint comparisons are case-insensitive by construction — there is
    no way to hold a non-normalized string in this type. Worker attributes
    and constraint literals share this type and therefore share the
    normalization invariant.
    """

    value: str | int | float

    def __post_init__(self) -> None:
        if isinstance(self.value, str):
            object.__setattr__(self, "value", self.value.strip().lower())

    def to_proto(self) -> job_pb2.AttributeValue:
        """Convert to protobuf representation."""
        proto = job_pb2.AttributeValue()
        if isinstance(self.value, str):
            proto.string_value = self.value
        elif isinstance(self.value, int):
            proto.int_value = self.value
        elif isinstance(self.value, float):
            proto.float_value = self.value
        return proto

    @staticmethod
    def from_proto(proto: job_pb2.AttributeValue) -> AttributeValue:
        """Convert from protobuf representation."""
        if proto.HasField("string_value"):
            return AttributeValue(proto.string_value)
        elif proto.HasField("int_value"):
            return AttributeValue(proto.int_value)
        elif proto.HasField("float_value"):
            return AttributeValue(proto.float_value)
        # Default to empty string if no value set
        return AttributeValue("")


class ConstraintOp(IntEnum):
    """Constraint operators for worker attribute matching.

    Used to define constraints that filter which workers can run a job.
    Each operator compares a worker attribute against a constraint value.

    Example:
        >>> # Match workers where region equals "us-central1"
        >>> Constraint.create(key="region", op=ConstraintOp.EQ, value="us-central1")
        >>> # Match workers with memory > 32GB
        >>> Constraint.create(key="memory_gb", op=ConstraintOp.GT, value=32)
        >>> # Match workers that have the "gpu" attribute set
        >>> Constraint.create(key="gpu", op=ConstraintOp.EXISTS)
    """

    EQ = 0
    NE = 1
    EXISTS = 2
    NOT_EXISTS = 3
    GT = 4
    GE = 5
    LT = 6
    LE = 7
    IN = 8

    def to_proto(self) -> job_pb2.ConstraintOp:
        """Convert to protobuf ConstraintOp enum value."""
        mapping = {
            ConstraintOp.EQ: job_pb2.CONSTRAINT_OP_EQ,
            ConstraintOp.NE: job_pb2.CONSTRAINT_OP_NE,
            ConstraintOp.EXISTS: job_pb2.CONSTRAINT_OP_EXISTS,
            ConstraintOp.NOT_EXISTS: job_pb2.CONSTRAINT_OP_NOT_EXISTS,
            ConstraintOp.GT: job_pb2.CONSTRAINT_OP_GT,
            ConstraintOp.GE: job_pb2.CONSTRAINT_OP_GE,
            ConstraintOp.LT: job_pb2.CONSTRAINT_OP_LT,
            ConstraintOp.LE: job_pb2.CONSTRAINT_OP_LE,
            ConstraintOp.IN: job_pb2.CONSTRAINT_OP_IN,
        }
        return mapping[self]


# Per-op arity bounds (lo, hi) for Constraint.values. hi=None means unbounded.
# Enforced in Constraint.__post_init__ so invalid constraints cannot be constructed.
_CONSTRAINT_ARITY: dict[ConstraintOp, tuple[int, int | None]] = {
    ConstraintOp.EXISTS: (0, 0),
    ConstraintOp.NOT_EXISTS: (0, 0),
    ConstraintOp.EQ: (1, 1),
    ConstraintOp.NE: (1, 1),
    ConstraintOp.GT: (1, 1),
    ConstraintOp.GE: (1, 1),
    ConstraintOp.LT: (1, 1),
    ConstraintOp.LE: (1, 1),
    ConstraintOp.IN: (1, None),
}


@dataclass(frozen=True)
class Constraint:
    """Worker constraint for job scheduling.

    Constraints filter which workers are eligible to run a job based on
    worker attributes. Workers must satisfy all constraints to be considered.

    `values` is a tuple of `AttributeValue` — a single type for both worker-side
    and constraint-side scalars. The arity of `values` is determined by `op`
    and validated at construction; downstream code can always index into
    `values` without None checks.

    Prefer ``Constraint.create(...)`` in call sites — it accepts raw
    ``value=``/``values=`` scalars and wraps them in ``AttributeValue``
    automatically. The primary constructor is used by ``from_proto`` and
    tests of the invariant itself.

    Example:
        >>> # Require a specific TPU pod
        >>> Constraint.create(key="tpu-name", op=ConstraintOp.EQ, value="my-tpu-pod")
        >>> # Require workers with at least 64GB memory
        >>> Constraint.create(key="memory_gb", op=ConstraintOp.GE, value=64)
        >>> # Require workers that have a GPU
        >>> Constraint.create(key="gpu", op=ConstraintOp.EXISTS)
        >>> # Require workers in one of several regions
        >>> Constraint.create(key="region", op=ConstraintOp.IN,
        ...                   values=["us-central1", "us-central2"])
    """

    key: str
    op: ConstraintOp
    values: tuple[AttributeValue, ...] = ()
    mode: int = job_pb2.CONSTRAINT_MODE_REQUIRED

    def __post_init__(self) -> None:
        lo, hi = _CONSTRAINT_ARITY[self.op]
        n = len(self.values)
        if n < lo or (hi is not None and n > hi):
            bound = str(hi) if hi is not None else "∞"
            raise ValueError(f"Constraint op {self.op.name} requires {lo}..{bound} values, got {n}")

    @property
    def is_soft(self) -> bool:
        return self.mode == job_pb2.CONSTRAINT_MODE_PREFERRED

    def to_proto(self) -> job_pb2.Constraint:
        """Convert to protobuf representation.

        Singular ops (EQ/NE/GT/GE/LT/LE) write `proto.value`; IN writes
        `proto.values`; EXISTS/NOT_EXISTS write neither.
        """
        proto = job_pb2.Constraint(key=self.key, op=self.op.to_proto(), mode=self.mode)
        if self.op == ConstraintOp.IN:
            for v in self.values:
                proto.values.append(v.to_proto())
        elif self.values:
            proto.value.CopyFrom(self.values[0].to_proto())
        return proto

    @staticmethod
    def from_proto(proto: job_pb2.Constraint) -> Constraint:
        """Convert from protobuf representation.

        Normalization (strip/lowercase for strings) happens inside
        AttributeValue.__post_init__, so constraint evaluation never needs
        to re-normalize.
        """
        op = ConstraintOp(proto.op)
        if op in (ConstraintOp.EXISTS, ConstraintOp.NOT_EXISTS):
            values: tuple[AttributeValue, ...] = ()
        elif op == ConstraintOp.IN:
            values = tuple(AttributeValue.from_proto(v) for v in proto.values)
        else:
            values = (AttributeValue.from_proto(proto.value),)
        return Constraint(key=proto.key, op=op, values=values, mode=proto.mode)

    @classmethod
    def create(
        cls,
        key: str,
        op: ConstraintOp,
        *,
        value: str | int | float | None = None,
        values: Sequence[str | int | float] | None = None,
        mode: int = job_pb2.CONSTRAINT_MODE_REQUIRED,
    ) -> Constraint:
        """Ergonomic factory: wraps raw scalars in AttributeValue automatically.

        - Singular ops (EQ/NE/GT/GE/LT/LE): pass ``value=``.
        - IN: pass ``values=``.
        - EXISTS/NOT_EXISTS: pass neither.

        Raw strings are normalized (stripped + lowercased) via
        AttributeValue.__post_init__.
        """
        if op in (ConstraintOp.EXISTS, ConstraintOp.NOT_EXISTS):
            if value is not None or values is not None:
                raise ValueError(f"op={op.name} takes no value/values")
            tup: tuple[AttributeValue, ...] = ()
        elif op == ConstraintOp.IN:
            if value is not None or values is None:
                raise ValueError("op=IN requires values=, not value=")
            tup = tuple(AttributeValue(v) for v in values)
        else:
            if value is None or values is not None:
                raise ValueError(f"op={op.name} requires value=, not values=")
            tup = (AttributeValue(value),)
        return cls(key=key, op=op, values=tup, mode=mode)


# ---------------------------------------------------------------------------
# Step 2 types: constraint helpers (depend on WellKnownAttribute)
# ---------------------------------------------------------------------------


def preemptible_constraint(preemptible: bool = True, soft: bool | None = None) -> Constraint:
    """Constraint requiring (or preferring) workers to be preemptible (or not).

    Args:
        preemptible: Whether to match preemptible or non-preemptible workers.
        soft: If True, the constraint is soft — matching workers are preferred
            but non-matching workers are still eligible. If None (default),
            the mode is inferred from preemptible:
            - preemptible=False → hard constraint, because non-preemptible jobs
              (e.g. coordinators) genuinely cannot tolerate spot eviction.
            - preemptible=True → soft constraint, because preemptible is a cost
              preference: we prefer spot capacity but can fall back to on-demand
              when spot is unavailable rather than leaving the job unscheduled.
    """
    if soft is None:
        # preemptible=True is a preference (soft), preemptible=False is a requirement (hard)
        soft = preemptible
    mode = job_pb2.CONSTRAINT_MODE_PREFERRED if soft else job_pb2.CONSTRAINT_MODE_REQUIRED
    return Constraint.create(key=WellKnownAttribute.PREEMPTIBLE, op=ConstraintOp.EQ, value=str(preemptible), mode=mode)


def zone_constraint(zone: str) -> Constraint:
    """Constraint requiring workers to be in a given zone."""
    if not zone:
        raise ValueError("zone must be non-empty")
    return Constraint.create(key=WellKnownAttribute.ZONE, op=ConstraintOp.EQ, value=zone)


def region_constraint(regions: list[str]) -> Constraint:
    """Constraint requiring workers to be in one of the given regions.

    Emits an EQ constraint for a single region or an IN constraint for multiple
    regions.

    Args:
        regions: Non-empty list of region strings. Must be a list, not a bare string.

    Raises:
        TypeError: If regions is a string (common mistake — pass [region] instead).
        ValueError: If regions is empty or contains empty strings.
    """
    if isinstance(regions, str):
        raise TypeError("region_constraint() requires a list of strings, not a bare string. Use [region] instead.")
    if not regions:
        raise ValueError("regions must be non-empty")
    for r in regions:
        if not r:
            raise ValueError("region must be non-empty")
    if len(regions) == 1:
        return Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value=regions[0])
    return Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=regions)


def device_variant_constraint(variants: Sequence[str]) -> Constraint:
    """Constraint requiring scheduling on workers with one of the given device variants.

    Args:
        variants: Non-empty sequence of device variant strings (e.g., ["v4-8", "v5p-8"]).

    Raises:
        TypeError: If variants is a string (common mistake — pass [variant] instead).
        ValueError: If variants is empty or contains empty strings.
    """
    if isinstance(variants, str):
        raise TypeError(
            "device_variant_constraint() requires a sequence of strings, not a bare string. Use [variant] instead."
        )
    if not variants:
        raise ValueError("variants must be non-empty")
    for v in variants:
        if not v:
            raise ValueError("variant must be non-empty")
    if len(variants) == 1:
        return Constraint.create(key=WellKnownAttribute.DEVICE_VARIANT, op=ConstraintOp.EQ, value=variants[0])
    return Constraint.create(key=WellKnownAttribute.DEVICE_VARIANT, op=ConstraintOp.IN, values=list(variants))


@dataclass(frozen=True)
class PlacementRequirements:
    """Canonical placement constraints derived from proto constraints.

    Combines device type, device variant, preemptible preference, and
    region/zone requirements into a single object for demand routing.
    The autoscaler uses this instead of carrying separate fields.
    """

    device_type: DeviceType | None
    device_variants: frozenset[str] | None
    preemptible: bool | None
    required_regions: frozenset[str] | None
    required_zones: frozenset[str] | None

    _KEY_TO_FIELD: ClassVar[dict[str, str]] = {
        "device-type": "device_type",
        "device-variant": "device_variants",
        "preemptible": "preemptible",
        "region": "required_regions",
        "zone": "required_zones",
    }

    def get(self, key: str) -> Any:
        """Look up a routing constraint value by its well-known key."""
        field_name = self._KEY_TO_FIELD.get(key)
        if field_name is None:
            return None
        return getattr(self, field_name)


# ---------------------------------------------------------------------------
# Shared helpers for extract_placement_requirements
# ---------------------------------------------------------------------------


def _collect_values(
    constraint: Constraint,
    *,
    allow_in: bool = True,
) -> list[str | int | float]:
    """Flatten a Constraint's value(s) into a list of raw scalars.

    EQ → [values[0].value], IN → [v.value for v in values]. Arity is already
    enforced by Constraint.__post_init__ so no None checks are needed here.
    """
    if constraint.op == ConstraintOp.EQ:
        return [constraint.values[0].value]
    if constraint.op == ConstraintOp.IN:
        if not allow_in:
            raise ValueError(f"{constraint.key} constraint must use EQ")
        return [v.value for v in constraint.values]
    raise ValueError(f"{constraint.key} constraint must use EQ or IN, got {constraint.op}")


def _extract_string_set(
    constraints: list[Constraint],
    key: str,
    *,
    transform: Callable[[str], str] = str.strip,
    reject_empty: bool = True,
) -> frozenset[str] | None:
    """Extract a set of string values from constraints sharing the same key."""
    values: set[str] = set()
    has_in = False
    for c in constraints:
        raw_vals = _collect_values(c)
        if c.op == ConstraintOp.IN:
            has_in = True
        for raw in raw_vals:
            val = transform(str(raw))
            if reject_empty and not val:
                raise ValueError(f"{key} constraint must be non-empty")
            values.add(val)
    if not has_in and len(values) > 1:
        raise ValueError(f"conflicting {key} constraints")
    return frozenset(values) if values else None


def _extract_preemptible(constraints: list[Constraint]) -> bool | None:
    values: set[bool] = set()
    for c in constraints:
        for raw in _collect_values(c, allow_in=False):
            s = str(raw).strip().lower()
            if s == "true":
                values.add(True)
            elif s == "false":
                values.add(False)
            else:
                raise ValueError("preemptible constraint must be 'true' or 'false'")
    if len(values) > 1:
        raise ValueError("conflicting preemptible constraints")
    return next(iter(values)) if values else None


def _extract_device_type(constraints: list[Constraint]) -> DeviceType | None:
    values = _extract_string_set(
        constraints,
        WellKnownAttribute.DEVICE_TYPE,
        transform=lambda s: s.strip().lower(),
        reject_empty=False,
    )
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"conflicting device-type constraints: {values}")
    raw = next(iter(values))
    try:
        return DeviceType(raw)
    except ValueError as e:
        raise ValueError(f"unknown device type: {raw}") from e


def extract_placement_requirements(constraints: Sequence[Constraint]) -> PlacementRequirements:
    """Extract canonical placement requirements from constraints.

    Groups constraints by key, then extracts each field using shared helpers.
    """
    by_key: dict[str, list[Constraint]] = {}
    for c in constraints:
        by_key.setdefault(c.key, []).append(c)

    return PlacementRequirements(
        device_type=_extract_device_type(by_key.get(WellKnownAttribute.DEVICE_TYPE, [])),
        device_variants=_extract_string_set(
            by_key.get(WellKnownAttribute.DEVICE_VARIANT, []),
            WellKnownAttribute.DEVICE_VARIANT,
        ),
        preemptible=_extract_preemptible(by_key.get(WellKnownAttribute.PREEMPTIBLE, [])),
        required_regions=_extract_string_set(
            by_key.get(WellKnownAttribute.REGION, []),
            WellKnownAttribute.REGION,
        ),
        required_zones=_extract_string_set(
            by_key.get(WellKnownAttribute.ZONE, []),
            WellKnownAttribute.ZONE,
        ),
    )


def merge_constraints(parent: Sequence[Constraint], child: Sequence[Constraint]) -> list[Constraint]:
    """Merge parent and child constraints with canonical-key override semantics."""

    merged_by_key: dict[str, list[Constraint]] = {}
    for constraint in parent:
        merged_by_key.setdefault(constraint.key, []).append(constraint)

    _CANONICAL_KEYS = frozenset(d.key for d in CONSTRAINT_REGISTRY.values() if d.canonical)
    for key in _CANONICAL_KEYS:
        child_for_key = [constraint for constraint in child if constraint.key == key]
        if child_for_key:
            merged_by_key[key] = child_for_key

    for constraint in child:
        if constraint.key in _CANONICAL_KEYS:
            continue
        existing = merged_by_key.setdefault(constraint.key, [])
        if constraint not in existing:
            existing.append(constraint)

    result: list[Constraint] = []
    for constraints_for_key in merged_by_key.values():
        result.extend(constraints_for_key)
    return result


# ---------------------------------------------------------------------------
# ConstraintDescriptor registry
# ---------------------------------------------------------------------------


class ConstraintKind(StrEnum):
    """Whether a constraint is a label match or a capacity-deducted resource."""

    TAG = "tag"
    CONSUMABLE = "consumable"


@dataclass(frozen=True)
class ConstraintDescriptor:
    """Single source of truth for a well-known constraint.

    Each well-known attribute gets one descriptor that declares its type,
    allowed operators, and (for routing constraints) how to match a scaling
    group's value against a requested value.
    """

    key: str
    kind: ConstraintKind
    python_type: type
    allowed_ops: frozenset[int]
    canonical: bool
    routing: bool


_EQ_IN = frozenset({job_pb2.CONSTRAINT_OP_EQ, job_pb2.CONSTRAINT_OP_IN})
_EQ_ONLY = frozenset({job_pb2.CONSTRAINT_OP_EQ})
_ALL_OPS = frozenset(
    {
        job_pb2.CONSTRAINT_OP_EQ,
        job_pb2.CONSTRAINT_OP_NE,
        job_pb2.CONSTRAINT_OP_EXISTS,
        job_pb2.CONSTRAINT_OP_NOT_EXISTS,
        job_pb2.CONSTRAINT_OP_GT,
        job_pb2.CONSTRAINT_OP_GE,
        job_pb2.CONSTRAINT_OP_LT,
        job_pb2.CONSTRAINT_OP_LE,
        job_pb2.CONSTRAINT_OP_IN,
    }
)


CONSTRAINT_REGISTRY: dict[str, ConstraintDescriptor] = {}


def _register(desc: ConstraintDescriptor) -> ConstraintDescriptor:
    CONSTRAINT_REGISTRY[desc.key] = desc
    return desc


_register(
    ConstraintDescriptor(
        key="device-type", kind=ConstraintKind.TAG, python_type=str, allowed_ops=_EQ_IN, canonical=True, routing=True
    )
)
_register(
    ConstraintDescriptor(
        key="device-variant", kind=ConstraintKind.TAG, python_type=str, allowed_ops=_EQ_IN, canonical=True, routing=True
    )
)
_register(
    ConstraintDescriptor(
        key="preemptible", kind=ConstraintKind.TAG, python_type=bool, allowed_ops=_EQ_ONLY, canonical=True, routing=True
    )
)
_register(
    ConstraintDescriptor(
        key="region", kind=ConstraintKind.TAG, python_type=str, allowed_ops=_EQ_IN, canonical=True, routing=True
    )
)
_register(
    ConstraintDescriptor(
        key="zone", kind=ConstraintKind.TAG, python_type=str, allowed_ops=_EQ_IN, canonical=True, routing=True
    )
)
_register(
    ConstraintDescriptor(
        key="tpu-name", kind=ConstraintKind.TAG, python_type=str, allowed_ops=_ALL_OPS, canonical=False, routing=False
    )
)
_register(
    ConstraintDescriptor(
        key="tpu-worker-id",
        kind=ConstraintKind.TAG,
        python_type=int,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
    )
)
_register(
    ConstraintDescriptor(
        key="tpu-topology",
        kind=ConstraintKind.TAG,
        python_type=str,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
    )
)
_register(
    ConstraintDescriptor(
        key="tpu-vm-count",
        kind=ConstraintKind.TAG,
        python_type=int,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
    )
)
_register(
    ConstraintDescriptor(
        key="gpu-variant", kind=ConstraintKind.TAG, python_type=str, allowed_ops=_ALL_OPS, canonical=False, routing=False
    )
)
_register(
    ConstraintDescriptor(
        key="gpu-count",
        kind=ConstraintKind.CONSUMABLE,
        python_type=int,
        allowed_ops=_ALL_OPS,
        canonical=False,
        routing=False,
    )
)


# Constraint keys that propagate from parent to child jobs via IRIS_JOB_CONSTRAINTS.
# Device constraints are NOT inherited — each child re-derives them from its own
# resource spec via constraints_from_resources(). Preemptible is per-job policy.
INHERITED_CONSTRAINT_KEYS: frozenset[str] = frozenset(
    {
        WellKnownAttribute.REGION,
        WellKnownAttribute.ZONE,
    }
)


# ---------------------------------------------------------------------------
# Resource-derived constraint generation
# ---------------------------------------------------------------------------


def constraints_from_resources(resources: job_pb2.ResourceSpecProto) -> list[Constraint]:
    """Auto-generate device constraints from a job's resource spec.

    Produces Constraint objects for device-type and device-variant when the
    resource spec carries a non-CPU device.  CPU jobs get no auto-generated
    device constraints since CPU resources are fungible across all workers.

    The controller merges these with explicit user constraints using
    merge_constraints(), where explicit constraints for canonical keys replace
    auto-generated ones.
    """
    constraints: list[Constraint] = []

    if not resources.HasField("device"):
        return constraints

    device_type = get_device_type(resources.device)
    if device_type != "cpu":
        constraints.append(Constraint.create(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value=device_type))

    variant = get_device_variant(resources.device)
    if variant and variant != "auto":
        constraints.append(Constraint.create(key=WellKnownAttribute.DEVICE_VARIANT, op=ConstraintOp.EQ, value=variant))

    return constraints


def validate_tpu_request(
    resources: job_pb2.ResourceSpecProto,
    constraints: Sequence[Constraint],
) -> str | None:
    """Check that a TPU job's chip count matches the VM shape of every candidate variant.

    A TPU VM is the atomic scheduling unit: the scheduler reserves chips from a
    worker's advertised capacity, but a single-VM slice (e.g. ``v6e-8``) cannot
    be shared between two jobs even if their combined chip count fits.

    An explicit ``device-variant`` constraint is authoritative for scheduling
    (it replaces the auto-generated constraint from the primary variant), so
    we validate the requested chip count against every effective candidate —
    not just the primary. This rejects submissions where:

    - any candidate variant's ``chips_per_vm`` differs from
      ``resources.device.tpu.count`` (e.g. primary ``v6e-4`` with
      ``device-variant EQ v6e-8`` would schedule on a single v6e-8 VM while
      reserving only 4 of its 8 chips), or
    - an IN constraint lists candidates with mismatched VM shapes
      (e.g. ``["v6e-4", "v6e-8"]``).

    Returns ``None`` if the request is valid, or a human-readable error
    message suitable for returning as ``INVALID_ARGUMENT``.
    """
    from iris.cluster.types import TpuTopologyInfo, get_tpu_topology

    if not resources.HasField("device") or not resources.device.HasField("tpu"):
        return None

    primary = resources.device.tpu.variant
    if not primary or primary == "auto":
        return None

    chips_requested = resources.device.tpu.count

    # Effective candidates: an explicit device-variant constraint overrides
    # the primary. Fall back to the primary when no such constraint exists.
    variants: list[str] = [primary]
    for c in constraints:
        if c.key != WellKnownAttribute.DEVICE_VARIANT:
            continue
        if c.op == ConstraintOp.IN:
            variants = [str(av.value) for av in c.values if av.value]
            break
        if c.op == ConstraintOp.EQ and c.values:
            variants = [str(c.values[0].value)]
            break

    topos: dict[str, TpuTopologyInfo] = {}
    for v in variants:
        try:
            topos[v] = get_tpu_topology(v)
        except ValueError:
            continue  # unknown variants fall through to the scheduler

    if not topos:
        return None

    mismatched = {
        v: topo.chips_per_vm for v, topo in topos.items() if chips_requested and chips_requested != topo.chips_per_vm
    }
    if mismatched:
        return (
            f"TPU chip count mismatch: requested {chips_requested} chips per replica, but "
            f"candidate variants have chips_per_vm={mismatched}. A TPU VM is indivisible; "
            "the per-replica chip count must equal every candidate variant's chips_per_vm."
        )

    shapes = {v: (topo.vm_count, topo.chips_per_vm) for v, topo in topos.items()}
    if len(set(shapes.values())) > 1:
        return (
            "TPU variant alternatives have incompatible VM shapes: "
            f"{ {v: {'vm_count': s[0], 'chips_per_vm': s[1]} for v, s in shapes.items()} }. "
            "All candidates must share vm_count and chips_per_vm; single-VM variants like "
            "v6e-8 or v5litepod-8 cannot be mixed with smaller variants because their VM is "
            "indivisible and would be shared between co-scheduled jobs."
        )

    return None


# ---------------------------------------------------------------------------
# Executor heuristic: auto-tag small CPU-only jobs as non-preemptible
# ---------------------------------------------------------------------------

# Thresholds for the executor heuristic.  A job that stays within all four
# limits is assumed to be an orchestrator/coordinator and is pinned to
# non-preemptible workers so it is not killed by spot reclamation.
_EXECUTOR_MAX_CPU_MILLICORES = 1000  # 1 core
_EXECUTOR_MAX_MEMORY_BYTES = 4 * 1024**3  # 4 GiB
_EXECUTOR_MAX_REPLICAS = 1


def looks_like_executor(
    resources: job_pb2.ResourceSpecProto,
    replicas: int,
) -> bool:
    """Return True when a job's resource shape matches the executor heuristic.

    Heuristic: no accelerators, single task, CPU ≤ 1 core, RAM ≤ 4 GiB.
    """
    has_accelerator = resources.HasField("device") and not resources.device.HasField("cpu")
    if has_accelerator:
        return False
    if replicas > _EXECUTOR_MAX_REPLICAS:
        return False
    if resources.cpu_millicores > _EXECUTOR_MAX_CPU_MILLICORES:
        return False
    if resources.memory_bytes > _EXECUTOR_MAX_MEMORY_BYTES:
        return False
    return True


def infer_preemptible_constraint(
    resources: job_pb2.ResourceSpecProto,
    replicas: int,
    existing_constraints: Sequence[Constraint],
) -> Constraint | None:
    """Return a non-preemptible constraint when the executor heuristic fires.

    Returns None if the user already set an explicit preemptible constraint or
    the job does not look like an executor.
    """
    for c in existing_constraints:
        if c.key == WellKnownAttribute.PREEMPTIBLE:
            return None

    if looks_like_executor(resources, replicas):
        return preemptible_constraint(False)
    return None


def accelerator_type_to_string(accel_type: int) -> str:
    """Convert AcceleratorType proto enum value to a scheduling string."""
    if accel_type == config_pb2.ACCELERATOR_TYPE_UNSPECIFIED:
        return "cpu"
    if accel_type == config_pb2.ACCELERATOR_TYPE_CPU:
        return "cpu"
    if accel_type == config_pb2.ACCELERATOR_TYPE_GPU:
        return "gpu"
    if accel_type == config_pb2.ACCELERATOR_TYPE_TPU:
        return "tpu"
    raise ValueError(f"Unknown accelerator type: {accel_type}")


def _compare_ordered(
    attr_value: str | int | float,
    target_value: str | int | float,
    op: str,
) -> bool:
    """Compare two attribute values with an ordering operator.

    Only numeric types (int, float) support ordered comparisons.
    Strings are not orderable (comparing "v4-8" > "v5" is not meaningful).

    Raises:
        ValueError: If either value is a string (ordered comparison not supported).
    """
    if isinstance(attr_value, str) or isinstance(target_value, str):
        raise ValueError(
            f"Ordered comparison ({op}) not supported for string attributes: "
            f"{attr_value!r} vs {target_value!r}. Use EQ or NE operators instead."
        )

    attr_num: int | float = attr_value
    target_num: int | float = target_value

    if op == "gt":
        return attr_num > target_num
    elif op == "ge":
        return attr_num >= target_num
    elif op == "lt":
        return attr_num < target_num
    elif op == "le":
        return attr_num <= target_num
    return False


def evaluate_constraint(
    attr: AttributeValue | None,
    constraint: Constraint,
) -> bool:
    """Evaluate a single constraint against an entity's attribute.

    Works for any entity (worker, scaling group, etc.) that has typed attributes.
    Constraint values are already normalized (stripped, lowercased) at ingestion.
    """
    op = constraint.op

    if op == ConstraintOp.EXISTS:
        return attr is not None
    if op == ConstraintOp.NOT_EXISTS:
        return attr is None

    if attr is None:
        return False

    match op:
        case ConstraintOp.EQ:
            return attr.value == constraint.values[0].value
        case ConstraintOp.NE:
            return attr.value != constraint.values[0].value
        case ConstraintOp.GT:
            return _compare_ordered(attr.value, constraint.values[0].value, "gt")
        case ConstraintOp.GE:
            return _compare_ordered(attr.value, constraint.values[0].value, "ge")
        case ConstraintOp.LT:
            return _compare_ordered(attr.value, constraint.values[0].value, "lt")
        case ConstraintOp.LE:
            return _compare_ordered(attr.value, constraint.values[0].value, "le")
        case ConstraintOp.IN:
            return any(attr.value == v.value for v in constraint.values)
        case _:
            return False


def is_cpu_device_type_constraint(c: Constraint) -> bool:
    """True if this constraint is device-type=cpu.

    CPU jobs match any scaling group, so this constraint is stripped
    before routing evaluation. Values are already normalized at ingestion.
    """
    return c.key == WellKnownAttribute.DEVICE_TYPE and c.op == ConstraintOp.EQ and c.values[0].value == "cpu"


def routing_constraints(constraints: Sequence[Constraint]) -> list[Constraint]:
    """Filter to routing-only constraints, stripping CPU device-type.

    Non-routing constraints (tpu-name, tpu-worker-id, etc.) and unknown
    constraints match individual workers, not scaling groups, so they are
    excluded from group routing.
    """
    result = []
    for c in constraints:
        if is_cpu_device_type_constraint(c):
            continue
        desc = CONSTRAINT_REGISTRY.get(c.key)
        if desc is None or not desc.routing:
            continue
        result.append(c)
    return result


def split_hard_soft(
    constraints: Sequence[Constraint],
) -> tuple[list[Constraint], list[Constraint]]:
    """Split constraints into (hard, soft) lists based on mode.

    Hard constraints filter candidates; soft constraints only influence ranking.
    """
    hard: list[Constraint] = []
    soft: list[Constraint] = []
    for c in constraints:
        if c.is_soft:
            soft.append(c)
        else:
            hard.append(c)
    return hard, soft


def soft_constraint_score(
    entity_attrs: dict[str, AttributeValue],
    soft_constraints: Sequence[Constraint],
) -> int:
    """Count how many soft constraints an entity satisfies.

    Higher score means better match. Used to rank candidates when soft
    constraints are present.
    """
    score = 0
    for c in soft_constraints:
        attr = entity_attrs.get(c.key)
        if evaluate_constraint(attr, c):
            score += 1
    return score


# ---------------------------------------------------------------------------
# ConstraintIndex: posting-list index for fast constraint matching
# ---------------------------------------------------------------------------


@dataclass
class ConstraintIndex:
    """Posting-list index for fast constraint matching over a set of entities.

    Entities are identified by plain strings. Each entity has a dict of
    AttributeValue attributes. The index supports fast EQ/IN/EXISTS/NOT_EXISTS
    lookups via posting lists, with a slow-path fallback for ordered operators.
    """

    _all_ids: frozenset[str]
    _discrete_lists: dict[str, dict[str | int | float, set[str]]]
    _entity_attributes: dict[str, dict[str, AttributeValue]]

    @classmethod
    def build(cls, entities: dict[str, dict[str, AttributeValue]]) -> ConstraintIndex:
        """Build index from entity_id -> attributes mapping."""
        discrete_lists: dict[str, dict[str | int | float, set[str]]] = {}
        for entity_id, attrs in entities.items():
            for key, attr_value in attrs.items():
                if key not in discrete_lists:
                    discrete_lists[key] = {}
                value = attr_value.value
                if value not in discrete_lists[key]:
                    discrete_lists[key][value] = set()
                discrete_lists[key][value].add(entity_id)
        return cls(
            _all_ids=frozenset(entities.keys()),
            _discrete_lists=discrete_lists,
            _entity_attributes=dict(entities),
        )

    def matching_entities(self, constraints: Sequence[Constraint]) -> set[str]:
        """Get entity IDs matching ALL constraints."""
        if not constraints:
            return set(self._all_ids)
        result: set[str] | None = None
        for constraint in constraints:
            matches = self._evaluate_constraint_set(constraint)
            if result is None:
                result = matches
            else:
                result = result & matches
            if not result:
                return set()
        return result or set()

    def _evaluate_constraint_set(self, constraint: Constraint) -> set[str]:
        """Evaluate a single constraint, returning matching entity IDs."""
        key = constraint.key
        op = constraint.op

        if op == ConstraintOp.EQ and key in self._discrete_lists:
            return self._discrete_lists[key].get(constraint.values[0].value, set())

        if op == ConstraintOp.EXISTS:
            if key in self._discrete_lists:
                result: set[str] = set()
                for entities in self._discrete_lists[key].values():
                    result.update(entities)
                return result
            return set()

        if op == ConstraintOp.NOT_EXISTS:
            if key in self._discrete_lists:
                has_attr: set[str] = set()
                for entities in self._discrete_lists[key].values():
                    has_attr.update(entities)
                return set(self._all_ids) - has_attr
            return set(self._all_ids)

        if op == ConstraintOp.IN and key in self._discrete_lists:
            in_result: set[str] = set()
            for val in constraint.values:
                in_result |= self._discrete_lists[key].get(val.value, set())
            return in_result

        # Slow path for NE, GT, GE, LT, LE, or non-indexed attributes
        result_set: set[str] = set()
        for entity_id, attrs in self._entity_attributes.items():
            attr = attrs.get(key)
            if evaluate_constraint(attr, constraint):
                result_set.add(entity_id)
        return result_set

    def entities_by_group(self, group_by: str, matching_ids: set[str]) -> dict[str, list[str]]:
        """Group entities by the specified attribute value."""
        groups: dict[str, list[str]] = defaultdict(list)
        if group_by not in self._discrete_lists:
            return groups
        for value, entities in self._discrete_lists[group_by].items():
            for entity_id in entities:
                if entity_id in matching_ids:
                    groups[str(value)].append(entity_id)
        return groups


@dataclass(frozen=True)
class ResourceCapacity:
    """Resource dimensions for capacity comparison.

    Used by both the autoscaler (ScalingGroup resource fit) and the scheduler
    (WorkerCapacity resource fit) to check whether a request fits available capacity.
    """

    cpu_millicores: int | None = None
    memory_bytes: int | None = None
    disk_bytes: int | None = None
    gpu_count: int | None = None
    tpu_count: int | None = None


def check_resource_fit(
    available: ResourceCapacity,
    required: ResourceCapacity,
) -> str | None:
    """Check if required resources fit within available capacity.

    Returns None if fit, human-readable reason string otherwise.

    A None value on the available side means "not configured / unlimited" and
    never rejects.  A None value on the required side means "not needed".
    """

    def _check(avail: int | None, req: int | None, name: str, fmt: str = "d") -> str | None:
        if req is None or req <= 0:
            return None
        if avail is None:
            return None
        if req > avail:
            return f"{name}: need {req:{fmt}}, available {avail:{fmt}}"
        return None

    for result in [
        _check(available.cpu_millicores, required.cpu_millicores, "cpu"),
        _check(available.memory_bytes, required.memory_bytes, "memory"),
        _check(available.disk_bytes, required.disk_bytes, "disk"),
        _check(available.gpu_count, required.gpu_count, "gpu"),
        _check(available.tpu_count, required.tpu_count, "tpu"),
    ]:
        if result is not None:
            return result
    return None


def resource_capacity_from_spec(spec: job_pb2.ResourceSpecProto) -> ResourceCapacity:
    """Extract ResourceCapacity from a job's ResourceSpecProto."""
    from iris.cluster.types import get_gpu_count, get_tpu_count

    return ResourceCapacity(
        cpu_millicores=spec.cpu_millicores,
        memory_bytes=spec.memory_bytes,
        disk_bytes=spec.disk_bytes,
        gpu_count=get_gpu_count(spec.device) if spec.HasField("device") else 0,
        tpu_count=get_tpu_count(spec.device) if spec.HasField("device") else 0,
    )


def worker_attributes_from_resources(resources: config_pb2.ScaleGroupResources) -> dict[str, str]:
    """Derive well-known worker attributes from scale group resources config.

    This ensures local workers advertise the same device-type, device-variant,
    and preemptible attributes that constraint matching expects.
    """
    attrs: dict[str, str] = {}
    attrs[WellKnownAttribute.DEVICE_TYPE] = accelerator_type_to_string(resources.device_type)
    if resources.device_variant:
        attrs[WellKnownAttribute.DEVICE_VARIANT] = resources.device_variant.lower()
    is_preemptible = resources.capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE
    attrs[WellKnownAttribute.PREEMPTIBLE] = str(is_preemptible).lower()
    return attrs
