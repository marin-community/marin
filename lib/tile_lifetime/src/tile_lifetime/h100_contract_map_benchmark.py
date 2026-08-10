# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable staging contract for generic H100 Contract/Map evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ArchitectureStatus(StrEnum):
    """Whether the harness conforms to the ordinary-JAX Shuttle architecture."""

    NONCONFORMING = "architecture_nonconforming"


class BackendVariant(StrEnum):
    """Backends required in every primary timing comparison."""

    ORDINARY_XLA = "ordinary_xla"
    SHUTTLE_SOURCE_ORDERED = "shuttle_source_ordered"
    SHUTTLE_FAST = "shuttle_fast"


class MeasurementBoundary(StrEnum):
    """Physical scopes measured separately for each backend."""

    KERNEL_ONLY = "kernel_only"
    LOGICAL_TRAINING_STEP = "logical_training_step"


class ScalarMapFamily(StrEnum):
    """Generic scalar programs used to vary Map structure."""

    SIGMOID_PRODUCT = "sigmoid_product"
    TANH_PRODUCT = "tanh_product"
    CUBIC_MIX = "cubic_mix"


class StructuralFeature(StrEnum):
    """Algebra features used to admit optional external comparators."""

    CONTRACT = "contract"
    MAP = "map"
    FOLD = "fold"
    NORMALIZED_EXP = "normalized_exp"
    ATTENTION_SCORE = "attention_score"
    SEGMENTED_CONTRACT = "segmented_contract"
    RELATION = "relation"
    TRANSPORT = "transport"


class ExternalComparator(StrEnum):
    """External denominators admitted only by structural overlap."""

    FA4 = "fa4"
    GRUG = "grug"


class NumericalReference(StrEnum):
    """Independent reference used by one numerical acceptance gate."""

    SOURCE_ORDERED_FP32 = "source_ordered_fp32"
    REAL_ALGEBRA_FP64 = "real_algebra_fp64"


class RepeatabilityMode(StrEnum):
    """Required relationship between untimed repeated executions."""

    BITWISE = "bitwise"
    BOUNDED_DRIFT = "bounded_drift"


@dataclass(frozen=True)
class StructuralCase:
    """Anonymous two-Contract/Map shape with no workload identity."""

    rows: int
    reduction: int
    features: int
    scalar_map: ScalarMapFamily

    def __post_init__(self) -> None:
        if min(self.rows, self.reduction, self.features) <= 0:
            raise ValueError("Contract/Map case dimensions must be positive")
        if self.rows % 2 == 0:
            raise ValueError("benchmark rows must be odd to avoid common sequence-tile signatures")
        if self.reduction % 8 or self.features % 8:
            raise ValueError("Contract dimensions must retain tensor-core-compatible multiples of eight")
        if type(self.scalar_map) is not ScalarMapFamily:
            raise TypeError("scalar_map must be a ScalarMapFamily")

    @property
    def case_id(self) -> str:
        """Return a structural digest instead of a caller-supplied workload key."""
        record = {
            "features": self.features,
            "reduction": self.reduction,
            "rows": self.rows,
            "scalar_map": self.scalar_map.value,
        }
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return f"contract_map_{digest[:16]}"


@dataclass(frozen=True)
class NumericalFloor:
    """Predeclared accuracy and repeatability bounds for one backend."""

    backend: BackendVariant
    reference: NumericalReference
    maximum_absolute_error: float
    mean_absolute_error: float
    maximum_ulp_distance: int
    mean_ulp_distance: float
    maximum_nonfinite_values: int
    repeatability: RepeatabilityMode
    repeat_maximum_absolute_error: float
    repeat_mean_absolute_error: float

    def __post_init__(self) -> None:
        absolute_bounds = (
            self.maximum_absolute_error,
            self.mean_absolute_error,
            self.mean_ulp_distance,
            self.repeat_maximum_absolute_error,
            self.repeat_mean_absolute_error,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in absolute_bounds):
            raise ValueError("numerical floors must be finite and nonnegative")
        if self.mean_absolute_error > self.maximum_absolute_error:
            raise ValueError("mean absolute error cannot exceed the maximum absolute error")
        if self.mean_ulp_distance > self.maximum_ulp_distance:
            raise ValueError("mean ULP distance cannot exceed the maximum ULP distance")
        if self.repeat_mean_absolute_error > self.repeat_maximum_absolute_error:
            raise ValueError("mean repeat drift cannot exceed maximum repeat drift")
        if self.maximum_ulp_distance < 0 or self.maximum_nonfinite_values < 0:
            raise ValueError("ULP and nonfinite bounds must be nonnegative")
        if self.repeatability is RepeatabilityMode.BITWISE and (
            self.repeat_maximum_absolute_error != 0.0 or self.repeat_mean_absolute_error != 0.0
        ):
            raise ValueError("bitwise repeatability requires zero drift bounds")


@dataclass(frozen=True)
class TimingProtocol:
    """Counterbalanced timing and process-isolated cache protocol."""

    compile_processes: int
    warmup_iterations: int
    steady_state_repeats: int
    iterations_per_sample: int
    persistent_cache_cold_processes: int
    persistent_cache_hit_processes: int
    isolate_persistent_cache_roots: bool
    retain_raw_samples: bool

    def __post_init__(self) -> None:
        values = (
            self.compile_processes,
            self.warmup_iterations,
            self.steady_state_repeats,
            self.iterations_per_sample,
            self.persistent_cache_cold_processes,
            self.persistent_cache_hit_processes,
        )
        if min(values) <= 0:
            raise ValueError("timing protocol counts must be positive")
        required_order_count = math.factorial(len(BackendVariant))
        if self.steady_state_repeats % required_order_count:
            raise ValueError(f"steady-state repeats must cover all {required_order_count} backend permutations equally")
        if not self.isolate_persistent_cache_roots or not self.retain_raw_samples:
            raise ValueError("cache isolation and raw samples are mandatory")

    @property
    def counterbalanced_orders(self) -> tuple[tuple[str, ...], ...]:
        """Return every backend permutation in deterministic order."""
        return tuple(tuple(backend.value for backend in order) for order in itertools.permutations(BackendVariant))


@dataclass(frozen=True)
class ResourceEvidence:
    """Required physical evidence for every backend and boundary."""

    ptx: bool = True
    sass: bool = True
    registers_per_thread: bool = True
    spill_load_bytes: bool = True
    spill_store_bytes: bool = True
    occupancy: bool = True
    static_and_dynamic_shared_memory: bool = True
    launch_count: bool = True
    unexpected_copy_count: bool = True

    def __post_init__(self) -> None:
        missing = tuple(name for name, value in asdict(self).items() if not value)
        if missing:
            raise ValueError(f"resource evidence cannot omit required fields: {missing}")


@dataclass(frozen=True)
class LogicalBoundaryEvidence:
    """Required accounting for the full logical training-step boundary."""

    input_layouts: bool = True
    output_layouts: bool = True
    layout_adapters: bool = True
    materialized_copies: bool = True
    saved_state_names_and_bytes: bool = True
    recompute_operations: bool = True

    def __post_init__(self) -> None:
        missing = tuple(name for name, value in asdict(self).items() if not value)
        if missing:
            raise ValueError(f"logical-boundary evidence cannot omit required fields: {missing}")


@dataclass(frozen=True)
class ComparatorDecision:
    """Structural admission result for one optional external comparator."""

    comparator: ExternalComparator
    admitted: bool
    required_features: tuple[StructuralFeature, ...]
    missing_features: tuple[StructuralFeature, ...]


@dataclass(frozen=True)
class BackendWiring:
    """Review state for one primary backend implementation."""

    backend: BackendVariant
    generated_backend_wired: bool
    resource_collectors_wired: bool
    reviewed: bool
    evidence_paths: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def execution_ready(self) -> bool:
        """Whether this backend satisfies every pre-launch prerequisite."""
        return self.generated_backend_wired and self.resource_collectors_wired and self.reviewed and not self.blockers


@dataclass(frozen=True)
class H100ContractMapBenchmarkPlan:
    """Closed architecture-nonconforming plan for the staged H100 harness."""

    schema_version: int
    architecture_status: ArchitectureStatus
    features: tuple[StructuralFeature, ...]
    cases: tuple[StructuralCase, ...]
    backends: tuple[BackendVariant, ...]
    boundaries: tuple[MeasurementBoundary, ...]
    numerical_floors: tuple[NumericalFloor, ...]
    timing: TimingProtocol
    resources: ResourceEvidence
    logical_boundary: LogicalBoundaryEvidence

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("the staged H100 benchmark schema version is fixed at one")
        if self.architecture_status is not ArchitectureStatus.NONCONFORMING:
            raise ValueError("this harness must remain architecture-nonconforming until ordinary-JAX integration")
        if self.features != (StructuralFeature.CONTRACT, StructuralFeature.MAP):
            raise ValueError("the initial sweep is limited to dense Contract/Map algebra")
        if len(self.cases) < 4 or len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("the shape sweep requires at least four distinct structural cases")
        if self.backends != tuple(BackendVariant):
            raise ValueError("every plan must compare ordinary XLA, SOURCE_ORDERED, and FAST in fixed order")
        if self.boundaries != tuple(MeasurementBoundary):
            raise ValueError("every plan must measure kernel-only and full logical boundaries")
        floor_backends = tuple(floor.backend for floor in self.numerical_floors)
        if floor_backends != self.backends:
            raise ValueError("numerical floors must cover every backend exactly once in backend order")


def comparator_decision(
    comparator: ExternalComparator,
    features: tuple[StructuralFeature, ...],
) -> ComparatorDecision:
    """Admit an external comparator only when the algebra overlaps."""
    requirements = {
        ExternalComparator.FA4: (
            StructuralFeature.ATTENTION_SCORE,
            StructuralFeature.NORMALIZED_EXP,
            StructuralFeature.FOLD,
        ),
        ExternalComparator.GRUG: (
            StructuralFeature.SEGMENTED_CONTRACT,
            StructuralFeature.RELATION,
            StructuralFeature.TRANSPORT,
        ),
    }
    required = requirements[comparator]
    available = set(features)
    missing = tuple(feature for feature in required if feature not in available)
    return ComparatorDecision(
        comparator=comparator,
        admitted=not missing,
        required_features=required,
        missing_features=missing,
    )


def default_h100_contract_map_benchmark_plan() -> H100ContractMapBenchmarkPlan:
    """Return the immutable reviewed staging plan."""
    cases = (
        StructuralCase(43, 104, 72, ScalarMapFamily.SIGMOID_PRODUCT),
        StructuralCase(131, 168, 104, ScalarMapFamily.TANH_PRODUCT),
        StructuralCase(269, 232, 136, ScalarMapFamily.CUBIC_MIX),
        StructuralCase(521, 328, 184, ScalarMapFamily.SIGMOID_PRODUCT),
    )
    return H100ContractMapBenchmarkPlan(
        schema_version=1,
        architecture_status=ArchitectureStatus.NONCONFORMING,
        features=(StructuralFeature.CONTRACT, StructuralFeature.MAP),
        cases=cases,
        backends=tuple(BackendVariant),
        boundaries=tuple(MeasurementBoundary),
        numerical_floors=(
            NumericalFloor(
                backend=BackendVariant.ORDINARY_XLA,
                reference=NumericalReference.REAL_ALGEBRA_FP64,
                maximum_absolute_error=0.03125,
                mean_absolute_error=0.002,
                maximum_ulp_distance=4,
                mean_ulp_distance=0.25,
                maximum_nonfinite_values=0,
                repeatability=RepeatabilityMode.BOUNDED_DRIFT,
                repeat_maximum_absolute_error=0.0078125,
                repeat_mean_absolute_error=0.0005,
            ),
            NumericalFloor(
                backend=BackendVariant.SHUTTLE_SOURCE_ORDERED,
                reference=NumericalReference.SOURCE_ORDERED_FP32,
                maximum_absolute_error=0.0078125,
                mean_absolute_error=0.0005,
                maximum_ulp_distance=1,
                mean_ulp_distance=0.05,
                maximum_nonfinite_values=0,
                repeatability=RepeatabilityMode.BITWISE,
                repeat_maximum_absolute_error=0.0,
                repeat_mean_absolute_error=0.0,
            ),
            NumericalFloor(
                backend=BackendVariant.SHUTTLE_FAST,
                reference=NumericalReference.REAL_ALGEBRA_FP64,
                maximum_absolute_error=0.03125,
                mean_absolute_error=0.002,
                maximum_ulp_distance=4,
                mean_ulp_distance=0.25,
                maximum_nonfinite_values=0,
                repeatability=RepeatabilityMode.BOUNDED_DRIFT,
                repeat_maximum_absolute_error=0.0078125,
                repeat_mean_absolute_error=0.0005,
            ),
        ),
        timing=TimingProtocol(
            compile_processes=3,
            warmup_iterations=10,
            steady_state_repeats=24,
            iterations_per_sample=100,
            persistent_cache_cold_processes=3,
            persistent_cache_hit_processes=3,
            isolate_persistent_cache_roots=True,
            retain_raw_samples=True,
        ),
        resources=ResourceEvidence(),
        logical_boundary=LogicalBoundaryEvidence(),
    )


def staged_backend_wiring() -> tuple[BackendWiring, ...]:
    """Return the reviewed blockers without probing an accelerator."""
    return (
        BackendWiring(
            backend=BackendVariant.ORDINARY_XLA,
            generated_backend_wired=True,
            resource_collectors_wired=False,
            reviewed=False,
            evidence_paths=("lib/tile_lifetime/benchmarks/h100_generated_contract_map_chain_training.py",),
            blockers=("XLA PTX/SASS, occupancy, launch, and unexpected-copy collectors are not wired",),
        ),
        BackendWiring(
            backend=BackendVariant.SHUTTLE_SOURCE_ORDERED,
            generated_backend_wired=False,
            resource_collectors_wired=False,
            reviewed=False,
            evidence_paths=(
                "lib/tile_lifetime/src/tile_lifetime/cuda_contract_map_chain_codegen.py",
                "lib/tile_lifetime/benchmarks/h100_generated_contract_map_chain_training.py",
            ),
            blockers=(
                "the retained generated FFI is a bounded one-CTA prototype, not the ordinary-JAX Shuttle seam",
                "resource collectors are not wired",
            ),
        ),
        BackendWiring(
            backend=BackendVariant.SHUTTLE_FAST,
            generated_backend_wired=False,
            resource_collectors_wired=False,
            reviewed=False,
            evidence_paths=(),
            blockers=("no generic generated FAST Contract/Map backend exists", "resource collectors are not wired"),
        ),
    )


def staging_manifest(*, shuttle_revision: str) -> dict[str, Any]:
    """Return a machine-readable plan that cannot be mistaken for run evidence."""
    if len(shuttle_revision) != 40 or any(character not in "0123456789abcdef" for character in shuttle_revision):
        raise ValueError("shuttle_revision must be a full lowercase Git SHA")
    plan = default_h100_contract_map_benchmark_plan()
    wiring = staged_backend_wiring()
    decisions = tuple(comparator_decision(comparator, plan.features) for comparator in ExternalComparator)
    return {
        "schema": "shuttle.h100_contract_map_backend_evidence.v1",
        "kind": "staged_plan_no_gpu_evidence",
        "shuttle_revision": shuttle_revision,
        "plan": asdict(plan),
        "case_ids": [case.case_id for case in plan.cases],
        "counterbalanced_orders": plan.timing.counterbalanced_orders,
        "external_comparators": [asdict(decision) for decision in decisions],
        "backend_wiring": [asdict(status) | {"execution_ready": status.execution_ready} for status in wiring],
        "execution_allowed": False,
    }


def require_gpu_execution_ready() -> None:
    """Refuse GPU execution until all three backends and collectors are reviewed."""
    blocked = tuple(status for status in staged_backend_wiring() if not status.execution_ready)
    if blocked:
        summary = "; ".join(f"{status.backend.value}: {', '.join(status.blockers)}" for status in blocked)
        raise RuntimeError(f"H100 execution is disabled for the staged architecture-nonconforming harness: {summary}")
    raise RuntimeError("H100 execution requires an ordinary-JAX Shuttle architecture review")
