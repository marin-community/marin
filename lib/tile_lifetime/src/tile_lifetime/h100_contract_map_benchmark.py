# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable staging contract for generic H100 Contract/Map evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping
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


REVIEWED_NUMERICAL_FLOORS = (
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
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


REVIEWED_NUMERICAL_FLOORS_SHA256 = _canonical_sha256([asdict(floor) for floor in REVIEWED_NUMERICAL_FLOORS])


@dataclass(frozen=True)
class CounterbalancedScheduleRow:
    """One predeclared position in the steady-state sample schedule."""

    sample_index: int
    cycle_index: int
    backend_order: tuple[BackendVariant, ...]


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

    @property
    def steady_state_schedule(self) -> tuple[CounterbalancedScheduleRow, ...]:
        """Return all 24 predeclared rows, four cycles over six orders."""
        orders = tuple(itertools.permutations(BackendVariant))
        cycles = self.steady_state_repeats // len(orders)
        return tuple(
            CounterbalancedScheduleRow(
                sample_index=cycle_index * len(orders) + order_index,
                cycle_index=cycle_index,
                backend_order=order,
            )
            for cycle_index in range(cycles)
            for order_index, order in enumerate(orders)
        )


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
    final_optimized_hlo_and_custom_call_manifest: bool = True
    kernel_block_and_occupancy_limits: bool = True
    ordered_kernel_names: bool = True
    copy_counts_and_bytes: bool = True
    command_environment_and_compiler_flags: bool = True
    source_revision_and_cache_identity: bool = True

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
    transposes: bool = True
    bitcasts: bool = True

    def __post_init__(self) -> None:
        missing = tuple(name for name, value in asdict(self).items() if not value)
        if missing:
            raise ValueError(f"logical-boundary evidence cannot omit required fields: {missing}")


@dataclass(frozen=True)
class EvidenceSectionSchema:
    """Required fields for one machine-readable result section."""

    name: str
    required_fields: tuple[str, ...]


RESULT_EVIDENCE_SECTIONS = (
    EvidenceSectionSchema(
        "identity",
        ("case_id", "backend", "measurement_boundary"),
    ),
    EvidenceSectionSchema(
        "artifacts",
        (
            "final_optimized_hlo_path",
            "final_optimized_hlo_sha256",
            "custom_call_manifest_path",
            "custom_call_manifest_sha256",
        ),
    ),
    EvidenceSectionSchema(
        "resources",
        ("kernel_records", "launch_count", "ordered_kernel_names"),
    ),
    EvidenceSectionSchema(
        "copies",
        (
            "device_to_device_count",
            "device_to_device_bytes",
            "host_to_device_count",
            "host_to_device_bytes",
            "unexpected_copy_count",
        ),
    ),
    EvidenceSectionSchema(
        "logical_boundary",
        (
            "input_layouts",
            "output_layouts",
            "layout_adapters",
            "materialized_copies",
            "transposes",
            "bitcasts",
            "saved_state_names_and_bytes",
            "recompute_operations",
        ),
    ),
    EvidenceSectionSchema(
        "provenance",
        ("command", "environment", "compiler_flags", "source_sha", "persistent_cache_identity"),
    ),
    EvidenceSectionSchema(
        "numerical",
        ("reviewed_floors_sha256", "floors_passed_before_timing", "outputs"),
    ),
    EvidenceSectionSchema(
        "timing",
        (
            "compile_samples_ns",
            "first_execution_samples_ns",
            "warmup_iterations",
            "warmup_samples_ns",
            "persistent_cache_cold_samples_ns",
            "persistent_cache_hit_samples_ns",
            "steady_state_schedule",
            "raw_samples",
        ),
    ),
)

KERNEL_RECORD_REQUIRED_FIELDS = (
    "name",
    "ptx_path",
    "ptx_sha256",
    "sass_path",
    "sass_sha256",
    "registers_per_thread",
    "spill_load_bytes",
    "spill_store_bytes",
    "static_shared_memory_bytes",
    "dynamic_shared_memory_bytes",
    "block_size",
    "active_blocks_per_sm",
    "limiting_occupancy_resource",
    "achieved_occupancy",
)
NUMERICAL_OUTPUT_ROLES = ("forward", "dx", "dw0", "dw1")
NUMERICAL_OUTPUT_REQUIRED_FIELDS = (
    "maximum_absolute_error",
    "mean_absolute_error",
    "maximum_ulp_distance",
    "mean_ulp_distance",
    "nonfinite_values",
    "repeat_hashes",
    "pairwise_drift",
)
PAIRWISE_DRIFT_REQUIRED_FIELDS = (
    "left_repeat_index",
    "right_repeat_index",
    "maximum_absolute_error",
    "mean_absolute_error",
    "maximum_ulp_distance",
    "mean_ulp_distance",
)
RAW_SAMPLE_REQUIRED_FIELDS = ("sample_index", "backend_order", "measurements_ns")


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_fields(mapping: Mapping[str, Any], required_fields: tuple[str, ...], context: str) -> None:
    missing = tuple(field for field in required_fields if field not in mapping)
    if missing:
        raise ValueError(f"{context} is missing required evidence fields: {missing}")


def _serialized_schedule(timing: TimingProtocol) -> list[dict[str, Any]]:
    return [asdict(row) for row in timing.steady_state_schedule]


def result_evidence_schema() -> dict[str, Any]:
    """Return the immutable result schema before any benchmark execution."""
    plan = default_h100_contract_map_benchmark_plan()
    return {
        "schema": "shuttle.h100_contract_map_result_evidence.v1",
        "required_sections": [section.name for section in RESULT_EVIDENCE_SECTIONS],
        "sections": [asdict(section) for section in RESULT_EVIDENCE_SECTIONS],
        "nested_records": {
            "kernel_record": KERNEL_RECORD_REQUIRED_FIELDS,
            "numerical_output_roles": NUMERICAL_OUTPUT_ROLES,
            "numerical_output": NUMERICAL_OUTPUT_REQUIRED_FIELDS,
            "pairwise_drift": PAIRWISE_DRIFT_REQUIRED_FIELDS,
            "raw_sample": RAW_SAMPLE_REQUIRED_FIELDS,
            "raw_sample_measurement_backends": tuple(BackendVariant),
            "raw_sample_measurement_boundaries": tuple(MeasurementBoundary),
        },
        "reviewed_numerical_floors": [asdict(floor) for floor in REVIEWED_NUMERICAL_FLOORS],
        "reviewed_numerical_floors_sha256": REVIEWED_NUMERICAL_FLOORS_SHA256,
        "steady_state_schedule": _serialized_schedule(plan.timing),
        "samples_per_backend_permutation": 4,
        "required_backend_boundary_records": [
            {"backend": backend.value, "measurement_boundary": boundary.value}
            for backend in BackendVariant
            for boundary in MeasurementBoundary
        ],
    }


def validate_result_evidence(payload: Mapping[str, Any]) -> None:
    """Reject incomplete or post-hoc result evidence before it can be reported."""
    for section_schema in RESULT_EVIDENCE_SECTIONS:
        if section_schema.name not in payload:
            raise ValueError(f"result evidence is missing required section: {section_schema.name}")
        section = _mapping(payload[section_schema.name], section_schema.name)
        _require_fields(section, section_schema.required_fields, section_schema.name)

    identity = _mapping(payload["identity"], "identity")
    valid_case_ids = {case.case_id for case in default_h100_contract_map_benchmark_plan().cases}
    if identity["case_id"] not in valid_case_ids:
        raise ValueError("identity.case_id must name a reviewed structural case")
    if identity["backend"] not in tuple(backend.value for backend in BackendVariant):
        raise ValueError("identity.backend must name a required backend")
    if identity["measurement_boundary"] not in tuple(boundary.value for boundary in MeasurementBoundary):
        raise ValueError("identity.measurement_boundary must name a required boundary")

    resources = _mapping(payload["resources"], "resources")
    kernel_records = resources["kernel_records"]
    if not isinstance(kernel_records, list) or not kernel_records:
        raise ValueError("resources.kernel_records must contain at least one kernel record")
    for index, record_value in enumerate(kernel_records):
        record = _mapping(record_value, f"resources.kernel_records[{index}]")
        _require_fields(record, KERNEL_RECORD_REQUIRED_FIELDS, f"resources.kernel_records[{index}]")
    kernel_names = tuple(record["name"] for record in kernel_records)
    if resources["launch_count"] != len(kernel_records) or tuple(resources["ordered_kernel_names"]) != kernel_names:
        raise ValueError("launch_count and ordered_kernel_names must match kernel_records")

    provenance = _mapping(payload["provenance"], "provenance")
    source_sha = provenance["source_sha"]
    if not isinstance(source_sha, str) or len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("provenance.source_sha must be a full lowercase Git SHA")

    numerical = _mapping(payload["numerical"], "numerical")
    if numerical["reviewed_floors_sha256"] != REVIEWED_NUMERICAL_FLOORS_SHA256:
        raise ValueError("numerical evidence does not use the reviewed floor digest")
    if numerical["floors_passed_before_timing"] is not True:
        raise ValueError("numerical floors must pass before timing")
    outputs = _mapping(numerical["outputs"], "numerical.outputs")
    if tuple(outputs) != NUMERICAL_OUTPUT_ROLES:
        raise ValueError("numerical.outputs must contain forward, dx, dw0, and dw1 in fixed order")
    for role in NUMERICAL_OUTPUT_ROLES:
        output = _mapping(outputs[role], f"numerical.outputs.{role}")
        _require_fields(output, NUMERICAL_OUTPUT_REQUIRED_FIELDS, f"numerical.outputs.{role}")
        if not output["repeat_hashes"]:
            raise ValueError(f"numerical.outputs.{role}.repeat_hashes cannot be empty")
        drift_records = output["pairwise_drift"]
        if not isinstance(drift_records, list) or not drift_records:
            raise ValueError(f"numerical.outputs.{role}.pairwise_drift cannot be empty")
        for index, drift_value in enumerate(drift_records):
            drift = _mapping(drift_value, f"numerical.outputs.{role}.pairwise_drift[{index}]")
            _require_fields(
                drift,
                PAIRWISE_DRIFT_REQUIRED_FIELDS,
                f"numerical.outputs.{role}.pairwise_drift[{index}]",
            )

    timing = _mapping(payload["timing"], "timing")
    expected_schedule = _serialized_schedule(default_h100_contract_map_benchmark_plan().timing)
    if timing["steady_state_schedule"] != expected_schedule:
        raise ValueError("timing.steady_state_schedule must equal the reviewed 24-row schedule")
    raw_samples = timing["raw_samples"]
    if not isinstance(raw_samples, list) or len(raw_samples) != len(expected_schedule):
        raise ValueError("timing.raw_samples must contain all 24 scheduled rows")
    expected_backends = tuple(backend.value for backend in BackendVariant)
    expected_boundaries = tuple(boundary.value for boundary in MeasurementBoundary)
    for index, (sample_value, schedule_row) in enumerate(zip(raw_samples, expected_schedule, strict=True)):
        sample = _mapping(sample_value, f"timing.raw_samples[{index}]")
        _require_fields(sample, RAW_SAMPLE_REQUIRED_FIELDS, f"timing.raw_samples[{index}]")
        if (
            sample["sample_index"] != schedule_row["sample_index"]
            or sample["backend_order"] != schedule_row["backend_order"]
        ):
            raise ValueError(f"timing.raw_samples[{index}] does not match its scheduled order")
        measurements = _mapping(sample["measurements_ns"], f"timing.raw_samples[{index}].measurements_ns")
        if tuple(measurements) != expected_backends:
            raise ValueError(f"timing.raw_samples[{index}] must contain all backends in fixed order")
        for backend in expected_backends:
            boundaries = _mapping(measurements[backend], f"timing.raw_samples[{index}].measurements_ns.{backend}")
            if tuple(boundaries) != expected_boundaries:
                raise ValueError(f"timing.raw_samples[{index}] backend measurements must contain both boundaries")


def validate_result_evidence_bundle(payloads: tuple[Mapping[str, Any], ...]) -> None:
    """Require one complete result for every backend and measurement boundary."""
    for payload in payloads:
        validate_result_evidence(payload)
    identities = tuple(
        (
            _mapping(payload["identity"], "identity")["backend"],
            _mapping(payload["identity"], "identity")["measurement_boundary"],
        )
        for payload in payloads
    )
    expected = tuple((backend.value, boundary.value) for backend in BackendVariant for boundary in MeasurementBoundary)
    if identities != expected:
        raise ValueError("result bundle must contain every backend and boundary exactly once in fixed order")


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
        if self.numerical_floors != REVIEWED_NUMERICAL_FLOORS:
            raise ValueError("numerical floors must equal the exact reviewed schema constant")


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
        numerical_floors=REVIEWED_NUMERICAL_FLOORS,
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
        "steady_state_schedule": _serialized_schedule(plan.timing),
        "reviewed_numerical_floors_sha256": REVIEWED_NUMERICAL_FLOORS_SHA256,
        "result_evidence_schema": result_evidence_schema(),
        "external_comparators": [asdict(decision) for decision in decisions],
        "backend_wiring": [asdict(status) | {"execution_ready": status.execution_ready} for status in wiring],
        "execution_allowed": False,
    }
