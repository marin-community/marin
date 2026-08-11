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


class CubinAvailability(StrEnum):
    """Whether a public backend artifact path exposed a cubin."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class CubinUnavailableReason(StrEnum):
    """Closed reasons for accepting an absent cubin."""

    PUBLIC_XLA_DUMP_OMITS_CUBIN = "public_xla_dump_omits_cubin"


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


class UlpAcceptanceMode(StrEnum):
    """Whether BF16 ULP diagnostics are an acceptance criterion."""

    HARD = "hard"
    DIAGNOSTIC_ONLY = "diagnostic_only"


class NumericalFloorError(ValueError):
    """A numerical rejection retaining the failed logical output role."""

    def __init__(self, message: str, *, output_name: str):
        super().__init__(message)
        self.output_name = output_name


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
class NumericalOutputFloor:
    """Absolute-error bounds for one logical training-step output."""

    output: str
    maximum_absolute_error: float
    mean_absolute_error: float

    def __post_init__(self) -> None:
        if self.output not in ("forward", "dx", "dw0", "dw1"):
            raise ValueError("numerical output floor must name one reviewed output")
        bounds = (self.maximum_absolute_error, self.mean_absolute_error)
        if any(not math.isfinite(value) or value < 0.0 for value in bounds):
            raise ValueError("absolute-error floors must be finite and nonnegative")
        if self.mean_absolute_error > self.maximum_absolute_error:
            raise ValueError("mean absolute error cannot exceed maximum absolute error")


@dataclass(frozen=True)
class NumericalFloor:
    """Predeclared accuracy and repeatability policy for one backend."""

    backend: BackendVariant
    reference: NumericalReference
    output_floors: tuple[NumericalOutputFloor, ...]
    ulp_acceptance: UlpAcceptanceMode
    maximum_ulp_distance: int | None
    mean_ulp_distance: float | None
    maximum_nonfinite_values: int
    repeatability: RepeatabilityMode
    repeat_maximum_absolute_error: float
    repeat_mean_absolute_error: float

    def __post_init__(self) -> None:
        if tuple(floor.output for floor in self.output_floors) != ("forward", "dx", "dw0", "dw1"):
            raise ValueError("output floors must cover forward, dx, dw0, and dw1 in fixed order")
        absolute_bounds = (self.repeat_maximum_absolute_error, self.repeat_mean_absolute_error)
        if any(not math.isfinite(value) or value < 0.0 for value in absolute_bounds):
            raise ValueError("numerical floors must be finite and nonnegative")
        if self.repeat_mean_absolute_error > self.repeat_maximum_absolute_error:
            raise ValueError("mean repeat drift cannot exceed maximum repeat drift")
        if self.maximum_nonfinite_values < 0:
            raise ValueError("nonfinite bounds must be nonnegative")
        if self.ulp_acceptance is UlpAcceptanceMode.HARD:
            if self.maximum_ulp_distance is None or self.mean_ulp_distance is None:
                raise ValueError("hard ULP acceptance requires maximum and mean limits")
            if self.maximum_ulp_distance < 0:
                raise ValueError("maximum ULP distance must be nonnegative")
            if not math.isfinite(self.mean_ulp_distance) or self.mean_ulp_distance < 0.0:
                raise ValueError("mean ULP distance must be finite and nonnegative")
            if self.mean_ulp_distance > self.maximum_ulp_distance:
                raise ValueError("mean ULP distance cannot exceed maximum ULP distance")
        elif self.maximum_ulp_distance is not None or self.mean_ulp_distance is not None:
            raise ValueError("diagnostic-only ULP policy cannot carry acceptance limits")
        if self.repeatability is RepeatabilityMode.BITWISE and (
            self.repeat_maximum_absolute_error != 0.0 or self.repeat_mean_absolute_error != 0.0
        ):
            raise ValueError("bitwise repeatability requires zero drift bounds")

    def output_floor(self, output: str) -> NumericalOutputFloor:
        """Return the immutable absolute-error floor for one output."""
        for floor in self.output_floors:
            if floor.output == output:
                return floor
        raise ValueError("output must name one reviewed numerical role")


def _uniform_output_floors(maximum: float, mean: float) -> tuple[NumericalOutputFloor, ...]:
    return tuple(
        NumericalOutputFloor(output=output, maximum_absolute_error=maximum, mean_absolute_error=mean)
        for output in ("forward", "dx", "dw0", "dw1")
    )


REVIEWED_NUMERICAL_FLOORS = (
    NumericalFloor(
        backend=BackendVariant.ORDINARY_XLA,
        reference=NumericalReference.REAL_ALGEBRA_FP64,
        output_floors=(
            NumericalOutputFloor("forward", 0.03125, 0.00390625),
            NumericalOutputFloor("dx", 0.03125, 0.00390625),
            NumericalOutputFloor("dw0", 0.03125, 0.00390625),
            NumericalOutputFloor("dw1", 0.0625, 0.00390625),
        ),
        ulp_acceptance=UlpAcceptanceMode.DIAGNOSTIC_ONLY,
        maximum_ulp_distance=None,
        mean_ulp_distance=None,
        maximum_nonfinite_values=0,
        repeatability=RepeatabilityMode.BOUNDED_DRIFT,
        repeat_maximum_absolute_error=0.0078125,
        repeat_mean_absolute_error=0.0005,
    ),
    NumericalFloor(
        backend=BackendVariant.SHUTTLE_SOURCE_ORDERED,
        reference=NumericalReference.SOURCE_ORDERED_FP32,
        output_floors=_uniform_output_floors(0.0078125, 0.0005),
        ulp_acceptance=UlpAcceptanceMode.HARD,
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
        output_floors=(
            NumericalOutputFloor("forward", 0.03125, 0.00390625),
            NumericalOutputFloor("dx", 0.03125, 0.00390625),
            NumericalOutputFloor("dw0", 0.03125, 0.00390625),
            NumericalOutputFloor("dw1", 0.0625, 0.00390625),
        ),
        ulp_acceptance=UlpAcceptanceMode.DIAGNOSTIC_ONLY,
        maximum_ulp_distance=None,
        mean_ulp_distance=None,
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
    cubin_availability: bool = True
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
            "device_to_host_count",
            "device_to_host_bytes",
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
    "cubin",
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
_MAX_NUMERICAL_DIAGNOSTIC_CHARS = 1024

LOGICAL_BOUNDARY_RECORD_SCHEMAS = (
    (
        "layout_adapter",
        (
            ("value", "canonical_string"),
            ("input_layout", "canonical_string"),
            ("output_layout", "canonical_string"),
            ("materialized", "bool"),
        ),
    ),
    (
        "materialized_copy",
        (
            ("source", "canonical_string"),
            ("destination", "canonical_string"),
            ("bytes", "nonnegative_int"),
        ),
    ),
    (
        "transpose",
        (
            ("input", "canonical_string"),
            ("output", "canonical_string"),
            ("permutation", "permutation_int_list"),
            ("materialized", "bool"),
        ),
    ),
    (
        "bitcast",
        (
            ("input", "canonical_string"),
            ("output", "canonical_string"),
            ("input_shape", "positive_int_list"),
            ("output_shape", "positive_int_list"),
        ),
    ),
    (
        "recompute_operation",
        (
            ("output", "canonical_string"),
            ("operation", "canonical_string"),
            ("launch_count", "nonnegative_int"),
        ),
    ),
)

_SHA256_LENGTH = 64
_GIT_SHA_LENGTH = 40


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_fields(mapping: Mapping[str, Any], required_fields: tuple[str, ...], context: str) -> None:
    missing = tuple(field for field in required_fields if field not in mapping)
    if missing:
        raise ValueError(f"{context} is missing required evidence fields: {missing}")


def _require_nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{context} must be a nonempty canonical string")
    return value


def _require_artifact_path(value: object, context: str) -> None:
    path = _require_nonempty_string(value, context)
    basename = path.rstrip("/").rsplit("/", 1)[-1]
    stem, separator, suffix = basename.rpartition(".")
    if not stem or not separator or not suffix or any(character in path for character in "\x00\r\n\t"):
        raise ValueError(f"{context} must name a concrete artifact file")


def _require_lowercase_hex_digest(value: object, length: int, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be exactly {length} lowercase hexadecimal characters")


def _validate_cubin_evidence(value: object, backend: object, context: str) -> None:
    cubin = _mapping(value, context)
    availability = cubin.get("availability")
    if availability == CubinAvailability.AVAILABLE.value:
        required = ("availability", "path", "sha256")
        if set(cubin) != set(required):
            raise ValueError(f"{context} available record must contain exactly {required}")
        _require_artifact_path(cubin["path"], f"{context}.path")
        _require_lowercase_hex_digest(cubin["sha256"], _SHA256_LENGTH, f"{context}.sha256")
        return
    if availability != CubinAvailability.UNAVAILABLE.value:
        raise ValueError(f"{context}.availability must name a closed cubin availability")
    required = ("availability", "unavailable_reason")
    if set(cubin) != set(required):
        raise ValueError(f"{context} unavailable record must contain exactly {required}")
    if backend != BackendVariant.ORDINARY_XLA.value:
        raise ValueError("generated backends require an available cubin")
    if cubin["unavailable_reason"] != CubinUnavailableReason.PUBLIC_XLA_DUMP_OMITS_CUBIN.value:
        raise ValueError(f"{context}.unavailable_reason must name the closed public-XLA reason")


def _require_nonempty_string_list(value: object, context: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a nonempty list")
    for index, item in enumerate(value):
        _require_nonempty_string(item, f"{context}[{index}]")


def _require_finite_nonnegative_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{context} must be a finite nonnegative number")
    return float(value)


def _require_nonnegative_integer(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _require_positive_integer_list(value: object, context: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a nonempty list of positive integers")
    if any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{context} must be a nonempty list of positive integers")
    return value


def _require_permutation(value: object, context: str) -> None:
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError(f"{context} must be a nonempty integer permutation")
    if sorted(value) != list(range(len(value))):
        raise ValueError(f"{context} must be a nonempty integer permutation")


def _require_exact_record(value: object, schema_name: str, context: str) -> Mapping[str, Any]:
    record = _mapping(value, context)
    schema = dict(next(fields for name, fields in LOGICAL_BOUNDARY_RECORD_SCHEMAS if name == schema_name))
    if set(record) != set(schema):
        raise ValueError(f"{context} must contain exactly the closed {schema_name} schema fields: {tuple(schema)}")
    return record


def _require_record_list(value: object, schema_name: str, context: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of closed {schema_name} records")
    return [_require_exact_record(record, schema_name, f"{context}[{index}]") for index, record in enumerate(value)]


def _validate_logical_boundary(logical_boundary: Mapping[str, Any]) -> None:
    required_fields = next(
        section.required_fields for section in RESULT_EVIDENCE_SECTIONS if section.name == "logical_boundary"
    )
    if set(logical_boundary) != set(required_fields):
        raise ValueError("logical_boundary must contain exactly its reviewed schema fields")
    _require_nonempty_string_list(logical_boundary["input_layouts"], "logical_boundary.input_layouts")
    _require_nonempty_string_list(logical_boundary["output_layouts"], "logical_boundary.output_layouts")

    adapters = _require_record_list(
        logical_boundary["layout_adapters"], "layout_adapter", "logical_boundary.layout_adapters"
    )
    for index, adapter in enumerate(adapters):
        context = f"logical_boundary.layout_adapters[{index}]"
        for field in ("value", "input_layout", "output_layout"):
            _require_nonempty_string(adapter[field], f"{context}.{field}")
        if type(adapter["materialized"]) is not bool:
            raise ValueError(f"{context}.materialized must be a bool")

    copies = _require_record_list(
        logical_boundary["materialized_copies"], "materialized_copy", "logical_boundary.materialized_copies"
    )
    for index, copy_record in enumerate(copies):
        context = f"logical_boundary.materialized_copies[{index}]"
        for field in ("source", "destination"):
            _require_nonempty_string(copy_record[field], f"{context}.{field}")
        _require_nonnegative_integer(copy_record["bytes"], f"{context}.bytes")

    transposes = _require_record_list(logical_boundary["transposes"], "transpose", "logical_boundary.transposes")
    for index, transpose in enumerate(transposes):
        context = f"logical_boundary.transposes[{index}]"
        for field in ("input", "output"):
            _require_nonempty_string(transpose[field], f"{context}.{field}")
        _require_permutation(transpose["permutation"], f"{context}.permutation")
        if type(transpose["materialized"]) is not bool:
            raise ValueError(f"{context}.materialized must be a bool")

    bitcasts = _require_record_list(logical_boundary["bitcasts"], "bitcast", "logical_boundary.bitcasts")
    for index, bitcast in enumerate(bitcasts):
        context = f"logical_boundary.bitcasts[{index}]"
        for field in ("input", "output"):
            _require_nonempty_string(bitcast[field], f"{context}.{field}")
        input_shape = _require_positive_integer_list(bitcast["input_shape"], f"{context}.input_shape")
        output_shape = _require_positive_integer_list(bitcast["output_shape"], f"{context}.output_shape")
        if math.prod(input_shape) != math.prod(output_shape):
            raise ValueError(f"{context} must preserve the element count")

    saved_state = _mapping(
        logical_boundary["saved_state_names_and_bytes"], "logical_boundary.saved_state_names_and_bytes"
    )
    for name, byte_count in saved_state.items():
        _require_nonempty_string(name, "logical_boundary.saved_state_names_and_bytes key")
        _require_nonnegative_integer(byte_count, f"logical_boundary.saved_state_names_and_bytes.{name}")

    recompute_operations = _require_record_list(
        logical_boundary["recompute_operations"],
        "recompute_operation",
        "logical_boundary.recompute_operations",
    )
    for index, operation in enumerate(recompute_operations):
        context = f"logical_boundary.recompute_operations[{index}]"
        for field in ("output", "operation"):
            _require_nonempty_string(operation[field], f"{context}.{field}")
        _require_nonnegative_integer(operation["launch_count"], f"{context}.launch_count")


def _require_timing_samples(value: object, context: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must contain at least one timing sample")
    for index, sample in enumerate(value):
        if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
            raise ValueError(f"{context}[{index}] must be a positive integer nanosecond sample")


def _reviewed_floor(backend: object) -> NumericalFloor:
    for floor in REVIEWED_NUMERICAL_FLOORS:
        if backend == floor.backend.value:
            return floor
    raise ValueError("identity.backend must name a required backend")


@dataclass(frozen=True)
class _PairwiseDrift:
    left_repeat_index: int
    right_repeat_index: int
    maximum_absolute_error: float
    mean_absolute_error: float
    maximum_ulp_distance: int
    mean_ulp_distance: float


@dataclass(frozen=True)
class _NumericalOutputSummary:
    maximum_absolute_error: float
    mean_absolute_error: float
    maximum_ulp_distance: int
    mean_ulp_distance: float
    nonfinite_values: int
    repeat_count: int
    repeat_identities_equal: bool
    repeat_maximum_absolute_error: float
    repeat_mean_absolute_error: float
    repeat_maximum_ulp_distance: int
    repeat_mean_ulp_distance: float


def _diagnostic_value(value: float | int | bool) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return repr(value)


def _numerical_floor_error(
    *,
    floor: NumericalFloor,
    floor_kind: str,
    case_id: str,
    measurement_boundary: str,
    output_name: str,
    metric: str,
    measured: float | int | bool,
    limit: float | int | bool,
    summary: _NumericalOutputSummary,
) -> NumericalFloorError:
    prefix = (
        "immutable bitwise-repeatability floor exceeded"
        if floor_kind == "bitwise-repeatability"
        else f"immutable {floor.backend.value} {floor_kind} floor exceeded"
    )
    fields = (
        ("case", case_id),
        ("backend", floor.backend.value),
        ("boundary", measurement_boundary),
        ("output", output_name),
        ("reference", floor.reference.value),
        ("metric", metric),
        ("measured", _diagnostic_value(measured)),
        ("limit", _diagnostic_value(limit)),
        ("maximum_absolute_error", _diagnostic_value(summary.maximum_absolute_error)),
        ("mean_absolute_error", _diagnostic_value(summary.mean_absolute_error)),
        ("maximum_ulp_distance", _diagnostic_value(summary.maximum_ulp_distance)),
        ("mean_ulp_distance", _diagnostic_value(summary.mean_ulp_distance)),
        ("nonfinite_values", _diagnostic_value(summary.nonfinite_values)),
        ("repeat_count", _diagnostic_value(summary.repeat_count)),
        ("repeat_identities_equal", _diagnostic_value(summary.repeat_identities_equal)),
        ("repeat_maximum_absolute_error", _diagnostic_value(summary.repeat_maximum_absolute_error)),
        ("repeat_mean_absolute_error", _diagnostic_value(summary.repeat_mean_absolute_error)),
        ("repeat_maximum_ulp_distance", _diagnostic_value(summary.repeat_maximum_ulp_distance)),
        ("repeat_mean_ulp_distance", _diagnostic_value(summary.repeat_mean_ulp_distance)),
    )
    diagnostic = f"{prefix}: " + " ".join(f"{name}={value}" for name, value in fields)
    if len(diagnostic) > _MAX_NUMERICAL_DIAGNOSTIC_CHARS:
        return NumericalFloorError(
            "numerical floor diagnostic exceeded the closed 1024-character bound",
            output_name=output_name,
        )
    return NumericalFloorError(diagnostic, output_name=output_name)


def _validate_numerical_output(
    output: Mapping[str, Any],
    floor: NumericalFloor,
    *,
    case_id: str,
    measurement_boundary: str,
    output_name: str,
) -> None:
    context = f"numerical.outputs.{output_name}"
    output_floor = floor.output_floor(output_name)
    maximum_absolute_error = _require_finite_nonnegative_number(
        output["maximum_absolute_error"], f"{context}.maximum_absolute_error"
    )
    mean_absolute_error = _require_finite_nonnegative_number(
        output["mean_absolute_error"], f"{context}.mean_absolute_error"
    )
    maximum_ulp_distance = _require_nonnegative_integer(
        output["maximum_ulp_distance"], f"{context}.maximum_ulp_distance"
    )
    mean_ulp_distance = _require_finite_nonnegative_number(output["mean_ulp_distance"], f"{context}.mean_ulp_distance")
    nonfinite_values = _require_nonnegative_integer(output["nonfinite_values"], f"{context}.nonfinite_values")

    repeat_hashes = output["repeat_hashes"]
    if not isinstance(repeat_hashes, list) or len(repeat_hashes) < 2:
        raise ValueError(f"{context}.repeat_hashes must contain at least two content identities")
    for index, digest in enumerate(repeat_hashes):
        _require_lowercase_hex_digest(digest, _SHA256_LENGTH, f"{context}.repeat_hashes[{index}]")

    drift_records = output["pairwise_drift"]
    if not isinstance(drift_records, list) or not drift_records:
        raise ValueError(f"{context}.pairwise_drift cannot be empty")
    expected_pairs = {
        (left, right) for left in range(len(repeat_hashes)) for right in range(left + 1, len(repeat_hashes))
    }
    observed_pairs: set[tuple[int, int]] = set()
    parsed_drift = []
    for index, drift_value in enumerate(drift_records):
        drift_context = f"{context}.pairwise_drift[{index}]"
        drift = _mapping(drift_value, drift_context)
        _require_fields(drift, PAIRWISE_DRIFT_REQUIRED_FIELDS, drift_context)
        left = _require_nonnegative_integer(drift["left_repeat_index"], f"{drift_context}.left_repeat_index")
        right = _require_nonnegative_integer(drift["right_repeat_index"], f"{drift_context}.right_repeat_index")
        pair = (left, right)
        if pair not in expected_pairs or pair in observed_pairs:
            raise ValueError(f"{drift_context} must identify one unique ordered repeat pair")
        observed_pairs.add(pair)
        record = _PairwiseDrift(
            left_repeat_index=left,
            right_repeat_index=right,
            maximum_absolute_error=_require_finite_nonnegative_number(
                drift["maximum_absolute_error"], f"{drift_context}.maximum_absolute_error"
            ),
            mean_absolute_error=_require_finite_nonnegative_number(
                drift["mean_absolute_error"], f"{drift_context}.mean_absolute_error"
            ),
            maximum_ulp_distance=_require_nonnegative_integer(
                drift["maximum_ulp_distance"], f"{drift_context}.maximum_ulp_distance"
            ),
            mean_ulp_distance=_require_finite_nonnegative_number(
                drift["mean_ulp_distance"], f"{drift_context}.mean_ulp_distance"
            ),
        )
        parsed_drift.append(record)
    if observed_pairs != expected_pairs:
        raise ValueError(f"{context}.pairwise_drift must cover every repeat pair exactly once")

    summary = _NumericalOutputSummary(
        maximum_absolute_error=maximum_absolute_error,
        mean_absolute_error=mean_absolute_error,
        maximum_ulp_distance=maximum_ulp_distance,
        mean_ulp_distance=mean_ulp_distance,
        nonfinite_values=nonfinite_values,
        repeat_count=len(repeat_hashes),
        repeat_identities_equal=len(set(repeat_hashes)) == 1,
        repeat_maximum_absolute_error=max(record.maximum_absolute_error for record in parsed_drift),
        repeat_mean_absolute_error=max(record.mean_absolute_error for record in parsed_drift),
        repeat_maximum_ulp_distance=max(record.maximum_ulp_distance for record in parsed_drift),
        repeat_mean_ulp_distance=max(record.mean_ulp_distance for record in parsed_drift),
    )

    floating_metric_limits = (
        ("maximum_absolute_error", maximum_absolute_error, output_floor.maximum_absolute_error),
        ("mean_absolute_error", mean_absolute_error, output_floor.mean_absolute_error),
    )
    for field, value, limit in floating_metric_limits:
        if value > limit:
            raise _numerical_floor_error(
                floor=floor,
                floor_kind="numerical",
                case_id=case_id,
                measurement_boundary=measurement_boundary,
                output_name=output_name,
                metric=field,
                measured=value,
                limit=limit,
                summary=summary,
            )
    if floor.ulp_acceptance is UlpAcceptanceMode.HARD:
        assert floor.maximum_ulp_distance is not None
        assert floor.mean_ulp_distance is not None
        for field, value, limit in (
            ("maximum_ulp_distance", maximum_ulp_distance, floor.maximum_ulp_distance),
            ("mean_ulp_distance", mean_ulp_distance, floor.mean_ulp_distance),
        ):
            if value > limit:
                raise _numerical_floor_error(
                    floor=floor,
                    floor_kind="numerical",
                    case_id=case_id,
                    measurement_boundary=measurement_boundary,
                    output_name=output_name,
                    metric=field,
                    measured=value,
                    limit=limit,
                    summary=summary,
                )
    if nonfinite_values > floor.maximum_nonfinite_values:
        raise _numerical_floor_error(
            floor=floor,
            floor_kind="numerical",
            case_id=case_id,
            measurement_boundary=measurement_boundary,
            output_name=output_name,
            metric="nonfinite_values",
            measured=nonfinite_values,
            limit=floor.maximum_nonfinite_values,
            summary=summary,
        )
    if floor.repeatability is RepeatabilityMode.BITWISE and not summary.repeat_identities_equal:
        raise _numerical_floor_error(
            floor=floor,
            floor_kind="bitwise-repeatability",
            case_id=case_id,
            measurement_boundary=measurement_boundary,
            output_name=output_name,
            metric="repeat_identities_equal",
            measured=False,
            limit=True,
            summary=summary,
        )
    if mean_absolute_error > maximum_absolute_error:
        raise ValueError(f"{context}.mean_absolute_error cannot exceed maximum_absolute_error")
    if mean_ulp_distance > maximum_ulp_distance:
        raise ValueError(f"{context}.mean_ulp_distance cannot exceed maximum_ulp_distance")

    for drift in parsed_drift:
        repeat_floating_limits = (
            ("maximum_absolute_error", drift.maximum_absolute_error, floor.repeat_maximum_absolute_error),
            ("mean_absolute_error", drift.mean_absolute_error, floor.repeat_mean_absolute_error),
        )
        for field, value, limit in repeat_floating_limits:
            if value > limit:
                raise _numerical_floor_error(
                    floor=floor,
                    floor_kind="repeat",
                    case_id=case_id,
                    measurement_boundary=measurement_boundary,
                    output_name=output_name,
                    metric=f"pairwise_drift[{drift.left_repeat_index}:{drift.right_repeat_index}].{field}",
                    measured=value,
                    limit=limit,
                    summary=summary,
                )
        if floor.ulp_acceptance is UlpAcceptanceMode.HARD:
            assert floor.maximum_ulp_distance is not None
            assert floor.mean_ulp_distance is not None
            repeat_maximum_ulp_limit = (
                0 if floor.repeatability is RepeatabilityMode.BITWISE else floor.maximum_ulp_distance
            )
            repeat_mean_ulp_limit = 0.0 if floor.repeatability is RepeatabilityMode.BITWISE else floor.mean_ulp_distance
            for field, value, limit in (
                ("mean_ulp_distance", drift.mean_ulp_distance, repeat_mean_ulp_limit),
                ("maximum_ulp_distance", drift.maximum_ulp_distance, repeat_maximum_ulp_limit),
            ):
                if value > limit:
                    raise _numerical_floor_error(
                        floor=floor,
                        floor_kind="repeat",
                        case_id=case_id,
                        measurement_boundary=measurement_boundary,
                        output_name=output_name,
                        metric=f"pairwise_drift[{drift.left_repeat_index}:{drift.right_repeat_index}].{field}",
                        measured=value,
                        limit=limit,
                        summary=summary,
                    )
        drift_context = f"{context}.pairwise_drift[{drift.left_repeat_index}:{drift.right_repeat_index}]"
        if drift.mean_absolute_error > drift.maximum_absolute_error:
            raise ValueError(f"{drift_context}.mean_absolute_error cannot exceed maximum_absolute_error")
        if drift.mean_ulp_distance > drift.maximum_ulp_distance:
            raise ValueError(f"{drift_context}.mean_ulp_distance cannot exceed maximum_ulp_distance")


def validate_backend_numerical_evidence(
    backend: BackendVariant,
    outputs: Mapping[str, Mapping[str, Any]],
    *,
    case_id: str,
    measurement_boundary: MeasurementBoundary,
) -> None:
    """Apply immutable per-output floors before a runner may begin timing."""
    if type(backend) is not BackendVariant:
        raise TypeError("backend must be a BackendVariant")
    if case_id not in {case.case_id for case in default_h100_contract_map_benchmark_plan().cases}:
        raise ValueError("case_id must name a reviewed structural case")
    if type(measurement_boundary) is not MeasurementBoundary:
        raise TypeError("measurement_boundary must be a MeasurementBoundary")
    if tuple(outputs) != NUMERICAL_OUTPUT_ROLES:
        raise ValueError("numerical outputs must contain forward, dx, dw0, and dw1 in fixed order")
    floor = _reviewed_floor(backend.value)
    for role in NUMERICAL_OUTPUT_ROLES:
        output = _mapping(outputs[role], f"numerical.outputs.{role}")
        _require_fields(output, NUMERICAL_OUTPUT_REQUIRED_FIELDS, f"numerical.outputs.{role}")
        _validate_numerical_output(
            output,
            floor,
            case_id=case_id,
            measurement_boundary=measurement_boundary.value,
            output_name=role,
        )


def _serialized_schedule(timing: TimingProtocol) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": row.sample_index,
            "cycle_index": row.cycle_index,
            "backend_order": [backend.value for backend in row.backend_order],
        }
        for row in timing.steady_state_schedule
    ]


def result_evidence_schema() -> dict[str, Any]:
    """Return the immutable result schema before any benchmark execution."""
    plan = default_h100_contract_map_benchmark_plan()
    return {
        "schema": "shuttle.h100_contract_map_result_evidence.v4",
        "required_sections": [section.name for section in RESULT_EVIDENCE_SECTIONS],
        "sections": [asdict(section) for section in RESULT_EVIDENCE_SECTIONS],
        "nested_records": {
            "kernel_record": KERNEL_RECORD_REQUIRED_FIELDS,
            "cubin": {
                CubinAvailability.AVAILABLE.value: ("availability", "path", "sha256"),
                CubinAvailability.UNAVAILABLE.value: ("availability", "unavailable_reason"),
            },
            "numerical_output_roles": NUMERICAL_OUTPUT_ROLES,
            "numerical_output": NUMERICAL_OUTPUT_REQUIRED_FIELDS,
            "pairwise_drift": PAIRWISE_DRIFT_REQUIRED_FIELDS,
            "raw_sample": RAW_SAMPLE_REQUIRED_FIELDS,
            "raw_sample_measurement_backends": tuple(BackendVariant),
            "raw_sample_measurement_boundaries": tuple(MeasurementBoundary),
            "logical_boundary_records": {name: dict(fields) for name, fields in LOGICAL_BOUNDARY_RECORD_SCHEMAS},
        },
        "reviewed_numerical_floors": [asdict(floor) for floor in REVIEWED_NUMERICAL_FLOORS],
        "reviewed_numerical_floors_sha256": REVIEWED_NUMERICAL_FLOORS_SHA256,
        "steady_state_schedule": _serialized_schedule(plan.timing),
        "samples_per_backend_permutation": 4,
        "required_result_records": [
            {"case_id": case.case_id, "backend": backend.value, "measurement_boundary": boundary.value}
            for case in plan.cases
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

    artifacts = _mapping(payload["artifacts"], "artifacts")
    artifact_fields = (
        ("final_optimized_hlo_path", "final_optimized_hlo_sha256"),
        ("custom_call_manifest_path", "custom_call_manifest_sha256"),
    )
    for path_field, digest_field in artifact_fields:
        _require_artifact_path(artifacts[path_field], f"artifacts.{path_field}")
        _require_lowercase_hex_digest(artifacts[digest_field], _SHA256_LENGTH, f"artifacts.{digest_field}")

    resources = _mapping(payload["resources"], "resources")
    kernel_records = resources["kernel_records"]
    if not isinstance(kernel_records, list) or not kernel_records:
        raise ValueError("resources.kernel_records must contain at least one kernel record")
    for index, record_value in enumerate(kernel_records):
        context = f"resources.kernel_records[{index}]"
        record = _mapping(record_value, context)
        _require_fields(record, KERNEL_RECORD_REQUIRED_FIELDS, context)
        _require_nonempty_string(record["name"], f"{context}.name")
        for path_field, digest_field in (("ptx_path", "ptx_sha256"), ("sass_path", "sass_sha256")):
            _require_artifact_path(record[path_field], f"{context}.{path_field}")
            _require_lowercase_hex_digest(record[digest_field], _SHA256_LENGTH, f"{context}.{digest_field}")
        _validate_cubin_evidence(record["cubin"], identity["backend"], f"{context}.cubin")
        for field in (
            "registers_per_thread",
            "spill_load_bytes",
            "spill_store_bytes",
            "static_shared_memory_bytes",
            "dynamic_shared_memory_bytes",
            "active_blocks_per_sm",
        ):
            _require_nonnegative_integer(record[field], f"{context}.{field}")
        block_size = record["block_size"]
        if (
            not isinstance(block_size, list)
            or len(block_size) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in block_size)
        ):
            raise ValueError(f"{context}.block_size must contain three positive integer dimensions")
        _require_nonempty_string(record["limiting_occupancy_resource"], f"{context}.limiting_occupancy_resource")
        occupancy = _require_finite_nonnegative_number(record["achieved_occupancy"], f"{context}.achieved_occupancy")
        if occupancy > 1.0:
            raise ValueError(f"{context}.achieved_occupancy cannot exceed one")
    kernel_names = tuple(record["name"] for record in kernel_records)
    launch_count = resources["launch_count"]
    _require_nonnegative_integer(launch_count, "resources.launch_count")
    if launch_count != len(kernel_records) or tuple(resources["ordered_kernel_names"]) != kernel_names:
        raise ValueError("launch_count and ordered_kernel_names must match kernel_records")

    copies = _mapping(payload["copies"], "copies")
    for field in (
        "device_to_device_count",
        "device_to_device_bytes",
        "host_to_device_count",
        "host_to_device_bytes",
        "device_to_host_count",
        "device_to_host_bytes",
        "unexpected_copy_count",
    ):
        _require_nonnegative_integer(copies[field], f"copies.{field}")
    if copies["unexpected_copy_count"] != 0:
        raise ValueError("copies.unexpected_copy_count must be zero")

    logical_boundary = _mapping(payload["logical_boundary"], "logical_boundary")
    _validate_logical_boundary(logical_boundary)

    provenance = _mapping(payload["provenance"], "provenance")
    _require_nonempty_string_list(provenance["command"], "provenance.command")
    environment = _mapping(provenance["environment"], "provenance.environment")
    if not environment:
        raise ValueError("provenance.environment must be nonempty")
    for key, value in environment.items():
        _require_nonempty_string(key, "provenance.environment key")
        _require_nonempty_string(value, f"provenance.environment.{key}")
    _require_nonempty_string_list(provenance["compiler_flags"], "provenance.compiler_flags")
    source_sha = provenance["source_sha"]
    _require_lowercase_hex_digest(source_sha, _GIT_SHA_LENGTH, "provenance.source_sha")
    _require_nonempty_string(provenance["persistent_cache_identity"], "provenance.persistent_cache_identity")

    numerical = _mapping(payload["numerical"], "numerical")
    if numerical["reviewed_floors_sha256"] != REVIEWED_NUMERICAL_FLOORS_SHA256:
        raise ValueError("numerical evidence does not use the reviewed floor digest")
    outputs = _mapping(numerical["outputs"], "numerical.outputs")
    if tuple(outputs) != NUMERICAL_OUTPUT_ROLES:
        raise ValueError("numerical.outputs must contain forward, dx, dw0, and dw1 in fixed order")
    floor = _reviewed_floor(identity["backend"])
    for role in NUMERICAL_OUTPUT_ROLES:
        context = f"numerical.outputs.{role}"
        output = _mapping(outputs[role], context)
        _require_fields(output, NUMERICAL_OUTPUT_REQUIRED_FIELDS, context)
        _validate_numerical_output(
            output,
            floor,
            case_id=identity["case_id"],
            measurement_boundary=identity["measurement_boundary"],
            output_name=role,
        )
    if numerical["floors_passed_before_timing"] is not True:
        raise ValueError("numerical floors must pass before timing")

    timing = _mapping(payload["timing"], "timing")
    timing_protocol = default_h100_contract_map_benchmark_plan().timing
    for field in (
        "compile_samples_ns",
        "first_execution_samples_ns",
        "warmup_samples_ns",
        "persistent_cache_cold_samples_ns",
        "persistent_cache_hit_samples_ns",
    ):
        _require_timing_samples(timing[field], f"timing.{field}")
    if timing["warmup_iterations"] != timing_protocol.warmup_iterations:
        raise ValueError("timing.warmup_iterations must equal the reviewed protocol")
    if len(timing["warmup_samples_ns"]) != timing_protocol.warmup_iterations:
        raise ValueError("timing.warmup_samples_ns must contain every reviewed warmup iteration")
    expected_schedule = _serialized_schedule(timing_protocol)
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
            for boundary in expected_boundaries:
                sample_context = f"timing.raw_samples[{index}].measurements_ns.{backend}.{boundary}"
                sample = boundaries[boundary]
                if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
                    raise ValueError(f"{sample_context} must be a positive integer nanosecond sample")


def validate_result_evidence_bundle(payloads: tuple[Mapping[str, Any], ...]) -> None:
    """Require one complete result for every reviewed case, backend, and boundary."""
    for payload in payloads:
        validate_result_evidence(payload)
    expected = tuple(
        (case.case_id, backend.value, boundary.value)
        for case in default_h100_contract_map_benchmark_plan().cases
        for backend in BackendVariant
        for boundary in MeasurementBoundary
    )
    identities = tuple(
        (
            _mapping(payload["identity"], "identity")["case_id"],
            _mapping(payload["identity"], "identity")["backend"],
            _mapping(payload["identity"], "identity")["measurement_boundary"],
        )
        for payload in payloads
    )
    if identities != expected:
        raise ValueError("result bundle must contain all 24 reviewed case, backend, and boundary records in fixed order")


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
        if self.schema_version != 2:
            raise ValueError("the staged H100 benchmark schema version is fixed at two")
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
        schema_version=2,
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
            resource_collectors_wired=True,
            reviewed=False,
            evidence_paths=("lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py",),
            blockers=("the executable collectors have no reviewed H100 result bundle",),
        ),
        BackendWiring(
            backend=BackendVariant.SHUTTLE_SOURCE_ORDERED,
            generated_backend_wired=False,
            resource_collectors_wired=True,
            reviewed=False,
            evidence_paths=(
                "lib/tile_lifetime/src/tile_lifetime/contract_map_backend.py",
                "lib/tile_lifetime/src/tile_lifetime/cuda_contract_map_backend_codegen.py",
                "lib/tile_lifetime/benchmarks/h100_contract_map_backend_training.py",
                "lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py",
            ),
            blockers=(
                "the multi-CTA direct FFI is not reached through the ordinary-JAX Shuttle transform",
                "the executable collectors have no reviewed H100 result bundle",
            ),
        ),
        BackendWiring(
            backend=BackendVariant.SHUTTLE_FAST,
            generated_backend_wired=False,
            resource_collectors_wired=True,
            reviewed=False,
            evidence_paths=(
                "lib/tile_lifetime/src/tile_lifetime/contract_map_backend.py",
                "lib/tile_lifetime/src/tile_lifetime/cuda_contract_map_backend_codegen.py",
                "lib/tile_lifetime/benchmarks/h100_contract_map_backend_training.py",
                "lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py",
            ),
            blockers=(
                "the fixed-tree FAST direct FFI is not reached through the ordinary-JAX Shuttle transform",
                "the executable collectors have no reviewed H100 result bundle",
            ),
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
        "schema": "shuttle.h100_contract_map_backend_evidence.v4",
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
