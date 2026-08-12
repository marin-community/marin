# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent numerical reference contract for the Target 1 rowwise program."""

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

SCHEMA_VERSION = 1
MAX_CONTRACT_BYTES = 128 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
BOUNDARIES = ("forward", "backward", "composed")
POLICIES = ("source_ordered", "fast")
SHAPES = ((2048, 4096), (7, 13))
INPUT_RANGES = ((-0.75, 0.875), (-0.625, 1.0), (-0.5, 1.125))
EPSILON = 1e-5
RELATIVE_SCALE_FLOOR = 2**-7
ANALYTIC_THRESHOLDS = {
    "max_absolute_error": 2**-6,
    "mean_absolute_error": 1e-6,
    "relative_linf_error": 2**-7,
    "max_bfloat16_ulp_error": 8,
}
OUTPUT_ROLES = {
    "forward": ("y",),
    "backward": ("dx", "dgamma"),
    "composed": ("y", "dx", "dgamma"),
}
FORMULAS = {
    "r": "r_i = 1 / sqrt(sum_j(binary64(x_ij) * binary64(x_ij)) / features + binary64(1e-5))",
    "y": "y_ij = bfloat16(binary64(x_ij) * r_i * binary64(gamma_j))",
    "dx": (
        "dx_ij = bfloat16(binary64(dy_ij) * binary64(gamma_j) * r_i - "
        "binary64(x_ij) * r_i**3 * sum_k(binary64(dy_ik) * binary64(x_ik) * binary64(gamma_k)) / features)"
    ),
    "dgamma": "dgamma_j = bfloat16(sum_i(binary64(dy_ij) * binary64(x_ij) * r_i))",
}
PROVENANCE = {
    "jax_version": "0.10.1",
    "jaxlib_version": "0.10.1",
    "jax_revision": "619764c15117fbefc4ba13ab941871cb514c23f6",
    "xla_revision": "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69",
    "numpy_version": "2.3.5",
    "ml_dtypes_version": "0.5.4",
}


@dataclass(frozen=True)
class ErrorMetrics:
    """Four numerical deviations required by the evaluation contract."""

    max_absolute_error: float
    mean_absolute_error: float
    relative_linf_error: float
    max_bfloat16_ulp_error: int


def fixed_inputs(rows: int, features: int, boundary: str) -> tuple[np.ndarray, ...]:
    """Construct the closed BF16 public inputs without using JAX."""
    if (rows, features) not in SHAPES:
        raise ValueError(f"undeclared shape: {(rows, features)!r}")
    if boundary not in BOUNDARIES:
        raise ValueError(f"undeclared boundary: {boundary!r}")
    dimensions = ((rows, features), (features,))
    if boundary != "forward":
        dimensions += ((rows, features),)
    return tuple(
        np.linspace(start, stop, math.prod(shape), dtype=np.float32).reshape(shape).astype(ml_dtypes.bfloat16)
        for shape, (start, stop) in zip(dimensions, INPUT_RANGES[: len(dimensions)], strict=True)
    )


def independent_reference(boundary: str, arguments: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Evaluate the closed-form forward and analytic VJP in binary64."""
    if boundary not in BOUNDARIES:
        raise ValueError(f"undeclared boundary: {boundary!r}")
    expected_arity = 2 if boundary == "forward" else 3
    if len(arguments) != expected_arity:
        raise ValueError(f"{boundary} requires {expected_arity} inputs")
    x = _bf16_input(arguments[0], "x", rank=2).astype(np.float64)
    gamma = _bf16_input(arguments[1], "gamma", rank=1).astype(np.float64)
    rows, features = x.shape
    if gamma.shape != (features,) or (rows, features) not in SHAPES:
        raise ValueError("input shapes differ from the closed shape contract")

    inverse = np.reciprocal(np.sqrt(np.sum(x * x, axis=1, keepdims=True, dtype=np.float64) / features + EPSILON))
    y = _to_bf16(x * inverse * gamma)
    if boundary == "forward":
        return (y,)

    dy = _bf16_input(arguments[2], "dy", rank=2).astype(np.float64)
    if dy.shape != x.shape:
        raise ValueError("dy must have the same shape as x")
    row_cotangent = np.sum(dy * x * gamma, axis=1, keepdims=True, dtype=np.float64)
    dx = _to_bf16(dy * gamma * inverse - x * inverse**3 * row_cotangent / features)
    dgamma = _to_bf16(np.sum(dy * x * inverse, axis=0, dtype=np.float64))
    if boundary == "backward":
        return dx, dgamma
    return y, dx, dgamma


def error_metrics(actual: np.ndarray, reference: np.ndarray) -> ErrorMetrics:
    """Measure one BF16 output against the independent BF16 reference."""
    actual = _bf16_output(actual, "actual")
    reference = _bf16_output(reference, "reference")
    if actual.shape != reference.shape:
        raise ValueError("actual and reference shapes differ")
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    absolute = np.abs(actual64 - reference64)
    scale = max(float(np.max(np.abs(reference64), initial=0.0)), RELATIVE_SCALE_FLOOR)
    return ErrorMetrics(
        max_absolute_error=float(np.max(absolute, initial=0.0)),
        mean_absolute_error=float(np.mean(absolute)) if absolute.size else 0.0,
        relative_linf_error=float(np.max(absolute, initial=0.0) / scale),
        max_bfloat16_ulp_error=int(np.max(np.abs(_ordered_bf16(actual) - _ordered_bf16(reference)), initial=0)),
    )


def require_accepted(metrics: ErrorMetrics, policy: str, *, identity_roundtrip_bitwise: bool) -> None:
    """Apply the policy-specific, predeclared local numerical gate."""
    if policy not in POLICIES:
        raise ValueError(f"undeclared policy: {policy!r}")
    if not identity_roundtrip_bitwise:
        if policy == "source_ordered":
            raise AssertionError("source_ordered must be bitwise-equal to ordinary JAX for the identity lowering")
        raise AssertionError("non-identity FAST numerical thresholds are unresolved")
    for name, limit in ANALYTIC_THRESHOLDS.items():
        if getattr(metrics, name) > limit:
            raise AssertionError(f"{policy} {name} exceeds the predeclared local analytic threshold")


def array_digest(value: np.ndarray) -> str:
    """Hash an array including its public dtype and shape."""
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def validate_contract(document: object) -> None:
    """Reject any drift in the closed, reviewable numerical contract."""
    root = _closed_mapping(
        document,
        "contract",
        {
            "schema_version",
            "contract_id",
            "boundaries",
            "shapes",
            "inputs",
            "reference",
            "outputs",
            "metrics",
            "policies",
            "provenance",
            "local_observation",
            "scorecard_effect",
        },
    )
    _equal(root["schema_version"], SCHEMA_VERSION, "schema_version")
    _equal(root["contract_id"], "target1_rowwise_bf16_numerical_oracle_v1", "contract_id")
    _equal(root["boundaries"], list(BOUNDARIES), "boundaries")
    _equal(root["shapes"], [[rows, features] for rows, features in SHAPES], "shapes")

    inputs = _closed_mapping(root["inputs"], "inputs", {"generator", "ranges", "digests"})
    _equal(inputs["generator"], "numpy.linspace(float32)->bfloat16", "inputs.generator")
    _equal(inputs["ranges"], [list(value) for value in INPUT_RANGES], "inputs.ranges")
    expected_digests = {
        f"{rows}x{features}/{boundary}": [array_digest(value) for value in fixed_inputs(rows, features, boundary)]
        for rows, features in SHAPES
        for boundary in BOUNDARIES
    }
    _equal(inputs["digests"], expected_digests, "inputs.digests")

    reference = _closed_mapping(
        root["reference"],
        "reference",
        {"implementation", "accumulation_dtype", "epsilon", "formulas", "output_digests", "role"},
    )
    _equal(reference["implementation"], "independent_numpy_closed_form", "reference.implementation")
    _equal(reference["accumulation_dtype"], "float64", "reference.accumulation_dtype")
    _equal(reference["epsilon"], EPSILON, "reference.epsilon")
    _equal(reference["formulas"], FORMULAS, "reference.formulas")
    expected_reference_digests = {
        f"{rows}x{features}/{boundary}": [
            array_digest(value) for value in independent_reference(boundary, fixed_inputs(rows, features, boundary))
        ]
        for rows, features in SHAPES
        for boundary in BOUNDARIES
    }
    _equal(reference["output_digests"], expected_reference_digests, "reference.output_digests")
    _equal(reference["role"], "numerical_reference_not_expert_performance_oracle", "reference.role")

    expected_outputs = {
        f"{rows}x{features}/{boundary}": [
            {"role": role, "shape": list(_output_shape(role, rows, features)), "dtype": "bfloat16"}
            for role in OUTPUT_ROLES[boundary]
        ]
        for rows, features in SHAPES
        for boundary in BOUNDARIES
    }
    _equal(root["outputs"], expected_outputs, "outputs")

    metrics = _closed_mapping(root["metrics"], "metrics", {"definitions", "relative_scale_floor"})
    _equal(
        metrics["definitions"],
        {
            "max_absolute_error": "max(abs(actual-reference))",
            "mean_absolute_error": "mean(abs(actual-reference))",
            "relative_linf_error": "max_absolute_error/max(max(abs(reference)),relative_scale_floor)",
            "max_bfloat16_ulp_error": "max distance between ordered finite bfloat16 encodings",
        },
        "metrics.definitions",
    )
    _equal(metrics["relative_scale_floor"], RELATIVE_SCALE_FLOOR, "metrics.relative_scale_floor")

    policies = _closed_mapping(root["policies"], "policies", set(POLICIES))
    for policy in POLICIES:
        value = _closed_mapping(
            policies[policy],
            f"policies.{policy}",
            {"identity_roundtrip", "analytic_thresholds", "non_identity_fast_status"},
        )
        _equal(value["identity_roundtrip"], "bitwise_ordinary_jax_required", f"policies.{policy}.identity_roundtrip")
        _equal(value["analytic_thresholds"], ANALYTIC_THRESHOLDS, f"policies.{policy}.analytic_thresholds")
        expected_status = "not_applicable" if policy == "source_ordered" else "unresolved_requires_contract_revision"
        _equal(value["non_identity_fast_status"], expected_status, f"policies.{policy}.non_identity_fast_status")

    _equal(root["provenance"], PROVENANCE, "provenance")
    observation = _closed_mapping(root["local_observation"], "local_observation", {"claim", "environment", "results"})
    _equal(observation["claim"], "local_cpu_reference_check_not_scorecard_evidence", "local_observation.claim")
    _equal(
        observation["environment"],
        {
            "backend_platform": "cpu",
            "device_kind_class": "cpu",
            "host_architecture": "arm64",
            "jax_enable_x64": False,
            "python_version": "3.12.11",
        },
        "local_observation.environment",
    )
    results = _closed_mapping(
        observation["results"],
        "local_observation.results",
        {f"{rows}x{features}/{boundary}" for rows, features in SHAPES for boundary in BOUNDARIES},
    )
    for rows, features in SHAPES:
        for boundary in BOUNDARIES:
            name = f"{rows}x{features}/{boundary}"
            records = results[name]
            if not isinstance(records, list) or len(records) != len(OUTPUT_ROLES[boundary]):
                raise ValueError(f"local_observation.results.{name} output arity drifted")
            for record, role in zip(records, OUTPUT_ROLES[boundary], strict=True):
                record = _closed_mapping(
                    record,
                    f"local_observation.results.{name}.{role}",
                    {"role", "output_digest", "metrics"},
                )
                _equal(record["role"], role, f"local_observation.results.{name}.{role}.role")
                if not isinstance(record["output_digest"], str) or not SHA256_PATTERN.fullmatch(record["output_digest"]):
                    raise ValueError(f"local_observation.results.{name}.{role}.output_digest drifted")
                recorded_metrics = _closed_mapping(
                    record["metrics"],
                    f"local_observation.results.{name}.{role}.metrics",
                    set(ANALYTIC_THRESHOLDS),
                )
                checked_metrics = ErrorMetrics(
                    max_absolute_error=_nonnegative_finite_number(
                        recorded_metrics["max_absolute_error"],
                        f"local_observation.results.{name}.{role}.max_absolute_error",
                    ),
                    mean_absolute_error=_nonnegative_finite_number(
                        recorded_metrics["mean_absolute_error"],
                        f"local_observation.results.{name}.{role}.mean_absolute_error",
                    ),
                    relative_linf_error=_nonnegative_finite_number(
                        recorded_metrics["relative_linf_error"],
                        f"local_observation.results.{name}.{role}.relative_linf_error",
                    ),
                    max_bfloat16_ulp_error=_nonnegative_integer(
                        recorded_metrics["max_bfloat16_ulp_error"],
                        f"local_observation.results.{name}.{role}.max_bfloat16_ulp_error",
                    ),
                )
                require_accepted(
                    checked_metrics,
                    "source_ordered",
                    identity_roundtrip_bitwise=True,
                )
    _equal(
        root["scorecard_effect"],
        {
            "status_changed": False,
            "reason": "No architecturally conforming hardware run or expert performance oracle is pinned.",
        },
        "scorecard_effect",
    )


def load_contract(path: Path) -> Mapping[str, Any]:
    """Load a contract while rejecting duplicate JSON keys."""
    payload = path.read_bytes()
    if len(payload) > MAX_CONTRACT_BYTES:
        raise ValueError("numerical contract exceeds the byte limit")
    document = json.loads(payload, object_pairs_hook=_unique_object)
    validate_contract(document)
    return document


def _bf16_input(value: np.ndarray, role: str, *, rank: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != ml_dtypes.bfloat16 or array.ndim != rank:
        raise ValueError(f"{role} must be rank-{rank} bfloat16")
    if not np.all(np.isfinite(array.astype(np.float32))):
        raise ValueError(f"{role} must be finite")
    return array


def _bf16_output(value: np.ndarray, role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != ml_dtypes.bfloat16 or not np.all(np.isfinite(array.astype(np.float32))):
        raise ValueError(f"{role} must be finite bfloat16")
    return array


def _to_bf16(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=ml_dtypes.bfloat16)


def _ordered_bf16(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value).view(np.uint16).astype(np.int32)
    return np.where(bits & 0x8000, 0x7FFF - (bits & 0x7FFF), 0x8000 + bits)


def _output_shape(role: str, rows: int, features: int) -> tuple[int, ...]:
    return (features,) if role == "dgamma" else (rows, features)


def _closed_mapping(value: object, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} fields drifted")
    return value


def _equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} drifted")


def _nonnegative_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
