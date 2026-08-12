# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-run numerical comparison contract for Target 1 hardware evidence."""

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
from target1_expert_oracle import load_contract as load_expert_contract
from target1_numerical_oracle import (
    RELATIVE_SCALE_FLOOR,
    fixed_inputs,
    independent_reference,
)
from target1_numerical_oracle import (
    load_contract as load_numerical_contract,
)

SCHEMA_VERSION = 1
MAX_CONTRACT_BYTES = 128 * 1024
REPOSITORY_ROOT = Path(__file__).parents[4]
CONTRACT_ID = "target1_rowwise_bf16_prerun_comparison_v1"
METRICS = (
    "max_absolute_error",
    "mean_absolute_error",
    "relative_linf_error",
    "max_bfloat16_ulp_error",
)
SHAPES = ((2048, 4096), (7, 13))
BOUNDARIES = {
    "forward": {"numerical_reference_boundary": "forward", "outputs": ["y"]},
    "backward_recompute": {
        "numerical_reference_boundary": "backward",
        "outputs": ["dx", "dgamma"],
    },
    "composed": {
        "numerical_reference_boundary": "composed",
        "outputs": ["y", "dx", "dgamma"],
    },
}
BACKEND_PAIRS = [
    {"forward": "transformer_engine", "backward": "transformer_engine"},
    {"forward": "cudnn", "backward": "cudnn"},
    {"forward": "transformer_engine", "backward": "cudnn"},
    {"forward": "cudnn", "backward": "transformer_engine"},
]
DEPENDENCIES = {
    "numerical_reference": {
        "contract_id": "target1_rowwise_bf16_numerical_oracle_v1",
        "path": "lib/shuttle/mlir/jax_patch/target1-rowwise-bf16-numerical-oracle-v1.json",
        "sha256": "709d5b40f45da79331535aca02872d7cc89ff93010deeb9a63e99cc8c49e8c3d",
    },
    "expert_oracle": {
        "contract_id": "target1_rowwise_bf16_te_2_17_expert_oracle_v1",
        "path": "lib/shuttle/mlir/jax_patch/target1-rowwise-bf16-te-2.17-expert-oracle-v1.json",
        "sha256": "87a893e91f43d9e52fdcc94a4305da829e99f41c0e692f03d1c787913b6868be",
    },
    "evaluation_manifest": {
        "manifest_id": "shuttle_evaluation_v1",
        "path": ".agents/projects/tile_lifetime_compiler/shuttle_evaluation_manifest_v1.json",
        "sha256": "bece8b3c766e75764242fb96eab4c47bdc7321be899662c2661eb4ad0d6109f4",
    },
}
SUBJECTS = {
    "independent_reference": {
        "subject_id": "independent_numpy_binary64_closed_form",
        "role": "reference_not_performance_oracle",
        "required_identity": [
            "numerical_reference_contract_id",
            "input_digests",
            "reference_output_digests",
        ],
    },
    "ordinary_jax": {
        "subject_id": "ordinary_jax_disabled_shuttle",
        "role": "architecture_roundtrip_and_numerical_baseline",
        "required_identity": [
            "marin_revision",
            "jax_version",
            "jaxlib_version",
            "jax_revision",
            "xla_revision",
            "cuda_plugin_identity",
            "pjrt_identity",
            "xla_build_identity",
        ],
    },
    "transformer_engine": {
        "subject_id": "transformer_engine_2_17_exact_c_api",
        "role": "matched_expert_numerical_and_performance_oracle",
        "required_identity": [
            "expert_oracle_contract_id",
            "source_commit",
            "resolved_library_sha256",
            "elf_build_id",
            "resolved_shared_library_dependencies",
            "cuda_identity",
            "cudnn_identity",
            "device_identity",
            "workspace_queries",
        ],
    },
    "shuttle_source_ordered": {
        "subject_id": "ordinary_jax_through_shuttle_source_ordered",
        "role": "candidate",
        "required_identity": [
            "marin_revision",
            "pipeline_abi_version",
            "compiler_options_sha256",
            "generated_executable_sha256",
            "invocation_abi_sha256",
            "persistent_cache_key",
        ],
    },
    "shuttle_fast": {
        "subject_id": "ordinary_jax_through_shuttle_fast",
        "role": "identity_candidate_only",
        "required_identity": [
            "marin_revision",
            "pipeline_abi_version",
            "compiler_options_sha256",
            "generated_executable_sha256",
            "invocation_abi_sha256",
            "persistent_cache_key",
            "identity_lowering_proof",
        ],
    },
}
RULES = {
    "reference_qualification": "every_metric_lte_shape_output_reference_limit",
    "transformer_engine": "qualify_each_of_24_runs_against_independent_reference_before_use_as_matched_oracle",
    "ordinary_jax": "qualify_against_independent_reference_and_repeat_bitwise",
    "te_backend_pairing": (
        "compare_each_candidate_output_to_each_of_four_qualified_matching_te_backend_pair_outputs_no_post_hoc_selector"
    ),
    "source_ordered": {
        "architecture": "bitwise_equal_to_matched_ordinary_jax",
        "reference": "qualify_against_independent_reference",
        "matched_oracle": "each_metric_lte_max(matched_te_metric,shape_output_dtype_floor)",
    },
    "fast_identity": {
        "status": "allowed_only_with_identity_lowering_proof",
        "architecture": "bitwise_equal_to_matched_ordinary_jax",
        "reference": "same_rules_as_source_ordered",
        "matched_oracle": "same_rules_as_source_ordered",
    },
    "fast_non_identity": {
        "status": "blocked_requires_new_reviewed_contract_before_execution_or_timing",
        "reason": "no_independent_non_identity_fast_error_bound_exists",
    },
}
REPEATABILITY = {
    "post_timing_invocations": 3,
    "output_rule": "all_public_output_bytes_bitwise_equal_across_post_timing_invocations",
    "placement": "outside_cuda_event_intervals_after_50_measured_invocations",
    "required_subjects": [
        "ordinary_jax",
        "transformer_engine",
        "shuttle_source_ordered",
        "shuttle_fast_identity",
    ],
    "timing_repeatability_status": "blocked_requires_separate_predeclared_cross_run_contract",
}
PERFORMANCE = {
    "status": "blocked_not_part_of_numerical_contract",
    "evaluation_manifest_ratios": {
        "oracle_latency_ratio": 1.2,
        "stretch_oracle_latency_ratio": 1.1,
    },
    "reason": "one_50_sample_record_per_te_backend_pair_does_not_establish_cross_run_repeatability",
}
AGGREGATION = {
    "schema": "closed_complete_hardware_matrix_v1",
    "te_results": {
        "count": 24,
        "path_pattern": "result-%02d.json",
        "coordinates": "derived_only_from_sealed_run_plan_positions_0_through_23",
        "output_records": 48,
    },
    "candidate_records": {
        "count": 36,
        "policies": ["ordinary_jax", "source_ordered", "fast_identity"],
        "coordinates": "two_shapes_times_six_boundary_output_roles_times_three_policies",
    },
    "matching": "each_source_ordered_and_identity_fast_output_against_all_four_qualified_te_backend_pairs",
    "post_hoc_te_selector": "forbidden",
    "result_sealing": (
        "exact_plan_binary_harness_build_te_elf_dependencies_cuda_cudnn_device_input_and_reference_identity_required"
    ),
}
EXECUTION_STATE = {
    "status": "prepared_not_executed",
    "launch_ready": False,
    "hardware_results": [],
    "scorecard_status_changed": False,
    "blockers": [
        "hardware_execution_missing",
        "scorecard_grade_provenance_missing",
        "ordinary_jax_and_shuttle_subject_results_missing",
        "non_identity_fast_threshold_unresolved",
        "performance_repeatability_contract_missing",
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dtype_floors() -> dict[str, dict[str, dict[str, float | int]]]:
    floors = {}
    for rows, features in SHAPES:
        shape = f"{rows}x{features}"
        references = independent_reference("composed", fixed_inputs(rows, features, "composed"))
        floors[shape] = {}
        for role, reference in zip(("y", "dx", "dgamma"), references, strict=True):
            positive = np.nextafter(reference, np.array(np.inf, dtype=ml_dtypes.bfloat16)).astype(np.float64)
            negative = np.nextafter(reference, np.array(-np.inf, dtype=ml_dtypes.bfloat16)).astype(np.float64)
            reference64 = reference.astype(np.float64)
            spacing = np.maximum(np.abs(positive - reference64), np.abs(negative - reference64))
            maximum = float(np.max(spacing))
            scale = max(float(np.max(np.abs(reference64))), RELATIVE_SCALE_FLOOR)
            floors[shape][role] = {
                "max_absolute_error": maximum,
                "mean_absolute_error": float(np.mean(spacing)),
                "relative_linf_error": maximum / scale,
                "max_bfloat16_ulp_error": 1,
            }
    return floors


def _reference_limits(
    numerical: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float | int]]]:
    floors = _dtype_floors()
    results = numerical["local_observation"]["results"]
    return {
        shape: {
            role: {metric: max(floors[shape][role][metric], observations[role][metric]) for metric in METRICS}
            for role in ("y", "dx", "dgamma")
        }
        for shape in floors
        for observations in [{record["role"]: record["metrics"] for record in results[f"{shape}/composed"]}]
    }


def _thresholds(numerical: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "derivation": {
            "dtype_floors": "one_adjacent_finite_bfloat16_step_around_each_pinned_reference_element",
            "reference_limits": "metricwise_max(dtype_floor,pinned_local_ordinary_jax_reference_error)",
            "local_observation_scope": "shape_and_input_specific_non_scorecard_cpu_evidence_not_a_general_dtype_floor",
        },
        "dtype_floors": _dtype_floors(),
        "reference_limits": _reference_limits(numerical),
    }


def validate_contract(document: object, *, repository_root: Path = REPOSITORY_ROOT) -> None:
    """Reject semantic or provenance drift in the pre-run comparison contract."""
    root = _closed(
        document,
        {
            "schema_version",
            "contract_id",
            "status",
            "dependencies",
            "run_matrix",
            "subjects",
            "metrics",
            "thresholds",
            "rules",
            "repeatability",
            "performance",
            "aggregation",
            "execution_state",
        },
        "contract",
    )
    _equal(root["schema_version"], SCHEMA_VERSION, "schema_version")
    _equal(root["contract_id"], CONTRACT_ID, "contract_id")
    _equal(root["status"], "predeclared_not_executed", "status")
    _equal(root["dependencies"], DEPENDENCIES, "dependencies")
    for dependency in DEPENDENCIES.values():
        path = repository_root / dependency["path"]
        if not path.is_file() or _sha256(path) != dependency["sha256"]:
            raise ValueError(f"dependency provenance drifted: {dependency['path']}")
    numerical = load_numerical_contract(repository_root / DEPENDENCIES["numerical_reference"]["path"])
    load_expert_contract(repository_root / DEPENDENCIES["expert_oracle"]["path"])

    _equal(
        root["run_matrix"],
        {
            "hardware": ["h100", "gb200_or_b200"],
            "shapes": [[rows, features] for rows, features in SHAPES],
            "boundaries": BOUNDARIES,
            "te_backend_pairs": BACKEND_PAIRS,
            "te_runs_per_hardware": 24,
            "total_te_runs": 48,
            "join_fields": [
                "comparison_contract_sha256",
                "hardware",
                "shape",
                "boundary",
                "forward_backend",
                "backward_backend",
                "input_digest_set",
                "output_role",
            ],
        },
        "run_matrix",
    )
    _equal(root["subjects"], SUBJECTS, "subjects")
    _equal(
        root["metrics"],
        {
            "names": list(METRICS),
            "relative_scale_floor": RELATIVE_SCALE_FLOOR,
            "comparison_order": "metricwise_no_scalar_aggregation",
        },
        "metrics",
    )
    _equal(root["thresholds"], _thresholds(numerical), "thresholds")
    _equal(root["rules"], RULES, "rules")
    _equal(root["repeatability"], REPEATABILITY, "repeatability")
    _equal(root["performance"], PERFORMANCE, "performance")
    _equal(root["aggregation"], AGGREGATION, "aggregation")
    _equal(root["execution_state"], EXECUTION_STATE, "execution_state")


def load_contract(path: Path, *, repository_root: Path = REPOSITORY_ROOT) -> Mapping[str, Any]:
    """Load the comparison contract while rejecting duplicate keys and oversized input."""
    payload = path.read_bytes()
    if len(payload) > MAX_CONTRACT_BYTES:
        raise ValueError("pre-run comparison contract exceeds the byte limit")
    document = json.loads(payload, object_pairs_hook=_unique_object)
    validate_contract(document, repository_root=repository_root)
    return document


def require_reference_qualified(
    metrics: Mapping[str, object], *, shape: str, role: str, contract: Mapping[str, Any]
) -> None:
    """Require one subject output to satisfy its predeclared reference limit."""
    checked = _checked_metrics(metrics)
    limits = contract["thresholds"]["reference_limits"][shape][role]
    for metric in METRICS:
        if checked[metric] > limits[metric]:
            raise AssertionError(f"{shape}/{role} {metric} exceeds the predeclared reference limit")


def require_matched_oracle(
    candidate: Mapping[str, object],
    matched_te: Mapping[str, object],
    *,
    shape: str,
    role: str,
    contract: Mapping[str, Any],
) -> None:
    """Apply the frozen metricwise expert-or-dtype-floor comparison."""
    require_reference_qualified(candidate, shape=shape, role=role, contract=contract)
    require_reference_qualified(matched_te, shape=shape, role=role, contract=contract)
    checked_candidate = _checked_metrics(candidate)
    checked_te = _checked_metrics(matched_te)
    floors = contract["thresholds"]["dtype_floors"][shape][role]
    for metric in METRICS:
        if checked_candidate[metric] > max(checked_te[metric], floors[metric]):
            raise AssertionError(f"{shape}/{role} {metric} exceeds the matched expert-or-dtype-floor limit")


def require_identity_roundtrip(
    *, policy: str, identity_lowering: bool, ordinary_digest: str, shuttle_digest: str
) -> None:
    """Enforce the only policies admitted by comparison schema version 1."""
    if policy not in ("source_ordered", "fast"):
        raise ValueError(f"unknown policy: {policy}")
    if policy == "fast" and not identity_lowering:
        raise AssertionError("non-identity FAST requires a new reviewed comparison contract")
    if ordinary_digest != shuttle_digest:
        raise AssertionError(f"{policy} must be bitwise equal to matched ordinary JAX")


def _checked_metrics(metrics: Mapping[str, object]) -> dict[str, float | int]:
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise ValueError("metric fields drifted")
    checked: dict[str, float | int] = {}
    for name in METRICS[:-1]:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite nonnegative number")
        checked[name] = float(value)
    ulp = metrics["max_bfloat16_ulp_error"]
    if isinstance(ulp, bool) or not isinstance(ulp, int) or ulp < 0:
        raise ValueError("max_bfloat16_ulp_error must be a nonnegative integer")
    checked["max_bfloat16_ulp_error"] = ulp
    if checked["mean_absolute_error"] > checked["max_absolute_error"]:
        raise ValueError("mean_absolute_error exceeds max_absolute_error")
    return checked


def _closed(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields drifted")
    return value


def _equal(actual: object, expected: object, name: str) -> None:
    if not _exact_value(actual, expected):
        raise ValueError(f"{name} drifted")


def _exact_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, float):
        return math.isfinite(actual) and actual == expected
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_exact_value(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_value(actual_item, expected_item) for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
