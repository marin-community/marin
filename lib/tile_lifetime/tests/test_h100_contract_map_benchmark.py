# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tile_lifetime.h100_contract_map_benchmark import (
    KERNEL_RECORD_REQUIRED_FIELDS,
    NUMERICAL_OUTPUT_REQUIRED_FIELDS,
    NUMERICAL_OUTPUT_ROLES,
    PAIRWISE_DRIFT_REQUIRED_FIELDS,
    RAW_SAMPLE_REQUIRED_FIELDS,
    RESULT_EVIDENCE_SECTIONS,
    REVIEWED_NUMERICAL_FLOORS_SHA256,
    ArchitectureStatus,
    BackendVariant,
    ExternalComparator,
    MeasurementBoundary,
    RepeatabilityMode,
    StructuralFeature,
    comparator_decision,
    default_h100_contract_map_benchmark_plan,
    result_evidence_schema,
    staging_manifest,
    validate_result_evidence,
    validate_result_evidence_bundle,
)

_CANONICAL_REVISION = "ca2091a4b27a366c4f3625cd339b21e139886450"


def _complete_result_evidence() -> dict[str, Any]:
    schema = result_evidence_schema()
    kernel = {
        "name": "contract_map_kernel",
        "ptx_path": "artifacts/contract_map.ptx",
        "ptx_sha256": "1" * 64,
        "sass_path": "artifacts/contract_map.sass",
        "sass_sha256": "2" * 64,
        "registers_per_thread": 64,
        "spill_load_bytes": 0,
        "spill_store_bytes": 0,
        "static_shared_memory_bytes": 0,
        "dynamic_shared_memory_bytes": 32768,
        "block_size": [256, 1, 1],
        "active_blocks_per_sm": 2,
        "limiting_occupancy_resource": "shared_memory",
        "achieved_occupancy": 0.5,
    }
    drift = {
        "left_repeat_index": 0,
        "right_repeat_index": 1,
        "maximum_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "maximum_ulp_distance": 0,
        "mean_ulp_distance": 0.0,
    }
    output_metrics = {
        "maximum_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "maximum_ulp_distance": 0,
        "mean_ulp_distance": 0.0,
        "nonfinite_values": 0,
        "repeat_hashes": ["3" * 64, "3" * 64],
        "pairwise_drift": [drift],
    }
    raw_samples = []
    for schedule_row in schema["steady_state_schedule"]:
        measurements = {
            backend.value: {boundary.value: 1000 for boundary in MeasurementBoundary} for backend in BackendVariant
        }
        raw_samples.append(
            {
                "sample_index": schedule_row["sample_index"],
                "backend_order": schedule_row["backend_order"],
                "measurements_ns": measurements,
            }
        )
    return {
        "identity": {
            "case_id": default_h100_contract_map_benchmark_plan().cases[0].case_id,
            "backend": BackendVariant.ORDINARY_XLA.value,
            "measurement_boundary": MeasurementBoundary.KERNEL_ONLY.value,
        },
        "artifacts": {
            "final_optimized_hlo_path": "artifacts/final.hlo",
            "final_optimized_hlo_sha256": "4" * 64,
            "custom_call_manifest_path": "artifacts/custom_calls.json",
            "custom_call_manifest_sha256": "5" * 64,
        },
        "resources": {
            "kernel_records": [kernel],
            "launch_count": 1,
            "ordered_kernel_names": ["contract_map_kernel"],
        },
        "copies": {
            "device_to_device_count": 0,
            "device_to_device_bytes": 0,
            "host_to_device_count": 0,
            "host_to_device_bytes": 0,
            "unexpected_copy_count": 0,
        },
        "logical_boundary": {
            "input_layouts": ["row_major", "column_major"],
            "output_layouts": ["row_major"],
            "layout_adapters": [],
            "materialized_copies": [],
            "transposes": [],
            "bitcasts": [],
            "saved_state_names_and_bytes": {"map_activation": 4096},
            "recompute_operations": [],
        },
        "provenance": {
            "command": ["benchmark", "--plan", "plan.json"],
            "environment": {"CUDA_VISIBLE_DEVICES": "0"},
            "compiler_flags": ["--enable_shuttle"],
            "source_sha": _CANONICAL_REVISION,
            "persistent_cache_identity": "cache-root-a",
        },
        "numerical": {
            "reviewed_floors_sha256": REVIEWED_NUMERICAL_FLOORS_SHA256,
            "floors_passed_before_timing": True,
            "outputs": {role: copy.deepcopy(output_metrics) for role in NUMERICAL_OUTPUT_ROLES},
        },
        "timing": {
            "compile_samples_ns": [100],
            "first_execution_samples_ns": [80],
            "warmup_iterations": 10,
            "warmup_samples_ns": [40] * 10,
            "persistent_cache_cold_samples_ns": [70],
            "persistent_cache_hit_samples_ns": [10],
            "steady_state_schedule": schema["steady_state_schedule"],
            "raw_samples": raw_samples,
        },
    }


def _complete_result_evidence_bundle() -> tuple[dict[str, Any], ...]:
    payloads = []
    for case in default_h100_contract_map_benchmark_plan().cases:
        for backend in BackendVariant:
            for boundary in MeasurementBoundary:
                payload = _complete_result_evidence()
                payload["identity"] = {
                    "case_id": case.case_id,
                    "backend": backend.value,
                    "measurement_boundary": boundary.value,
                }
                payloads.append(payload)
    return tuple(payloads)


def test_default_plan_requires_three_backends_two_boundaries_and_anonymous_irregular_cases() -> None:
    plan = default_h100_contract_map_benchmark_plan()

    assert plan.architecture_status is ArchitectureStatus.NONCONFORMING
    assert plan.backends == tuple(BackendVariant)
    assert plan.boundaries == tuple(MeasurementBoundary)
    assert len(plan.cases) == 4
    assert len({case.case_id for case in plan.cases}) == 4
    assert all(case.case_id.startswith("contract_map_") for case in plan.cases)
    assert all(case.rows % 2 == 1 and case.reduction % 8 == 0 and case.features % 8 == 0 for case in plan.cases)


def test_plan_rejects_missing_resource_or_logical_boundary_evidence() -> None:
    plan = default_h100_contract_map_benchmark_plan()

    with pytest.raises(ValueError, match="resource evidence cannot omit"):
        replace(plan.resources, ptx=False)
    with pytest.raises(ValueError, match="logical-boundary evidence cannot omit"):
        replace(plan.logical_boundary, saved_state_names_and_bytes=False)


def test_dense_contract_map_plan_excludes_attention_and_routed_comparators() -> None:
    plan = default_h100_contract_map_benchmark_plan()

    fa4 = comparator_decision(ExternalComparator.FA4, plan.features)
    grug = comparator_decision(ExternalComparator.GRUG, plan.features)

    assert not fa4.admitted
    assert fa4.missing_features == (
        StructuralFeature.ATTENTION_SCORE,
        StructuralFeature.NORMALIZED_EXP,
        StructuralFeature.FOLD,
    )
    assert not grug.admitted
    assert grug.missing_features == (
        StructuralFeature.SEGMENTED_CONTRACT,
        StructuralFeature.RELATION,
        StructuralFeature.TRANSPORT,
    )


def test_staging_manifest_is_structural_and_records_every_counterbalanced_order() -> None:
    manifest = staging_manifest(shuttle_revision=_CANONICAL_REVISION)
    serialized = json.dumps(manifest, sort_keys=True)
    schedule = manifest["steady_state_schedule"]

    assert manifest["kind"] == "staged_plan_no_gpu_evidence"
    assert not manifest["execution_allowed"]
    assert len(manifest["counterbalanced_orders"]) == 6
    assert len({tuple(order) for order in manifest["counterbalanced_orders"]}) == 6
    assert len(schedule) == 24
    assert Counter(tuple(row["backend_order"]) for row in schedule) == {
        tuple(order): 4 for order in manifest["counterbalanced_orders"]
    }
    assert manifest["reviewed_numerical_floors_sha256"] == REVIEWED_NUMERICAL_FLOORS_SHA256
    assert "workload" not in serialized
    assert "model_name" not in serialized
    assert all(not decision["admitted"] for decision in manifest["external_comparators"])


def test_execute_gpu_refuses_in_fresh_process_before_package_or_jax_import(tmp_path: Path) -> None:
    import_marker = tmp_path / "forbidden-import.txt"
    output_path = tmp_path / "plan.json"
    (tmp_path / "sitecustomize.py").write_text(
        """
import os
import sys

class _RejectAcceleratorStack:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'jax', 'jaxlib', 'tile_lifetime'}:
            with open(os.environ['FORBIDDEN_IMPORT_MARKER'], 'w') as marker:
                marker.write(fullname)
            raise RuntimeError(f'forbidden preflight import: {fullname}')
        return None

sys.meta_path.insert(0, _RejectAcceleratorStack())
""".lstrip()
    )
    script = Path(__file__).parents[1] / "benchmarks" / "h100_contract_map_backend_evidence.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    environment["FORBIDDEN_IMPORT_MARKER"] = str(import_marker)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--shuttle-revision",
            _CANONICAL_REVISION,
            "--json-output",
            str(output_path),
            "--execute-gpu",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    assert "H100 execution is disabled" in result.stderr
    assert not import_marker.exists()
    assert not output_path.exists()


def test_plan_rejects_missing_backend_or_measurement_boundary() -> None:
    plan = default_h100_contract_map_benchmark_plan()

    with pytest.raises(ValueError, match="every plan must compare"):
        replace(plan, backends=plan.backends[:-1])
    with pytest.raises(ValueError, match="every plan must measure"):
        replace(plan, boundaries=plan.boundaries[:-1])


def test_timing_protocol_requires_full_permutation_balance_and_raw_cache_evidence() -> None:
    plan = default_h100_contract_map_benchmark_plan()
    timing = plan.timing

    with pytest.raises(ValueError, match="backend permutations"):
        replace(timing, steady_state_repeats=25)
    with pytest.raises(ValueError, match="cache isolation"):
        replace(timing, isolate_persistent_cache_roots=False)
    with pytest.raises(ValueError, match="raw samples"):
        replace(timing, retain_raw_samples=False)


def test_numerical_floor_rejects_posthoc_or_inconsistent_accuracy_bounds() -> None:
    floor = default_h100_contract_map_benchmark_plan().numerical_floors[1]
    assert floor.backend is BackendVariant.SHUTTLE_SOURCE_ORDERED
    assert floor.repeatability is RepeatabilityMode.BITWISE

    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(floor, maximum_absolute_error=-1.0)
    with pytest.raises(ValueError, match="mean absolute error"):
        replace(floor, mean_absolute_error=floor.maximum_absolute_error + 1.0)
    with pytest.raises(ValueError, match="mean ULP distance"):
        replace(floor, mean_ulp_distance=floor.maximum_ulp_distance + 1.0)
    with pytest.raises(ValueError, match="bitwise repeatability"):
        replace(floor, repeat_maximum_absolute_error=0.01)


def test_plan_rejects_valid_relaxed_replacement_for_reviewed_numerical_floor() -> None:
    plan = default_h100_contract_map_benchmark_plan()
    relaxed = replace(plan.numerical_floors[0], maximum_absolute_error=0.0625)

    with pytest.raises(ValueError, match="exact reviewed schema constant"):
        replace(plan, numerical_floors=(relaxed, *plan.numerical_floors[1:]))


@pytest.mark.parametrize(
    ("section", "field"),
    [(section.name, field) for section in RESULT_EVIDENCE_SECTIONS for field in section.required_fields],
)
def test_result_evidence_rejects_each_missing_section_field(section: str, field: str) -> None:
    payload = _complete_result_evidence()
    del payload[section][field]

    with pytest.raises(ValueError, match="missing required evidence fields"):
        validate_result_evidence(payload)


@pytest.mark.parametrize("section", [section.name for section in RESULT_EVIDENCE_SECTIONS])
def test_result_evidence_rejects_each_missing_section(section: str) -> None:
    payload = _complete_result_evidence()
    del payload[section]

    with pytest.raises(ValueError, match="missing required section"):
        validate_result_evidence(payload)


@pytest.mark.parametrize("field", KERNEL_RECORD_REQUIRED_FIELDS)
def test_result_evidence_rejects_each_missing_kernel_field(field: str) -> None:
    payload = _complete_result_evidence()
    del payload["resources"]["kernel_records"][0][field]

    with pytest.raises(ValueError, match="missing required evidence fields"):
        validate_result_evidence(payload)


@pytest.mark.parametrize("role", NUMERICAL_OUTPUT_ROLES)
@pytest.mark.parametrize("field", NUMERICAL_OUTPUT_REQUIRED_FIELDS)
def test_result_evidence_rejects_each_missing_numerical_output_field(role: str, field: str) -> None:
    payload = _complete_result_evidence()
    del payload["numerical"]["outputs"][role][field]

    with pytest.raises(ValueError, match="missing required evidence fields"):
        validate_result_evidence(payload)


@pytest.mark.parametrize("field", PAIRWISE_DRIFT_REQUIRED_FIELDS)
def test_result_evidence_rejects_each_missing_pairwise_drift_field(field: str) -> None:
    payload = _complete_result_evidence()
    del payload["numerical"]["outputs"]["forward"]["pairwise_drift"][0][field]

    with pytest.raises(ValueError, match="missing required evidence fields"):
        validate_result_evidence(payload)


@pytest.mark.parametrize("field", RAW_SAMPLE_REQUIRED_FIELDS)
def test_result_evidence_rejects_each_missing_raw_sample_field(field: str) -> None:
    payload = _complete_result_evidence()
    del payload["timing"]["raw_samples"][0][field]

    with pytest.raises(ValueError, match="missing required evidence fields"):
        validate_result_evidence(payload)


def test_result_evidence_accepts_complete_reviewed_payload() -> None:
    # A real collector crosses JSON before the public validator consumes its record.
    payload = json.loads(json.dumps(_complete_result_evidence()))

    validate_result_evidence(payload)


def test_result_evidence_schema_names_all_24_required_records() -> None:
    schema = result_evidence_schema()
    required_records = schema["required_result_records"]

    assert len(required_records) == 24
    assert required_records == [
        {"case_id": case.case_id, "backend": backend.value, "measurement_boundary": boundary.value}
        for case in default_h100_contract_map_benchmark_plan().cases
        for backend in BackendVariant
        for boundary in MeasurementBoundary
    ]


def test_result_evidence_bundle_requires_every_backend_and_boundary() -> None:
    payloads = _complete_result_evidence_bundle()
    validate_result_evidence_bundle(payloads)

    with pytest.raises(ValueError, match="all 24 reviewed case, backend, and boundary records"):
        validate_result_evidence_bundle(payloads[:-1])


def test_result_evidence_bundle_rejects_six_records_for_only_one_case() -> None:
    payloads = _complete_result_evidence_bundle()[: len(BackendVariant) * len(MeasurementBoundary)]

    with pytest.raises(ValueError, match="all 24 reviewed case, backend, and boundary records"):
        validate_result_evidence_bundle(payloads)


def test_result_evidence_rejects_relaxed_floor_digest_and_post_numerical_timing() -> None:
    payload = _complete_result_evidence()
    payload["numerical"]["reviewed_floors_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reviewed floor digest"):
        validate_result_evidence(payload)

    payload = _complete_result_evidence()
    payload["numerical"]["floors_passed_before_timing"] = False
    with pytest.raises(ValueError, match="must pass before timing"):
        validate_result_evidence(payload)


def test_result_evidence_rejects_incomplete_or_reordered_raw_schedule() -> None:
    payload = _complete_result_evidence()
    payload["timing"]["raw_samples"].pop()
    with pytest.raises(ValueError, match="all 24 scheduled rows"):
        validate_result_evidence(payload)

    payload = _complete_result_evidence()
    payload["timing"]["raw_samples"][0]["backend_order"] = list(reversed(tuple(BackendVariant)))
    with pytest.raises(ValueError, match="scheduled order"):
        validate_result_evidence(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_absolute_error", 0.03126),
        ("mean_absolute_error", 0.00201),
        ("maximum_ulp_distance", 5),
        ("mean_ulp_distance", 0.26),
        ("nonfinite_values", 1),
    ),
)
def test_result_evidence_rejects_measured_output_that_exceeds_backend_floor(field: str, value: float) -> None:
    payload = _complete_result_evidence()
    payload["numerical"]["outputs"]["dw1"][field] = value

    with pytest.raises(ValueError, match="immutable ordinary_xla numerical floor"):
        validate_result_evidence(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_result_evidence_rejects_nonfinite_measured_metrics(value: float) -> None:
    payload = _complete_result_evidence()
    payload["numerical"]["outputs"]["forward"]["maximum_absolute_error"] = value

    with pytest.raises(ValueError, match="finite nonnegative"):
        validate_result_evidence(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_absolute_error", 0.007813),
        ("mean_absolute_error", 0.000501),
        ("maximum_ulp_distance", 5),
        ("mean_ulp_distance", 0.26),
    ),
)
def test_result_evidence_rejects_repeat_drift_that_exceeds_backend_floor(field: str, value: float) -> None:
    payload = _complete_result_evidence()
    payload["numerical"]["outputs"]["dx"]["pairwise_drift"][0][field] = value

    with pytest.raises(ValueError, match="immutable ordinary_xla repeat floor"):
        validate_result_evidence(payload)


def test_result_evidence_rejects_source_ordered_repeat_content_drift() -> None:
    payload = _complete_result_evidence()
    payload["identity"]["backend"] = BackendVariant.SHUTTLE_SOURCE_ORDERED.value
    payload["numerical"]["outputs"]["forward"]["repeat_hashes"][1] = "6" * 64

    with pytest.raises(ValueError, match="bitwise-repeatability floor"):
        validate_result_evidence(payload)


def test_result_evidence_rejects_unexpected_physical_copy() -> None:
    payload = _complete_result_evidence()
    payload["copies"]["unexpected_copy_count"] = 1

    with pytest.raises(ValueError, match="unexpected_copy_count must be zero"):
        validate_result_evidence(payload)


@pytest.mark.parametrize(
    ("container_path", "field", "value"),
    (
        (("artifacts",), "final_optimized_hlo_path", ""),
        (("artifacts",), "custom_call_manifest_path", "manifest"),
        (("resources", "kernel_records", 0), "ptx_path", " "),
        (("resources", "kernel_records", 0), "sass_path", "artifacts/"),
    ),
)
def test_result_evidence_rejects_non_artifact_paths(
    container_path: tuple[str | int, ...], field: str, value: str
) -> None:
    payload = _complete_result_evidence()
    container: Any = payload
    for component in container_path:
        container = container[component]
    container[field] = value

    with pytest.raises(ValueError, match=r"canonical string|concrete artifact file"):
        validate_result_evidence(payload)


@pytest.mark.parametrize(
    ("container_path", "field", "value"),
    (
        (("artifacts",), "final_optimized_hlo_sha256", "4" * 63),
        (("artifacts",), "custom_call_manifest_sha256", "G" * 64),
        (("resources", "kernel_records", 0), "ptx_sha256", "A" * 64),
        (("resources", "kernel_records", 0), "sass_sha256", "2" * 65),
        (("numerical", "outputs", "forward", "repeat_hashes"), 0, "not-a-hash"),
    ),
)
def test_result_evidence_rejects_malformed_content_identity(
    container_path: tuple[str | int, ...], field: str | int, value: str
) -> None:
    payload = _complete_result_evidence()
    container: Any = payload
    for component in container_path:
        container = container[component]
    container[field] = value

    with pytest.raises(ValueError, match="lowercase hexadecimal characters"):
        validate_result_evidence(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("command", []),
        ("environment", {}),
        ("compiler_flags", []),
        ("source_sha", "0" * 39),
        ("persistent_cache_identity", ""),
    ),
)
def test_result_evidence_rejects_empty_or_malformed_provenance(field: str, value: object) -> None:
    payload = _complete_result_evidence()
    payload["provenance"][field] = value

    with pytest.raises(ValueError, match=r"nonempty|lowercase hexadecimal characters"):
        validate_result_evidence(payload)


@pytest.mark.parametrize(
    "field",
    (
        "compile_samples_ns",
        "first_execution_samples_ns",
        "warmup_samples_ns",
        "persistent_cache_cold_samples_ns",
        "persistent_cache_hit_samples_ns",
    ),
)
def test_result_evidence_rejects_empty_timing_sample_lists(field: str) -> None:
    payload = _complete_result_evidence()
    payload["timing"][field] = []

    with pytest.raises(ValueError, match="at least one timing sample"):
        validate_result_evidence(payload)
