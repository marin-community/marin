# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import pytest

from shuttle.evaluation_manifest import (
    ArchitectureConformance,
    EvaluationManifestError,
    Hardware,
    Status,
    load_evaluation_manifest,
    validate_evaluation_manifest,
)

SOURCE_ROOT = Path(__file__).parents[3]
MANIFEST = SOURCE_ROOT / ".agents/projects/tile_lifetime_compiler/shuttle_evaluation_manifest_v1.json"
BASELINE_COMMIT = "92b5dd7da84492d3766a51b8cb915a08b2a9e56a"


def _document() -> dict[str, object]:
    return json.loads(MANIFEST.read_text())


def test_checked_in_scorecard_covers_the_complete_current_matrix_without_promoting_gpu_prototypes() -> None:
    manifest = load_evaluation_manifest(MANIFEST, source_root=SOURCE_ROOT)

    assert manifest.baseline_commit == BASELINE_COMMIT
    assert {target.id for target in manifest.targets} == set(range(11))
    assert len(manifest.cells) == 116
    assert {target.id: target.status for target in manifest.targets} == {
        0: Status.PARTIAL,
        1: Status.BLOCKED,
        2: Status.BLOCKED,
        3: Status.BLOCKED,
        4: Status.BLOCKED,
        5: Status.BLOCKED,
        6: Status.BLOCKED,
        7: Status.BLOCKED,
        8: Status.BLOCKED,
        9: Status.BLOCKED,
        10: Status.BLOCKED,
    }
    target_zero_cells = [cell for cell in manifest.cells if cell.identity.target_id == 0]
    cpu_cells = [cell for cell in target_zero_cells if cell.identity.hardware is Hardware.CPU]
    assert len(cpu_cells) == 4
    assert all(cell.architecture_status is Status.ACCEPTED for cell in cpu_cells)
    assert all(cell.numerical_status is Status.PARTIAL for cell in cpu_cells)
    assert all(cell.performance_status is Status.NOT_STARTED for cell in cpu_cells)
    assert all(cell.status is Status.PARTIAL for cell in cpu_cells)
    excluded_evidence = manifest.excluded_evidence[0].evidence
    assert excluded_evidence.architecture_conformance is ArchitectureConformance.NONCONFORMING
    assert excluded_evidence.source_commit == BASELINE_COMMIT
    assert excluded_evidence.record_commit == BASELINE_COMMIT
    assert all(excluded_evidence.id not in cell.evidence_ids for cell in manifest.cells)
    target_one = next(target for target in manifest.targets if target.id == 1)
    target_one_shape = next(shape for shape in manifest.shapes if shape.id == "rmsnorm_bf16_2048x4096")
    target_one_cells = [cell for cell in manifest.cells if cell.identity.target_id == 1]
    assert target_one.required_shapes == ("rmsnorm_bf16_2048x4096",)
    assert target_one_shape.declaration_status is Status.ACCEPTED
    assert target_one_shape.specification == (
        "x:bf16[2048,4096], gamma:bf16[4096], dy:bf16[2048,4096], y:bf16[2048,4096], "
        "dx:bf16[2048,4096], dgamma:bf16[4096], epsilon:f32=1e-5, layout:row_major_contiguous"
    )
    assert len(target_one_cells) == 12
    assert {cell.identity.shape for cell in target_one_cells} == {"rmsnorm_bf16_2048x4096"}
    assert {cell.identity.boundary.value for cell in target_one_cells} == {
        "forward",
        "backward",
        "composed_forward_backward",
    }
    assert {cell.identity.policy.value for cell in target_one_cells} == {"source_ordered", "fast"}
    assert {cell.identity.hardware.value for cell in target_one_cells} == {"h100", "gb200_or_b200"}
    assert all(cell.status is Status.BLOCKED for cell in target_one_cells)
    assert all(cell.architecture_status is Status.NOT_STARTED for cell in target_one_cells)
    assert all(cell.numerical_status is Status.NOT_STARTED for cell in target_one_cells)
    assert all(cell.performance_status is Status.NOT_STARTED for cell in target_one_cells)
    assert all(
        {blocker.value for blocker in cell.blockers} == {"architecture_path_unavailable", "oracle_not_pinned"}
        for cell in target_one_cells
    )
    assert all("pending" not in cell.identity.shape for cell in target_one_cells)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "complete"),
        ("architecture_status", "passing"),
        ("numerical_status", "passing"),
        ("performance_status", "passing"),
        ("boundary", "inference"),
        ("policy", "approximate"),
        ("hardware", "future_gpu"),
    ],
)
def test_scorecard_rejects_unknown_cell_enums(field: str, value: str) -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell[field] = value

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_duplicate_cell_identity() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cells.append(copy.deepcopy(cells[0]))

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_missing_cell_dimension() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    del cell["shape"]

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_an_incomplete_required_cross_product() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cells.pop()

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


@pytest.mark.parametrize(("dimension", "value"), [("required_boundaries", ["forward"]), ("required_hardware", ["cpu"])])
def test_scorecard_rejects_a_self_consistent_shrink_of_normative_target_dimensions(
    dimension: str, value: list[str]
) -> None:
    document = _document()
    targets = document["targets"]
    cells = document["cells"]
    assert isinstance(targets, list)
    assert isinstance(cells, list)
    target = targets[0]
    assert isinstance(target, dict)
    target[dimension] = value
    if dimension == "required_boundaries":
        document["cells"] = [
            cell
            for cell in cells
            if not isinstance(cell, dict) or cell["target_id"] != 0 or cell["boundary"] == "forward"
        ]
    else:
        document["cells"] = [
            cell for cell in cells if not isinstance(cell, dict) or cell["target_id"] != 0 or cell["hardware"] == "cpu"
        ]

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_a_weakened_performance_policy() -> None:
    document = _document()
    thresholds = document["acceptance_thresholds"]
    assert isinstance(thresholds, dict)
    thresholds["oracle_latency_ratio"] = 100.0
    thresholds["stretch_oracle_latency_ratio"] = 99.0

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_cell_acceptance_without_all_gate_acceptance() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["status"] = "accepted"

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_target_acceptance_until_every_required_cell_is_accepted() -> None:
    document = _document()
    targets = document["targets"]
    assert isinstance(targets, list)
    target = targets[0]
    assert isinstance(target, dict)
    target["status"] = "accepted"

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_evidence_from_a_different_cell_identity() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = next(item for item in cells if isinstance(item, dict) and item["hardware"] == "h100")
    cell["evidence_ids"] = ["cpu_ordinary_jax_acceptance6"]

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_performance_acceptance_without_a_measured_artifact_claim() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["status"] = "accepted"
    cell["architecture_status"] = "accepted"
    cell["numerical_status"] = "accepted"
    cell["performance_status"] = "accepted"
    cell["blockers"] = []

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_promotion_beyond_the_artifact_derived_gate_status() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["numerical_status"] = "accepted"

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_an_accepted_cell_that_retains_blockers() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["status"] = "accepted"
    cell["architecture_status"] = "accepted"
    cell["numerical_status"] = "accepted"
    cell["performance_status"] = "accepted"

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_nonaccepted_gate_claims_without_evidence_or_consistent_aggregate_status() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["status"] = "not_started"
    cell["architecture_status"] = "partial"
    cell["numerical_status"] = "failed"
    cell["performance_status"] = "blocked"
    cell["evidence_ids"] = []
    cell["blockers"] = []

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_architecture_nonconforming_evidence_from_a_cell() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["evidence_ids"] = ["direct_h100_contract_map_backend"]

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_nonconforming_evidence_in_the_active_registry() -> None:
    document = _document()
    evidence = document["evidence"]
    assert isinstance(evidence, list)
    item = evidence[0]
    assert isinstance(item, dict)
    item["architecture_conformance"] = "nonconforming"

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_unknown_evidence_formats() -> None:
    document = _document()
    evidence = document["evidence"]
    assert isinstance(evidence, list)
    item = evidence[0]
    assert isinstance(item, dict)
    item["format"] = "unreviewed_result"

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_evidence_with_fabricated_commit_provenance() -> None:
    document = _document()
    evidence = document["evidence"]
    assert isinstance(evidence, list)
    item = evidence[0]
    assert isinstance(item, dict)
    item["record_commit"] = "0" * 40

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_artifact_content_that_does_not_match_its_digest() -> None:
    document = _document()
    evidence = document["evidence"]
    assert isinstance(evidence, list)
    item = evidence[0]
    assert isinstance(item, dict)
    item["sha256"] = "0" * 64

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_scorecard_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    duplicate = MANIFEST.read_text().replace(
        '  "schema_version": 1,',
        '  "schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    manifest = tmp_path / "duplicate.json"
    manifest.write_text(duplicate)

    with pytest.raises(EvaluationManifestError):
        load_evaluation_manifest(manifest, source_root=SOURCE_ROOT)


def test_scorecard_rejects_execution_against_an_undeclared_representative_shape() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = next(item for item in cells if isinstance(item, dict) and item["target_id"] == 2)
    cell["status"] = "partial"
    cell["evidence_ids"] = ["cpu_ordinary_jax_acceptance6"]

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


@pytest.mark.parametrize(
    ("dimension", "retained"),
    [
        ("required_boundaries", {"forward", "backward"}),
        ("required_policies", {"source_ordered"}),
        ("required_hardware", {"h100"}),
        ("required_shapes", set()),
    ],
)
def test_target_one_matrix_rejects_a_self_consistent_dimension_shrink(dimension: str, retained: set[str]) -> None:
    document = _document()
    targets = document["targets"]
    cells = document["cells"]
    assert isinstance(targets, list)
    assert isinstance(cells, list)
    target = next(item for item in targets if isinstance(item, dict) and item["id"] == 1)
    target[dimension] = [value for value in target[dimension] if value in retained]
    cell_field = {
        "required_boundaries": "boundary",
        "required_policies": "policy",
        "required_hardware": "hardware",
        "required_shapes": "shape",
    }[dimension]
    document["cells"] = [
        cell for cell in cells if not isinstance(cell, dict) or cell["target_id"] != 1 or cell[cell_field] in retained
    ]

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_target_one_matrix_rejects_a_self_consistent_shape_change() -> None:
    document = _document()
    shapes = document["shapes"]
    targets = document["targets"]
    cells = document["cells"]
    assert isinstance(shapes, list)
    assert isinstance(targets, list)
    assert isinstance(cells, list)
    shape = next(item for item in shapes if isinstance(item, dict) and item["id"] == "rmsnorm_bf16_2048x4096")
    target = next(item for item in targets if isinstance(item, dict) and item["id"] == 1)
    shape["id"] = "rmsnorm_bf16_7x13"
    target["required_shapes"] = ["rmsnorm_bf16_7x13"]
    for cell in cells:
        if isinstance(cell, dict) and cell["target_id"] == 1:
            cell["shape"] = "rmsnorm_bf16_7x13"

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_declared_shape_rejects_a_representative_shape_blocker() -> None:
    document = _document()
    cells = document["cells"]
    assert isinstance(cells, list)
    cell = next(item for item in cells if isinstance(item, dict) and item["target_id"] == 1)
    cell["blockers"].append("representative_shape_not_declared")

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)


def test_declared_target_rejects_a_representative_shape_blocker() -> None:
    document = _document()
    targets = document["targets"]
    assert isinstance(targets, list)
    target = next(item for item in targets if isinstance(item, dict) and item["id"] == 1)
    target["blockers"].append("representative_shape_not_declared")

    with pytest.raises(EvaluationManifestError):
        validate_evaluation_manifest(document, source_root=SOURCE_ROOT)
