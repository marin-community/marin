# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from experiments.post_training.task_curriculum.assignment import (
    AssignmentStatus,
    AssignmentThresholds,
    assign_hierarchically,
)
from experiments.post_training.task_curriculum.catalog import (
    CatalogError,
    CurriculumUnit,
    load_curriculum,
    load_macro_catalog,
)
from experiments.post_training.task_curriculum.evaluate import evaluate_catalog
from experiments.post_training.task_curriculum.rubric import RubricConfig, UnitEvidence, evaluate_unit
from experiments.post_training.task_curriculum.semantic_key import SemanticKey

CURRICULUM_ROOT = Path("experiments/post_training/task_curriculum")


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


def _macro_catalog(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "macros.json",
        {
            "schema_version": "task-curriculum-macro-areas-v1",
            "snapshot": "test-v1",
            "source_url": "https://example.test",
            "source_issue": "https://example.test/issue",
            "methodology": {
                "vocabulary_macro_area_count": 2,
                "active_macro_area_count": 2,
                "active_micro_area_count": 2,
                "tasktrove_source_count": 1,
                "tasktrove_row_count": 1,
                "count_semantics": "test",
            },
            "macro_areas": [
                {
                    "id": "C01",
                    "name": "Math",
                    "micro_areas": [
                        {
                            "id": "C01.1",
                            "name": "Arithmetic",
                            "taxonomy_unit_count": 1,
                            "credited_task_count": 1,
                            "mapped_source_count": 1,
                            "coverage_status": "covered",
                        }
                    ],
                },
                {
                    "id": "C02",
                    "name": "Writing",
                    "micro_areas": [
                        {
                            "id": "C02.1",
                            "name": "Editing",
                            "taxonomy_unit_count": 1,
                            "credited_task_count": 0,
                            "mapped_source_count": 0,
                            "coverage_status": "gap",
                        }
                    ],
                },
            ],
        },
    )


def _unit(unit_id: str = "add") -> CurriculumUnit:
    return CurriculumUnit(
        id=unit_id,
        name="Add integers",
        outcome="Add two integers",
        macro_area_id="C01",
        micro_area_ids=("C01.1",),
        includes=("integer addition",),
        excludes=("symbolic algebra",),
        prerequisites=(),
        positive_examples=("add 1 and 2", "add 3 and 4", "add 5 and 6"),
    )


def _unit_json(unit_id: str, prerequisites: list[str]) -> dict[str, object]:
    return {
        "id": unit_id,
        "name": f"Unit {unit_id}",
        "outcome": f"Complete unit {unit_id}",
        "macro_area_id": "C01",
        "micro_area_ids": ["C01.1"],
        "includes": [f"member of {unit_id}"],
        "excludes": [f"non-member of {unit_id}"],
        "prerequisites": prerequisites,
        "positive_examples": [f"{unit_id} example 1", f"{unit_id} example 2", f"{unit_id} example 3"],
    }


def _rubric() -> RubricConfig:
    return RubricConfig(
        version="test",
        full_support_examples=3,
        full_support_sources=2,
        partial_support_examples=2,
        partial_support_sources=1,
        full_generation_validity=0.95,
        partial_generation_validity=0.8,
        minimum_ready_total=10,
        require_full_evidence_support=True,
        require_full_generation_validity=True,
    )


def test_curriculum_rejects_micro_area_from_another_macro(tmp_path: Path) -> None:
    macro_catalog = load_macro_catalog(_macro_catalog(tmp_path))
    curriculum_path = _write_json(
        tmp_path / "curriculum.json",
        {
            "schema_version": "task-curriculum-v1",
            "version": "test",
            "macro_snapshot": "test-v1",
            "notes": [],
            "units": [
                {
                    "id": "add",
                    "name": "Add integers",
                    "outcome": "Add two integers",
                    "macro_area_id": "C01",
                    "micro_area_ids": ["C02.1"],
                    "includes": ["integer addition"],
                    "excludes": ["symbolic algebra"],
                    "prerequisites": [],
                    "positive_examples": ["add 1 and 2", "add 3 and 4", "add 5 and 6"],
                }
            ],
        },
    )

    with pytest.raises(CatalogError):
        load_curriculum(curriculum_path, macro_catalog)


def test_curriculum_rejects_prerequisite_cycle(tmp_path: Path) -> None:
    macro_catalog = load_macro_catalog(_macro_catalog(tmp_path))
    curriculum_path = _write_json(
        tmp_path / "curriculum.json",
        {
            "schema_version": "task-curriculum-v1",
            "version": "test",
            "macro_snapshot": "test-v1",
            "notes": [],
            "units": [_unit_json("a", ["b"]), _unit_json("b", ["a"])],
        },
    )

    with pytest.raises(CatalogError):
        load_curriculum(curriculum_path, macro_catalog)


@pytest.mark.parametrize(
    ("task_vector", "unit_anchors", "expected"),
    [
        ((1.0, 0.0), {"add": ((1.0, 0.0),)}, AssignmentStatus.ASSIGNED),
        ((1.0, 0.0), {}, AssignmentStatus.COVERAGE_GAP),
        ((0.0, 1.0), {"add": ((1.0, 0.0),)}, AssignmentStatus.OUTSIDE_INVENTORY),
    ],
)
def test_hierarchical_assignment_distinguishes_assignment_gaps_and_inventory(
    task_vector: tuple[float, float],
    unit_anchors: dict[str, tuple[tuple[float, float], ...]],
    expected: AssignmentStatus,
) -> None:
    assignment = assign_hierarchically(
        task_vector,
        task_vector,
        macro_anchors={"C01": ((1.0, 0.0),)},
        unit_anchors=unit_anchors,
        units_by_id={"add": _unit()},
        thresholds=AssignmentThresholds(
            max_macro_distance=0.4,
            min_macro_margin=0.0,
            max_unit_distance=0.2,
            min_unit_margin=0.0,
        ),
    )

    assert assignment.status == expected


def test_hierarchical_assignment_abstains_between_neighboring_units() -> None:
    units = {"add": _unit("add"), "subtract": _unit("subtract")}
    assignment = assign_hierarchically(
        (1.0, 0.0),
        (1.0, 0.0),
        macro_anchors={"C01": ((1.0, 0.0),)},
        unit_anchors={"add": ((1.0, 0.01),), "subtract": ((1.0, -0.01),)},
        units_by_id=units,
        thresholds=AssignmentThresholds(
            max_macro_distance=0.4,
            min_macro_margin=0.0,
            max_unit_distance=0.2,
            min_unit_margin=0.01,
        ),
    )

    assert assignment.status == AssignmentStatus.AMBIGUOUS
    assert assignment.macro_area_id == "C01"
    assert assignment.unit_id is None


def test_hierarchical_assignment_uses_separate_macro_and_unit_projections() -> None:
    units = {"add": _unit("add"), "subtract": _unit("subtract")}
    assignment = assign_hierarchically(
        macro_vector=(1.0, 0.0),
        unit_vector=(0.0, 1.0),
        macro_anchors={"C01": ((1.0, 0.0),)},
        unit_anchors={"add": ((1.0, 0.0),), "subtract": ((0.0, 1.0),)},
        units_by_id=units,
        thresholds=AssignmentThresholds(
            max_macro_distance=0.4,
            min_macro_margin=0.0,
            max_unit_distance=0.2,
            min_unit_margin=0.01,
        ),
    )

    assert assignment.status == AssignmentStatus.ASSIGNED
    assert assignment.macro_area_id == "C01"
    assert assignment.unit_id == "subtract"


def test_rubric_requires_cross_source_evidence_and_nontrivial_difficulty() -> None:
    result = evaluate_unit(
        UnitEvidence(
            unit_id="add",
            observable_outcome=2,
            distinct_boundary=2,
            generation_feasible=2,
            reviewed_examples=4,
            source_count=1,
            generated_valid=3,
            generated_total=3,
            solve_successes=6,
            solve_attempts=6,
            difficulty_ordering_plausible=True,
        ),
        _rubric(),
    )

    assert result.scores["evidence_support"] == 1
    assert result.scores["difficulty_gradient"] == 0
    assert not result.ready


def test_rubric_accepts_supported_valid_unit_with_mixed_solve_results() -> None:
    result = evaluate_unit(
        UnitEvidence(
            unit_id="add",
            observable_outcome=2,
            distinct_boundary=2,
            generation_feasible=2,
            reviewed_examples=4,
            source_count=2,
            generated_valid=3,
            generated_total=3,
            solve_successes=3,
            solve_attempts=6,
            difficulty_ordering_plausible=True,
        ),
        _rubric(),
    )

    assert result.total == 12
    assert result.ready


@pytest.mark.parametrize(
    "overrides",
    [
        {"reviewed_examples": 2, "source_count": 3},
        {"generated_valid": 4, "generated_total": 3},
        {"solve_successes": 7, "solve_attempts": 6},
    ],
)
def test_rubric_rejects_impossible_evidence_counts(overrides: dict[str, int]) -> None:
    values = {
        "unit_id": "add",
        "observable_outcome": 2,
        "distinct_boundary": 2,
        "generation_feasible": 2,
        "reviewed_examples": 4,
        "source_count": 2,
        "generated_valid": 3,
        "generated_total": 3,
        "solve_successes": 3,
        "solve_attempts": 6,
        "difficulty_ordering_plausible": True,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        evaluate_unit(UnitEvidence(**values), _rubric())


def test_catalog_evaluation_rejects_inconsistent_macro_totals(tmp_path: Path) -> None:
    evidence = json.loads((CURRICULUM_ROOT / "pilot_evaluation.json").read_text())
    evidence["macro_assignment"]["blind_reference"]["exact_macro"] += 1
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)

    with pytest.raises(ValueError):
        evaluate_catalog(
            CURRICULUM_ROOT / "macro_areas.json",
            CURRICULUM_ROOT / "math_v0.json",
            CURRICULUM_ROOT / "rubric_v1.json",
            evidence_path,
        )


def test_semantic_key_preserves_subject_for_macro_routing_and_operation_for_units() -> None:
    key = SemanticKey(
        summary="Find a triangle's inradius from its coordinates.",
        hardest_part="combine coordinate distances with the area-semiperimeter relation",
        required_operations=("compute side lengths", "compute area", "derive inradius"),
        subject_hint="math.geometry",
        answer_form="real number",
    )

    assert key.macro_embedding_text() == (
        "Subject: math.geometry. Task: Find a triangle's inradius from its coordinates. "
        "Central operation: combine coordinate distances with the area-semiperimeter relation"
    )
    assert key.unit_embedding_text() == "combine coordinate distances with the area-semiperimeter relation"
