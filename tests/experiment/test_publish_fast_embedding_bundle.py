# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from publish_fast_embedding_bundle import release_evidence_decision  # noqa: E402


def release_reports() -> tuple[dict, dict, dict, dict]:
    training = {
        "final_model_sha256": "model-sha",
        "validation_decision": {
            "semantic_mean_delta": 0.1,
            "gates": {"semantic": True, "rank": True},
            "passed": True,
        },
    }
    level = {"student_all_gates_passed": True}
    evaluation = {
        "student_config": "full",
        "student_training_name": "semantic",
        "student_rung": "50k",
        "student_model": "student",
        "documents": 10_000,
        "label_version": "adjudicated",
        "adjudication_review_url": "memory://adjudication.json",
        "source_metadata_used_as_quality_target": False,
        "model_metadata": {"student": {"final_model_sha256": "model-sha"}},
        "variants": {
            "compact": {
                "parent": level,
                "leaf": dict(level),
                "form": dict(level),
                "production_buckets": dict(level),
            }
        },
    }
    speed = {
        "mode": "cpu",
        "jax_backend": "cpu",
        "compute_dtype": "float32",
        "config_name": "full",
        "teacher": "semantic",
        "rung": "50k",
        "training_report": {"final_model_sha256": "model-sha"},
        "measurement_valid": True,
        "student_to_baseline_ratio": 0.85,
    }
    blind = {
        "student_model": "student",
        "claude_model": "claude-opus-5",
        "package_sha256": "package-sha",
        "overall": {"documents": 200, "release_gate_passed": True},
        "code": {"release_gate_passed": True},
        "non_english": {"release_gate_passed": True},
        "other": {"release_gate_passed": True},
    }
    return training, evaluation, speed, blind


def decision(*reports: dict) -> dict[str, bool]:
    return release_evidence_decision(
        *reports,
        config_name="full",
        training_name="semantic",
        rung="50k",
        student_model="student",
        blind_package_sha256="package-sha",
    )


def test_release_evidence_accepts_one_exact_student_that_passes_all_gates() -> None:
    assert all(decision(*release_reports()).values())


@pytest.mark.parametrize(
    ("report_index", "path"),
    [
        (0, ("validation_decision", "gates", "rank")),
        (0, ("validation_decision", "passed")),
        (1, ("variants", "compact", "production_buckets", "student_all_gates_passed")),
        (2, ("measurement_valid",)),
        (2, ("compute_dtype",)),
        (3, ("non_english", "release_gate_passed")),
    ],
)
def test_release_evidence_rejects_failed_training_bucket_speed_or_blind_gate(
    report_index: int, path: tuple[str, ...]
) -> None:
    reports = list(release_reports())
    value = reports[report_index]
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = False

    assert not all(decision(*reports).values())
