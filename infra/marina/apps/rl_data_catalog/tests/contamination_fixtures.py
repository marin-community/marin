# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lexical evidence cases shared by loader and installed-app browser regressions."""

from typing import Any

from finestore.rl.curation import LexicalContaminationEvidence, SourceDeclaration

from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

PIN = "a" * 40


def contamination_declarations() -> list[tuple[str, SourceDeclaration]]:
    clean = decontamination_evidence("fixture", PIN, 3).model_dump(mode="json")
    flagged = {**clean, "contaminated_tasks": 1, "contaminated_fraction": 1 / 3, "receipt": None}
    remediated: dict[str, Any] = {
        **flagged,
        "receipt": {
            "use": "train",
            "status": "remediated",
            "retained_tasks": 2,
            "dropped_tasks": 1,
            "release": {
                **clean["receipt"]["release"],
                "drop_manifest_uri": "s3://test/excluded.json",
                "original_private_dataset_id": "original",
                "original_private_dataset_revision": "b" * 40,
            },
        },
    }
    cases: list[tuple[str, str, dict[str, Any] | None, bool, str]] = [
        ("unmeasured", "NOT MEASURED", None, False, PIN),
        ("clean", "NO ISSUES", clean, False, PIN),
        ("remediated", "REMEDIATED", remediated, False, PIN),
        ("eval", "MEASURED, EVAL", flagged, True, PIN),
        ("required", "REMEDIATION REQUIRED", flagged, False, PIN),
        ("stale-pin", "REMEDIATION REQUIRED", remediated, False, "c" * 40),
        ("eval-zero", "NO ISSUES", {**clean, "receipt": None}, True, PIN),
        (
            "public-release",
            "REMEDIATION REQUIRED",
            {
                **remediated,
                "receipt": {
                    **remediated["receipt"],
                    "release": {
                        **remediated["receipt"]["release"],
                        "private": False,
                    },
                },
            },
            False,
            PIN,
        ),
        (
            "receipt-eval",
            "MEASURED, EVAL",
            {
                **flagged,
                "receipt": {
                    "use": "eval",
                    "status": "measured (benchmark, not remediated)",
                    "retained_tasks": 3,
                    "dropped_tasks": 0,
                    "release": None,
                },
            },
            False,
            PIN,
        ),
    ]
    declarations = []
    for name, expected, evidence, benchmark, revision in cases:
        measured = LexicalContaminationEvidence.model_validate(evidence) if evidence is not None else None
        count = measured.receipt.retained_tasks if measured is not None and measured.receipt is not None else 3
        declarations.append(
            (
                expected,
                SourceDeclaration(
                    id=f"Marin:{name}",
                    name=name,
                    display_name=name,
                    origin="Marin",
                    build_path="fixture",
                    upstream_dataset=name,
                    upstream_name=name,
                    family="fixture",
                    tags=[],
                    task_count=count,
                    notes="Lexical evidence fixture",
                    quality_result=(
                        {
                            "reviewed": 2,
                            "issue_count": 0,
                            "issues": [],
                            "repair_recommendation": "Continue content review",
                        }
                        if name == "eval"
                        else None
                    ),
                    dataset_id="fixture",
                    dataset_revision=revision,
                    url=None,
                    type="RLVR",
                    turns="Single-turn",
                    is_benchmark=benchmark,
                    kind="Dataset",
                    status="Available",
                    pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": "1", "files": []},
                    daytona_safety=None,
                    current_release={"repo": "fixture", "revision": revision, "url": "https://example.org/pin"},
                    within_source_dedup=deduplication_evidence(
                        "fixture",
                        revision,
                        "1",
                        count,
                    ),
                    lexical_contamination=measured,
                ),
            )
        )
    return declarations
