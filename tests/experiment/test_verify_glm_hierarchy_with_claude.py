# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import verify_glm_hierarchy_with_claude as verifier  # noqa: E402
from verify_glm_hierarchy_with_claude import (  # noqa: E402
    REVIEW_CHUNK_MARKER,
    claude_assignments,
    comparison,
    parse_claude_envelope,
    review_indices,
    review_package_from_chunks,
)


def test_claude_assignments_corrects_invalid_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    package = {
        "taxonomy": {
            "parents": [{"bucket_id": "SCIENCE"}, {"bucket_id": "ARTS"}],
            "leaves": [
                {"bucket_id": "BIOLOGY", "parent_id": "SCIENCE"},
                {"bucket_id": "FICTION", "parent_id": "ARTS"},
            ],
            "forms": [{"bucket_id": "RESEARCH"}],
        },
        "documents": [{"sample_index": 1, "text": "Biology"}],
    }
    invalid = {
        "sample_index": 1,
        "primary_parent_id": "SCIENCE",
        "secondary_parent_ids": [],
        "primary_leaf_id": "BIOLOGY",
        "secondary_leaf_ids": ["FICTION"],
        "form_id": "RESEARCH",
        "confidence": 0.8,
        "rationale": "Biology.",
    }
    corrected = invalid | {"secondary_parent_ids": ["ARTS"]}
    prompts = []

    def run(*args, **kwargs):
        prompts.append(kwargs["input"])
        assignment = invalid if len(prompts) == 1 else corrected
        output = json.dumps(
            {
                "is_error": False,
                "result": json.dumps({"assignments": [assignment]}),
                "modelUsage": {"claude-opus-5": {"inputTokens": 10}},
                "total_cost_usd": 0.25,
            }
        )
        return verifier.subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", run)

    review = claude_assignments(package, "claude-opus-5", batch_size=20, max_budget_usd=2)

    assert review.assignments == [corrected]
    assert review.cost_usd == 0.5
    assert "secondary leaf under an unselected parent" in prompts[1]


def test_review_indices_keep_representative_and_stress_samples_separate() -> None:
    assignments = [
        {"sample_index": index, "confidence": confidence} for index, confidence in enumerate([0.8, 0.1, 0.7, 0.2, 0.6])
    ]

    samples = review_indices(assignments, representative_size=2, stress_size=2)

    assert len(samples["representative"]) == 2
    assert len(samples["stress"]) == 2
    assert set(samples["representative"]).isdisjoint(samples["stress"])
    remaining = set(range(5)) - set(samples["representative"])
    assert samples["stress"] == sorted(remaining, key=lambda index: assignments[index]["confidence"])[:2]


def test_comparison_measures_parent_leaf_and_form_agreement() -> None:
    package = {
        "taxonomy": {
            "parents": [{"bucket_id": "SCIENCE"}, {"bucket_id": "ARTS"}],
            "leaves": [
                {"bucket_id": "BIOLOGY", "parent_id": "SCIENCE"},
                {"bucket_id": "FICTION", "parent_id": "ARTS"},
            ],
            "forms": [{"bucket_id": "RESEARCH"}, {"bucket_id": "NARRATIVE"}],
        },
        "documents": [{"sample_index": 1, "text": "a"}, {"sample_index": 2, "text": "b"}],
        "samples": {"representative": [1], "stress": [2]},
        "glm_assignments": [
            {
                "sample_index": 1,
                "primary_parent_id": "SCIENCE",
                "secondary_parent_ids": [],
                "primary_leaf_id": "BIOLOGY",
                "secondary_leaf_ids": [],
                "form_id": "RESEARCH",
                "confidence": 0.9,
                "rationale": "Biology paper.",
            },
            {
                "sample_index": 2,
                "primary_parent_id": "ARTS",
                "secondary_parent_ids": ["SCIENCE"],
                "primary_leaf_id": "FICTION",
                "secondary_leaf_ids": ["BIOLOGY"],
                "form_id": "NARRATIVE",
                "confidence": 0.6,
                "rationale": "Story.",
            },
        ],
    }
    claude_rows = [
        {
            "sample_index": 1,
            "primary_parent_id": "SCIENCE",
            "secondary_parent_ids": [],
            "primary_leaf_id": "BIOLOGY",
            "secondary_leaf_ids": [],
            "form_id": "RESEARCH",
            "confidence": 0.8,
            "rationale": "Paper.",
        },
        {
            "sample_index": 2,
            "primary_parent_id": "SCIENCE",
            "secondary_parent_ids": ["ARTS"],
            "primary_leaf_id": "BIOLOGY",
            "secondary_leaf_ids": ["FICTION"],
            "form_id": "RESEARCH",
            "confidence": 0.5,
            "rationale": "Scientific fiction.",
        },
    ]

    result = comparison(package, claude_rows)

    assert result["representative"]["primary_parent_exact_agreement"] == 1.0
    assert result["representative"]["form_exact_agreement"] == 1.0
    assert result["stress"]["primary_parent_exact_agreement"] == 0.0
    assert result["stress"]["any_parent_overlap_fraction"] == 1.0
    assert result["stress"]["any_leaf_overlap_fraction"] == 1.0
    assert result["stress"]["form_exact_agreement"] == 0.0


def test_comparison_rejects_leaf_under_wrong_parent() -> None:
    package = {
        "taxonomy": {
            "parents": [{"bucket_id": "SCIENCE"}, {"bucket_id": "ARTS"}],
            "leaves": [{"bucket_id": "BIOLOGY", "parent_id": "SCIENCE"}],
            "forms": [{"bucket_id": "RESEARCH"}],
        },
        "documents": [{"sample_index": 1, "text": "a"}],
        "samples": {"representative": [1]},
        "glm_assignments": [{"sample_index": 1}],
    }
    claude_rows = [
        {
            "sample_index": 1,
            "primary_parent_id": "ARTS",
            "secondary_parent_ids": [],
            "primary_leaf_id": "BIOLOGY",
            "secondary_leaf_ids": [],
            "form_id": "RESEARCH",
            "confidence": 0.8,
            "rationale": "Test.",
        }
    ]

    with pytest.raises(ValueError, match="wrong parent"):
        comparison(package, claude_rows)


def test_parse_claude_envelope_records_exact_model_and_cost() -> None:
    output = json.dumps(
        {
            "is_error": False,
            "result": '{"assignments":[{"sample_index":1}]}',
            "modelUsage": {"claude-opus-5": {"inputTokens": 10}},
            "total_cost_usd": 0.25,
        }
    )

    review = parse_claude_envelope(output, "claude-opus-5")

    assert review.assignments == [{"sample_index": 1}]
    assert review.model_usage_batches == [{"claude-opus-5": {"inputTokens": 10}}]
    assert review.cost_usd == 0.25


def test_review_package_from_chunks_accepts_task_prefixes_and_reorders_chunks() -> None:
    package = {"documents": [{"sample_index": 1, "text": "private"}]}
    encoded = base64.b64encode(gzip.compress(json.dumps(package).encode())).decode()
    split = len(encoded) // 2
    output = "\n".join(
        [
            f"task prefix {REVIEW_CHUNK_MARKER}0001/0002:{encoded[split:]}",
            f"task prefix {REVIEW_CHUNK_MARKER}0000/0002:{encoded[:split]}",
        ]
    )

    assert review_package_from_chunks(output) == package
