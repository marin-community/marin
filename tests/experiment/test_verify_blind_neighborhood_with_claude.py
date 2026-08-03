# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents/projects/luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import verify_blind_neighborhood_with_claude as verifier  # noqa: E402
from export_blind_neighborhood_review import REVIEW_CHUNK_MARKER  # noqa: E402
from verify_blind_neighborhood_with_claude import (  # noqa: E402
    ClaudeNeighborhoodReview,
    comparison,
    load_review_checkpoint,
    package_from_chunks,
    public_items,
    write_review_checkpoint,
    write_review_result,
)


def package() -> dict:
    return {
        "reference_model": "teacher",
        "student_model": "student",
        "items": [
            {
                "sample_index": 1,
                "query": "query one",
                "sets": {"A": ["neighbor one"], "B": ["neighbor two"]},
                "student_side": "A",
                "glm_primary_parent_id": "P",
                "glm_form_id": "CODE",
            },
            {
                "sample_index": 2,
                "query": "query two",
                "sets": {"A": ["neighbor three"], "B": ["neighbor four"]},
                "student_side": "B",
                "glm_primary_parent_id": "Q",
                "glm_form_id": "GENERAL_PROSE",
            },
        ],
    }


def test_package_chunks_round_trip_out_of_order() -> None:
    encoded = base64.b64encode(gzip.compress(json.dumps(package()).encode())).decode()
    split = len(encoded) // 2
    output = "\n".join(
        (
            f"{REVIEW_CHUNK_MARKER}0001/0002:{encoded[split:]}",
            f"{REVIEW_CHUNK_MARKER}0000/0002:{encoded[:split]}",
        )
    )

    assert package_from_chunks(output) == package()


def test_public_items_remove_hidden_model_and_label_truth() -> None:
    items = public_items(package()["items"])

    assert set(items[0]) == {"sample_index", "query", "set_a", "set_b"}
    assert "student" not in json.dumps(items)
    assert "glm_" not in json.dumps(items)


def test_review_checkpoint_round_trip_and_input_binding(tmp_path: Path) -> None:
    review_package = package()
    checkpoint = tmp_path / "review.json"
    decisions = [
        {"sample_index": 1, "choice": "A", "query_language": "en", "code_central": True, "rationale": "x"},
        {"sample_index": 2, "choice": "B", "query_language": "fr", "code_central": False, "rationale": "y"},
    ]
    review = ClaudeNeighborhoodReview(decisions, [{"claude-opus-5": {"inputTokens": 10}}], 0.25)

    write_review_checkpoint(checkpoint, review_package, "claude-opus-5", 10, review)

    assert load_review_checkpoint(checkpoint, review_package, "claude-opus-5", 10) == review
    assert load_review_checkpoint(checkpoint, review_package, "claude-opus-5", 1) == review
    changed = review_package | {"items": [review_package["items"][0] | {"query": "changed"}, review_package["items"][1]]}
    with pytest.raises(ValueError, match="different review inputs"):
        load_review_checkpoint(checkpoint, changed, "claude-opus-5", 10)


def test_write_review_result_replaces_complete_json(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_text("stale")

    write_review_result(output, {"rationale": "cohérent", "score": 0.75})

    assert json.loads(output.read_text()) == {"rationale": "cohérent", "score": 0.75}
    assert not output.with_name("result.json.tmp").exists()


def test_claude_decisions_corrects_an_incomplete_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    review_package = package()
    first = {
        "sample_index": 1,
        "choice": "A",
        "query_language": "en",
        "code_central": True,
        "rationale": "x",
    }
    second = {
        "sample_index": 2,
        "choice": "B",
        "query_language": "fr",
        "code_central": False,
        "rationale": "y",
    }
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs["input"])
        decisions = [first] if len(calls) == 1 else [first, second]
        output = json.dumps(
            {
                "is_error": False,
                "result": json.dumps({"decisions": decisions}),
                "modelUsage": {"claude-opus-5": {"inputTokens": 10}},
                "total_cost_usd": 0.25,
            }
        )
        return verifier.subprocess.CompletedProcess(args[0], 0, stdout=output, stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", run)

    review = verifier.claude_decisions(review_package, "claude-opus-5", batch_size=10, max_budget_usd=2)

    assert review.decisions == [first, second]
    assert review.cost_usd == 0.5
    assert "sample indices differ" in calls[1]


def test_comparison_scores_randomized_student_sides_and_content_groups() -> None:
    decisions = [
        {"sample_index": 1, "choice": "A", "query_language": "en", "code_central": True, "rationale": "x"},
        {"sample_index": 2, "choice": "TIE", "query_language": "fr", "code_central": False, "rationale": "y"},
    ]

    result = comparison(package(), decisions)

    assert result["package_sha256"] == verifier.review_package_sha256(package())
    assert result["overall"]["student_wins"] == 1
    assert result["overall"]["ties"] == 1
    assert result["overall"]["student_win_plus_half_tie_fraction"] == pytest.approx(0.75)
    assert isinstance(result["overall"]["release_gate_passed"], bool)
    assert result["code"]["documents"] == 1
    assert result["non_english"]["documents"] == 1
