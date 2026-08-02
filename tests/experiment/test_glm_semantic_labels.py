# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import glm_semantic_labels as semantic_labels  # noqa: E402
from glm_semantic_labels import (  # noqa: E402
    OTHER_BUCKET_ID,
    Assignment,
    completion,
    parse_buckets,
    parse_json_object,
    review_indices,
    source_quotas,
)


def test_source_quotas_are_balanced_and_exact() -> None:
    quotas = source_quotas([f"source-{index}" for index in range(146)], 1_000)

    assert sum(quotas.values()) == 1_000
    assert set(quotas.values()) == {6, 7}
    assert list(quotas.values()).count(7) == 124


def test_parse_json_object_accepts_plain_and_fenced_json() -> None:
    value = {"bucket": "SCIENCE"}

    assert parse_json_object(json.dumps(value)) == value
    assert parse_json_object(f"```json\n{json.dumps(value)}\n```") == value


def test_parse_buckets_rejects_duplicate_identifiers() -> None:
    row = {"bucket_id": OTHER_BUCKET_ID, "name": "Other", "definition": "Other", "include": [], "exclude": []}

    with pytest.raises(ValueError, match="duplicate"):
        parse_buckets({"buckets": [row, row]})


def test_review_indices_select_low_confidence_across_buckets() -> None:
    assignments = [
        Assignment(index, f"BUCKET_{index % 3}", [], "English", "article", confidence, "reason")
        for index, confidence in enumerate((0.9, 0.8, 0.7, 0.1, 0.2, 0.3))
    ]

    selected = review_indices(assignments, 3)

    assert set(selected) == {3, 4, 5}


def test_completion_doubles_token_limit_for_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    class Response:
        ok = True

        def json(self) -> dict:
            content = '{"answer":' if len(requests) < 3 else '{"answer": 42}'
            return {"choices": [{"message": {"content": content}, "finish_reason": "length"}]}

    def post(*args, **kwargs) -> Response:
        requests.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(semantic_labels.requests, "post", post)

    result = completion("http://server", [{"role": "user", "content": "prompt"}], max_tokens=128, seed=42)

    assert result == {"answer": 42}
    assert [request["max_tokens"] for request in requests] == [128, 256, 512]


def test_assignment_accepts_missing_optional_rationale(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "primary_bucket_id": "SCIENCE",
        "secondary_bucket_ids": [],
        "language": "English",
        "document_type": "article",
        "confidence": 0.8,
    }
    monkeypatch.setattr(semantic_labels, "completion", lambda *args, **kwargs: payload)
    document = semantic_labels.SampleDocument(0, "hash", "hidden", "standard", 0, "Science text")
    bucket = semantic_labels.Bucket("SCIENCE", "Science", "Science", [], [])

    assignment = semantic_labels.assign_document("http://server", document, [bucket])

    assert assignment.primary_bucket_id == "SCIENCE"
    assert assignment.rationale == ""
