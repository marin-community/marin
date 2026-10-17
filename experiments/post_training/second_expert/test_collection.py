# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Captured document identity and the historical RS selector must survive consolidation."""

import hashlib
import json

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.second_expert.audit_collection import audit_one
from experiments.post_training.second_expert.collect_teacher import selected_prompts
from experiments.post_training.second_expert.grading import grade_response


def test_repeated_source_ids_keep_distinct_prompts_and_canonical_answers():
    requests = [
        {"custom_id": "doc-a", "source_id": "wrong-shared-id", "gold": "A", "b_request": {"messages": ["options-a"]}},
        {"custom_id": "doc-b", "source_id": "wrong-shared-id", "gold": "A", "b_request": {"messages": ["options-b"]}},
    ]
    canonical = [
        {"custom_id": "doc-a", "question_id": 11, "canonical_gold": "B", "group": "same-question", "category": "math"},
        {"custom_id": "doc-b", "question_id": 12, "canonical_gold": "C", "group": "same-question", "category": "math"},
    ]
    selected = selected_prompts(requests, canonical, ["doc-b", "doc-a"])
    assert [(row["custom_id"], row["source_id"], row["gold"]) for row in selected] == [
        ("doc-a", "11", "B"),
        ("doc-b", "12", "C"),
    ]
    assert [row["b_request"] for row in selected] == [row["b_request"] for row in requests]
    assert requests[1]["gold"] == "A"


def test_reasoning_first_match_is_primary_even_when_final_answer_differs():
    record = {
        "custom_id": "doc-a::0",
        "prompt_id": "doc-a",
        "status_code": 200,
        "response": {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "The answer is (B).\n<|end_think|>\nThe answer is (C)."},
                }
            ]
        },
    }
    grade = grade_response(record, "B")
    assert grade["correct"]
    assert grade["finish_reason"] == "length"
    assert grade["pred"] == "B"
    assert grade["content_only_pred"] == "C"
    assert grade["content_only_disagrees"]


def test_local_collection_audit_preserves_selected_wrong_final_and_response_hash(tmp_path):
    prompt = {"custom_id": "doc-a", "gold": "B", "b_request": {"messages": [{"role": "user", "content": "Q"}]}}
    record = {
        "custom_id": "doc-a::0",
        "prompt_id": "doc-a",
        "sample": 0,
        "request": {"messages": prompt["b_request"]["messages"]},
        "status_code": 200,
        "response": {
            "choices": [
                {
                    "finish_reason": "length",
                    "token_ids": [10, 20, 30],
                    "message": {"content": "The answer is (B).<|end_think|>The answer is (C)."},
                }
            ],
        },
    }
    name = hashlib.sha256(b"doc-a::0").hexdigest() + ".json"
    raw = (json.dumps(record) + "\n").encode()
    (tmp_path / "responses").mkdir()
    (tmp_path / "grades").mkdir()
    (tmp_path / "responses" / name).write_bytes(raw)
    (tmp_path / "grades" / name).write_text(json.dumps(grade_response(record, "B")))
    result = audit_one(StoragePath(str(tmp_path)), prompt, 0)
    assert result["correct"] and result["pred"] == "B"
    assert result["content_only_pred"] == "C"
    assert result["completion_tokens"] == 3
    assert result["response_sha256"] == hashlib.sha256(raw).hexdigest()
