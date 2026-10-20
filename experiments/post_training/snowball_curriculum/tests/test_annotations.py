# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import cast

from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.review_requests import RequestObservation, RequestOutput

from experiments.post_training.snowball_curriculum.annotations import (
    AnnotationModel,
    annotate_inputs,
    public_input,
    quality_input,
)


def test_public_annotation_projection_excludes_private_contract():
    task = numeric_answer_task(
        RawRow(id="example", source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"), data={}),
        prompt="Calculate the sum of two and three.",
        answer="987654321",
        tolerance_abs=0,
        tolerance_rel=0,
    )
    task = cast(TaskSpec, task)
    private = quality_input(task)
    public = public_input(task, "fixture")
    assert "987654321" in private
    assert "987654321" not in public["public_input"]
    assert "grader" not in json.loads(public["public_input"])
    assert "Calculate the sum" in public["public_input"]


def test_annotation_cache_reuses_results_and_schema_failures_are_repaired(tmp_path):
    model = AnnotationModel(model="fixture-flash", revision="snapshot-1", endpoint="http://fixture/v1")

    def submit(requests):
        observations = []
        for request in requests:
            repaired = len(request["body"]["messages"]) > 2
            value = {
                "summary": (
                    (
                        "Elementary arithmetic addition of two integer values using the supplied operands, "
                        "returning a single numerical result under the public answer contract."
                    )
                    if repaired
                    else "Too short."
                ),
                "domain_tags": ["arithmetic"],
                "skill_tags": ["addition"],
                "interaction_mode": "text",
                "difficulty": "easy",
                "difficulty_rationale": "Single addition operation",
                "difficulty_confidence": 0.8,
            }
            body = {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "type": "function",
                                    "function": {"name": "annotate_task", "arguments": json.dumps(value)},
                                }
                            ],
                        },
                    }
                ]
            }
            observations.append(
                RequestObservation(
                    (request["custom_id"],),
                    json.dumps({"custom_id": request["custom_id"], "response": {"status_code": 200, "body": body}}),
                )
            )
        return RequestOutput(tuple(requests), observations)

    initial, traces = annotate_inputs(
        {"a": "Add integers", "b": "Sum numbers"},
        stage="annotation",
        model=model,
        submit=submit,
        cache_root=str(tmp_path / "cache"),
    )
    assert set(initial) == {"a", "b"}
    assert all(row["difficulty"] == "easy" for row in initial.values())
    assert len(traces) == 2
    assert all(15 <= len(row["summary"].split()) <= 40 for row in initial.values())

    repeated, repeated_traces = annotate_inputs(
        {"a": "Add integers", "b": "Sum numbers"},
        stage="annotation",
        model=model,
        submit=submit,
        cache_root=str(tmp_path / "cache"),
    )
    # Invalid first responses are retried; validated repairs are reused without another provider repair.
    assert repeated == initial
    assert set(repeated_traces[1]["cache_hits"]) == {"a", "b"}
