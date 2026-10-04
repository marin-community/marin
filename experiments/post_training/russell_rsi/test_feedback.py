# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from experiments.post_training.russell_rsi.feedback import (
    CodingSkill,
    FeedbackAnalysis,
    SkillEvidence,
    failure_evidence,
    generation_feedback,
)


def test_generation_feedback_cannot_include_private_analyst_evidence():
    private = "/workspace/private_case.py expected answer 123 task-id-secret"
    analysis = FeedbackAnalysis(
        skills=[
            SkillEvidence(skill=CodingSkill.TYPES, confidence=0.9, evidence=private),
            SkillEvidence(skill=CodingSkill.TYPES, confidence=0.8, evidence=private),
            SkillEvidence(skill=CodingSkill.PARSING, confidence=0.3, evidence=private),
        ]
    )
    feedback = json.loads(generation_feedback(analysis))
    assert feedback == {
        "skills": [
            {"label": "types", "description": "Preserve input and output types, including null and Boolean values."}
        ]
    }
    assert private not in generation_feedback(analysis)
    with pytest.raises(ValidationError):
        FeedbackAnalysis.model_validate({"skills": [{"skill": private, "confidence": 0.9, "evidence": private}]})


def test_failure_evidence_excludes_execution_errors_and_passed_rollouts(tmp_path):
    traces = tmp_path / "traces.jsonl"
    failed = {
        "messages": [{"role": "user", "content": "private task prompt"}],
        "grade": {"status": "graded", "reward": 0},
        "failure": None,
        "interrupted_operation": None,
    }
    traces.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {**failed, "grade": {"status": "graded", "reward": 1}},
                {**failed, "failure": {"exception_type": "SetupError"}},
                {**failed, "interrupted_operation": "grade"},
                {**failed, "grade": {"status": "error", "reward": None}},
                failed,
            ]
        )
        + "\n"
    )
    assert [json.loads(record) for record in failure_evidence(str(traces))] == [
        {"task_prompt": failed["messages"][0], "trajectory": [], "grade": failed["grade"], "stop_reason": None}
    ]
