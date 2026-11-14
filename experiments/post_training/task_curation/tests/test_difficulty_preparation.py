# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge qualification happens before candidate execution or environment staging."""

import pytest
from taskcompendium.grader import verifyit_package
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation import difficulty_preparation


@pytest.mark.parametrize(
    "api,model,reason",
    [
        ("responses", "", "production_judge_api_not_qualified:responses"),
        ("chat_completions", "", None),
        ("chat_completions", "fixture-current-glm", None),
        ("chat_completions", "glm-5.3", "production_judge_route_not_qualified:glm-5.3"),
    ],
)
def test_prepare_source_qualifies_judge_api_and_current_model_before_staging(
    tmp_path, monkeypatch, difficulty_source, api, model, reason
):
    monkeypatch.setattr(difficulty_preparation, "GLM_MODEL", "fixture-current-glm")
    task = TaskSpec(
        id="judge-task",
        context=ConversationInput(events=(TextMessage(role="user", content="Original question"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(
            JudgeSpec(
                api=api,
                model=model,
                rubric="labels",
                references=("Original answer",),
                system_prompt="Grade the original answer.",
                prompt_template="Reference: {reference}\nCandidate: {candidate}",
                label_scores={"PASS": 1.0, "FAIL": 0.0},
            )
        ).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    source = difficulty_source(task)
    sample = source.sample
    task_sha = sample.tasks[0].task_sha256
    # The frozen panel reader is the I/O boundary; qualification and staging remain real.
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([task], [{}]))
    prepared = difficulty_preparation.prepare_source(source, tmp_path / "tasks")
    (result,) = prepared.tasks
    assert result.reason == reason
    assert bool(list((tmp_path / "tasks").rglob("task-spec.json"))) == (reason is None)
    assert (result.path is None) == (reason is not None)
