# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge qualification happens before candidate execution or environment staging.

Also pins the frozen-receipt vintage handling: the 2026.11.13.8 queue's
receipts were frozen when an absent execution_requirements serialized as an
explicit null, and prepare must reproduce that digest space (hosted-v13 died
here before the vintage fix).
"""

import pytest
from finestore.rl.difficulty import SampleTask
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
from experiments.post_training.task_curation.difficulty_contracts import (
    CONTRACT_VINTAGE_NULL_DEFAULTS,
    ContractVintageError,
    difficulty_task_contract,
)


@pytest.mark.parametrize(
    "api,model,reason",
    [
        ("responses", "", "judge_api_unqualified:responses"),
        ("chat_completions", "", None),
        ("chat_completions", "glm-5.3", None),
        ("chat_completions", "MiniMaxAI/MiniMax-M3-MXFP8", None),
        ("chat_completions", "openai/gpt-oss-120b", None),
        ("chat_completions", "gpt-4o-mini", "judge_route_unqualified:gpt-4o-mini"),
    ],
)
def test_prepare_source_qualifies_judge_api_and_current_model_before_staging(
    tmp_path, monkeypatch, difficulty_source, api, model, reason
):
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
    # The frozen panel reader is the I/O boundary; qualification and staging remain real.
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([task], [{}]))
    prepared = difficulty_preparation.prepare_source(source, tmp_path / "tasks")
    (result,) = prepared.tasks
    assert result.reason == reason
    assert bool(list((tmp_path / "tasks").rglob("task-spec.json"))) == (reason is None)
    assert (result.path is None) == (reason is not None)


def _with_frozen_sha(source, task, task_sha256):
    """A source whose sample receipt binds the given digest for the single task."""
    frozen_sample = source.sample.model_copy(update={"tasks": [SampleTask(task_id=task.id, task_sha256=task_sha256)]})
    return source.model_copy(update={"sample": frozen_sample})


def test_prepare_source_recovers_frozen_null_defaults_receipts(tmp_path, monkeypatch, difficulty_source):
    """A 2026.11.13.8-style receipt (absent execution_requirements as null) still prepares.

    hosted-v13 died here: the serializer stopped emitting the absent field's
    explicit null, so the recomputed digest no longer matched the frozen
    receipt. Preparation must resolve the receipt's vintage instead.
    """
    task = TaskSpec(
        id="frozen-era-task",
        context=ConversationInput(events=(TextMessage(role="user", content="Original question"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(
            JudgeSpec(
                api="chat_completions",
                model="glm-5.3",
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
    frozen = difficulty_task_contract(task, {}, CONTRACT_VINTAGE_NULL_DEFAULTS)
    source = _with_frozen_sha(source, task, frozen["task_sha256"])
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([task], [{}]))
    prepared = difficulty_preparation.prepare_source(source, tmp_path / "tasks")
    assert [item.task_id for item in prepared.tasks] == [task.id]


def test_prepare_source_names_a_drifted_task(tmp_path, monkeypatch, difficulty_source):
    """A task whose bytes no recorded vintage reproduces is named with both digests."""
    task = TaskSpec(
        id="drifted-task",
        context=ConversationInput(events=(TextMessage(role="user", content="Original question"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(
            JudgeSpec(
                api="chat_completions",
                model="",
                rubric="labels",
                references=("Original answer",),
                system_prompt="Grade the original answer.",
                prompt_template="Reference: {reference}\nCandidate: {candidate}",
                label_scores={"PASS": 1.0, "FAIL": 0.0},
            )
        ).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    source = _with_frozen_sha(difficulty_source(task), task, "f" * 64)
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([task], [{}]))
    with pytest.raises(ContractVintageError) as error:
        difficulty_preparation.prepare_source(source, tmp_path / "tasks")
    message = str(error.value)
    assert "drifted-task" in message
    assert "ffff" in message
    assert canonical_sha256(task.model_dump(mode="json"))[:16] in message
