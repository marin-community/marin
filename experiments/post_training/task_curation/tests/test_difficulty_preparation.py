# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge qualification happens before candidate execution or environment staging."""

import pytest
from finestore.rl.difficulty import DifficultySample
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
from experiments.post_training.task_curation.difficulty_samples import DifficultySource
from experiments.post_training.task_curation.sample_qc_artifacts import SamplePanel
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source


@pytest.mark.parametrize(
    "api,model,reason",
    [
        ("responses", "", "production_judge_api_not_qualified:responses"),
        ("chat_completions", "", None),
        ("chat_completions", "fixture-current-glm", None),
        ("chat_completions", "glm-5.3", "production_judge_route_not_qualified:glm-5.3"),
    ],
)
def test_prepare_source_qualifies_judge_api_and_current_model_before_staging(tmp_path, monkeypatch, api, model, reason):
    monkeypatch.setattr(difficulty_preparation, "GLM_MODEL", "fixture-current-glm")
    declaration = number_source(tmp_path / "declaration").declaration
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
    task_sha = canonical_sha256(task.model_dump(mode="json"))
    panel = SamplePanel(
        source_id=declaration.id,
        dataset_id=declaration.current_release.repo,
        dataset_revision=declaration.current_release.revision,
        converter_version=declaration.pipeline.version,
        input_artifact="fixture:population",
        task_specs_sha256=task_sha,
        scanned_task_specs_sha256=task_sha,
        retained_tasks=1,
        sample_seed=0,
        sample_size=1,
        panel_path="fixture:panel",
        panel_sha256="a" * 64,
        evidence_url="fixture:inventory",
    )
    sample = DifficultySample(
        source_id=declaration.id,
        release=declaration.current_release,
        converter_version=declaration.pipeline.version,
        converter_revision="b" * 40,
        population_count=1,
        task_specs_sha256=task_sha,
        drop_manifest={"uri": "fixture:drops", "sha256": "c" * 64},
        seed=0,
        pre_oracle_filter=False,
        resample_after_oracle_filter=False,
        tasks=[{"task_id": task.id, "task_sha256": task_sha}],
    )
    source = DifficultySource(
        declaration=declaration,
        panel=panel,
        sample=sample,
        sample_receipt={"uri": "fixture:sample", "sha256": "d" * 64},
        backend="local",
    )
    # The frozen panel reader is the I/O boundary; qualification and staging remain real.
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([task], [{}]))
    prepared = difficulty_preparation.prepare_source(source, tmp_path / "tasks")
    (result,) = prepared.tasks
    assert result.reason == reason
    assert result.judge_model == model
    assert prepared.sample == sample
    assert bool(list((tmp_path / "tasks").rglob("task-spec.json"))) == (reason is None)
    assert (result.path is None) == (reason is not None)
    assert canonical_sha256(task.model_dump(mode="json")) == task_sha
