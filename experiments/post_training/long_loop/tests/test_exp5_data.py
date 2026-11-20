# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

import pytest
from click.testing import CliRunner
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    verifyit_spec,
)

from experiments.post_training.long_loop.exp5 import main, prepared_curriculum, prepared_pools
from experiments.post_training.long_loop.exp5_data import hardmath_task, ifbench_row
from experiments.post_training.task_curation.exp4_curriculum import CARRY_OVER_SHARES
from experiments.post_training.task_curation.exp5_curriculum import (
    AGENTIC_TRAIN_SOURCE,
    HARDMATH_LABEL,
    IFBENCH_LABEL,
    IFBENCH_SOURCE,
    IFBENCH_SOURCE_REVISION,
    OPENRESEARCHER_LABEL,
    PENDING_SOURCES,
    exp5_rl_curriculum,
)


def test_ifbench_stamping_keeps_task_content_and_rejects_nonempty_allocations():
    task = TaskSpec(
        id="ifbench:fixture",
        source=Source(
            dataset=IFBENCH_SOURCE,
            revision=IFBENCH_SOURCE_REVISION,
            row="fixture:0",
            importer_revision="1",
        ),
        context=ConversationInput(events=(TextMessage(role="user", content="Answer with exactly two words."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="stamping fixture"),
    )
    atlas = {**task.model_dump(mode="json"), "execution_requirements": None, "integrity_contract": None}
    row, removed = ifbench_row({"task_spec": json.dumps(atlas), "runtime": "{}"})
    assert TaskSpec.model_validate_json(row["task_spec"]) == task
    assert row["teacher_route"] == IFBENCH_LABEL
    assert dict(removed) == {"execution_requirements": 1, "integrity_contract": 1}
    with pytest.raises(ValueError, match="Cannot discard IFBench execution_requirements"):
        ifbench_row({"task_spec": json.dumps({**atlas, "execution_requirements": {"cpus": 8}})})


@pytest.mark.parametrize("difficulty", [None, "easy", "hard"])
def test_hardmath_base_preserves_equation_references_across_difficulty_labels(difficulty):
    task = hardmath_task({"question": "Solve for x.", "ground_truths": r"\boxed{x=3}", "difficulty": difficulty}, 0)
    assert isinstance(task, TaskSpec)
    assert isinstance(task.grader, VerifyitGrader)
    assert verifyit_spec(task.grader).expected == "x=3"
    assert task.context.events[0].content == "Solve for x."


def test_exp5_policy_keeps_the_unavailable_five_percent_and_exact_release_pins():
    curriculum = exp5_rl_curriculum()
    weights = curriculum.weights()
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12)
    assert weights[AGENTIC_TRAIN_SOURCE] == 0.20
    assert {label: weights[label] for label in CARRY_OVER_SHARES} == dict(CARRY_OVER_SHARES)
    assert weights[IFBENCH_LABEL] == weights[OPENRESEARCHER_LABEL] == weights[HARDMATH_LABEL] == 0.05
    pins = {entry.label: entry.pin for entry in curriculum.entries}
    assert pins[IFBENCH_LABEL].revision == "1b4076f63af29e34aee43e16781784890b1e9835"
    assert pins[OPENRESEARCHER_LABEL].revision == "5bee39eb26e0fae3e797d9523ac1b8012f5b95a3"
    assert pins[HARDMATH_LABEL].name == "pafitis/HARDMath_processed_training"
    assert pins[HARDMATH_LABEL].revision == "937e9f10356e31e854f6efb9a2507f1e200c8b25"
    assert set(PENDING_SOURCES) == {OPENRESEARCHER_LABEL}


def test_exp5_preparation_reserves_the_missing_source_and_refuses_submission(tmp_path):
    prepared = prepared_curriculum().weights()
    assert OPENRESEARCHER_LABEL not in prepared
    assert math.isclose(sum(prepared.values()), 0.95, rel_tol=0, abs_tol=1e-12)
    assert prepared[AGENTIC_TRAIN_SOURCE] == 0.20
    result = CliRunner().invoke(
        main,
        [
            "--round",
            str(Path(__file__).parents[1] / "rounds/exp5-setup.yaml"),
            "--source-revision",
            "0" * 40,
            "--validation-output",
            str(tmp_path),
            "--version",
            "2026.11.20.91",
            "--run",
        ],
    )
    assert result.exit_code == 2
    assert not list(tmp_path.iterdir())


def test_exp5_refuses_a_different_swe_population_before_constructing_training_inputs(monkeypatch):
    def substituted_bytes(path):
        if str(path).endswith("manifest.json"):
            return b'{"rows":5608,"source_counts":{"clone-only":5608}}'
        return b"{}"

    monkeypatch.setattr(StoragePath, "read_bytes", substituted_bytes)
    with pytest.raises(ValueError, match="Exp4 SWE pool manifest changed"):
        prepared_pools({})
