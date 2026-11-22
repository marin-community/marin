# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math
import subprocess
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
from experiments.post_training.long_loop.round_runner import load_round
from experiments.post_training.task_curation.exp5_curriculum import (
    AGENTIC_TRAIN_SOURCE,
    HARDMATH_LABEL,
    IFBENCH_LABEL,
    IFBENCH_SOURCE,
    IFBENCH_SOURCE_REVISION,
    OPENRESEARCHER_LABEL,
    PENDING_SOURCES,
    TARGET_SHARES,
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


def _exp5_validation():
    """A synthetic full-mixture holdout: the object-level policy test needs no pools."""
    from marin.rl.curriculum import PoolRow, draw_validation_split

    labels = dict(TARGET_SHARES)
    rows = {
        label: tuple(PoolRow(id=f"{label}-{index}", sha256=f"{index:064x}") for index in range(60))
        for label in labels
    }
    return draw_validation_split(dict(labels), rows, size=200, seed=20261020).validation


def test_exp5_policy_keeps_the_unavailable_five_percent():
    curriculum = exp5_rl_curriculum(_exp5_validation())
    weights = curriculum.weights()
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12)
    assert weights[AGENTIC_TRAIN_SOURCE] == 0.20
    assert weights[IFBENCH_LABEL] == weights[OPENRESEARCHER_LABEL] == weights[HARDMATH_LABEL] == 0.05


def test_exp5_preparation_reserves_the_missing_source_and_refuses_submission(tmp_path):
    validation = _exp5_validation()
    prepared = prepared_curriculum(
        dataclasses.replace(
            validation,
            size=sum(1 for row in validation.rows if row.label not in PENDING_SOURCES),
            rows=tuple(row for row in validation.rows if row.label not in PENDING_SOURCES),
        )
    ).weights()
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
            "2026.11.20.97",
            "--run",
        ],
    )
    assert result.exit_code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.integration
def test_exp5_sft_stage_constructs_only_training_and_its_inputs(tmp_path):
    workspace = Path(__file__).resolve().parents[4]
    round_path = Path(__file__).parents[1] / "rounds/exp5-setup.yaml"
    spec = load_round(round_path)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
    result = CliRunner().invoke(
        main,
        [
            "--round",
            str(round_path),
            "--source-revision",
            revision,
            "--validation-output",
            str(tmp_path),
            "--version",
            spec.version,
            "--stage",
            "sft",
        ],
    )
    assert result.exit_code == 0, result.output
    graph = json.loads((tmp_path / "graph.json").read_text())
    assert graph[-1]["stage"] == f"{spec.name}/sft"
    assert not any(
        row["stage"].endswith(("/rl", "/eval", "/archive", "/sft-hf")) or "prepared-" in row["stage"] for row in graph
    )
    resolved = json.loads((tmp_path / "resolved-plan.json").read_text())
    assert resolved["exp5_admission"]["launch_ready"]
    assert resolved["exp5_admission"]["pending_sources"]
    assert math.isclose(resolved["exp5_admission"]["reserved_share"], 0.05)
    assert resolved["execution"]["coordinator"]["priority"] == "production"


def test_exp5_refuses_a_different_swe_population_before_constructing_training_inputs(monkeypatch):
    def substituted_bytes(path):
        if str(path).endswith("manifest.json"):
            return b'{"rows":5608,"source_counts":{"clone-only":5608}}'
        return b"{}"

    monkeypatch.setattr(StoragePath, "read_bytes", substituted_bytes)
    with pytest.raises(ValueError, match="Exp4 SWE pool manifest changed"):
        prepared_pools({})
