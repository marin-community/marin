# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from functools import partial

import pytest
from shellbox.machine import DockerImage, MachineSpec
from taskcompendium.datasets.nemotron_ultra import normalization
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, Source, TextMessage
from taskcompendium.pipeline.models import CheckStatus, ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import multiple_choice_binding
from lib.taskcompendium.tests.test_runtime import LocalGradingMachines

from .test_calendar_binding import SourceScoreMachines, grade, grader_image


@pytest.fixture
def row():
    return RawRow(
        "mcqa",
        Source(dataset="fixture", revision="pinned", row="1", importer_revision="1"),
        {
            "dataset": "fixture",
            "agent_ref": {"name": "mcqa_simple_agent"},
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": "Which color is the sky? A: Blue. B: Red.",
                    }
                ]
            },
            "expected_answer": "A",
            "options": [{"A": "Blue"}, {"B": "Red"}],
        },
    )


def task_for(row):
    result = multiple_choice_binding.normalize_isolated(
        row,
        image="fixture@sha256:" + "1" * 64,
        normalize_task=partial(normalization.normalize, selector="fixture", family="qa-multiple-choice"),
    )
    assert isinstance(result, NormalizedTask)
    return result.task


@pytest.mark.parametrize(
    "mode,candidate,expected",
    [
        ("strict_single_letter_boxed", r"\boxed{A}", 1.0),
        ("strict_single_letter_boxed", "A", 0.0),
        ("strict_single_letter_boxed", r"\boxed{Blue}", 0.0),
        ("lenient_boxed", r"\boxed{\text{Blue}}", 1.0),
        ("lenient_answer_colon", "Answer: Blue", 1.0),
        ("lenient_answer_colon_md", "**Answer**: A", 1.0),
        ("strict_single_letter_boxed", r"\boxed{B}", 0.0),
        ("strict_single_letter_boxed", r"<think>\boxed{A}</think>\boxed{B}", 0.0),
        ("strict_single_letter_boxed", r"<think>unfinished \boxed{A}", 0.0),
    ],
)
def test_original_modes_and_reasoning_extraction(row, mode, candidate, expected):
    row.data["grading_mode"] = mode
    result = grade(task_for(row), candidate)
    assert (result.status, result.reward) == (Outcome.GRADED, expected)


@pytest.mark.parametrize("label", ["أ", "অ", "\uff21"])
def test_original_custom_regex_multilingual_labels(row, label):
    row.data["template_metadata"] = {"output_regex": r"choice=(.+)"}
    task = task_for(row)
    assert grade(task, "choice=" + label).reward == 1.0
    assert multiple_choice_binding.reference_answer(grader_config(task)["contract"]) is None


def test_ambiguous_regex_writes_no_reward_rather_than_scoring_a_wrong_answer(row):
    row.data["template_metadata"] = {"output_regex": r"(A)(B)"}
    result = grade(task_for(row), "AB")
    assert (result.status, result.failure) == (Outcome.INFRA_ERROR, GradingFailure.MISSING_REWARD)
    assert result.reward is None
    assert "unambiguous answer capture" in result.diagnostics["stderr"]


def test_private_contract_and_original_modules_stay_out_of_public_task(row):
    task = task_for(row)
    original = normalization.normalize(row, "fixture", "qa-multiple-choice")
    assert isinstance(original, NormalizedTask)
    assert task.context == original.task.context
    config = grader_config(task)
    assert config["contract"]["expected_answer"] == "A"
    assert config["contract"]["options"] == row.data["options"]
    assert not task.resources.worker
    assert not task.resources.oracle
    private_paths = {resource.path for resource in task.resources.verifier}
    assert private_paths == {"source_callable.py", "invocation.json", "config.json"}
    assert grade(task, r"\boxed{A}").reward == 1.0


def test_different_original_agent_is_not_replaced(row):
    row.data["agent_ref"]["name"] = "math_with_judge_agent"
    result = multiple_choice_binding.normalize_isolated(
        row,
        image="fixture",
        normalize_task=partial(normalization.normalize, selector="fixture", family="qa-multiple-choice"),
    )
    assert isinstance(result, ImportRejection)
    assert result.reason == "unsupported_native_text_agent"


@pytest.mark.parametrize(
    "mode",
    [
        "strict_single_letter_boxed",
        "lenient_boxed",
        "lenient_answer_colon",
        "lenient_answer_colon_md",
    ],
)
def test_constructed_witness_follows_original_mode(row, mode):
    row.data["grading_mode"] = mode
    task = task_for(row)
    assert grade(task, multiple_choice_binding.reference_answer(grader_config(task)["contract"])).reward == 1.0


@pytest.mark.asyncio
async def test_submitted_package_cannot_shadow_installed_source_grader(row, tmp_path):
    pytest.importorskip(
        "skyrl_gym.envs.nemotron_ultra.answer_extraction", reason="Requires installed original scorer assets"
    )
    task = task_for(row)
    path = "/app/skyrl_gym/__init__.py"
    task = task.model_copy(update={"output_paths": (*task.output_paths, path)})
    attempt = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=r"\boxed{A}"))),
        {path: b'raise RuntimeError("Actor package loaded")\n'},
    )
    result = await grade_in_sandbox(
        task, attempt, LocalGradingMachines(tmp_path), MachineSpec(DockerImage(grader_image(task)))
    )
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


def test_binding_preserves_worker_and_oracle_resource_roles(row):
    original = normalization.normalize(row, "fixture", "qa-multiple-choice")
    assert isinstance(original, NormalizedTask)
    worker = (inline_resource("public-scaffold.txt", b"public context"),)
    oracle = (inline_resource("private-evidence.txt", b"private lineage"),)
    original = replace(
        original,
        task=original.task.model_copy(
            update={"resources": original.task.resources.model_copy(update={"worker": worker, "oracle": oracle})}
        ),
    )
    result = multiple_choice_binding.normalize_isolated(
        row, image="fixture@sha256:" + "1" * 64, normalize_task=lambda _: original
    )
    assert isinstance(result, NormalizedTask)
    assert result.task.resources.worker == worker
    assert result.task.resources.oracle == oracle
    assert grade(result.task, r"\boxed{A}").reward == 1.0


@pytest.mark.parametrize("pattern,candidate", [("(", r"\boxed{A}"), ("(A)|(B)", "A"), ("(A)(B)?", "A")])
def test_original_regex_fallback_and_inactive_captures_remain_valid(row, pattern, candidate):
    row.data["template_metadata"] = {"output_regex": pattern}
    result = grade(task_for(row), candidate)
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


@pytest.mark.parametrize("verdict_status", ["invalid_task", "infra_error"])
@pytest.mark.asyncio
async def test_isolated_source_failure_is_not_a_scored_control(row, verdict_status):
    task = task_for(row)
    machines = SourceScoreMachines(verdict_status=verdict_status)
    report = await multiple_choice_binding.isolated_checks(
        task,
        factory=machines,
        machine_spec=MachineSpec(DockerImage(grader_image(task))),
        timeout=10,
    )
    assert {check.check: check.status for check in report.checks}["reference"] == CheckStatus.INFRA_ERROR
    assert len(machines.machines) == 1 and machines.machines[0].closed
