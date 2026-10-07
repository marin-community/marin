# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from shellbox.machine import DockerImage, ExitReason, MachineSpec, Result
from taskcompendium.datasets.nemotron_ultra import normalization
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, ConversationTrace, EnvironmentRequirements, Source, TextMessage
from taskcompendium.pipeline.models import CheckStatus, ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.runtime.task_grading import grade_task
from taskcompendium.submission import FinalAction, PlainText

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import calendar_binding
from lib.taskcompendium.tests.test_executable_ingestion import GradingMachine, GradingMachines
from lib.taskcompendium.tests.test_runtime import LocalGradingMachines

PLAIN = PlainText(id="plain")
VALID = [
    {"event_id": 0, "start_time": "10:00", "duration": 30},
    {"event_id": 1, "start_time": "11:00", "duration": 30},
]


@dataclass
class SourceScoreMachine(GradingMachine):
    async def run(self, command):
        if command.argv[0] == "rm":
            self.files.pop(command.argv[-1], None)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        if command.argv[0] == "test":
            return Result(0 if command.argv[-1] in self.files else 1, b"", b"", False, False, ExitReason.EXITED)
        if command.argv[0] == "python3" and self.verdict_status == "infra_error":
            raise OSError("Grading service failed")
        result = await super().run(command)
        if command.argv[0] == "python3":
            if self.verdict_status == "invalid_task":
                return Result(1, b"", b"Invalid source contract", False, False, ExitReason.EXITED)
            verdict = json.loads(self.files["/logs/verifier/verdict.json"])
            self.files["/logs/verifier/score.json"] = json.dumps(
                {"reward": verdict["reward"], "detail": verdict["detail"]}
            ).encode()
        return result


@dataclass
class SourceScoreMachines(GradingMachines):
    async def create(self, spec):
        machine = SourceScoreMachine(verdict_status=self.verdict_status)
        self.machines.append(machine)
        return machine


@pytest.fixture
def row():
    return RawRow(
        "calendar",
        Source(dataset="fixture", revision="pinned", row="1", importer_revision="1"),
        {
            "dataset": "fixture",
            "agent_ref": {"name": "calendar_simple_agent"},
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": "Return a schedule as JSON."},
                    {"role": "user", "content": "Schedule event 0 after 10am and event 1 at 11am."},
                ]
            },
            "exp_cal_state": {
                "0": {"duration": 30, "constraint": "after 10am", "min_time": "09:00", "max_time": "12:00"},
                "1": {"duration": 30, "constraint": "at 11am", "min_time": "09:00", "max_time": "12:00"},
            },
        },
    )


def task_for(row):
    result = calendar_binding.normalize_isolated(
        row,
        image="fixture@sha256:" + "1" * 64,
        normalize_task=partial(normalization.normalize, selector="fixture", family="instruction-following"),
    )
    assert isinstance(result, NormalizedTask)
    return result.task


def grade(task, candidate):
    pytest.importorskip(
        "skyrl_gym.envs.nemotron_ultra.answer_extraction", reason="Requires installed original scorer assets"
    )
    image = task.verifier.environment_requirements.docker_image or "fixture@sha256:" + "1" * 64
    task = task.model_copy(
        update={
            "verifier": task.verifier.model_copy(
                update={
                    "environment_requirements": EnvironmentRequirements(
                        docker_image=image, compatible_backends=(LocalGradingMachines.backend,)
                    )
                }
            )
        }
    )
    event = TextMessage(role="assistant", content=candidate) if isinstance(candidate, str) else candidate
    convention = FinalAction(id="fixture") if task.answer_type == AnswerType.NATIVE_ACTION else PLAIN
    with TemporaryDirectory() as directory:
        return grade_task(
            task,
            convention,
            ConversationTrace(events=(*task.context.events, event)),
            RuntimeEvidence({}, "{}"),
            machine_factory=LocalGradingMachines(Path(directory)),
            machine_spec=MachineSpec(DockerImage(image)),
        )


@pytest.mark.parametrize(
    "candidate,reward,reason",
    [
        (json.dumps(VALID), 1.0, "pass"),
        ("A schedule follows: " + json.dumps(VALID), 1.0, "pass"),
        (
            json.dumps([{**VALID[0], "start_time": "10:45"}, VALID[1]]),
            0.0,
            "conflicting_events",
        ),
        (json.dumps([{**VALID[0], "start_time": "09:00"}, VALID[1]]), 0.0, "constraint_violated"),
        (json.dumps([VALID[0]]), 0.0, "different_number_of_events"),
        ("<think>reasoning</think>" + json.dumps(VALID), 1.0, "pass"),
        ("<think>unfinished " + json.dumps(VALID), 0.0, "no_json_list"),
        ("[" + "{}" * 100, 0.0, "no_json_list"),
    ],
)
def test_original_calendar_schedule_constraints_and_reasoning_gate(row, candidate, reward, reason):
    result = grade(task_for(row), candidate)
    assert (result.status, result.reward) == (Outcome.GRADED, reward)
    assert result.detail is not None
    assert result.detail["reason"] == reason


def test_empty_calendar_retains_original_vacuous_success_after_reasoning_extraction(row):
    row.data["exp_cal_state"] = {}
    task = task_for(row)
    assert grade(task, "No scheduled events.").reward == 1.0
    assert grade(task, "<think>reasoning</think>").reward == 1.0


def test_invalid_source_calendar_constraint_is_invalid_task(row):
    row.data["exp_cal_state"]["0"]["constraint"] = "unknown constraint"
    result = grade(task_for(row), json.dumps(VALID))
    assert result.status == Outcome.INVALID_TASK
    assert result.reward is None
    assert result.error is not None and "Unknown calendar constraint" in result.error


def test_calendar_request_and_constraints_keep_original_private_boundary(row):
    original = normalization.normalize(row, "fixture", "instruction-following")
    assert isinstance(original, NormalizedTask)
    task = task_for(row)
    assert task.context == original.task.context
    assert grader_config(task)["contract"]["exp_cal_state"] == row.data["exp_cal_state"]
    assert task.resources.worker == original.task.resources.worker
    assert task.resources.oracle == original.task.resources.oracle
    assert {resource.path for resource in task.resources.verifier} == {
        "source_callable.py",
        "invocation.json",
        "config.json",
    }
    assert task.environment_requirements.docker_image is None
    assert task.verifier.environment_requirements.docker_image == "fixture@sha256:" + "1" * 64


def test_semantic_judge_agent_cannot_use_calendar_binding(row):
    data = {**row.data, "agent_ref": {"name": "multichallenge_simple_agent"}}
    result = calendar_binding.normalize_isolated(
        replace(row, data=data),
        image="fixture",
        normalize_task=partial(normalization.normalize, selector="fixture", family="instruction-following"),
    )
    assert isinstance(result, ImportRejection)
    assert result.reason == "unsupported_native_text_agent"


@pytest.mark.parametrize(
    "verdict_status,expected",
    [("scored", CheckStatus.PASS), ("invalid_task", CheckStatus.FAIL), ("infra_error", CheckStatus.INFRA_ERROR)],
)
@pytest.mark.asyncio
async def test_runtime_and_negative_control_never_invent_positive_witness(row, verdict_status, expected):
    task = task_for(row)
    machines = SourceScoreMachines(verdict_status=verdict_status)
    image = task.verifier.environment_requirements.docker_image
    assert image is not None
    report = await calendar_binding.isolated_checks(
        task, factory=machines, machine_spec=MachineSpec(DockerImage(image)), timeout=10
    )
    assert [(check.check, check.status) for check in report.checks] == [
        ("native_runtime", expected),
        ("positive_witness", CheckStatus.SKIPPED),
    ]
    assert all(machine.closed for machine in machines.machines)
    assert [machine.files[calendar_binding.ANSWER_PATH] for machine in machines.machines] == [
        b"[]",
    ]
