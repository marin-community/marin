# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submitted code runs without private assets; only replay outputs reach grading."""


import pytest
from shellbox.machine import DockerImage, MachineSpec

from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    GraderMachine,
    GradingAttempt,
    PlainText,
    ReplayStage,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.task_grading import grade_task

from .grading_machines import GRADER_IMAGE, REPLAY_COMMAND, REPLAY_IMAGE, ReplayMachines, completed_trace

FORBIDDEN_VARIABLE = "REPLAY_FORBIDDEN_SECRET"


@pytest.fixture
def replay_task():
    grader = ScriptGrader(
        placement=GraderMachine(
            environment=EnvironmentRequirements(
                docker_image=GRADER_IMAGE,
                command_semantics=CommandSemantics.LINUX_PROCESS,
            )
        ),
        argv=("python3", "/tests/grade.py"),
        cwd="/app",
        answer_path=None,
        timeout=30,
        replay=ReplayStage(
            environment=EnvironmentRequirements(
                docker_image=REPLAY_IMAGE,
                command_semantics=CommandSemantics.LINUX_PROCESS,
            ),
            argv=REPLAY_COMMAND,
            cwd="/app",
            timeout=10,
            artifacts=(VerifierArtifact(source="/app/outputs", target="/app/outputs", kind=ArtifactKind.DIRECTORY),),
        ),
    )
    grade = b"""
from pathlib import Path
import os
assert os.environ['REPLAY_FORBIDDEN_SECRET'] == 'grader-only'
assert not Path('solution.py').exists()
assert not Path('outputs/actor-only.txt').exists()
assert Path('../tests/gold.txt').read_text() == '42'
assert Path('outputs/first.txt').read_text() == '42'
assert Path('outputs/second.txt').read_text() == '43'
print(1.0)
"""
    return TaskSpec.model_validate(
        {
            "id": "replay-files",
            "source": Source(dataset="test/replay", revision="1", row="0", importer_revision="1"),
            "context": ConversationInput(events=(TextMessage(role="user", content="Write solution.py."),)),
            "answer_type": AnswerType.FILE,
            "answer_format": PlainText(),
            "environment_requirements": grader.replay.environment,
            "grader": grader,
            "output_paths": ("/app/solution.py",),
            "resources": ResourceGroups(
                worker=(inline_resource("app/input.txt", b"42"),),
                oracle=(inline_resource("oracle.py", b"private reference program"),),
                verifier=(inline_resource("grade.py", grade), inline_resource("gold.txt", b"42")),
            ),
        }
    )


def run_program(tmp_path, task, program, *, hang=False):
    machines = ReplayMachines(tmp_path / "machines", hang=hang)
    result = grade_task(
        task,
        GradingAttempt(
            completed_trace(task),
            {
                "/app/solution.py": program,
                "/app/outputs/actor-only.txt": b"must not reach the grader",
            },
        ),
        machine_factory=machines,
        machine_spec=MachineSpec(DockerImage(GRADER_IMAGE), env={FORBIDDEN_VARIABLE: "grader-only"}),
    )
    return result, machines.machines


def test_replay_isolated_from_references_credentials_and_original_actor_outputs(tmp_path, replay_task):
    program = b"""
from pathlib import Path
import os
assert 'REPLAY_FORBIDDEN_SECRET' not in os.environ
assert not Path('../tests/gold.txt').exists()
assert not Path('../solution/oracle.py').exists()
Path('outputs').mkdir()
Path('outputs/first.txt').write_text(Path('input.txt').read_text())
Path('outputs/second.txt').write_text('43')
"""
    task = TaskSpec.model_validate_json(replay_task.model_dump_json())
    result, machines = run_program(tmp_path, task, program)
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0), result
    assert len(machines) == 2
    assert all(machine.closed for machine in machines)
    assert result.diagnostics["replay"]["exit_code"] == 0


@pytest.mark.parametrize("program", [b"raise RuntimeError('candidate error')", b"pass"])
def test_failed_program_or_missing_replay_outputs_scores_zero_without_starting_grader(tmp_path, replay_task, program):
    result, machines = run_program(tmp_path, replay_task, program)
    assert result.status in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE)
    assert result.reward == 0.0
    assert len(machines) == 1 and machines[0].closed


def test_replay_setup_failure_is_ungraded(tmp_path, replay_task):
    wire = replay_task.model_dump()
    wire["grader"]["replay"]["environment"]["setup_commands"] = ["exit 19"]
    task = TaskSpec.model_validate(wire)
    result, machines = run_program(tmp_path, task, b"pass")
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)
    assert len(machines) == 1 and machines[0].closed


def test_total_verifier_deadline_cancels_replay_and_closes_machine(tmp_path, replay_task):
    wire = replay_task.model_dump()
    wire["grader"]["timeout"] = 0.1
    task = TaskSpec.model_validate(wire)
    result, machines = run_program(tmp_path, task, b"pass", hang=True)
    assert (result.status, result.reward, result.failure) == (Outcome.INFRA_ERROR, None, GradingFailure.TIMEOUT)
    assert len(machines) == 1 and machines[0].closed


def test_replay_cannot_transfer_symbolic_links_to_grader(tmp_path, replay_task):
    program = b"""
from pathlib import Path
Path('outputs').mkdir()
Path('outputs/first.txt').symlink_to('../input.txt')
"""
    result, machines = run_program(tmp_path, replay_task, program)
    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    assert len(machines) == 1 and machines[0].closed
