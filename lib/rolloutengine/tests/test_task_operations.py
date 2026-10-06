# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task machines, shell grading, and the shell tool outside an engine rollout."""

import json

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command
from taskcompendium.environment import (
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    HealthcheckSpec,
    ShellVerifierSpec,
)
from taskcompendium.execution import TaskExecution
from taskcompendium.grading_result import Outcome

from rolloutengine.cleanup import Cleanup
from rolloutengine.grading import shell_grade
from rolloutengine.machines import task_machine
from rolloutengine.shell_tool import SHELL_TOOL_NAME, shell_observation, shell_tool_definition

from .test_rollout import RecordingShellSimFactory, ReplayModel, engine, file_task

SHELLSIM = {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}
GRADER = ShellVerifierSpec(argv=("sh", "/private/grade.sh"), timeout=5)
GRADER_FILES = (
    EnvironmentFile(
        path="/private/grade.sh",
        content=b'if [ "$(cat /workspace/answer)" = "$(cat /workspace/expected)" ]; then echo 1; else echo 0; fi',
    ),
)


def prepared_environment() -> EnvironmentSpec:
    return EnvironmentSpec(
        kind=EnvironmentKind.SHELLSIM,
        files=(EnvironmentFile(path="/workspace/seed", content=b"12"),),
        setup=(EnvironmentCommand(argv=("sh", "-c", "cp /workspace/seed /workspace/expected"), timeout=5),),
        healthcheck=HealthcheckSpec(
            command=EnvironmentCommand(argv=("test", "-f", "/workspace/expected"), timeout=5),
            interval=0,
            start_period=0,
            start_interval=0,
            retries=1,
        ),
    )


async def test_task_machine_prepares_the_environment_and_closes_on_exit():
    factory = RecordingShellSimFactory()
    cleanup = Cleanup(5)
    async with task_machine(prepared_environment(), {EnvironmentKind.SHELLSIM: factory}, cleanup) as machine:
        assert machine is not None
        result = await machine.run(Command(("cat", "/workspace/expected"), timeout=5))
        assert result.stdout == b"12"
    assert cleanup.errors == []
    with pytest.raises(RuntimeError, match="closed"):
        await factory.machines[0].run(Command(("true",)))


async def test_task_machine_yields_no_machine_for_a_null_environment():
    async with task_machine(EnvironmentSpec(kind=EnvironmentKind.NULL), {}, Cleanup(5)) as machine:
        assert machine is None


async def test_task_machine_records_a_failed_close_without_raising():
    class FailingClose:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            return await self.machine.run(command)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def close(self):
            await self.machine.close()
            raise OSError("close report failed")

    class Factory:
        async def create(self, spec):
            return FailingClose(await ShellSimMachineFactory().create(spec))

    cleanup = Cleanup(5)
    async with task_machine(
        EnvironmentSpec(kind=EnvironmentKind.SHELLSIM), {EnvironmentKind.SHELLSIM: Factory()}, cleanup
    ):
        pass
    assert [(error.operation, error.exception_type) for error in cleanup.errors] == [("machine_close", "OSError")]


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0)])
async def test_shell_grade_installs_private_files_and_reads_the_reward(answer, reward):
    async with task_machine(prepared_environment(), SHELLSIM, Cleanup(5)) as machine:
        assert machine is not None
        await machine.run(Command(("sh", "-c", f"echo {answer} > /workspace/answer"), timeout=5))
        grade = await shell_grade(GRADER, ({"role": "assistant", "content": "Done."},), machine, GRADER_FILES)
    assert (grade.status, grade.reward) == (Outcome.GRADED, reward)


async def test_shell_grade_passes_the_transcript_on_stdin():
    verifier = ShellVerifierSpec(
        argv=("sh", "-c", 'if grep -q \'"content": "twelve"\'; then echo 1; else echo 0; fi'), timeout=5
    )
    messages = ({"role": "user", "content": "Spell 12."}, {"role": "assistant", "content": "twelve"})
    async with task_machine(EnvironmentSpec(kind=EnvironmentKind.SHELLSIM), SHELLSIM, Cleanup(5)) as machine:
        assert machine is not None
        grade = await shell_grade(verifier, messages, machine, ())
    assert (grade.status, grade.reward) == (Outcome.GRADED, 1.0)


async def test_shell_tool_contract_matches_the_engine_session():
    command = "echo 12 > /workspace/answer && cat /workspace/answer"
    model = ReplayModel(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": SHELL_TOOL_NAME, "arguments": json.dumps({"command": command})},
                    }
                ],
            },
            {"role": "assistant", "content": "Completed."},
        ]
    )
    await engine(model, SHELLSIM).run(file_task(), execution=TaskExecution())

    assert shell_tool_definition() in model.requests[0].options["tools"]
    async with task_machine(EnvironmentSpec(kind=EnvironmentKind.SHELLSIM), SHELLSIM, Cleanup(5)) as machine:
        assert machine is not None
        result = await machine.run(Command(("sh", "-c", command), timeout=5))
    assert model.requests[1].messages[-1]["content"] == shell_observation(result)
