# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared machines and the shell tool outside an engine rollout."""

import json
from dataclasses import dataclass, field

import pytest
from shellbox.machine import Command
from taskcompendium.grading_result import Outcome
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.resources import inline_resource

from rolloutengine.contracts import LENGTH_STOP_REASON, ModelTurn
from rolloutengine.machines import prepare_machine
from rolloutengine.shell_tool import SHELL_TOOL_NAME, shell_observation, shell_tool_definition

from .test_rollout import (
    FixtureImageFactory,
    RecordingShellSimFactory,
    ReplayModel,
    engine,
    file_task,
    lowered,
    machine_runtime,
    shell_call,
)

SEEDED = EnvironmentRequirements(setup_commands=("cp /workspace/seed /workspace/expected",))


class FailingCloseFactory:
    async def create(self, spec):
        machine = await FixtureImageFactory().create(spec)

        class FailingClose:
            async def run(self, command):
                return await machine.run(command)

            async def upload(self, source, target):
                await machine.upload(source, target)

            async def download(self, source, target):
                await machine.download(source, target)

            async def close(self):
                await machine.close()
                raise OSError("close report failed")

        return FailingClose()


async def test_prepare_machine_installs_resources_runs_setup_and_closes_on_exit():
    factory = RecordingShellSimFactory()
    resources = (inline_resource("workspace/seed", b"12"),)
    async with prepare_machine(SEEDED, machine_runtime(), resources, {"local": factory}, cleanup_timeout=5) as machine:
        result = await machine.run(Command(("cat", "/workspace/expected"), timeout=5))
    assert result.stdout == b"12"
    with pytest.raises(RuntimeError, match="closed"):
        await factory.machines[0].run(Command(("true",)))


async def test_prepare_machine_raises_a_close_failure_after_a_clean_body():
    with pytest.raises(RuntimeError, match="machine_close"):
        async with prepare_machine(
            EnvironmentRequirements(), machine_runtime(), (), {"local": FailingCloseFactory()}, cleanup_timeout=5
        ):
            pass


async def test_prepare_machine_keeps_the_body_failure_over_a_close_failure():
    with pytest.raises(ValueError, match="body"):
        async with prepare_machine(
            EnvironmentRequirements(), machine_runtime(), (), {"local": FailingCloseFactory()}, cleanup_timeout=5
        ):
            raise ValueError("body")


async def test_prepare_machine_closes_the_machine_when_setup_fails():
    factory = RecordingShellSimFactory()
    failing = EnvironmentRequirements(setup_commands=("exit 3",))
    with pytest.raises(RuntimeError, match="setup command failed"):
        async with prepare_machine(failing, machine_runtime(), (), {"local": factory}, cleanup_timeout=5):
            pass
    with pytest.raises(RuntimeError, match="closed"):
        await factory.machines[0].run(Command(("true",)))


async def test_shell_tool_contract_matches_the_engine_session():
    command = "echo 12 > /workspace/answer && cat /workspace/answer"
    model = ReplayModel([shell_call(command), {"role": "assistant", "content": "Completed."}])
    spec = lowered(file_task(), machine=machine_runtime(), verifier_machine=machine_runtime())
    await engine(model, {"local": FixtureImageFactory()}).run(spec)

    assert shell_tool_definition() in model.requests[0].options["tools"]
    async with prepare_machine(
        EnvironmentRequirements(), machine_runtime(), (), {"local": FixtureImageFactory()}, cleanup_timeout=5
    ) as machine:
        result = await machine.run(Command(("sh", "-c", command), timeout=5))
    assert model.requests[1].messages[-1]["content"] == shell_observation(result)


async def test_length_cut_tool_call_is_graded_without_execution():
    @dataclass
    class BudgetCutModel:
        requests: list = field(default_factory=list)

        async def complete(self, request):
            self.requests.append(request)
            call = {"name": SHELL_TOOL_NAME, "arguments": json.dumps({"command": "echo 12 > /workspace/answer"})}
            message = {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": call}]}
            return ModelTurn(message, (10, 11), (20,), None, LENGTH_STOP_REASON)

    task = file_task()
    task = task.model_copy(
        update={"resources": task.resources.model_copy(update={"all": (inline_resource("workspace/answer", b"0"),)})}
    )
    model = BudgetCutModel()
    spec = lowered(task, machine=machine_runtime(), verifier_machine=machine_runtime())
    result = await engine(model, {"local": FixtureImageFactory()}).run(spec)
    assert len(model.requests) == 1
    assert result.stop_reason == LENGTH_STOP_REASON
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 0.0)
