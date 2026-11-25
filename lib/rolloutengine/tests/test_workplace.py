# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from typing import cast

import pytest
from shellbox.machine import Machine
from taskcompendium.models import EnvironmentRequirements, FunctionDefinition, ProviderRequirement

from rolloutengine.harness.shell import ShellHarness, ToolContext
from rolloutengine.harness.workplace import WORKPLACE_ACTION_INTERFACE, WorkplaceProvider, workplace_prompt
from rolloutengine.recorder import harness_rollout
from rolloutengine.test_support import ReplayModel, arithmetic_task, rollout_limits, tool_call


def workplace_task():
    tools = [
        {"type": "function", "name": f"native_{i}", "parameters": {"type": "object", "properties": {}, "required": []}}
        for i in range(27)
    ]
    request = {
        "input": [{"role": "user", "content": "What is six plus six?"}],
        "tools": tools,
        "temperature": 1.0,
        "parallel_tool_calls": False,
    }
    provider = ProviderRequirement(
        action_interface=WORKPLACE_ACTION_INTERFACE,
        initial_state={
            "native_max_steps": 6,
            "native_model_interface": "responses_api",
            "native_request": request,
            "runtime": "/workspace/workplace/runtime.py",
            "state_root": "/workspace/workplace/session",
        },
    )
    return arithmetic_task(
        environment_requirements=EnvironmentRequirements(tool_providers={"workplace": provider}),
        interaction_tools=tuple(
            FunctionDefinition.model_validate({k: v for k, v in tool.items() if k != "type"}) for tool in tools
        ),
    )


class NativeMachine:
    def __init__(self):
        self.calls = []

    async def run(self, command):
        self.calls.append(command)
        return SimpleNamespace(
            exit_code=0, stdout=b'{"native_result":true}\n', stderr=b"", stdout_truncated=False, stderr_truncated=False
        )


def native_prompt(task, shell_tool, tools):
    return workplace_prompt(task)


@pytest.mark.asyncio
async def test_sixth_native_tool_observation_survives_without_a_seventh_model_request():
    task = workplace_task()
    machine = NativeMachine()
    model = ReplayModel([tool_call("native_0", {"task": i}, call_id=str(i)) for i in range(7)])
    harness = ShellHarness(
        command_timeout=None,
        tools={WORKPLACE_ACTION_INTERFACE: WorkplaceProvider()},
        prompt=native_prompt,
        source_turn_limit=6,
    )

    rollout = await harness_rollout(
        harness, task, model.complete, limits=rollout_limits(max_turns=None), machine=cast(Machine, machine)
    )

    assert len(machine.calls) == len(model.requests) == 6
    assert json.loads(machine.calls[-1].stdin) == {"name": "native_0", "arguments": '{"task": 5}'}
    assert machine.calls[-1].argv == (
        "/opt/workplace-env/bin/python",
        "/workspace/workplace/runtime.py",
        "call",
        "--root",
        "/workspace/workplace/session",
    )
    observations = [
        observation for step in rollout.conversations[0].steps for observation in step.transition.observations
    ]
    assert [item["tool_call_id"] for item in observations] == [str(i) for i in range(6)]
    assert json.loads(observations[-1]["content"]) == {"native_result": True}
    assert model.requests[0].messages == ({"role": "user", "content": "What is six plus six?"},)
    assert model.requests[0].options["parallel_tool_calls"] is False
    assert model.requests[0].options["temperature"] == 1.0
    assert len(model.requests[0].options["tools"]) == 27


@pytest.mark.asyncio
async def test_native_provider_refuses_missing_state_machine_and_unknown_calls():
    task = workplace_task()
    requirement = task.environment_requirements.tool_providers["workplace"]
    provider = WorkplaceProvider()
    with pytest.raises(ValueError, match="state machine"):
        await provider.open(ToolContext(None, requirement))
    machine = NativeMachine()
    session = await provider.open(ToolContext(cast(Machine, machine), requirement))
    with pytest.raises(ValueError, match="Undeclared native Workplace tool"):
        await session.call("shell", {"command": "true"})
    assert not machine.calls
