# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retained Math-TIR tasks lower to their required LOCAL backend and serve their stateful Python tool."""

import asyncio
from pathlib import Path

import pytest
from rolloutengine.harness.shell import ToolContext
from shellbox.machine import Backend, ExitReason, MachineSpec, RegistryImage, Result, UnsupportedMachineSpec
from skyrl_gym.nemotron_python_tools import (
    NEMOTRON_PYTHON_ACTION_INTERFACE,
    NemotronPythonProvider,
    NemotronPythonSession,
)
from skyrl_train.dataset.tasks import canonical_task_spec
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.environment import validate_machine_spec

FIXTURE = Path(__file__).with_name("fixtures") / "math_tir_task.json"
LOCAL_MACHINE = {
    "backend": "local",
    "cpus": 1,
    "memory_mb": 8192,
    "storage_mb": None,
    "gpus": 0,
    "network": "deny",
    "user": None,
    "startup_timeout": 600,
    "cleanup_timeout": 60,
}
DAYTONA_MACHINE = {**LOCAL_MACHINE, "backend": "daytona"}
CONFIG = {
    "default_harness": "shellbox",
    "harness_by_dataset": {},
    "max_turns_by_dataset": {},
    "task_machine_by_dataset": {"nvidia/Nemotron-RL-Ultra-Training-Blends": DAYTONA_MACHINE},
    "task_machine_by_backend": {"local": LOCAL_MACHINE},
    "session": {
        "max_turns": 8,
        "max_total_tokens": None,
        "model_turn_timeout": 60,
        "tool_turn_timeout": 90,
        "total_turn_timeout": 600,
        "attempt_timeout": 900,
        "verifier_timeout": 60,
        "cleanup_timeout": 60,
        "command_timeout": 30,
        "shell_tool": {},
    },
    "image_verifier": DAYTONA_MACHINE,
    "package_verifier": LOCAL_MACHINE,
}


def math_tir_task() -> TaskSpec:
    return TaskSpec.model_validate_json(FIXTURE.read_text())


def test_required_backend_round_trips_and_selects_its_machine():
    task = math_tir_task()
    assert task.environment_requirements.required_backend is Backend.LOCAL
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task
    # The declared backend wins over a same-dataset binding for other Ultra components.
    assert canonical_task_spec(task, CONFIG).task_machine.backend == "local"
    with pytest.raises(ValueError, match="required local backend"):
        canonical_task_spec(task, {**CONFIG, "task_machine_by_backend": {}})


class _Factory:
    def __init__(self, backend):
        self.backend = backend
        self.checked = False

    def check_environment(self, requirements, spec):
        self.checked = True


def test_machine_validation_refuses_another_backend():
    requirements = math_tir_task().environment_requirements
    spec = MachineSpec(RegistryImage(requirements.docker_image))
    with pytest.raises(UnsupportedMachineSpec, match="requires the local backend"):
        validate_machine_spec(requirements, _Factory(Backend.DAYTONA), spec)
    local = _Factory(Backend.LOCAL)
    validate_machine_spec(requirements, local, spec)
    assert local.checked


class _Kernel:
    def __init__(self):
        self.calls = []

    async def execute(self, code, *, timeout):
        self.calls.append((code, timeout))
        return Result(0, b"2\n", b"", False, False, ExitReason.EXITED)


def test_provider_advertises_declared_tool_and_runs_calls_in_one_kernel():
    task = math_tir_task()
    requirement = next(iter(task.environment_requirements.tool_providers.values()))
    assert requirement.action_interface == NEMOTRON_PYTHON_ACTION_INTERFACE
    assert NemotronPythonProvider().tools(requirement) == task.interaction_tools
    with pytest.raises(ValueError, match="requires a task machine"):
        asyncio.run(NemotronPythonProvider().open(ToolContext(None, requirement)))
    kernel = _Kernel()
    session = NemotronPythonSession(kernel)
    assert asyncio.run(session.call("stateful_python_code_exec", {"code": "print(1+1)"})).content == "2"
    assert kernel.calls == [("print(1+1)", 10.0)]
    assert "string code" in asyncio.run(session.call("stateful_python_code_exec", {"code": 3})).content
    assert "Unknown tool" in asyncio.run(session.call("shell", {"code": "x"})).content
