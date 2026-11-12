# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from shellbox.machine import Command, ExitReason, Result
from verifyit.spec import ExactSpec

from taskcompendium.grading_result import GradingFailure
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ScriptGrader
from taskcompendium.runtime import grading


class TimedMachine:
    def __init__(self):
        self.timeouts = []

    async def run(self, command: Command) -> Result:
        self.timeouts.append(command.timeout)
        return Result(124, b"partial output", b"cut", False, False, ExitReason.TIMED_OUT)


@pytest.mark.parametrize("kind", ["script", "verifyit"])
def test_grader_diagnostics_measure_only_actual_command(monkeypatch, kind):
    clock = iter((100.0, 163.0))
    monkeypatch.setattr(grading, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    machine = TimedMachine()
    if kind == "script":
        grader = ScriptGrader(
            argv=("work",),
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "a" * 64
            ),
            timeout=600,
        )
        result = asyncio.run(grading._script_reward(machine, grader, 600, None))
    else:
        result = asyncio.run(grading._verifyit_reward(machine, ExactSpec(expected=("yes",)), "/workspace", 600))
    assert result.failure is GradingFailure.GRADER_TIMEOUT
    assert result.diagnostics["grader_elapsed_seconds"] == 63.0
    assert result.diagnostics["grader_timeout_seconds"] == 600
    assert result.diagnostics["exit_code"] == 124
    assert machine.timeouts == [600]
