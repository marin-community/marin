# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import PurePosixPath
from typing import override

import pytest
from lib.taskforge.tests.resources.test_rollouts_grading import twelve
from rolloutengine.models import GradedRollout, Model, Rollout, RolloutInterrupted, RolloutOperation, TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task, rollout_spec
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.effects.events import Event, Retrying
from taskforge.effects.issues import Restart, Transient
from taskforge.resources.env import CommandResult, FakeCommand, FakeEnv, RunCommand, run_command
from taskforge.resources.rollouts import Engine, RolloutResource, SolverRollout, run_rollout


class Notifications:
    def __init__(self) -> None:
        self.events: list[Event[None]] = []

    def __call__(self, event: Event[None]) -> None:
        self.events.append(event)


class BusyEnv(FakeEnv):
    def __init__(self, notes: Notifications) -> None:
        super().__init__(
            "image",
            {"build": FakeCommand(CommandResult(0, "done", "", 0, False), {PurePosixPath("/output"): b"done"})},
            notes,
        )
        self.busy = True

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        if self.busy:
            self.busy = False
            raise Transient("busy")
        return await super().run(cmd, timeout=timeout)


async def test_command_executor_retries_busy_machine_and_preserves_mutation() -> None:
    notes = Notifications()
    env = BusyEnv(notes)
    event = RunCommand("build", 5)
    assert (await run_command(event, env, notify=notes)).stdout == "done"
    assert await env.read(PurePosixPath("/output")) == b"done"
    assert notes.events == [Retrying(event, 1, "busy")]


async def test_command_executor_stops_unresolved_machine_at_retry_bound() -> None:
    notes = Notifications()
    env = FakeEnv("image", {"build": Transient("busy")}, notes)
    initial = env.fingerprint()
    event = RunCommand("build", 5)
    with pytest.raises(Restart):
        await run_command(event, env, notify=notes)
    assert env.fingerprint() == initial
    assert notes.events == [Retrying(event, 1, "busy"), Retrying(event, 2, "busy")]


class StartingEngine(Engine):
    def __init__(self) -> None:
        self.starting = True

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        if self.starting:
            self.starting = False
            raise RolloutInterrupted(Rollout("trial", (), "interrupted"), RolloutOperation.START)
        return GradedRollout(Rollout("trial", (), "stop"), GradeResult(Outcome.GRADED, 1))


async def test_rollout_executor_retries_startup_and_returns_grade() -> None:
    notes = Notifications()
    event = SolverRollout(rollout_spec(arithmetic_task()), 1, 0)
    resource = RolloutResource(StartingEngine(), twelve, "config")
    assert (await run_rollout(event, resource, notify=notes)).grade.reward == 1
    assert len(notes.events) == 1
    assert isinstance(notes.events[0], Retrying) and notes.events[0].request == event
