"""Grading provider failures retain generated answers and safe cause evidence."""

import asyncio
import json

import pytest
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.external import AgentHarness
from rolloutengine.models import GradedRolloutInterrupted, Prompt
from rolloutengine.test_support import ReplayModel, arithmetic_task, machine_runtime, rollout_spec
from shellbox.machine import Backend
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ScriptGrader

from skyrl_train.rollouts.machines import OwnedMachineFactory
from skyrl_train.rollouts.task_worker import _failed_rollout


class FailingGraderMachine:
    def __init__(self):
        self.closed = False

    async def run(self, command):
        try:
            raise TimeoutError("private-command-argument")
        except TimeoutError as error:
            raise RuntimeError("private-command-argument") from error

    async def upload(self, source, target):
        pass

    async def close(self):
        self.closed = True


class GraderFactory:
    backend = Backend.DOCKER

    def __init__(self):
        self.machine = FailingGraderMachine()

    def check_environment(self, requirements, spec):
        pass

    async def create(self, spec):
        return self.machine


def test_grading_provider_failure_preserves_answer_and_safe_nested_causes():
    async def agent(_context, complete):
        await complete(Prompt(({"role": "user", "content": "Return 12."},), {}))

    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image="fixture@sha256:" + "0" * 64,
    )
    task = arithmetic_task(grader=ScriptGrader(environment=environment, argv=("python3", "/tests/grade.py")))
    spec = rollout_spec(task, harness="agent", verifier_machine=machine_runtime(backend="fixture"))
    factory = GraderFactory()
    owned = OwnedMachineFactory(factory, "fixture")
    engine = RolloutEngine({"fixture": owned}, {"agent": AgentHarness(agent)})
    model = ReplayModel([{"role": "assistant", "content": "12"}])

    async def run():
        with pytest.raises(GradedRolloutInterrupted) as caught:
            await engine.run_task(spec, model.complete)
        return _failed_rollout(caught.value, spec)

    rollout = asyncio.run(run())
    assert rollout.response_token_ids == (20,)
    assert len(model.requests) == 1
    assert factory.machine.closed
    assert rollout.failure.exception_type == "TaskMachineError"
    diagnostics = rollout.failure.diagnostics
    assert diagnostics["operation"] == "grade"
    assert diagnostics["machine_operation"] == "run"
    assert [cause["exception_type"] for cause in diagnostics["machine_error_causes"]] == ["RuntimeError", "TimeoutError"]
    assert "private-command-argument" not in json.dumps(diagnostics)
