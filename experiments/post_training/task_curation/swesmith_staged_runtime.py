# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskCompendium session for staged SWE-Smith local image artifacts.

Register ``session_factory`` and ``LocalArtifactDockerFactory`` with RolloutEngine.
Image locators use local-artifact/...@sha256:<Docker-config-ID>, not a registry
manifest digest. This factory resolves only explicitly supplied local artifacts;
the generic Docker/Daytona factories must not be used with these locators.
"""

import asyncio
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from rolloutengine.contracts import ModelTurn, TaskSessionContext, ToolCallCompleted, Transition
from rolloutengine.task_session import session_start
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, DockerImage, MachineSpec
from taskcompendium.chat import assistant_message
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import AssistantToolCalls, FunctionCall
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import resolve_shell_tool, run_shell_call

from experiments.post_training.task_curation.swesmith_staged import (
    DockerPilot,
    is_test_path,
    pytest_result,
    sha256,
    stage1_program,
)

SESSION_NAME = "swesmith-staged-v1"
BACKEND_NAME = "swesmith-local-artifact-docker"


class LocalArtifactDockerFactory(DockerMachineFactory):
    def __init__(self, images: dict[str, str]):
        super().__init__()
        self.images = images
        for locator, image in images.items():
            if locator != "local-artifact/swesmith@" + image:
                raise ValueError("Local artifact locator must carry the exact Docker image ID")

    async def create(self, spec: MachineSpec):
        if not isinstance(spec.source, RegistryImage) or spec.source.reference not in self.images:
            raise ValueError("No pinned local image registered for task")
        return await super().create(replace(spec, source=DockerImage(self.images[spec.source.reference])))


def repair_grade(contract: dict, outputs: dict[str, bytes], logs: Path) -> dict:
    baseline = contract["tracked_sha256"]
    allowed = set(contract["changed_source_paths"])
    if any(is_test_path(path) or path in contract["verifier"]["paths"] or path not in baseline for path in allowed):
        return {"passed": False, "error": "invalid_task_repair_scope_contains_tests_or_untracked_paths"}
    violations = [
        path
        for path, digest in baseline.items()
        if "/testbed/" + path not in outputs or (path not in allowed and sha256(outputs["/testbed/" + path]) != digest)
    ]
    if violations:
        return {"passed": False, "integrity_violations": sorted(violations)}
    pilot = DockerPilot(logs)
    with pilot.container(contract["verifier_image"]) as container:
        pilot.upload(container, {path: outputs["/testbed/" + path] for path in allowed})
        return pytest_result(pilot, container, contract["verifier"], "candidate")


class StagedSession:
    def __init__(self, context: TaskSessionContext, logs: Path):
        self.context = context
        self.machine = context.machine
        if self.machine is None:
            raise ValueError("Staged tasks require a native actor machine")
        self.shell = resolve_shell_tool(context.lowered.task, context.lowered.session.shell_tool)
        if self.shell is None:
            raise ValueError("Staged tasks require a shell tool")
        private = context.lowered.task.resources.verifier
        resource = next(item for item in private if item.path == "staged-contract.json")
        self.contract = json.loads(resource_bytes(resource))
        self.logs = logs

    async def prepare(self):
        return session_start(self.context.lowered.task, self.shell, ())

    async def advance(self, turn: ModelTurn) -> Transition:
        message = assistant_message(turn.message)
        if not isinstance(message, AssistantToolCalls):
            return Transition(done=True)
        observations = []
        for call in message.calls:
            started = time.monotonic()
            if call.name != self.shell.definition.name:
                observation = json.dumps({"error": "Unknown tool"})
            else:
                observation = await run_shell_call(
                    self.machine,
                    FunctionCall(name=call.name, arguments=call.arguments),
                    self.shell,
                    timeout=self.context.lowered.session.command_timeout,
                )
            self.context.events.emit(
                ToolCallCompleted(call.call_id, call.name, len(observation.encode()), False, time.monotonic() - started)
            )
            observations.append({"role": "tool", "tool_call_id": call.call_id, "content": observation})
        return Transition(done=False, observations=tuple(observations))

    async def grade(self, messages, *, stop_reason: str, outputs: dict[str, bytes]) -> GradeResult:
        if self.contract["stage"] == "stage1":
            result = await self.machine.run(
                Command(
                    ("/usr/local/bin/python", "-"), cwd="/tmp", stdin=stage1_program(self.contract).encode(), timeout=60
                )
            )
            if result.exit_code:
                return GradeResult(
                    Outcome.INFRA_ERROR, None, detail={"inspection_error": result.stderr.decode(errors="replace")}
                )
            try:
                detail = json.loads(result.stdout)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                return GradeResult(Outcome.INFRA_ERROR, None, error=f"Inspector output is invalid: {exc}")
        else:
            try:
                detail = await asyncio.to_thread(repair_grade, self.contract, outputs, self.logs)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                return GradeResult(Outcome.INFRA_ERROR, None, error=str(exc))
            if "error" in detail:
                status = Outcome.INVALID_TASK if detail["error"].startswith("invalid_task_") else Outcome.INFRA_ERROR
                return GradeResult(status, None, error=detail["error"], detail=detail)
        return GradeResult(Outcome.GRADED, float(detail["passed"]), passed=detail["passed"], detail=detail)

    async def close(self) -> None:
        pass


def session_factory(logs: Path):
    """Create per-attempt session factories with caller-owned private log directories."""

    def create(context: TaskSessionContext) -> StagedSession:
        return StagedSession(context, logs / (str(time.time_ns()) + "-" + context.lowered.task.id.rsplit("/", 1)[-1]))

    return create
