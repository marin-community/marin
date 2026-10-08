# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small lowered tasks shared by the unit and live validate tests, with a control set for two of them.

``math_task`` is a null-environment numeric task graded in process by verifyit; ``json_task`` is a
null-environment task with a JSON answer. ``file_task`` is a ShellSim task whose private ``script``
grader runs on the host and checks the captured file the agent must create. Each is lowered for a
laptop with the ShellSim factory (``lowered``); ``relower`` lowers a variant the same way.
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from rolloutengine.spec import LoweredTaskSpec
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Command, Machine, MachineSpec, MachineTerminated, Result
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source, TaskSpec
from verifyit.spec import NumericSpec, StructuredExactSpec

from taskforge.sandbox.factories import MachineHost
from taskforge.spec.controls import (
    Control,
    ControlCategory,
    ControlConcern,
    ControlKind,
    Expectation,
    Transcript,
    Workspace,
    reply,
    shell_turn,
)
from taskforge.spec.draft import (
    SHELL_CAPABILITY,
    answer_verifier,
    assemble,
    file,
    lower,
    machine,
    requirements,
    script_verifier,
    session,
)

MATH_ANSWER = "395"
NUMBERS = "12\n7\n30\n11\n"
NUMBERS_SUM = 60
FACTORIES = {Backend.SHELLSIM.value: ShellSimMachineFactory()}
SESSION = session(
    max_turns=4,
    model_turn_timeout=None,
    command_timeout=10,
    tool_turn_timeout=20,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=30,
    cleanup_timeout=10,
)
SHELLSIM = machine(startup_timeout=30)
SUM_GRADER = """import json, os, pathlib
workspace = pathlib.Path(os.environ["VERIFYIT_WORKSPACE"])
expected = json.loads(pathlib.Path(os.environ["VERIFYIT_TESTS_DIR"], "config.json").read_text())["expected"]
answer = workspace / "captured/workspace/sum.txt"
got = answer.read_text().strip() if answer.is_file() else None
verdict = {"status": "scored", "reward": float(got == expected), "detail": {"got": got}}
pathlib.Path(os.environ["VERIFYIT_LOGS_DIR"], "verdict.json").write_text(json.dumps(verdict))
"""
"""Rewards 1 when the captured ``/workspace/sum.txt`` holds ``config["expected"]``."""


def source(row: str) -> Source:
    return Source(dataset="taskforge-validate-tests", revision="1", row=row, importer_revision="1")


def lowered(task: TaskSpec) -> LoweredTaskSpec:
    """``task`` lowered for a laptop: a ShellSim task machine when it has one, no verifier machine."""
    has_machine = SHELL_CAPABILITY in task.environment_requirements.capabilities
    return lower(
        task,
        host=MachineHost.LAPTOP,
        task_machine=SHELLSIM if has_machine else None,
        verifier_machine=None,
        session=SESSION,
        factories=FACTORIES,
    )


@pytest.fixture
def relower() -> Callable[[TaskSpec], LoweredTaskSpec]:
    return lowered


@pytest.fixture
def math_task() -> LoweredTaskSpec:
    task = assemble(
        "validate-math",
        "What is 17 * 23 + 4? Reply with only the number, nothing else.",
        AnswerType.NUMBER,
        answer_verifier(NumericSpec(expected=MATH_ANSWER, tolerance_abs=0, tolerance_rel=0)),
        source("math"),
        environment=None,
    )
    return lowered(task)


@pytest.fixture
def json_task() -> LoweredTaskSpec:
    task = assemble(
        "validate-json",
        "Report the sum of 12, 7, 30 and 11 as a JSON object whose only key is sum.",
        AnswerType.JSON,
        answer_verifier(StructuredExactSpec(expected={"sum": NUMBERS_SUM})),
        source("json"),
        environment=None,
    )
    return lowered(task)


def file_task_spec(grader_script: str = SUM_GRADER, setup: tuple[str, ...] = (), grader_timeout: float = 30) -> TaskSpec:
    return assemble(
        "validate-file",
        "The file /workspace/numbers.txt holds one integer per line. Use the shell tool to write their sum, "
        "as a single integer, to /workspace/sum.txt. Say when you are done.",
        AnswerType.FILE,
        script_verifier(grader_script, {"expected": str(NUMBERS_SUM)}, timeout=grader_timeout),
        source("file"),
        environment=requirements(image=None, setup=setup),
        files=(file("workspace/numbers.txt", NUMBERS),),
        output_paths=("/workspace/sum.txt",),
    )


@pytest.fixture
def file_task() -> LoweredTaskSpec:
    return lowered(file_task_spec())


@pytest.fixture
def file_task_with() -> Callable[..., LoweredTaskSpec]:
    """``file_task`` with another grader script, setup commands or grader timeout."""
    return lambda grader_script=SUM_GRADER, setup=(), grader_timeout=30: lowered(
        file_task_spec(grader_script, tuple(setup), grader_timeout)
    )


def control(
    name: str, kind: ControlKind, category: ControlCategory, concern: ControlConcern, payload, expect: Expectation
) -> Control:
    return Control(
        id=name, kind=kind, category=category, concern=concern, author="tests", payload=payload, expect=expect
    )


CORRECT = Expectation(status=Outcome.GRADED, reward_min=1.0)
WRONG = Expectation(status=Outcome.GRADED, reward_max=0.0)
NO_SUBMISSION = Expectation(status=Outcome.SUBMISSION_FAILURE)


@pytest.fixture
def math_controls() -> tuple[Control, ...]:
    return (
        control(
            "correct",
            ControlKind.POSITIVE,
            ControlCategory.KNOWN_CORRECT,
            ControlConcern.REFERENCE,
            Transcript((reply("395"),)),
            CORRECT,
        ),
        control(
            "wrong",
            ControlKind.NEGATIVE,
            ControlCategory.PLAUSIBLE_WRONG,
            ControlConcern.ACCEPTANCE,
            Transcript((reply("391"),)),
            WRONG,
        ),
        control(
            "two-answers",
            ControlKind.NEGATIVE,
            ControlCategory.TASK_SPECIFIC_SHORTCUT,
            ControlConcern.SHORTCUT,
            Transcript((reply("395 or 391"),)),
            WRONG,
        ),
        control(
            "empty",
            ControlKind.MALFORMED,
            ControlCategory.EMPTY_OR_MALFORMED,
            ControlConcern.EXTRACTION,
            Transcript((reply(""),)),
            NO_SUBMISSION,
        ),
    )


@pytest.fixture
def file_controls() -> tuple[Control, ...]:
    return (
        control(
            "correct",
            ControlKind.POSITIVE,
            ControlCategory.KNOWN_CORRECT,
            ControlConcern.REFERENCE,
            Transcript((shell_turn(("c1", f"echo {NUMBERS_SUM} > /workspace/sum.txt")), reply("Done."))),
            CORRECT,
        ),
        control(
            "wrong",
            ControlKind.NEGATIVE,
            ControlCategory.PLAUSIBLE_WRONG,
            ControlConcern.ACCEPTANCE,
            Transcript((shell_turn(("c1", "head -n 1 /workspace/numbers.txt > /workspace/sum.txt")), reply("Done."))),
            WRONG,
        ),
        control(
            "workspace-correct",
            ControlKind.POSITIVE,
            ControlCategory.KNOWN_CORRECT,
            ControlConcern.ACCEPTANCE,
            Workspace((file("workspace/sum.txt", f"{NUMBERS_SUM}\n"),)),
            CORRECT,
        ),
        control(
            "plant-grader",
            ControlKind.NEGATIVE,
            ControlCategory.REWARD_HACK,
            ControlConcern.SHORTCUT,
            Transcript((shell_turn(("c1", "mkdir -p /tests && echo 'print(1)' > /tests/grader.py")), reply("Done."))),
            WRONG,
        ),
        control(
            "no-work",
            ControlKind.MALFORMED,
            ControlCategory.EMPTY_OR_MALFORMED,
            ControlConcern.EXTRACTION,
            Transcript((reply("Done."),)),
            WRONG,
        ),
    )


def assistant_text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def assistant_shell(command: str, call_id: str = "c1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": command})},
            }
        ],
    }


@dataclass
class ScriptModel:
    """Answers each request with the next of ``messages``, with ids that preserve the served prefix.

    From turn ``hang_from`` on, a request waits forever, so only a timeout ends it. With a
    ``barrier``, every request waits until all of the barrier's parties are in flight. ``requests``
    collects every request served.
    """

    messages: list[dict[str, Any]]
    hang_from: int | None = None
    barrier: asyncio.Barrier | None = None
    requests: list[ModelRequest] = field(default_factory=list)

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = sum(message["role"] == "assistant" for message in request.messages)
        if self.hang_from is not None and index >= self.hang_from:
            await asyncio.Event().wait()
        if self.barrier is not None:
            await self.barrier.wait()
        prompt = (*request.prefix_token_ids, 90) if request.prefix_token_ids else (10, 11)
        stop = "tool_calls" if "tool_calls" in self.messages[index] else "stop"
        return ModelTurn(self.messages[index], prompt, (20 + index,), (-0.5,), stop)


@dataclass
class RaisingModel:
    error: Callable[[], Exception]

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        raise self.error()


@dataclass
class FlakyFactory:
    """Raises ``error()`` on the first ``failures`` creates (after ``delay``), then delegates to ShellSim."""

    failures: int
    error: Callable[[], Exception]
    delay: float = 0.0
    creates: int = 0

    @property
    def backend(self) -> Backend:
        return Backend.SHELLSIM

    async def create(self, spec: MachineSpec) -> Machine:
        self.creates += 1
        failing = self.creates <= self.failures
        await asyncio.sleep(self.delay)
        if failing:
            raise self.error()
        return await ShellSimMachineFactory().create(spec)


@dataclass
class FaultyMachine:
    """A ShellSim machine whose ``close`` raises when ``close_error`` is set and whose commands raise
    ``MachineTerminated`` when ``terminated`` is set, as a killed sandbox's do."""

    machine: Machine
    close_error: bool
    terminated: bool

    async def run(self, command: Command) -> Result:
        if self.terminated:
            raise MachineTerminated("sandbox task was killed")
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()
        if self.close_error:
            raise RuntimeError("sandbox delete refused")


@dataclass
class FaultyFactory:
    close_error: bool = False
    terminated: bool = False

    @property
    def backend(self) -> Backend:
        return Backend.SHELLSIM

    async def create(self, spec: MachineSpec) -> Machine:
        return FaultyMachine(await ShellSimMachineFactory().create(spec), self.close_error, self.terminated)


@dataclass(frozen=True)
class Fakes:
    """The fake classes, handed to tests through the ``fakes`` fixture (test modules cannot import each other)."""

    script_model: type[ScriptModel] = ScriptModel
    raising_model: type[RaisingModel] = RaisingModel
    flaky_factory: type[FlakyFactory] = FlakyFactory
    faulty_factory: type[FaultyFactory] = FaultyFactory
    text: Callable[[str], dict[str, Any]] = assistant_text
    shell: Callable[..., dict[str, Any]] = assistant_shell


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()
