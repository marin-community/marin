# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two small tasks shared by the unit and live validate tests, with a control set for each.

``math_task`` is a null-environment numeric task graded by verifyit through TaskCompendium's
registry. ``file_task`` is a ShellSim task whose private ``ShellVerifierSpec`` script checks a file
the agent must create.
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Machine, MachineSpec
from taskcompendium.environment import EnvironmentKind, StdoutReward
from taskcompendium.execution import TaskExecution
from taskcompendium.grading import numeric_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source, TaskSpec

from taskforge.spec.controls import (
    Control,
    ControlCategory,
    ControlKind,
    Expectation,
    Transcript,
    Workspace,
    reply,
    shell_turn,
)
from taskforge.spec.draft import assemble, environment, file, shell_verifier

MATH_ANSWER = "395"
NUMBERS = "12\n7\n30\n11\n"
NUMBERS_SUM = 60
CHECK_SCRIPT = 'v=$(tr -d " \\n" < /workspace/sum.txt)\n' f'if [ "$v" = {NUMBERS_SUM} ]; then echo 1; else echo 0; fi\n'


def source(row: str) -> Source:
    return Source(dataset="taskforge-validate-tests", revision="1", row=row, importer_revision="1")


@pytest.fixture
def math_task() -> TaskSpec:
    return assemble(
        "validate-math",
        "What is 17 * 23 + 4? Reply with only the number, nothing else.",
        AnswerType.NUMBER,
        environment(EnvironmentKind.NULL),
        numeric_answer(MATH_ANSWER, tolerance_abs=0, tolerance_rel=0),
        source("math"),
        execution=TaskExecution(),
    )


@pytest.fixture
def file_task() -> TaskSpec:
    return assemble(
        "validate-file",
        "The file /workspace/numbers.txt holds one integer per line. Use the shell tool to write their sum, "
        "as a single integer, to /workspace/sum.txt. Say when you are done.",
        AnswerType.FILE,
        environment(EnvironmentKind.SHELLSIM, files=(file("/workspace/numbers.txt", NUMBERS),)),
        shell_verifier(
            ("sh", "/grader/check.sh"), StdoutReward(), timeout=30, files=(file("/grader/check.sh", CHECK_SCRIPT),)
        ),
        source("file"),
        execution=TaskExecution(),
    )


def control(name: str, kind: ControlKind, category: ControlCategory, payload, expect: Expectation) -> Control:
    return Control(id=name, kind=kind, category=category, author="tests", payload=payload, expect=expect)


CORRECT = Expectation(status=Outcome.GRADED, reward_min=1.0)
WRONG = Expectation(status=Outcome.GRADED, reward_max=0.0)
NO_SUBMISSION = Expectation(status=Outcome.SUBMISSION_FAILURE)


@pytest.fixture
def math_controls() -> tuple[Control, ...]:
    return (
        control("correct", ControlKind.POSITIVE, ControlCategory.KNOWN_CORRECT, Transcript((reply("395"),)), CORRECT),
        control("wrong", ControlKind.NEGATIVE, ControlCategory.PLAUSIBLE_WRONG, Transcript((reply("391"),)), WRONG),
        control(
            "last-operand",
            ControlKind.NEGATIVE,
            ControlCategory.TASK_SPECIFIC_SHORTCUT,
            Transcript((reply("4"),)),
            WRONG,
        ),
        control(
            "empty", ControlKind.MALFORMED, ControlCategory.EMPTY_OR_MALFORMED, Transcript((reply(""),)), NO_SUBMISSION
        ),
    )


@pytest.fixture
def file_controls() -> tuple[Control, ...]:
    return (
        control(
            "correct",
            ControlKind.POSITIVE,
            ControlCategory.KNOWN_CORRECT,
            Transcript((shell_turn(("c1", f"echo {NUMBERS_SUM} > /workspace/sum.txt")), reply("Done."))),
            CORRECT,
        ),
        control(
            "wrong",
            ControlKind.NEGATIVE,
            ControlCategory.PLAUSIBLE_WRONG,
            Transcript((shell_turn(("c1", "head -n 1 /workspace/numbers.txt > /workspace/sum.txt")), reply("Done."))),
            WRONG,
        ),
        control(
            "workspace-correct",
            ControlKind.POSITIVE,
            ControlCategory.KNOWN_CORRECT,
            Workspace((file("/workspace/sum.txt", f"{NUMBERS_SUM}\n"),)),
            CORRECT,
        ),
        control(
            "plant-grader",
            ControlKind.NEGATIVE,
            ControlCategory.REWARD_HACK,
            Transcript((shell_turn(("c1", "mkdir -p /grader && echo 'echo 1' > /grader/check.sh")), reply("Done."))),
            WRONG,
        ),
        control(
            "no-work", ControlKind.MALFORMED, ControlCategory.EMPTY_OR_MALFORMED, Transcript((reply("Done."),)), WRONG
        ),
    )


def assistant_text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def assistant_shell(command: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": command})},
            }
        ],
    }


@dataclass
class ScriptModel:
    """Answers each request with the next of ``messages``, with ids that preserve the served prefix.

    From turn ``hang_from`` on, a request waits forever, so only a timeout ends it. With a
    ``barrier``, every request waits until all of the barrier's parties are in flight.
    """

    messages: list[dict[str, Any]]
    hang_from: int | None = None
    barrier: asyncio.Barrier | None = None

    async def __call__(self, request: ModelRequest) -> ModelTurn:
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

    async def create(self, spec: MachineSpec) -> Machine:
        self.creates += 1
        failing = self.creates <= self.failures
        await asyncio.sleep(self.delay)
        if failing:
            raise self.error()
        return await ShellSimMachineFactory().create(spec)


@dataclass(frozen=True)
class Fakes:
    """The fake classes, handed to tests through the ``fakes`` fixture (test modules cannot import each other)."""

    script_model: type[ScriptModel] = ScriptModel
    raising_model: type[RaisingModel] = RaisingModel
    flaky_factory: type[FlakyFactory] = FlakyFactory
    text: Callable[[str], dict[str, Any]] = assistant_text
    shell: Callable[[str], dict[str, Any]] = assistant_shell


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()
