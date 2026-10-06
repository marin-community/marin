# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small tasks shared by the unit and live validate tests, with a control set for two of them.

``math_task`` is a null-environment numeric task graded by verifyit through TaskCompendium's
registry; ``json_task`` is a null-environment task with a JSON answer. ``file_task`` is a ShellSim
task whose private ``ShellVerifierSpec`` script checks a file the agent must create. ``rounds`` builds
validation-round inputs: a ``TaskDraft`` around a task, a ``ValidationPolicy`` and a ``ValidationSite``.
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from rigging.timing import ExponentialBackoff
from rolloutengine.contracts import ModelRequest, ModelTurn
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ExitReason, Machine, MachineSpec, Result
from taskcompendium.environment import EnvironmentKind, StdoutReward
from taskcompendium.execution import TaskExecution
from taskcompendium.grading import numeric_answer, structured_exact
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source, TaskSpec
from taskcompendium.submission import SubmissionConvention

from taskforge.build.run import Provenance, TaskDraft
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.policy import LLMPolicy
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
from taskforge.spec.draft import assemble, environment, file, shell_verifier
from taskforge.validate.adversary import AdversaryRole
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.solver import ValidationSite
from taskforge.validate.trials import Deadlines

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
def json_task() -> TaskSpec:
    return assemble(
        "validate-json",
        "Report the sum of 12, 7, 30 and 11 as a JSON object whose only key is sum.",
        AnswerType.JSON,
        environment(EnvironmentKind.NULL),
        structured_exact({"sum": NUMBERS_SUM}),
        source("json"),
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
            Workspace((file("/workspace/sum.txt", f"{NUMBERS_SUM}\n"),)),
            CORRECT,
        ),
        control(
            "plant-grader",
            ControlKind.NEGATIVE,
            ControlCategory.REWARD_HACK,
            ControlConcern.SHORTCUT,
            Transcript((shell_turn(("c1", "mkdir -p /grader && echo 'echo 1' > /grader/check.sh")), reply("Done."))),
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

    async def create(self, spec: MachineSpec) -> Machine:
        self.creates += 1
        failing = self.creates <= self.failures
        await asyncio.sleep(self.delay)
        if failing:
            raise self.error()
        return await ShellSimMachineFactory().create(spec)


@dataclass
class FaultyMachine:
    """A ShellSim machine whose ``close`` raises when ``close_error`` is set and whose ``failing_argv``
    command exits 1."""

    machine: Machine
    close_error: bool
    failing_argv: tuple[str, ...] | None

    async def run(self, command: Command) -> Result:
        if command.argv == self.failing_argv:
            return Result(1, b"", b"", False, False, ExitReason.EXITED)
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def open_shell(self):
        return await self.machine.open_shell()

    async def close(self) -> None:
        await self.machine.close()
        if self.close_error:
            raise RuntimeError("sandbox delete refused")


@dataclass
class FaultyFactory:
    close_error: bool = False
    failing_argv: tuple[str, ...] | None = None

    async def create(self, spec: MachineSpec) -> Machine:
        return FaultyMachine(await ShellSimMachineFactory().create(spec), self.close_error, self.failing_argv)


@dataclass(frozen=True)
class Fakes:
    """The fake classes, handed to tests through the ``fakes`` fixture (test modules cannot import each other)."""

    script_model: type[ScriptModel] = ScriptModel
    raising_model: type[RaisingModel] = RaisingModel
    flaky_factory: type[FlakyFactory] = FlakyFactory
    faulty_factory: type[FaultyFactory] = FaultyFactory
    text: Callable[[str], dict[str, Any]] = assistant_text
    shell: Callable[[str], dict[str, Any]] = assistant_shell


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


def draft_of(task: TaskSpec, controls: tuple[Control, ...], convention: SubmissionConvention) -> TaskDraft:
    """``task`` as a built draft; the provenance names no real program."""
    provenance = Provenance(
        item_id=task.id,
        proposal_digest="proposal",
        program_digest="program",
        sdk_version="tests",
        model="tests",
        policy_digest="policy",
        round=0,
        steps=(),
        resources=(),
    )
    return TaskDraft(task, TaskExecution(), convention, controls, provenance)


def validation_policy(
    k: int = 3, adversary_k: int = 2, max_retries: int = 0, token_contract_retries: int = 0
) -> ValidationPolicy:
    return ValidationPolicy(
        k=k,
        adversary_k=adversary_k,
        roles=tuple(AdversaryRole),
        band=CalibrationBand(0.125, 0.875),
        sampling=LLMPolicy(max_continuations=0),
        deadlines=Deadlines(agent_timeout=30, attempt_timeout=60),
        max_retries=max_retries,
        token_contract_retries=token_contract_retries,
        retry_backoff=ExponentialBackoff(initial=0.001, maximum=0.001),
    )


def validation_site(directory: Path) -> ValidationSite:
    return ValidationSite("item", 0, directory / "evidence", JsonlLedger(directory / "ledger"))


@dataclass(frozen=True)
class Rounds:
    """Builders for validation-round inputs, handed to tests through the ``rounds`` fixture."""

    draft: Callable[..., TaskDraft] = draft_of
    policy: Callable[..., ValidationPolicy] = validation_policy
    site: Callable[[Path], ValidationSite] = validation_site


@pytest.fixture
def rounds() -> Rounds:
    return Rounds()
