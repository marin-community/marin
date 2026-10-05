# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for builder tests: a small proposal, a recording ledger, and build services."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import pytest
from rigging.timing import ExponentialBackoff
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind

from taskforge.build.sdk import BuildServices
from taskforge.ledger.records import LedgerEntry
from taskforge.llm.agent import AgentTool
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.proposal.model import TaskProposal, parse

PROPOSAL = """---
id: "d00.arithmetic.products/1"
source: {kind: capability, ref: "d00.arithmetic.products", hash: "abc123"}
environment: reasoning
verification: simple
grounding: unverified
research:
  - {kind: web, purpose: "confirm the multiplication table"}
build:
  - "grader": "checks the final ANSWER line"
resources: ["grader/grade.py"]
null_reason: null
---
## Task
Multiply 6 by 7 and end the reply with `ANSWER = <n>`.

## Realism and workflow
A clerk checks a product.

## Research plan
Confirm the table.

## Build plan
Write the question and the grader.

## Grader design and controls
Full credit only for ANSWER = 42.

## Risks and null conditions
None.
"""


class ListLedger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


@pytest.fixture
def proposal() -> TaskProposal:
    return parse(PROPOSAL)


@pytest.fixture
def ledger() -> ListLedger:
    return ListLedger()


@pytest.fixture
def services(ledger: ListLedger) -> Callable:
    """``async with services(base_url, web_tools=()) as s``: build services over a GLM endpoint."""

    @asynccontextmanager
    async def make(
        base_url: str = "http://127.0.0.1:9/v1", web_tools: tuple[AgentTool, ...] = ()
    ) -> AsyncIterator[BuildServices]:
        endpoint = GlmEndpoint(base_url=base_url, token="test-token", pool=Pool.HIGH)
        async with GlmClient(endpoint, backoff=ExponentialBackoff(initial=0.001, maximum=0.001)) as client:
            yield BuildServices(
                client=client,
                policy=LLMPolicy(),
                factories={EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
                ledger=ledger,
                web_tools=web_tools,
            )

    return make


GRADE = """
import json, sys
messages = json.load(sys.stdin)
final = [m for m in messages if m.get("role") == "assistant"][-1].get("content") or ""
print(1.0 if final.strip().endswith("ANSWER = 42") else 0.0)
"""

PROGRAM = """
from taskcompendium.environment import EnvironmentKind, EnvironmentSpec, StdoutReward
from taskcompendium.grading import Outcome
from taskcompendium.models import AnswerType, Source, TaskSpec

GRADE = GRADE_SOURCE


@step(StepRole.ENVIRONMENT)
async def machine(b: Build) -> EnvironmentSpec:
    return spec.environment(EnvironmentKind.SHELLSIM, files=(spec.file("/workspace/question.txt", "6 * 7"),))


@step(StepRole.GRADER)
async def grader(b: Build, env: EnvironmentSpec) -> Grader:
    verifier = spec.shell_verifier(
        argv=("python3", "/grader/grade.py"),
        reward=StdoutReward(),
        timeout=60,
        files=(spec.file("/grader/grade.py", GRADE),),
    )
    reference = await b.try_grader(env, verifier, "question", "ANSWER = 42")
    b.check(reference.reward == 1.0, f"reference scored {reference.reward}")
    b.emit("grader/grade.py", GRADE.encode())
    return Grader(verifier=verifier, answer_contract="End with ANSWER = <n>.", reference_reply="ANSWER = 42")


@step(StepRole.ASSEMBLE)
async def assemble(b: Build, env: EnvironmentSpec, graded: Grader) -> TaskSpec:
    return spec.assemble(
        task_id=b.item_id,
        instruction="Compute the product in question.txt. " + graded.answer_contract,
        answer_type=AnswerType.TEXT,
        environment=env,
        verifier=graded.verifier,
        source=Source(dataset="test", revision="r1", row="0", importer_revision="test"),
    )


def control(id, kind, category, text, **expect):
    return controls.Control(
        id=id,
        kind=kind,
        category=category,
        author="test",
        payload=controls.Transcript((controls.reply(text),)),
        expect=controls.Expectation(status=Outcome.GRADED, **expect),
    )


@step(StepRole.CONTROLS)
async def fixed_controls(b: Build, task: TaskSpec) -> tuple[controls.Control, ...]:
    return (
        control("gold", K.POSITIVE, C.KNOWN_CORRECT, "ANSWER = 42", reward_min=1.0),
        control("empty", K.MALFORMED, C.EMPTY_OR_MALFORMED, "", reward_max=0.0),
        control("off-by-one", K.NEGATIVE, C.PLAUSIBLE_WRONG, "ANSWER = 41", reward_max=0.0),
        control("sum", K.NEGATIVE, C.TASK_SPECIFIC_SHORTCUT, "ANSWER = 13", reward_max=0.0),
    )


K, C = controls.ControlKind, controls.ControlCategory


async def build(b: Build) -> BuildOutput:
    env = await machine(b)
    graded = await grader(b, env)
    task = await assemble(b, env, graded)
    return BuildOutput(task=task, controls=await fixed_controls(b, task))
""".replace(
    "GRADE_SOURCE", repr(GRADE)
)


@pytest.fixture
def program_source() -> str:
    """A builder program without model calls: a ShellSim task with a script grader and four controls."""
    return PROGRAM
