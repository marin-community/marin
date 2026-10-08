# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for builder tests: a small proposal and build services over the shared ledger."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import pytest
from rigging.timing import ExponentialBackoff
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend

from taskforge.builder.sdk import BuildServices
from taskforge.ledger.records import Ledger
from taskforge.llm.agent import AgentTool
from taskforge.llm.client import GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.proposal.model import TaskProposal, parse
from taskforge.sandbox.factories import MachineHost

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


@pytest.fixture
def proposal() -> TaskProposal:
    return parse(PROPOSAL)


@pytest.fixture
def services(ledger: Ledger) -> Callable:
    """``async with services(base_url, web_tools=()) as s``: laptop build services with only ShellSim."""

    @asynccontextmanager
    async def make(
        base_url: str = "http://127.0.0.1:9/v1", web_tools: tuple[AgentTool, ...] = ()
    ) -> AsyncIterator[BuildServices]:
        endpoint = GlmEndpoint(base_url=base_url, token="test-token", pool=Pool.HIGH)
        async with GlmClient(endpoint, backoff=ExponentialBackoff(initial=0.001, maximum=0.001)) as client:
            yield BuildServices(
                client=client,
                policy=LLMPolicy(),
                host=MachineHost.LAPTOP,
                factories={Backend.SHELLSIM.value: ShellSimMachineFactory()},
                images=None,
                ledger=ledger,
                web_tools=web_tools,
            )

    return make


GRADE = """
import json, os, pathlib
final = pathlib.Path(os.environ["VERIFYIT_WORKSPACE"], "answer.txt").read_text().strip()
verdict = {"status": "scored", "reward": float(final.endswith("ANSWER = 42")), "detail": {}}
pathlib.Path(os.environ["VERIFYIT_LOGS_DIR"], "verdict.json").write_text(json.dumps(verdict))
"""

PROGRAM = """
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, EnvironmentRequirements, Source, TaskSpec
from taskcompendium.submission import PlainText

CONVENTION = PlainText(id="plain_text")
MACHINE = spec.machine(startup_timeout=60)
SESSION = spec.session(
    max_turns=8,
    model_turn_timeout=None,
    command_timeout=None,
    tool_turn_timeout=None,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=60,
    cleanup_timeout=30,
)
FILES = (spec.file("workspace/question.txt", "6 * 7"),)

GRADE = GRADE_SOURCE


@step(StepRole.ENVIRONMENT)
async def machine(b: Build) -> EnvironmentRequirements:
    return spec.requirements(image=None)


@step(StepRole.GRADER)
async def grader(b: Build, env: EnvironmentRequirements) -> Grader:
    package = spec.script_verifier(GRADE, {}, timeout=60)
    reference = await b.try_grader(env, package, AnswerType.TEXT, CONVENTION, "question", "ANSWER = 42", files=FILES)
    b.check(reference.reward == 1.0, f"reference scored {reference.reward}")
    b.emit("grader/grader.py", GRADE.encode())
    return Grader(package=package, answer_contract="End with ANSWER = <n>.", reference_reply="ANSWER = 42")


@step(StepRole.ASSEMBLE)
async def assemble(b: Build, env: EnvironmentRequirements, graded: Grader) -> TaskSpec:
    return spec.assemble(
        task_id=b.item_id,
        instruction="Compute the product in question.txt. " + graded.answer_contract,
        answer_type=AnswerType.TEXT,
        grader=graded.package,
        source=Source(dataset="test", revision="r1", row="0", importer_revision="test"),
        environment=env,
        files=FILES,
    )


def control(id, kind, category, concern, text, **expect):
    return controls.Control(
        id=id,
        kind=kind,
        category=category,
        concern=concern,
        author="test",
        payload=controls.Transcript((controls.reply(text),)),
        expect=controls.Expectation(status=Outcome.GRADED, **expect),
    )


@step(StepRole.CONTROLS)
async def fixed_controls(b: Build, task: TaskSpec) -> tuple[controls.Control, ...]:
    return (
        control("gold", K.POSITIVE, C.KNOWN_CORRECT, N.REFERENCE, "ANSWER = 42", reward_min=1.0),
        control("empty", K.MALFORMED, C.EMPTY_OR_MALFORMED, N.EXTRACTION, "", reward_max=0.0),
        control("off-by-one", K.NEGATIVE, C.PLAUSIBLE_WRONG, N.ACCEPTANCE, "ANSWER = 41", reward_max=0.0),
        control("sum", K.NEGATIVE, C.TASK_SPECIFIC_SHORTCUT, N.SHORTCUT, "ANSWER = 13", reward_max=0.0),
    )


K, C, N = controls.ControlKind, controls.ControlCategory, controls.ControlConcern


async def build(b: Build) -> BuildOutput:
    env = await machine(b)
    graded = await grader(b, env)
    task = await assemble(b, env, graded)
    lowered = b.lower(task, task_machine=MACHINE, verifier_machine=None, session=SESSION)
    fixed = await fixed_controls(b, task)
    return BuildOutput(task=task, lowered=lowered, convention=CONVENTION, controls=fixed)
""".replace(
    "GRADE_SOURCE", repr(GRADE)
)


@pytest.fixture
def program_source() -> str:
    """A builder program without model calls: a ShellSim task with a host-run script grader and four controls."""
    return PROGRAM
