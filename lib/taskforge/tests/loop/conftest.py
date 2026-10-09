# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fakes at the loop's I/O boundaries: a proposal source, a rubric, a solver model and a tokenizer.

The author is the real ``builder.author.author`` against the ``fake_glm`` router, which serves scripted
``submit_build_program`` calls; the builder programs it returns make no model call and build a ShellSim
task whose Python grader, run in a verifier machine on the ShellSim-backed fixture image factory, gives
full credit only to ``ANSWER = 42`` (a lenient program's grader to any ``ANSWER = <int>``). Validation runs
on ShellSim through RolloutEngine. The adversary is the real agent
loop against the same router: ``adversary_turns`` queues its turns after the author's, and every
``submit`` it makes is graded by the task's real verifier. ``Loop`` assembles a run root and its
``LoopServices`` for one test.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend

from taskforge.builder.author import SUBMIT_TOOL
from taskforge.builder.sdk import BuildServices
from taskforge.builder.template import standard
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import LedgerEntry
from taskforge.llm.client import Completion, FinishReason, GlmClient, GlmEndpoint, GlmUnavailable, Pool, Usage
from taskforge.llm.policy import LLMPolicy
from taskforge.loop.policy import LoopPolicy
from taskforge.loop.program import LEDGER_DIR, LoopServices
from taskforge.proposal.model import TaskProposal, parse, render
from taskforge.proposal.source import ProposalBatch, SlotFailure, SlotProposal
from taskforge.review.rules import BandChoice, BandRule, BandRules
from taskforge.sandbox.factories import LOCAL_DOCKER, SHELLSIM, MachineHost
from taskforge.triage.checks import CheckContext, CheckResult
from taskforge.triage.program import Repair as TriageRepair
from taskforge.triage.program import RubricAssessment
from taskforge.triage.verdict import ModelCall, RubricAxis, RubricResult, TriageDecision, Verdict
from taskforge.validate.adversary import NO_SHORTCUT_LINE, SUBMIT_TOOL_NAME, AdversaryContext
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff
from tests.sandbox.fixture_images import FixtureImageFactory
from tests.validate.conftest import TemplateTokenizer

CORRECT = "ANSWER = 42"
WRONG = "ANSWER = 41"
FACTORIES = {Backend.SHELLSIM.value: ShellSimMachineFactory(), Backend.DOCKER.value: FixtureImageFactory()}
"""ShellSim task machines; verifier machines on the ShellSim-backed fixture image factory."""
FAST = RetryBackoff(initial=0.001, maximum=0.001, factor=1.5, jitter=0.1)

PROPOSAL = """---
id: "d00.arithmetic.products/{slot}"
source: {{kind: capability, ref: "d00.arithmetic.products", hash: "abc123"}}
environment: reasoning
verification: simple
grounding: unverified
research:
  - {{kind: web, purpose: "confirm the multiplication table"}}
build:
  - "grader": "checks the final ANSWER line"
resources: ["grader/grade.py"]
null_reason: null
---
## Task
Multiply 6 by 7 and end the reply with `ANSWER = <n>`.{note}

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

GRADE = """
import pathlib
final = pathlib.Path("/app/answer.txt").read_text().strip()
print(float(final.endswith("ANSWER = 42")))
"""

LENIENT_GRADE = """
import pathlib, re
final = pathlib.Path("/app/answer.txt").read_text().strip()
print(float(bool(re.search(r"ANSWER = -?[0-9]+$", final))))
"""

PROGRAM = """
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, EnvironmentRequirements, PlainText, Source, TaskSpec

ANSWER_FORMAT = PlainText()
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
    package = spec.python_grader(
        GRADE, {}, environment=spec.grader_environment(None), answer_path=spec.ANSWER_PATH, timeout=GRADER_TIMEOUT
    )
    reference = await b.try_grader(env, package, AnswerType.TEXT, ANSWER_FORMAT, "question", "ANSWER = 42", files=FILES)
    b.check(reference.reward == 1.0, f"reference scored {reference.reward}")
    return Grader(package=package, answer_contract="End with ANSWER = <n>.", reference_reply="ANSWER = 42")


@step(StepRole.ASSEMBLE)
async def assemble(b: Build, env: EnvironmentRequirements, graded: Grader) -> TaskSpec:
    return spec.assemble(
        task_id=b.item_id,
        instruction="Compute the product in question.txt. " + graded.answer_contract,
        answer_type=AnswerType.TEXT,
        answer_format=ANSWER_FORMAT,
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
        control("off-by-one", K.NEGATIVE, C.PLAUSIBLE_WRONG, N.ACCEPTANCE, WRONG_REPLY, reward_max=0.0),
        control("sum", K.NEGATIVE, C.TASK_SPECIFIC_SHORTCUT, N.SHORTCUT, SUM_REPLY, reward_max=0.0),
    )


K, C, N = controls.ControlKind, controls.ControlCategory, controls.ControlConcern
# The lenient grader accepts any ANSWER line, so its negative controls carry none.
WRONG_REPLY = "The product is 41." if LENIENT else "ANSWER = 41"
SUM_REPLY = "The sum is 13." if LENIENT else "ANSWER = 13"


async def build(b: Build) -> BuildOutput:
    env = await machine(b)
    graded = await grader(b, env)
    task = await assemble(b, env, graded)
    verifier_machine = None if spec.grading_environment(task) is None else MACHINE
    lowered = b.lower(task, task_machine=MACHINE, verifier_machine=verifier_machine, session=SESSION)
    return BuildOutput(task=task, lowered=lowered, controls=await fixed_controls(b, task))
"""


def program(grader_timeout: int = 60, lenient: bool = False) -> str:
    """A builder program without model calls; a different ``grader_timeout`` builds a different task.

    A ``lenient`` program's grader accepts any ``ANSWER = <int>`` line; its negative controls end on no
    such line, so they hold under it.
    """
    return (
        PROGRAM.replace("GRADE_SOURCE", repr(LENIENT_GRADE if lenient else GRADE))
        .replace("GRADER_TIMEOUT", str(grader_timeout))
        .replace("LENIENT", repr(lenient))
    )


def proposal(slot: int = 1, note: str = "") -> TaskProposal:
    return parse(PROPOSAL.format(slot=slot, note=note))


def submit(fake_glm, source: str) -> None:
    """Queue one ``submit_build_program`` reply on the fake router."""
    fake_glm.stream(tool_calls=((SUBMIT_TOOL, json.dumps({"source": source, "notes": "n"})),), finish="tool_calls")


type AdversaryTurn = str | tuple[str, str] | tuple[str, str, list[str]]


def adversary_turns(fake_glm, *turns: AdversaryTurn) -> None:
    """Queue one adversary agent loop on the fake router, one streamed reply per turn.

    ``("shell", command)`` and ``("submit", reply, files)`` are tool-call turns; a ``str`` is the final
    text reply that ends the loop. With no turns the adversary replies ``NO_SHORTCUT`` at once.
    """
    for turn in turns or (NO_SHORTCUT_LINE,):
        match turn:
            case str(text):
                fake_glm.stream(content=text)
            case ("shell", command):
                fake_glm.stream(tool_calls=(("shell", json.dumps({"command": command})),), finish="tool_calls")
            case (name, reply, files):
                assert name == SUBMIT_TOOL_NAME, turn
                arguments = json.dumps({"reply": reply, "files": files})
                fake_glm.stream(tool_calls=((SUBMIT_TOOL_NAME, arguments),), finish="tool_calls")
            case _:
                raise ValueError(f"not an adversary turn: {turn!r}")


def authored(fake_glm) -> int:
    """Requests the router served to the author (each offers ``submit_build_program``)."""
    return sum(
        any(tool["function"]["name"] == SUBMIT_TOOL for tool in request.get("tools", ()))
        for request in fake_glm.requests
    )


def model_call(completion_tokens: int = 0) -> ModelCall:
    return ModelCall(Usage(10, completion_tokens, 0, 0), 0.1, FinishReason.STOP)


def rubric_result(decision: TriageDecision) -> RubricResult:
    score = 5 if decision is TriageDecision.ACCEPT else 2
    return RubricResult(
        scores=tuple((axis, score) for axis in RubricAxis),
        critical_failures=(),
        issues=(),
        required_changes=(),
        recommendation=decision,
    )


@dataclass
class FakeRubric:
    """Decides each assessment from ``decisions`` in order; a repair appends a note to the task section."""

    decisions: list[TriageDecision]
    tokens_out: int = 0
    raise_on_assess: Exception | None = None
    assessed: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)

    async def assess(self, p: TaskProposal, structural: tuple[CheckResult, ...]) -> RubricAssessment:
        if self.raise_on_assess is not None:
            error, self.raise_on_assess = self.raise_on_assess, None
            raise error
        self.assessed.append(p.digest)
        decision = self.decisions.pop(0)
        return RubricAssessment((rubric_result(decision),), (model_call(self.tokens_out),))

    async def repair(self, p: TaskProposal, verdict: Verdict) -> TriageRepair:
        self.repaired.append(p.digest)
        return TriageRepair(replace(p, body=p.body + f"\nRepair {len(self.repaired)}.\n"), model_call())


def completion(content: str) -> Completion:
    return Completion(content, "", (), FinishReason.STOP, Usage(10, 5, 0, 0), 0.1, 0.01, 0.05, 0, ())


def slot_request(idea: str, slot: int) -> tuple[dict[str, str], ...]:
    return ({"role": "user", "content": f"write slot {slot} of {idea}"},)


@dataclass
class FakeSource:
    """Serves ``batches`` in order; each is a tuple of slot outcomes (a proposal, or an error string).

    Every batch carries a planning request and completion; a proposal slot one completion, or two
    with a ``repair_error`` when its slot is in ``repaired``; a failed slot two completions.
    """

    batches: list[tuple[TaskProposal | str, ...]]
    repaired: frozenset[int] = frozenset()
    calls: int = 0

    async def propose(self, idea: str, n: int) -> ProposalBatch:
        self.calls += 1
        slots = tuple(self.outcome(idea, slot, outcome) for slot, outcome in enumerate(self.batches.pop(0)))
        return ProposalBatch(({"role": "user", "content": f"plan {n} slots for {idea}"},), (completion("plan"),), slots)

    def outcome(self, idea: str, slot: int, outcome: TaskProposal | str) -> SlotProposal | SlotFailure:
        request = slot_request(idea, slot)
        if isinstance(outcome, str):
            return SlotFailure(slot, outcome, request, (completion("not a proposal"), completion("still not one")))
        if slot in self.repaired:
            first = completion("not a proposal")
            return SlotProposal(slot, outcome, request, (first, completion(render(outcome))), "front matter missing")
        return SlotProposal(slot, outcome, request, (completion(render(outcome)),), None)


def describe_idea(idea: str) -> dict[str, object]:
    return {"idea": idea, "kind": "test"}


@dataclass
class RolloutFake:
    """The solver's model: one text turn per trial, cycling through ``solver``.

    ``error`` makes every call raise it instead. Token ids extend each request's served prefix.
    """

    solver: tuple[str, ...] = (CORRECT, WRONG)
    error: Callable[[], BaseException] | None = None
    calls: int = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if self.error is not None:
            raise self.error()
        reply = self.solver[self.calls % len(self.solver)]
        self.calls += 1
        prompt = (*request.prefix_token_ids, 90) if request.prefix_token_ids else (10, 11)
        return ModelTurn({"role": "assistant", "content": reply}, prompt, (20,), (-0.5,), "stop")


def validation_policy() -> ValidationPolicy:
    return ValidationPolicy(
        k=4,
        adversary_k=1,
        adversary_submissions=4,
        adversary_repair_submissions=2,
        band=CalibrationBand(0.125, 0.875),
        sampling=LLMPolicy(max_continuations=0),
        deadlines=Deadlines(total_turn_timeout=30, attempt_timeout=60),
        max_retries=0,
        token_contract_retries=0,
        retry_backoff=FAST,
    )


def loop_policy(**changes: Any) -> LoopPolicy:
    policy = LoopPolicy(
        proposals_per_idea=2,
        max_idea_reproposals=1,
        max_triage_repairs=1,
        max_build_revisions=2,
        max_repairs=1,
        max_validation_retries=1,
        max_build_retries=1,
        retry_backoff=FAST,
        output_token_budget=1_000_000,
        band_rules=BandRules(BandRule(1, BandChoice.REJECT), BandRule(1, BandChoice.REJECT)),
        validation=validation_policy(),
    )
    return replace(policy, **changes)


def no_context(proposal: TaskProposal) -> str:
    return ""


@dataclass
class Loop:
    """A run root under ``root`` with the fakes a test drives; ``services()`` opens its ``LoopServices``.

    ``context`` is the consumer's adversary context; by default there is none.
    """

    root: Path
    fake_glm: Any
    rubric: FakeRubric = field(default_factory=lambda: FakeRubric([TriageDecision.ACCEPT] * 4))
    source: FakeSource = field(default_factory=lambda: FakeSource([]))
    model: RolloutFake = field(default_factory=RolloutFake)
    tokenizer: TemplateTokenizer = field(default_factory=TemplateTokenizer)
    context: AdversaryContext = no_context

    @asynccontextmanager
    async def services(self, width: int = 8) -> AsyncIterator[LoopServices]:
        endpoint = GlmEndpoint(base_url=self.fake_glm.base_url, token="test-token", pool=Pool.HIGH)
        ledger = JsonlLedger(self.root / LEDGER_DIR)
        async with GlmClient(endpoint, backoff=FAST.schedule()) as client:
            yield LoopServices(
                client=client,
                source=self.source,
                describe_idea=describe_idea,
                checks=(),
                rubric=self.rubric,
                check_context=CheckContext(allowed_combinations=frozenset()),
                template=standard,
                build=BuildServices(
                    client=client,
                    policy=LLMPolicy(),
                    host=MachineHost.LAPTOP,
                    factories=FACTORIES,
                    images=None,
                    ledger=ledger,
                ),
                engine=EngineSettings(
                    factories=FACTORIES,
                    capabilities={Backend.SHELLSIM.value: SHELLSIM, Backend.DOCKER.value: LOCAL_DOCKER},
                    max_turns=6,
                    command_timeout=10,
                    tool_turn_timeout=20,
                    model_turn_timeout=30,
                    cleanup_timeout=10,
                ),
                rollout_models=lambda _: self.model,
                adversary_context=self.context,
                tokenize=self.tokenizer,
                ledger=ledger,
                root=self.root,
                slots=asyncio.Semaphore(width),
            )

    def entries(self, item_id: str) -> list[LedgerEntry]:
        return list(read_entries(self.root / LEDGER_DIR / f"{item_id}.jsonl"))


@dataclass(frozen=True)
class Programs:
    """The builder program sources and proposals tests script, through the ``programs`` fixture."""

    source: Callable[..., str] = program
    proposal: Callable[..., TaskProposal] = proposal
    submit: Callable[[Any, str], None] = submit
    adversary_turns: Callable[..., None] = adversary_turns
    authored: Callable[[Any], int] = authored
    model_call: Callable[..., ModelCall] = model_call
    policy: Callable[..., LoopPolicy] = loop_policy


@pytest.fixture
def programs() -> Programs:
    return Programs()


@pytest.fixture
def loop(tmp_path, fake_glm) -> Loop:
    return Loop(tmp_path / "run", fake_glm)


@pytest.fixture
def unavailable() -> Callable[[], BaseException]:
    return lambda: GlmUnavailable("router drained", ())
