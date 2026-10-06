# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fakes at the loop's I/O boundaries: a proposal source, a rubric, a rollout model and a tokenizer.

The author is the real ``build.author.author`` against the ``fake_glm`` router, which serves scripted
``submit_build_program`` calls; the builder programs it returns make no model call and build a ShellSim
task whose shell grader gives full credit only to ``ANSWER = 42``. Validation runs on ShellSim through
RolloutEngine. ``Loop`` assembles a run root and its ``LoopServices`` for one test.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.submission import PlainText

from taskforge.build.author import SUBMIT_TOOL
from taskforge.build.sdk import BuildServices
from taskforge.build.template import standard
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import LedgerEntry
from taskforge.llm.client import FinishReason, GlmClient, GlmEndpoint, GlmUnavailable, Pool, Usage
from taskforge.llm.policy import LLMPolicy
from taskforge.loop.policy import LoopPolicy
from taskforge.loop.program import LEDGER_DIR, LoopServices
from taskforge.proposal.model import TaskProposal, parse
from taskforge.proposal.source import ProposalBatch, SlotFailure, SlotProposal
from taskforge.sandbox.factories import SHELLSIM
from taskforge.triage.checks import CheckContext, CheckResult
from taskforge.triage.program import Repair as TriageRepair
from taskforge.triage.program import RubricAssessment
from taskforge.triage.verdict import ModelCall, RubricAxis, RubricResult, TriageDecision, Verdict
from taskforge.validate.adversary import ROLE_PREAMBLES, SENTINEL_REPLIES, AdversaryRole
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff

CORRECT = "ANSWER = 42"
WRONG = "ANSWER = 41"
CONVENTION = PlainText(id="plain_text")
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
import json, sys
messages = json.load(sys.stdin)
final = [m for m in messages if m.get("role") == "assistant"][-1].get("content") or ""
print(1.0 if final.strip().endswith("ANSWER = 42") else 0.0)
"""

PROGRAM = """
from taskcompendium.environment import EnvironmentKind, EnvironmentSpec, StdoutReward
from taskcompendium.execution import StageExecution, TaskExecution
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source, StageRewardStrategy, TaskSpec
from taskcompendium.submission import PlainText

STAGES = STAGE_NAMES
EXECUTION = TaskExecution(stages={name: StageExecution() for name in STAGES})
CONVENTION = PlainText(id="plain_text")

GRADE = GRADE_SOURCE


@step(StepRole.ENVIRONMENT)
async def machine(b: Build) -> EnvironmentSpec:
    return spec.environment(EnvironmentKind.SHELLSIM, files=(spec.file("/workspace/question.txt", "6 * 7"),))


@step(StepRole.GRADER)
async def grader(b: Build, env: EnvironmentSpec) -> Grader:
    verifier = spec.shell_verifier(
        argv=("python3", "/grader/grade.py"),
        reward=StdoutReward(),
        timeout=GRADER_TIMEOUT,
        files=(spec.file("/grader/grade.py", GRADE),),
    )
    reference = await b.try_grader(env, verifier, AnswerType.TEXT, CONVENTION, "question", "ANSWER = 42")
    b.check(reference.reward == 1.0, f"reference scored {reference.reward}")
    return Grader(verifier=verifier, answer_contract="End with ANSWER = <n>.", reference_reply="ANSWER = 42")


@step(StepRole.ASSEMBLE)
async def assemble(b: Build, env: EnvironmentSpec, graded: Grader) -> TaskSpec:
    stages = tuple(
        spec.stage(name, graded.verifier, instruction=None if index == 0 else "Check it again.")
        for index, name in enumerate(STAGES)
    )
    return spec.assemble(
        task_id=b.item_id,
        instruction="Compute the product in question.txt. " + graded.answer_contract,
        answer_type=AnswerType.TEXT,
        environment=env,
        verifier=spec.staged(StageRewardStrategy.FINAL) if stages else graded.verifier,
        source=Source(dataset="test", revision="r1", row="0", importer_revision="test"),
        execution=EXECUTION,
        stages=stages,
    )


def control(id, kind, category, text, stage, **expect):
    return controls.Control(
        id=f"{id}-{stage}",
        kind=kind,
        category=category,
        author="test",
        payload=controls.Transcript((controls.reply(text),)),
        expect=controls.Expectation(status=Outcome.GRADED, **expect),
        stage=stage,
    )


@step(StepRole.CONTROLS)
async def fixed_controls(b: Build, task: TaskSpec) -> tuple[controls.Control, ...]:
    return tuple(
        c
        for stage in range(max(1, len(STAGES)))
        for c in (
            control("gold", K.POSITIVE, C.KNOWN_CORRECT, "ANSWER = 42", stage, reward_min=1.0),
            control("empty", K.MALFORMED, C.EMPTY_OR_MALFORMED, "", stage, reward_max=0.0),
            control("off-by-one", K.NEGATIVE, C.PLAUSIBLE_WRONG, "ANSWER = 41", stage, reward_max=0.0),
            control("sum", K.NEGATIVE, C.TASK_SPECIFIC_SHORTCUT, "ANSWER = 13", stage, reward_max=0.0),
        )
    )


K, C = controls.ControlKind, controls.ControlCategory


async def build(b: Build) -> BuildOutput:
    env = await machine(b)
    graded = await grader(b, env)
    task = await assemble(b, env, graded)
    return BuildOutput(task=task, execution=EXECUTION, convention=CONVENTION, controls=await fixed_controls(b, task))
"""


def program(grader_timeout: int = 60, staged: bool = False) -> str:
    """A builder program without model calls; a different ``grader_timeout`` builds a different task."""
    names = ("one", "two") if staged else ()
    return (
        PROGRAM.replace("GRADE_SOURCE", repr(GRADE))
        .replace("GRADER_TIMEOUT", str(grader_timeout))
        .replace("STAGE_NAMES", repr(names))
    )


def proposal(slot: int = 1, note: str = "") -> TaskProposal:
    return parse(PROPOSAL.format(slot=slot, note=note))


def submit(fake_glm, source: str) -> None:
    """Queue one ``submit_build_program`` reply on the fake router."""
    fake_glm.stream(tool_calls=((SUBMIT_TOOL, json.dumps({"source": source, "notes": "n"})),), finish="tool_calls")


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


@dataclass
class FakeSource:
    """Serves ``batches`` in order; each is a tuple of slot outcomes (a proposal, or an error string)."""

    batches: list[tuple[TaskProposal | str, ...]]
    calls: int = 0

    async def propose(self, idea: str, n: int) -> ProposalBatch:
        self.calls += 1
        slots = tuple(
            (
                SlotProposal(slot, outcome, (), (), None)
                if isinstance(outcome, TaskProposal)
                else SlotFailure(slot, outcome, (), ())
            )
            for slot, outcome in enumerate(self.batches.pop(0))
        )
        return ProposalBatch((), (), slots)


ROLE_REPLIES = {
    AdversaryRole.SHORTCUT: SENTINEL_REPLIES[AdversaryRole.SHORTCUT],
    AdversaryRole.LEAK: SENTINEL_REPLIES[AdversaryRole.LEAK],
    AdversaryRole.AMBIGUITY: WRONG,
}


@dataclass
class RolloutFake:
    """Replies with one text turn: solver trials cycle through ``solver``; a role's trials reply ``roles[role]``.

    ``error`` makes every call raise it instead; a role request first awaits ``before_role``. Token ids
    extend each request's served prefix.
    """

    solver: tuple[str, ...] = (CORRECT, WRONG)
    roles: dict[AdversaryRole, str] = field(default_factory=lambda: dict(ROLE_REPLIES))
    error: Callable[[], BaseException] | None = None
    before_role: Callable[[], Awaitable[None]] | None = None
    calls: int = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if self.error is not None:
            raise self.error()
        system = next((m["content"] for m in request.messages if m["role"] == "system"), "")
        role = next((role for role, preamble in ROLE_PREAMBLES.items() if system.startswith(preamble)), None)
        if role is None:
            reply = self.solver[self.calls % len(self.solver)]
            self.calls += 1
        else:
            if self.before_role is not None:
                await self.before_role()
            reply = self.roles[role]
        prompt = (*request.prefix_token_ids, 90) if request.prefix_token_ids else (10, 11)
        return ModelTurn({"role": "assistant", "content": reply}, prompt, (20,), (-0.5,), "stop")


ROLE_IDS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}


def render_ids(messages) -> tuple[int, ...]:
    ids: list[int] = []
    for message in messages:
        ids.append(ROLE_IDS[message["role"]])
        ids.extend(json.dumps({key: message[key] for key in ("content", "tool_calls") if key in message}).encode())
    return tuple(ids)


@dataclass
class TemplateTokenizer:
    """The server's chat template, deterministically, for control replay; counts its calls."""

    calls: int = 0

    async def prompt_ids(self, messages, options):
        self.calls += 1
        return (*render_ids(messages), ROLE_IDS["assistant"])

    async def rendered_ids(self, messages, options):
        self.calls += 1
        return render_ids(messages)


class Crash(BaseException):
    """Stands in for the process dying mid-phase: not an ``Exception``, so no FAILED event is written."""


def validation_policy() -> ValidationPolicy:
    return ValidationPolicy(
        k=4,
        adversary_k=1,
        roles=tuple(AdversaryRole),
        band=CalibrationBand(0.125, 0.875),
        sampling=LLMPolicy(max_continuations=0),
        deadlines=Deadlines(agent_timeout=30, attempt_timeout=60),
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
        retry_backoff=FAST,
        output_token_budget=1_000_000,
        validation=validation_policy(),
    )
    return replace(policy, **changes)


@dataclass
class Loop:
    """A run root under ``root`` with the fakes a test drives; ``services()`` opens its ``LoopServices``."""

    root: Path
    fake_glm: Any
    rubric: FakeRubric = field(default_factory=lambda: FakeRubric([TriageDecision.ACCEPT] * 4))
    source: FakeSource = field(default_factory=lambda: FakeSource([]))
    model: RolloutFake = field(default_factory=RolloutFake)
    tokenizer: TemplateTokenizer = field(default_factory=TemplateTokenizer)

    @asynccontextmanager
    async def services(self, width: int = 8) -> AsyncIterator[LoopServices]:
        endpoint = GlmEndpoint(base_url=self.fake_glm.base_url, token="test-token", pool=Pool.HIGH)
        ledger = JsonlLedger(self.root / LEDGER_DIR)
        async with GlmClient(endpoint, backoff=FAST.schedule()) as client:
            yield LoopServices(
                client=client,
                source=self.source,
                checks=(),
                rubric=self.rubric,
                check_context=CheckContext(allowed_combinations=frozenset()),
                template=standard,
                build=BuildServices(
                    client=client,
                    policy=LLMPolicy(),
                    factories={EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
                    ledger=ledger,
                ),
                engine=EngineSettings(
                    factories={EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
                    capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
                    max_turns=6,
                    command_timeout=10,
                    cleanup_timeout=10,
                    conventions=(CONVENTION,),
                ),
                rollout_model=self.model,
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


@pytest.fixture
def crash() -> Callable[[], BaseException]:
    return Crash
