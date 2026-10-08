# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A whole run on fakes at the I/O boundary: a fake proposal source and rubric, a scripted GLM server
for the author, a builder program without model calls on ShellSim, and a scripted rollout model and
tokenizer for validation.

``queue_run`` builds ``LoopServices`` over a run root the way ``queue.job.run_job`` does and runs
``run_queue``; the fakes are handed to tests through fixtures because test modules cannot import each
other.
"""

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.submission import PlainText

from taskforge.build.author import SUBMIT_TOOL
from taskforge.build.sdk import BuildServices
from taskforge.build.template import standard
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.client import FinishReason, GlmClient, GlmEndpoint, Pool, Usage
from taskforge.llm.policy import LLMPolicy
from taskforge.loop.policy import LoopPolicy
from taskforge.loop.program import LEDGER_DIR, LoopServices
from taskforge.proposal.model import TaskProposal, parse
from taskforge.proposal.source import ProposalBatch, SlotProposal
from taskforge.queue.run import FailedItems, RunSummary, run_queue
from taskforge.sandbox.factories import SHELLSIM
from taskforge.triage.checks import ALL_COMBINATIONS, CheckContext
from taskforge.triage.program import RubricAssessment
from taskforge.triage.verdict import ModelCall, RubricAxis, RubricResult, TriageDecision
from taskforge.validate.adversary import SENTINEL_REPLIES, AdversaryRole
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff

PROPOSAL = """---
id: "IDEA/SLOT"
source: {kind: capability, ref: "IDEA", hash: "abc123"}
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

GRADE = """
import json, sys
messages = json.load(sys.stdin)
final = [m for m in messages if m.get("role") == "assistant"][-1].get("content") or ""
print(1.0 if final.strip().endswith("ANSWER = 42") else 0.0)
"""

PROGRAM = """
from taskcompendium.environment import EnvironmentKind, EnvironmentSpec, StdoutReward
from taskcompendium.execution import TaskExecution
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, Source, TaskSpec
from taskcompendium.submission import PlainText

EXECUTION = TaskExecution()
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
        timeout=60,
        files=(spec.file("/grader/grade.py", GRADE),),
    )
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
        execution=EXECUTION,
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
    return BuildOutput(task=task, execution=EXECUTION, convention=CONVENTION, controls=await fixed_controls(b, task))
""".replace(
    "GRADE_SOURCE", repr(GRADE)
)
# PROGRAM, with a grader step that tries the reference reply on a machine.
MACHINE_PROGRAM = PROGRAM.replace(
    "    return Grader(",
    '    await b.try_grader(env, verifier, AnswerType.TEXT, CONVENTION, "question", "ANSWER = 42")\n    return Grader(',
)

CALL = ModelCall(Usage(100, 50, 40, 0), wall_time=0.1, finish_reason=FinishReason.TOOL_CALLS)
YIELDS = 20
ROLE_IDS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
FAST = RetryBackoff(initial=0.001, maximum=0.001, factor=1.5, jitter=0.1)


def proposal(idea: str, slot: int) -> TaskProposal:
    return parse(PROPOSAL.replace("IDEA", idea).replace("SLOT", str(slot)))


def rubric_sample(decision: TriageDecision) -> RubricResult:
    score = 5 if decision is TriageDecision.ACCEPT else 1
    return RubricResult(tuple((axis, score) for axis in RubricAxis), (), (), (), decision)


@dataclass
class FakeSource:
    """Proposes ``n`` copies of the multiplication proposal for an idea; ``failing`` ideas raise."""

    failing: frozenset[str] = frozenset()
    calls: list[str] = field(default_factory=list)

    async def propose(self, idea: str, n: int) -> ProposalBatch:
        self.calls.append(idea)
        if idea in self.failing:
            raise RuntimeError(f"source broke on {idea}")
        slots = tuple(SlotProposal(slot, proposal(idea, slot), (), (), None) for slot in range(n))
        return ProposalBatch((), (), slots)


@dataclass
class FakeRubric:
    """One rubric sample per assessment: ``decisions[item id]``, else ``default``; ``failing`` ids raise.

    ``in_flight`` and ``peak`` count concurrent assessments; with ``barrier`` each assessment waits until
    ``barrier.parties`` are in flight, so a test can tell what the width lets overlap.
    """

    default: TriageDecision
    decisions: dict[str, TriageDecision] = field(default_factory=dict)
    failing: set[str] = field(default_factory=set)
    barrier: asyncio.Barrier | None = None
    assessed: list[str] = field(default_factory=list)
    in_flight: int = 0
    peak: int = 0

    async def assess(self, p: TaskProposal, structural: object) -> RubricAssessment:
        self.assessed.append(p.header.id)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.barrier is not None:
                await self.barrier.wait()
            for _ in range(YIELDS):  # let every other runnable assessment start before this one ends
                await asyncio.sleep(0)
            if p.header.id in self.failing:
                raise RuntimeError(f"rubric broke on {p.header.id}")
            return RubricAssessment((rubric_sample(self.decisions.get(p.header.id, self.default)),), (CALL,))
        finally:
            self.in_flight -= 1

    async def repair(self, p: TaskProposal, verdict: object) -> Any:
        raise AssertionError("the queue tests never repair at triage")


def render(messages) -> tuple[int, ...]:
    ids: list[int] = []
    for message in messages:
        ids.append(ROLE_IDS[message["role"]])
        ids.extend(json.dumps({key: message[key] for key in ("content", "tool_calls") if key in message}).encode())
    return tuple(ids)


@dataclass
class TemplateTokenizer:
    """The server's chat template, deterministically."""

    async def prompt_ids(self, messages, options):
        return (*render(messages), ROLE_IDS["assistant"])

    async def rendered_ids(self, messages, options):
        return render(messages)


@dataclass
class SolverModel:
    """Replies ``ANSWER = 42`` to every other solver request and ``ANSWER = 41`` to the rest; each adversary
    role replies its sentinel (the ambiguity role a wrong answer), so a round is calibrated.

    A role in ``solves`` instead reads the question with a shell command and submits ``ANSWER = 42``: an
    honest solve against its orders, which the calibration tiers ``NOTED``. While ``unavailable`` is set
    every request raises it instead. With ``hang`` set, every request waits until the test cancels the run.
    """

    unavailable: Callable[[], Exception] | None = None
    hang: bool = False
    solves: frozenset[AdversaryRole] = frozenset()
    solver_calls: int = 0
    requests: int = 0
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        self.requests += 1
        self.started.set()
        if self.hang:
            await asyncio.Event().wait()
        if self.unavailable is not None:
            raise self.unavailable()
        first = request.messages[0]
        role = next(
            (r for r in AdversaryRole if first["role"] == "system" and SENTINEL_REPLIES[r] in str(first["content"])),
            None,
        )
        if role is None:
            self.solver_calls += 1
            text = "ANSWER = 42" if self.solver_calls % 2 else "ANSWER = 41"
        elif role in self.solves:
            text = "ANSWER = 42"
        else:
            text = SENTINEL_REPLIES.get(role, "ANSWER = 7")
        prompt = (*request.prefix_token_ids, 90) if request.prefix_token_ids else (10, 11)
        if role in self.solves and request.messages[-1]["role"] != "tool":
            return ModelTurn(READ_QUESTION, prompt, (21,), (-0.5,), "tool_calls")
        return ModelTurn({"role": "assistant", "content": text}, prompt, (20,), (-0.5,), "stop")


READ_QUESTION = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "shell", "arguments": json.dumps({"command": "cat /workspace/question.txt"})},
        }
    ],
}


def loop_policy(k: int = 4, max_validation_retries: int = 0, max_build_retries: int = 0) -> LoopPolicy:
    validation = ValidationPolicy(
        k=k,
        adversary_k=1,
        adversary_output_tokens=32768,
        roles=tuple(AdversaryRole),
        band=CalibrationBand(0.125, 0.875),
        sampling=LLMPolicy(max_continuations=0),
        deadlines=Deadlines(agent_timeout=30, attempt_timeout=60),
        max_retries=0,
        token_contract_retries=0,
        retry_backoff=FAST,
    )
    return LoopPolicy(
        proposals_per_idea=1,
        max_idea_reproposals=0,
        max_triage_repairs=0,
        max_build_revisions=1,
        max_repairs=1,
        max_validation_retries=max_validation_retries,
        max_build_retries=max_build_retries,
        retry_backoff=FAST,
        output_token_budget=1_000_000,
        validation=validation,
    )


@dataclass
class QueueRun:
    """Runs a queue over ``root`` with the given fakes; the GLM server answers every author request.

    Builds run on ``build_factories``; trials always run on ShellSim.
    """

    root: Path
    glm_base_url: str
    source: FakeSource
    rubric: FakeRubric
    model: SolverModel
    build_factories: Mapping[EnvironmentKind, MachineFactory]

    async def __call__(
        self, ideas: Mapping[str, str], policy: LoopPolicy, width: int, failed: FailedItems = FailedItems.SKIP
    ) -> RunSummary:
        ledger = JsonlLedger(self.root / LEDGER_DIR)
        factories = {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}
        endpoint = GlmEndpoint(base_url=self.glm_base_url, token="test-token", pool=Pool.HIGH)
        async with GlmClient(endpoint, backoff=FAST.schedule()) as client:
            services = LoopServices(
                client=client,
                source=self.source,
                describe_idea=lambda idea: {"idea": idea},
                checks=(),
                rubric=self.rubric,
                check_context=CheckContext(allowed_combinations=ALL_COMBINATIONS),
                template=standard,
                build=BuildServices(client=client, policy=LLMPolicy(), factories=self.build_factories, ledger=ledger),
                engine=EngineSettings(
                    factories=factories,
                    capabilities={EnvironmentKind.SHELLSIM: SHELLSIM},
                    max_turns=4,
                    command_timeout=30,
                    cleanup_timeout=30,
                    conventions=(PlainText(id="plain_text"),),
                ),
                rollout_models=lambda _: self.model,
                tokenize=TemplateTokenizer(),
                ledger=ledger,
                root=self.root,
                slots=asyncio.Semaphore(width),
            )
            return await run_queue(ideas, policy, services, failed)


@pytest.fixture
def queue_run(tmp_path, fake_glm) -> Callable[..., QueueRun]:
    """``queue_run(rubric=..., model=..., source=..., build_factories=...)``: a ``QueueRun``.

    Builds default to a ShellSim factory.
    """

    def make(
        rubric: FakeRubric | None = None,
        model: SolverModel | None = None,
        source: FakeSource | None = None,
        build_factories: Mapping[EnvironmentKind, MachineFactory] | None = None,
    ) -> QueueRun:
        return QueueRun(
            root=tmp_path / "run",
            glm_base_url=fake_glm.base_url,
            source=source or FakeSource(),
            rubric=rubric or FakeRubric(TriageDecision.ACCEPT),
            model=model or SolverModel(),
            build_factories=(
                {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()} if build_factories is None else build_factories
            ),
        )

    return make


@pytest.fixture
def author_replies(fake_glm) -> Callable[[int], None]:
    """``author_replies(n, source=PROGRAM)`` queues ``n`` author completions that submit ``source``."""

    def queue(n: int, source: str = PROGRAM) -> None:
        for _ in range(n):
            fake_glm.stream(
                tool_calls=((SUBMIT_TOOL, json.dumps({"source": source, "notes": "builds 6*7"})),),
                finish="tool_calls",
            )

    return queue


@dataclass(frozen=True)
class Fakes:
    rubric: type[FakeRubric] = FakeRubric
    source: type[FakeSource] = FakeSource
    model: type[SolverModel] = SolverModel
    policy: Callable[..., LoopPolicy] = loop_policy
    machine_program: str = MACHINE_PROGRAM


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()
