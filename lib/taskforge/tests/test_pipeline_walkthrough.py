# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the coordinator with a three-step toy workpiece and controlled stage failures.

``numbers`` writes a resource and ``grader`` the grader, and the ``task`` sink uses both. ``SOURCE``
holds the same program as module text, as an author writing Python would return it, and passes the
restricted loader.
"""

import asyncio
import base64
import inspect
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import pytest
from pydantic import BaseModel, JsonValue
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend
from taskcompendium.grader import verifyit_package
from taskcompendium.models import AnswerType, InlineFile, PlainText, Source, TaskResource, TaskSpec
from verifyit.spec import NumericSpec

from taskforge import content_hash
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.client import Completion, FinishReason, ModelUnavailable, StructuredResult, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError
from taskforge.sandbox.factories import MachineHost
from taskforge.spec.draft import assemble, requirements
from taskforge.stage10_proposal.model import (
    Environment,
    ProposalFormatError,
    SolverMachine,
    TaskProposal,
    parse,
    render,
)
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage20_triage.verdict import TriageDecision, Verdict
from taskforge.stage30_author.author import PROGRAM_FILE, AuthoredProgram, Revision, RevisionKind, static_author
from taskforge.stage30_author.loader import SdkExports, load_program_source
from taskforge.stage30_author.program import (
    RLEnvBuildProgram,
    rl_env_build_program,
    rl_env_build_sink,
    rl_env_build_step,
)
from taskforge.stage40_build_with_program.context import (
    BuildExpired,
    BuildFailure,
    BuildInfrastructureFailure,
    ExpiredClock,
    RLEnvBuildContext,
)
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.outputs import Grader
from taskforge.stage40_build_with_program.run import (
    DRAFT_DIR,
    TASK_FILE,
    ResearchRequirement,
    TaskDraft,
    build_model_digest,
    run_build,
)
from taskforge.stage40_build_with_program.services import AgentLimits, RLEnvBuildServices
from taskforge.stage40_build_with_program.step_store import StoreCorruption
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision, Modify, RejectKind
from taskforge.stage_coordinator.coordinator import (
    AUTHORED_DIR,
    DECISION_FILE,
    FAILURES_DIR,
    REPAIRED_DIR,
    STORE_DIR,
    VALIDATION_FILE,
    StageCoordinator,
    WorkpieceBudget,
    run_workpiece,
)
from taskforge.stage_coordinator.event_log import InMemoryEventLog, JsonEventStore
from taskforge.stage_coordinator.run import InfrastructureRetry, run_workpieces
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge.stage_coordinator.workpiece_machine import EXIT, Stage, Terminal, WorkpieceEvent, derive_state

WORKPIECE = "toy-0"
PROPOSAL = parse(
    """---
id: "toy/0"
source: {kind: capability, ref: "toy", hash: "0"}
grounding: unverified
null_reason: null
environment: shellsim
solver_machine: shellsim
verification: simple
research_need: not_needed
research_reason: "the numbers are in the task's own file"
research: []
build:
  - "numbers": "the file holding the two numbers"
  - "grader": "checks the product"
resources: ["workspace/numbers.txt"]
---
## Task
Multiply the two numbers in /workspace/numbers.txt. The file holds two positive integers on one
line, separated by a space. Read them, multiply them, and reply with the product alone inside the
answer box, written as a plain integer with no units, no separators and no explanation around it.

## Realism and workflow
A clerk checks a product before an invoice goes out. The clerk opens the order file, reads the
quantity and the unit price, multiplies them, and compares the result with the total the system
printed. The task keeps that one step: read two numbers from a file and report their product.

## Research plan
None. The numbers are written by the build into the task's own file, and the product is computed
from them, so no external fact, document or dataset is needed. The proposal declares no research
items, and the body names no lookup in any outside source for the builder or the solver.

## Build plan
Write the numbers and the grader. The build writes workspace/numbers.txt with the two numbers six
and seven, and a grader that reads the answer box and compares the reply with the product. The
build checks its own file by recomputing the product before it publishes the task.

## Grader design and controls
Full credit only for 42. The grader is an exact match on the integer in the answer box after
stripping surrounding whitespace. A positive control replies 42 and earns full credit; a negative
control replies 13, the sum of the numbers, and earns nothing; an empty reply earns nothing.

## Risks and null conditions
None. The numbers are small and fixed, so the product cannot overflow and the answer format is
unambiguous. A solver that guesses cannot reach the product by chance often, and the file holds
no hint of the answer. No condition makes this task impossible to build in ShellSim.
"""
)
BUDGET = WorkpieceBudget(max_triage_repairs=0, max_build_revisions=1, max_validation_retries=1)
REPAIRING = replace(BUDGET, max_triage_repairs=1)
"""``BUDGET`` for a consumer using the REPAIR surface: one rewrite, then a REPAIR verdict rejects."""
SAMPLING = LLMPolicy()
SERVICES = RLEnvBuildServices(
    host=MachineHost.LAPTOP,
    factories={Backend.SHELLSIM.value: ShellSimMachineFactory()},
    images=None,
    web_tools=(),
    ledger=None,
    reasoning_replay=ReasoningReplay.KEEP,
    agent_limits=AgentLimits(max_turns=1, max_tokens=1024, max_calls=1),
)
SUMMARY = {"graded": 4, "solved": 2}


class UnusedModel:
    """The model of a program whose steps call no model."""

    model = "unused"

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        raise AssertionError("no step calls the model")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("no step calls the model")


@rl_env_build_step
async def numbers(context: RLEnvBuildContext) -> TaskResource:
    return TaskResource(
        path="workspace/numbers.txt", source=InlineFile(content_base64=base64.b64encode(b"6 7\n").decode())
    )


@rl_env_build_step
async def grader(context: RLEnvBuildContext) -> Grader:
    return Grader(
        package=verifyit_package(NumericSpec(expected="42", tolerance_abs=0.0, tolerance_rel=0.0)),
        answer_contract="Reply with the product alone.",
        reference_reply="42",
        reference_files=(),
        secret_values=("42",),
        rejection_ceiling=0.0,
        prototyped=(),
    )


@rl_env_build_sink
async def task(context: RLEnvBuildContext, numbers: TaskResource, grader: Grader, proposal: TaskProposal) -> TaskSpec:
    return assemble(
        context.workpiece_id,
        proposal.body,
        AnswerType.TEXT,
        PlainText(),
        grader.package,
        Source(dataset="taskforge", revision="test", row=context.workpiece_id, importer_revision="test"),
        environment=requirements(image=None),
        files=(numbers,),
    )


@rl_env_build_sink
async def broken(context: RLEnvBuildContext, proposal: TaskProposal) -> TaskSpec:
    raise BuildFailure("the numbers do not multiply to the answer")


PROGRAM = rl_env_build_program(numbers, grader, sink=task)
FAILING = rl_env_build_program(sink=broken)
SOURCE_IMPORTS = """\
import base64

from taskcompendium.grader import verifyit_package
from taskcompendium.models import AnswerType, InlineFile, PlainText, Source, TaskResource, TaskSpec
from verifyit.spec import NumericSpec

from taskforge.spec.draft import assemble, requirements
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink, rl_env_build_step
from taskforge.stage40_build_with_program.context import RLEnvBuildContext
from taskforge.stage40_build_with_program.outputs import Grader
"""
SOURCE = "\n\n".join(
    [
        SOURCE_IMPORTS,
        inspect.getsource(numbers.fn),
        inspect.getsource(grader.fn),
        inspect.getsource(task.fn),
        "PROGRAM = rl_env_build_program(numbers, grader, sink=task)\n",
    ]
)
"""``PROGRAM`` as module text. The steps keep their text, so the loaded program has ``PROGRAM``'s digest."""


async def validate(draft: TaskDraft) -> dict[str, int]:
    return SUMMARY


async def review(summary: object) -> Decision:
    return Accept(band=BandOutcome.IN_BAND)


NO_SDK_NAMES = SdkExports(defining_modules={})
"""The walkthrough's programs read no name they leave unbound, so the loader has no SDK name to suggest."""


def load_toy_program(source: str) -> RLEnvBuildProgram:
    return load_program_source(source, NO_SDK_NAMES)


def no_call(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("this stage does not run")


STAGES = StageFunctions(
    triage=preflight,
    repair=no_call,
    author=static_author(PROGRAM),
    load_program=load_toy_program,
    build=run_build,
    validate=validate,
    review=review,
)
UNUSED_MODEL_DIGEST = build_model_digest(UnusedModel(), SAMPLING, SERVICES)


class RecordingAuthor:
    """Returns ``programs`` in turn, the last one from then on, and records each revision and author state."""

    def __init__(self, *programs: RLEnvBuildProgram):
        self.programs = list(programs)
        self.revisions: list[Revision | None] = []
        self.states: list[Mapping[str, JsonValue]] = []

    async def __call__(
        self, proposal: TaskProposal, revision: Revision | None, author_state: Mapping[str, JsonValue]
    ) -> AuthoredProgram:
        self.revisions.append(revision)
        self.states.append(author_state)
        return AuthoredProgram(self.programs.pop(0) if len(self.programs) > 1 else self.programs[0], None)


class SourceAuthor:
    """Returns ``source`` with the program loaded from it, and records each revision asked for."""

    def __init__(self, source: str = SOURCE) -> None:
        self.source = source
        self.revisions: list[Revision | None] = []

    async def __call__(
        self, proposal: TaskProposal, revision: Revision | None, author_state: Mapping[str, JsonValue]
    ) -> AuthoredProgram:
        self.revisions.append(revision)
        return AuthoredProgram(load_toy_program(self.source), self.source)


HOST_FAILURE = BuildInfrastructureFailure("image_pull", "the sandbox image did not pull")


class FailsOnce:
    """A build function raising ``error`` on its first call and running ``run_build`` after."""

    def __init__(self, error: Exception = HOST_FAILURE) -> None:
        self.error = error
        self.calls = 0

    async def __call__(self, *args: Any, **kwargs: Any) -> TaskDraft:
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return await run_build(*args, **kwargs)


class SaveFailsOnce(JsonEventStore):
    """A JSON event store whose first save finds the storage unreachable."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.saves = 0

    def save(self, workpiece_id: str, events: tuple[WorkpieceEvent, ...]) -> None:
        self.saves += 1
        if self.saves == 1:
            raise OSError("the event store is unreachable")
        super().save(workpiece_id, events)


def coordinator(
    root: Path,
    stages: StageFunctions = STAGES,
    budget: WorkpieceBudget = BUDGET,
    digest: str = UNUSED_MODEL_DIGEST,
    research: ResearchRequirement = ResearchRequirement.OFF,
) -> StageCoordinator:
    return StageCoordinator(
        stages, budget, root, JsonEventStore(root), UnusedModel(), SAMPLING, SERVICES, research, digest
    )


def exits(log: InMemoryEventLog) -> list[tuple[str, str | None]]:
    """Each event's kind with its exit, or with its terminal for ``TERMINAL``."""
    return [(event.kind, event.attrs.get(EXIT, event.attrs.get("terminal"))) for event in log.events()]


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


ACCEPTED_EXITS = [
    ("opened", None),
    ("triaged", "accept"),
    ("authored", "authored"),
    ("built", "built"),
    ("validated", "validated"),
    ("decided", "accept"),
    ("terminal", "accepted"),
]


async def test_run_workpiece_records_an_event_per_stage_and_a_relaunch_changes_nothing(tmp_path):
    log = InMemoryEventLog()
    assert await run_workpiece(coordinator(tmp_path), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED

    assert exits(log) == ACCEPTED_EXITS
    assert JsonEventStore(tmp_path).load(WORKPIECE) == log.events()
    round_dir = tmp_path / "workpieces" / WORKPIECE / "rounds" / "0"
    for name in (f"{AUTHORED_DIR}/0/program.json", f"{DRAFT_DIR}/{TASK_FILE}", VALIDATION_FILE, DECISION_FILE):
        assert (round_dir / name).is_file(), name
    assert (tmp_path / STORE_DIR / "workpieces" / WORKPIECE / "steps" / "task").is_dir()

    before = tree_bytes(tmp_path)
    untouched = StageFunctions(no_call, no_call, no_call, no_call, no_call, no_call, no_call)
    relaunch = await run_workpiece(coordinator(tmp_path, untouched), WORKPIECE, PROPOSAL, InMemoryEventLog())
    assert relaunch is Terminal.ACCEPTED
    assert tree_bytes(tmp_path) == before


NULL_PROPOSAL = replace(
    PROPOSAL,
    header=replace(PROPOSAL.header, build=(), resources=(), null_reason="the capability has no gradable artifact here"),
)
RISKS_HEADING = "## Risks and null conditions\n"
THIN_PROPOSAL = replace(PROPOSAL, body=PROPOSAL.body[: PROPOSAL.body.index(RISKS_HEADING)] + RISKS_HEADING + "None.\n")
MACHINELESS_SHELLSIM = replace(PROPOSAL, header=replace(PROPOSAL.header, solver_machine=SolverMachine.NONE))


@pytest.mark.parametrize(
    ("proposal", "reason"),
    [
        (THIN_PROPOSAL, "required_sections: '## Risks and null conditions' has 1 words, fewer than 40"),
        (NULL_PROPOSAL, "null proposal: the capability has no gradable artifact here"),
    ],
)
async def test_the_default_preflight_rejects_a_proposal_with_the_failing_check_before_any_later_stage(
    tmp_path, proposal, reason
):
    assert proposal.header.environment is Environment.SHELLSIM
    log = InMemoryEventLog()
    stages = replace(STAGES, author=no_call, load_program=no_call, build=no_call, validate=no_call, review=no_call)
    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, proposal, log) is Terminal.REJECTED

    assert exits(log) == [("opened", None), ("triaged", "reject"), ("terminal", "rejected")]
    triaged, terminal = log.events()[1:]
    assert triaged.attrs["reason"] == terminal.attrs["reason"]
    assert terminal.attrs["reason"].startswith("triage rejected the proposal: ")
    assert reason in terminal.attrs["reason"]


async def test_a_retry_after_a_failed_save_of_a_finished_workpiece_writes_its_events(tmp_path):
    store = SaveFailsOnce(tmp_path)
    log = InMemoryEventLog()
    with pytest.raises(OSError, match="unreachable"):
        await run_workpiece(replace(coordinator(tmp_path), event_store=store), WORKPIECE, PROPOSAL, log)
    assert store.load(WORKPIECE) is None

    untouched = StageFunctions(no_call, no_call, no_call, no_call, no_call, no_call, no_call)
    retried = replace(coordinator(tmp_path, untouched), event_store=store)
    assert await run_workpiece(retried, WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert store.load(WORKPIECE) == log.events()
    assert exits(log) == ACCEPTED_EXITS


async def test_an_author_returning_source_runs_once_and_the_build_loads_its_program_file(tmp_path):
    author = SourceAuthor()
    loaded: list[str] = []

    def load_program(source: str) -> RLEnvBuildProgram:
        loaded.append(source)
        return load_toy_program(source)

    log = InMemoryEventLog()
    stages = replace(STAGES, author=author, load_program=load_program)
    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED

    assert exits(log) == ACCEPTED_EXITS
    assert (len(author.revisions), loaded) == (1, [SOURCE])
    assert (
        tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / AUTHORED_DIR / "0" / PROGRAM_FILE
    ).read_text() == SOURCE
    assert log.events()[2].digest == PROGRAM.digest


OFF = ResearchRequirement.OFF


@pytest.mark.parametrize(
    ("proposal", "budget", "digest", "research", "refusal"),
    [
        (
            replace(PROPOSAL, body=PROPOSAL.body.replace("Multiply", "Add")),
            BUDGET,
            UNUSED_MODEL_DIGEST,
            OFF,
            "opened for proposal",
        ),
        (PROPOSAL, replace(BUDGET, max_build_revisions=2), UNUSED_MODEL_DIGEST, OFF, "opened under budget"),
        (PROPOSAL, BUDGET, "another-model", OFF, "opened with model digest"),
        (PROPOSAL, BUDGET, UNUSED_MODEL_DIGEST, ResearchRequirement.ENFORCED, "opened with research off"),
    ],
    ids=["proposal", "budget", "model digest", "research"],
)
async def test_a_relaunch_with_another_proposal_budget_model_digest_or_research_is_refused(
    tmp_path, proposal, budget, digest, research, refusal
):
    await run_workpiece(coordinator(tmp_path), WORKPIECE, PROPOSAL, InMemoryEventLog())
    before = tree_bytes(tmp_path)

    relaunch = coordinator(tmp_path, budget=budget, digest=digest, research=research)
    with pytest.raises(ValueError, match=f"{WORKPIECE} was {refusal}"):
        await run_workpiece(relaunch, WORKPIECE, proposal, InMemoryEventLog())
    assert tree_bytes(tmp_path) == before


async def test_a_program_other_than_the_authored_one_fails_the_workpiece_at_build(tmp_path):
    stages = replace(STAGES, author=RecordingAuthor(PROGRAM, FAILING))
    with pytest.raises(ValueError, match="now returns program"):
        await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, InMemoryEventLog())

    state = derive_state(JsonEventStore(tmp_path).load(WORKPIECE) or ())
    assert (state.stage, state.terminal) == (Stage.BUILD, Terminal.FAILED)


async def test_a_program_file_changed_on_disk_fails_the_workpiece_at_build(tmp_path):
    stages = replace(STAGES, author=SourceAuthor(), build=FailsOnce())
    with pytest.raises(BuildInfrastructureFailure):
        await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, InMemoryEventLog())
    program_file = tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / AUTHORED_DIR / "0" / PROGRAM_FILE
    program_file.write_text(SOURCE.replace("6 7", "6 8"))

    with pytest.raises(ValueError, match="holds program"):
        await run_workpiece(
            coordinator(tmp_path, replace(stages, author=no_call)), WORKPIECE, PROPOSAL, InMemoryEventLog()
        )
    state = derive_state(JsonEventStore(tmp_path).load(WORKPIECE) or ())
    assert (state.stage, state.terminal) == (Stage.BUILD, Terminal.FAILED)


async def test_a_build_failure_costs_a_revision_and_the_next_one_rejects_the_workpiece(tmp_path):
    author = RecordingAuthor(FAILING)
    log = InMemoryEventLog()
    terminal = await run_workpiece(coordinator(tmp_path, replace(STAGES, author=author)), WORKPIECE, PROPOSAL, log)

    assert terminal is Terminal.REJECTED
    assert exits(log) == [
        ("opened", None),
        ("triaged", "accept"),
        ("authored", "authored"),
        ("build_failed", "revise"),
        ("authored", "authored"),
        ("build_failed", "reject"),
        ("terminal", "rejected"),
    ]
    assert log.events()[-1].attrs["kind"] == "budget"
    asked = [None if revision is None else revision.kind for revision in author.revisions]
    assert asked == [None, None, RevisionKind.BUILD_FAILURE, RevisionKind.BUILD_FAILURE]
    revision = author.revisions[-1]
    assert revision is not None
    assert revision.detail.startswith(
        "BuildFailure: the numbers do not multiply to the answer at broken, program line "
        + str(inspect.getsourcelines(broken.fn)[1] + 2)
        + ': raise BuildFailure("the numbers do not multiply to the answer")\n'
    )


NUMBERS_FILE = 'InlineFile(content_base64=base64.b64encode(b"6 7\\n").decode())'


def program_line(source: str, text: str) -> str:
    """The ``program line N: <line>`` part of a failure's first line, for the line of ``source`` holding ``text``."""
    lines = source.splitlines()
    number = next(number for number, line in enumerate(lines, start=1) if text in line)
    return f"program line {number}: {lines[number - 1].strip()}"


async def failure_first_lines(root: Path, source: str) -> tuple[str, str, str]:
    """Run a workpiece authoring ``source`` to its rejection, and return the first line of its failure text.

    It returns the line from ``build-failures/1.txt``, from the revision detail the second author
    call received, and from the ``failure`` attr of the first ``BUILD_FAILED`` event.
    """
    author = SourceAuthor(source)
    log = InMemoryEventLog()
    assert (
        await run_workpiece(coordinator(root, replace(STAGES, author=author)), WORKPIECE, PROPOSAL, log)
        is Terminal.REJECTED
    )
    revision = author.revisions[1]
    assert revision is not None and revision.kind is RevisionKind.BUILD_FAILURE
    text = (root / "workpieces" / WORKPIECE / "rounds" / "0" / FAILURES_DIR / "1.txt").read_text()
    assert text == revision.detail
    built = next(event for event in log.events() if event.kind == "build_failed")
    return text.splitlines()[0], revision.detail.splitlines()[0], built.attrs["failure"]


async def test_a_name_error_in_a_step_reports_its_program_line_to_the_author(tmp_path):
    """The loader refuses a name that nothing binds, so the step declares it global and never assigns it."""
    declared = "async def numbers(context: RLEnvBuildContext) -> TaskResource:\n"
    source = SOURCE.replace(NUMBERS_FILE, "InlineFile(content_base64=encoded_numbers)").replace(
        declared, declared + "    global encoded_numbers\n", 1
    )
    assert source.count("global encoded_numbers") == 1
    expected = "NameError: name 'encoded_numbers' is not defined at numbers, " + program_line(
        source, "content_base64=encoded_numbers"
    )
    assert await failure_first_lines(tmp_path, source) == (expected,) * 3


async def test_an_os_error_raised_by_a_step_is_the_programs_failure(tmp_path):
    """A step raising a missing file's error is a program bug, not the storage fault the OSError pass-through covers.

    The restricted loader gives a program no ``open``, so the step raises the error itself.
    """
    missing = '    raise FileNotFoundError(2, "No such file or directory", "/workspace/does-not-exist.txt")\n'
    source = SOURCE.replace("    return TaskResource(\n", missing + "    return TaskResource(\n", 1)
    assert source != SOURCE
    expected = (
        "FileNotFoundError: [Errno 2] No such file or directory: '/workspace/does-not-exist.txt' at numbers, "
        + program_line(source, "does-not-exist")
    )
    assert await failure_first_lines(tmp_path, source) == (expected,) * 3


async def test_a_library_error_called_from_a_step_still_reports_the_program_line(tmp_path):
    source = SOURCE.replace(NUMBERS_FILE, '"6 7"')
    assert source != SOURCE
    first_line, detail_line, attr = await failure_first_lines(tmp_path, source)
    assert first_line == detail_line == attr
    assert first_line.startswith("ValidationError: 1 validation error for TaskResource in library code at pydantic/")
    assert first_line.endswith(", called from numbers, " + program_line(source, "return TaskResource("))


CUT_REPLY = Completion(
    content='{"source": "import base64\\n',
    reasoning="",
    tool_calls=(),
    finish_reason=FinishReason.LENGTH,
    usage=Usage(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, cached_tokens=0),
    wall_time=0.0,
    ttft=0.0,
    decode_time=0.0,
    continuations=0,
    attempts=(),
)


class CutAuthor:
    """An author whose first ``cuts`` replies are cut at the output limit, returning ``SOURCE`` after.

    It records each revision asked for.
    """

    def __init__(self, cuts: int) -> None:
        self.cuts = cuts
        self.revisions: list[Revision | None] = []

    async def __call__(
        self, proposal: TaskProposal, revision: Revision | None, author_state: Mapping[str, JsonValue]
    ) -> AuthoredProgram:
        self.revisions.append(revision)
        if len(self.revisions) <= self.cuts:
            raise StructuredOutputError(ValueError("the reply was cut at the output limit"), (CUT_REPLY, CUT_REPLY))
        return AuthoredProgram(load_toy_program(SOURCE), SOURCE)


async def test_an_author_reply_cut_after_its_repairs_costs_a_revision_and_the_author_samples_again(tmp_path):
    author = CutAuthor(cuts=1)
    log = InMemoryEventLog()

    assert (
        await run_workpiece(coordinator(tmp_path, replace(STAGES, author=author)), WORKPIECE, PROPOSAL, log)
        is Terminal.ACCEPTED
    )
    assert exits(log) == [*ACCEPTED_EXITS[:2], ("author_failed", "revise"), *ACCEPTED_EXITS[2:]]
    revision = author.revisions[1]
    assert revision is not None and revision.kind is RevisionKind.AUTHOR_FAILURE
    assert "the reply was cut at the output limit" in revision.detail
    round_dir = tmp_path / "workpieces" / WORKPIECE / "rounds" / "0"
    assert (round_dir / FAILURES_DIR / "1.txt").read_text() == revision.detail
    assert (round_dir / AUTHORED_DIR / "1" / PROGRAM_FILE).read_text() == SOURCE


async def test_author_replies_cut_past_the_budget_reject_the_workpiece_on_its_budget(tmp_path):
    author = CutAuthor(cuts=2)
    log = InMemoryEventLog()

    assert (
        await run_workpiece(coordinator(tmp_path, replace(STAGES, author=author)), WORKPIECE, PROPOSAL, log)
        is Terminal.REJECTED
    )
    assert exits(log) == [
        *ACCEPTED_EXITS[:2],
        ("author_failed", "revise"),
        ("author_failed", "reject"),
        ("terminal", "rejected"),
    ]
    assert log.events()[-1].attrs["kind"] == RejectKind.BUDGET
    assert [None if revision is None else revision.kind for revision in author.revisions] == [
        None,
        RevisionKind.AUTHOR_FAILURE,
    ]


async def unavailable_author(
    proposal: TaskProposal, revision: Revision | None, author_state: Mapping[str, JsonValue]
) -> AuthoredProgram:
    raise ModelUnavailable("the hold ran out", ())


async def test_an_unavailable_model_fails_the_workpiece_at_the_author_with_its_revisions_unspent(tmp_path):
    with pytest.raises(ModelUnavailable):
        await run_workpiece(
            coordinator(tmp_path, replace(STAGES, author=unavailable_author)), WORKPIECE, PROPOSAL, InMemoryEventLog()
        )
    state = derive_state(JsonEventStore(tmp_path).load(WORKPIECE) or ())
    assert (state.stage, state.terminal, state.build_revisions) == (Stage.AUTHOR, Terminal.FAILED, 0)
    assert not (tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / FAILURES_DIR).exists()


@pytest.mark.parametrize(
    "error",
    [
        HOST_FAILURE,
        FileNotFoundError("blobs/ab/abcdef is missing"),
        BuildExpired("ran for more than 1800 seconds without returning", ExpiredClock.STEP, "numbers"),
    ],
    ids=["build host", "missing file", "expired clock"],
)
async def test_a_failure_outside_the_program_fails_the_workpiece_at_build_and_a_relaunch_resumes_there(tmp_path, error):
    stages = replace(STAGES, author=SourceAuthor(), build=FailsOnce(error))
    with pytest.raises(type(error), match=str(error)):
        await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, InMemoryEventLog())
    state = derive_state(JsonEventStore(tmp_path).load(WORKPIECE) or ())
    assert (state.stage, state.terminal, state.build_revisions) == (Stage.BUILD, Terminal.FAILED, 0)

    log = InMemoryEventLog()
    resumed = replace(stages, author=no_call)
    assert await run_workpiece(coordinator(tmp_path, resumed), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert exits(log) == [*ACCEPTED_EXITS[:3], ("terminal", "failed"), *ACCEPTED_EXITS[3:]]


async def builds_then_the_host_fails(*args: Any, **kwargs: Any) -> TaskDraft:
    await run_build(*args, **kwargs)
    raise HOST_FAILURE


async def test_a_corrupt_memo_fails_the_workpiece_at_build_with_its_revisions_unspent(tmp_path):
    stages = replace(STAGES, build=builds_then_the_host_fails)
    with pytest.raises(BuildInfrastructureFailure):
        await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, InMemoryEventLog())
    memo = next((tmp_path / STORE_DIR).glob("workpieces/*/steps/*/*.json"))
    memo.write_bytes(memo.read_bytes()[:20])

    log = InMemoryEventLog()
    with pytest.raises(StoreCorruption, match="does not validate"):
        await run_workpiece(coordinator(tmp_path), WORKPIECE, PROPOSAL, log)
    assert exits(log) == [*ACCEPTED_EXITS[:3], ("terminal", "failed"), ("terminal", "failed")]
    assert derive_state(log.events()).build_revisions == 0
    assert not (tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / FAILURES_DIR).exists()


async def fails_in_the_program(*args: Any, **kwargs: Any) -> TaskDraft:
    raise BuildFailure("the numbers do not multiply to the answer", "task")


async def test_a_relaunch_after_a_killed_re_author_builds_the_program_its_authored_event_records(tmp_path):
    first = replace(STAGES, author=SourceAuthor(), build=FailsOnce())
    with pytest.raises(BuildInfrastructureFailure):
        await run_workpiece(coordinator(tmp_path, first), WORKPIECE, PROPOSAL, InMemoryEventLog())

    other = SOURCE.replace("6 7", "6 8")
    killed = coordinator(tmp_path, replace(STAGES, author=SourceAuthor(other), build=fails_in_the_program))
    with pytest.raises(OSError, match="unreachable"):
        await run_workpiece(
            replace(killed, event_store=SaveFailsOnce(tmp_path)), WORKPIECE, PROPOSAL, InMemoryEventLog()
        )
    authored = tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / AUTHORED_DIR
    assert (authored / "1" / PROGRAM_FILE).read_text() == other

    log = InMemoryEventLog()
    resumed = replace(STAGES, author=no_call)
    assert await run_workpiece(coordinator(tmp_path, resumed), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert exits(log) == [*ACCEPTED_EXITS[:3], ("terminal", "failed"), *ACCEPTED_EXITS[3:]]
    assert (authored / "0" / PROGRAM_FILE).read_text() == SOURCE


class UnavailableOnce:
    """A validate function whose first call finds the model unavailable."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, draft: TaskDraft) -> dict[str, int]:
        self.calls += 1
        if self.calls == 1:
            raise ModelUnavailable("the hold ran out", ())
        return SUMMARY


async def test_an_unavailable_model_fails_the_workpiece_at_validation_and_a_relaunch_resumes_there(tmp_path):
    validate_once = UnavailableOnce()
    with pytest.raises(ModelUnavailable):
        await run_workpiece(
            coordinator(tmp_path, replace(STAGES, validate=validate_once)), WORKPIECE, PROPOSAL, InMemoryEventLog()
        )
    state = derive_state(JsonEventStore(tmp_path).load(WORKPIECE) or ())
    assert (state.stage, state.terminal) == (Stage.VALIDATE, Terminal.FAILED)

    resumed = replace(STAGES, author=no_call, build=no_call, validate=validate_once)
    log = InMemoryEventLog()
    assert await run_workpiece(coordinator(tmp_path, resumed), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert exits(log) == [*ACCEPTED_EXITS[:4], ("terminal", "failed"), *ACCEPTED_EXITS[4:]]


async def test_a_modify_decision_re_authors_the_workpiece_in_round_one_with_the_findings(tmp_path):
    decisions: list[Decision] = [Modify(findings=("the file states the product",)), Accept(band=BandOutcome.IN_BAND)]
    author = RecordingAuthor(PROGRAM)

    async def review_in_turn(summary: object) -> Decision:
        return decisions.pop(0)

    stages = replace(STAGES, author=author, review=review_in_turn)
    log = InMemoryEventLog()

    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert [(event.kind, event.round) for event in log.events()][5:8] == [("decided", 0), ("authored", 1), ("built", 1)]
    assert author.revisions[2:] == [Revision(kind=RevisionKind.MODIFY, detail="the file states the product")] * 2
    assert (tmp_path / "workpieces" / WORKPIECE / "rounds" / "1" / DRAFT_DIR / TASK_FILE).is_file()


class TriageInTurn:
    """A triage function returning a verdict of each of ``decisions`` in turn, recording each proposal triaged.

    Each ACCEPT verdict carries ``author_state``.
    """

    def __init__(self, *decisions: TriageDecision, author_state: Mapping[str, JsonValue] | None = None) -> None:
        self.decisions = list(decisions)
        self.author_state = author_state or {}
        self.proposals: list[TaskProposal] = []

    async def __call__(self, proposal: TaskProposal) -> Verdict:
        self.proposals.append(proposal)
        decision = self.decisions.pop(0)
        state = self.author_state if decision is TriageDecision.ACCEPT else {}
        return Verdict(proposal.header.id, proposal.digest, decision, (), (f"triage says {decision}",), state)


class MachineRepair:
    """A deterministic repair of one header defect, recording each call.

    A shellsim proposal declaring no solver machine is rewritten to declare ShellSim. Any other
    proposal has no rewrite here and raises ``ProposalFormatError``.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, proposal: TaskProposal, verdict: Verdict) -> TaskProposal:
        self.calls += 1
        header = proposal.header
        if header.environment is not Environment.SHELLSIM or header.solver_machine is not SolverMachine.NONE:
            raise ProposalFormatError(f"no header fix applies to {header.id}")
        return replace(proposal, header=replace(header, solver_machine=SolverMachine.SHELLSIM))


async def test_a_repair_verdict_triages_the_repaired_proposal_and_the_later_stages_run_it_across_a_relaunch(
    tmp_path,
):
    triage, repair = TriageInTurn(TriageDecision.REPAIR, TriageDecision.ACCEPT), MachineRepair()
    stages = replace(STAGES, triage=triage, repair=repair)
    failing = coordinator(tmp_path, replace(stages, author=unavailable_author), REPAIRING)
    with pytest.raises(ModelUnavailable):
        await run_workpiece(failing, WORKPIECE, MACHINELESS_SHELLSIM, InMemoryEventLog())

    author = RecordingAuthor(PROGRAM)
    proposals: list[TaskProposal] = []

    async def build(program: RLEnvBuildProgram, proposal: TaskProposal, *args: Any, **kwargs: Any) -> TaskDraft:
        proposals.append(proposal)
        return await run_build(program, proposal, *args, **kwargs)

    log = InMemoryEventLog()
    resumed = replace(stages, triage=no_call, repair=no_call, author=author, build=build)
    resumed_coordinator = coordinator(tmp_path, resumed, REPAIRING)
    assert await run_workpiece(resumed_coordinator, WORKPIECE, MACHINELESS_SHELLSIM, log) is Terminal.ACCEPTED

    assert exits(log) == [
        *ACCEPTED_EXITS[:1],
        ("triaged", "repair"),
        ("triaged", "accept"),
        ("terminal", "failed"),
        *ACCEPTED_EXITS[2:],
    ]
    repaired, accepted = log.events()[1:3]
    assert (repaired.digest, repaired.attrs["triaged_digest"]) == (
        content_hash.digest(PROPOSAL),
        content_hash.digest(MACHINELESS_SHELLSIM),
    )
    assert (repaired.attrs["decision"], repaired.attrs["repairs"]) == ("repair", "1")
    assert (accepted.digest, accepted.attrs["repairs"]) == (content_hash.digest(PROPOSAL), "1")
    assert (tmp_path / "workpieces" / WORKPIECE / repaired.attrs["proposal_file"]).read_text() == render(PROPOSAL)
    assert repaired.attrs["proposal_file"] == f"{REPAIRED_DIR}/1.md"
    assert (triage.proposals, repair.calls, proposals) == ([MACHINELESS_SHELLSIM, PROPOSAL], 1, [PROPOSAL])
    assert derive_state(log.events()).triage_repairs == 1


async def test_a_second_repair_verdict_rejects_the_workpiece_without_a_second_repair(tmp_path):
    triage, repair = TriageInTurn(TriageDecision.REPAIR, TriageDecision.REPAIR), MachineRepair()
    stages = replace(STAGES, triage=triage, repair=repair, author=no_call)
    log = InMemoryEventLog()

    assert (
        await run_workpiece(coordinator(tmp_path, stages, REPAIRING), WORKPIECE, MACHINELESS_SHELLSIM, log)
        is Terminal.REJECTED
    )
    assert exits(log) == [("opened", None), ("triaged", "repair"), ("triaged", "reject"), ("terminal", "rejected")]
    rejected, closed = log.events()[2:]
    assert (rejected.digest, rejected.attrs["decision"], rejected.attrs["repairs"]) == (
        content_hash.digest(PROPOSAL),
        "repair",
        "1",
    )
    assert closed.attrs == {
        "terminal": "rejected",
        "reason": "triage returned REPAIR with 1 of 1 rewrites spent",
        "kind": RejectKind.TASK,
    }
    assert (triage.proposals, repair.calls) == ([MACHINELESS_SHELLSIM, PROPOSAL], 1)


async def test_a_rewrite_that_is_not_a_valid_proposal_rejects_the_workpiece_with_its_error(tmp_path):
    repair = MachineRepair()
    stages = replace(STAGES, triage=TriageInTurn(TriageDecision.REPAIR), repair=repair, author=no_call)
    log = InMemoryEventLog()

    assert await run_workpiece(coordinator(tmp_path, stages, REPAIRING), WORKPIECE, PROPOSAL, log) is Terminal.REJECTED
    assert exits(log) == [("opened", None), ("triaged", "reject"), ("terminal", "rejected")]
    assert log.events()[-1].attrs["reason"] == (
        f"the rewrite for the REPAIR verdict is not a valid proposal: no header fix applies to {PROPOSAL.header.id}"
    )
    assert repair.calls == 1
    assert not (tmp_path / "workpieces" / WORKPIECE / REPAIRED_DIR).exists()


async def test_under_the_default_budget_a_repair_verdict_rejects_without_a_rewrite(tmp_path):
    log = InMemoryEventLog()
    stages = replace(STAGES, triage=TriageInTurn(TriageDecision.REPAIR), repair=no_call)

    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, log) is Terminal.REJECTED
    assert exits(log) == [("opened", None), ("triaged", "reject"), ("terminal", "rejected")]
    assert log.events()[-1].attrs["reason"] == "triage returned REPAIR with 0 of 0 rewrites spent"
    assert not (tmp_path / "workpieces" / WORKPIECE / REPAIRED_DIR).exists()


AUTHOR_STATE: Mapping[str, JsonValue] = {"obligations": ["derive 42 in the grader design"], "checked": 3}
"""Author state a consumer's triage could attach on accept; the package's triage attaches none."""


async def test_a_verdicts_author_state_reaches_every_author_call_through_the_event_log_across_a_relaunch(tmp_path):
    triage = TriageInTurn(TriageDecision.ACCEPT, author_state=AUTHOR_STATE)
    with pytest.raises(ModelUnavailable):
        await run_workpiece(
            coordinator(tmp_path, replace(STAGES, triage=triage, author=unavailable_author)),
            WORKPIECE,
            PROPOSAL,
            InMemoryEventLog(),
        )
    triaged = (JsonEventStore(tmp_path).load(WORKPIECE) or ())[1]
    assert triaged.attrs["author_state"] == content_hash.canonical_json(AUTHOR_STATE)

    author = RecordingAuthor(PROGRAM)
    resumed = replace(STAGES, triage=no_call, author=author, build=FailsOnce(BuildFailure("wrong product", "task")))
    log = InMemoryEventLog()
    assert await run_workpiece(coordinator(tmp_path, resumed), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED

    assert [None if revision is None else revision.kind for revision in author.revisions] == [
        None,
        None,
        RevisionKind.BUILD_FAILURE,
        RevisionKind.BUILD_FAILURE,
    ]
    assert author.states == [AUTHOR_STATE] * 4
    assert derive_state(log.events()).author_state == AUTHOR_STATE


async def test_the_default_triage_records_an_empty_author_state(tmp_path):
    author = RecordingAuthor(PROGRAM)
    log = InMemoryEventLog()

    assert await run_workpiece(coordinator(tmp_path, replace(STAGES, author=author)), WORKPIECE, PROPOSAL, log) is (
        Terminal.ACCEPTED
    )
    assert log.events()[1].attrs["author_state"] == "{}"
    assert (author.states, derive_state(log.events()).author_state) == ([{}, {}], {})


RETRY = InfrastructureRetry(max_attempts=3, backoff_initial=0.001, backoff_maximum=0.001, backoff_factor=1.0)
ARRIVAL_TIMEOUT = 10
"""Seconds a proposal iterable waits for a workpiece's progress before the test fails instead of hanging."""


async def arriving(*workpiece_ids: str) -> AsyncIterator[tuple[str, TaskProposal]]:
    for workpiece_id in workpiece_ids:
        yield workpiece_id, PROPOSAL


def stored_log(root: Path, workpiece_id: str) -> InMemoryEventLog:
    log = InMemoryEventLog()
    log.restore(JsonEventStore(root).load(workpiece_id) or ())
    return log


class FailsFor:
    """A build function raising ``error`` on the first ``times`` builds of ``workpiece_id``, counted in ``calls``.

    Every other build runs ``run_build``.
    """

    def __init__(self, workpiece_id: str, times: int, error: Exception = HOST_FAILURE) -> None:
        self.workpiece_id = workpiece_id
        self.times = times
        self.error = error
        self.calls = 0

    async def __call__(
        self, program: RLEnvBuildProgram, proposal: TaskProposal, workpiece_id: str, *args: Any, **kwargs: Any
    ) -> TaskDraft:
        if workpiece_id == self.workpiece_id:
            self.calls += 1
            if self.calls <= self.times:
                raise self.error
        return await run_build(program, proposal, workpiece_id, *args, **kwargs)


async def test_run_workpieces_fails_a_workpiece_in_place_and_cancels_no_sibling_in_flight(tmp_path):
    waiting: list[str] = []
    siblings_waiting, failed = asyncio.Event(), asyncio.Event()

    async def validate_or_crash(draft: TaskDraft) -> dict[str, int]:
        if draft.task.id == "toy-1":
            await siblings_waiting.wait()
            failed.set()
            raise RuntimeError("the validator crashed")
        waiting.append(draft.task.id)
        if len(waiting) == 2:
            siblings_waiting.set()
        await failed.wait()
        return SUMMARY

    stages = replace(STAGES, validate=validate_or_crash)
    terminals = await run_workpieces(coordinator(tmp_path, stages), arriving("toy-0", "toy-1", "toy-2"), RETRY)

    assert terminals == {"toy-0": Terminal.ACCEPTED, "toy-1": Terminal.FAILED, "toy-2": Terminal.ACCEPTED}
    assert exits(stored_log(tmp_path, "toy-1")) == [*ACCEPTED_EXITS[:4], ("terminal", "failed")]
    for sibling in ("toy-0", "toy-2"):
        assert exits(stored_log(tmp_path, sibling)) == ACCEPTED_EXITS


@pytest.mark.parametrize(
    ("failures", "terminal", "builds"),
    [(2, Terminal.ACCEPTED, 3), (5, Terminal.FAILED, 3)],
    ids=["recovers", "attempts spent"],
)
async def test_run_workpieces_retries_a_host_failure_in_process_while_its_attempts_last(
    tmp_path, failures, terminal, builds
):
    build = FailsFor("toy-1", failures)
    terminals = await run_workpieces(
        coordinator(tmp_path, replace(STAGES, build=build)), arriving("toy-0", "toy-1"), RETRY
    )

    assert terminals == {"toy-0": Terminal.ACCEPTED, "toy-1": terminal}
    assert build.calls == builds
    log = stored_log(tmp_path, "toy-1")
    assert exits(log).count(("terminal", "failed")) == min(failures, RETRY.max_attempts)
    assert derive_state(log.events()).build_revisions == 0


async def test_run_workpieces_retries_an_unavailable_model_at_the_stage_that_raised(tmp_path):
    validate_once = UnavailableOnce()
    stages = replace(STAGES, validate=validate_once)

    assert await run_workpieces(coordinator(tmp_path, stages), arriving(WORKPIECE), RETRY) == {
        WORKPIECE: Terminal.ACCEPTED
    }
    assert exits(stored_log(tmp_path, WORKPIECE)) == [*ACCEPTED_EXITS[:4], ("terminal", "failed"), *ACCEPTED_EXITS[4:]]
    assert validate_once.calls == 2


@pytest.mark.parametrize(
    "error",
    [
        BuildInfrastructureFailure(InfrastructureCause.NO_FACTORY, "this host has no local machine factory"),
        BuildExpired("ran for more than 1800 seconds without returning", ExpiredClock.STEP, "numbers"),
    ],
    ids=["host rejection", "expired clock"],
)
async def test_run_workpieces_does_not_retry_a_failure_another_attempt_would_meet_again(tmp_path, error):
    build = FailsFor(WORKPIECE, RETRY.max_attempts, error)
    stages = replace(STAGES, build=build)

    assert await run_workpieces(coordinator(tmp_path, stages), arriving(WORKPIECE), RETRY) == {
        WORKPIECE: Terminal.FAILED
    }
    assert build.calls == 1


async def test_run_workpieces_starts_a_workpiece_before_the_next_proposal_arrives(tmp_path):
    order: list[str] = []
    triaged = asyncio.Event()

    async def triage(proposal: TaskProposal) -> Verdict:
        order.append("triaged")
        triaged.set()
        return await preflight(proposal)

    async def proposals() -> AsyncIterator[tuple[str, TaskProposal]]:
        yield "toy-0", PROPOSAL
        async with asyncio.timeout(ARRIVAL_TIMEOUT):
            await triaged.wait()
        order.append("toy-1 arrives")
        yield "toy-1", PROPOSAL

    terminals = await run_workpieces(coordinator(tmp_path, replace(STAGES, triage=triage)), proposals(), RETRY)

    assert terminals == {"toy-0": Terminal.ACCEPTED, "toy-1": Terminal.ACCEPTED}
    assert order == ["triaged", "toy-1 arrives", "triaged"]


class SignallingStore(JsonEventStore):
    """A JSON event store setting ``saved`` once it has saved a workpiece's events."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.saved = asyncio.Event()

    def save(self, workpiece_id: str, events: tuple[WorkpieceEvent, ...]) -> None:
        super().save(workpiece_id, events)
        self.saved.set()


async def test_run_workpieces_returns_only_once_its_proposals_are_exhausted_and_every_workpiece_is_terminal(tmp_path):
    store = SignallingStore(tmp_path)

    async def proposals() -> AsyncIterator[tuple[str, TaskProposal]]:
        yield "toy-0", PROPOSAL
        async with asyncio.timeout(ARRIVAL_TIMEOUT):
            await store.saved.wait()
        yield "toy-1", PROPOSAL

    terminals = await run_workpieces(replace(coordinator(tmp_path), event_store=store), proposals(), RETRY)

    assert terminals == {"toy-0": Terminal.ACCEPTED, "toy-1": Terminal.ACCEPTED}
    for workpiece_id in terminals:
        assert exits(stored_log(tmp_path, workpiece_id)) == ACCEPTED_EXITS


async def test_run_workpieces_refuses_a_repeated_workpiece_id_before_starting_it_and_finishes_the_first(tmp_path):
    triage = TriageInTurn(TriageDecision.ACCEPT)
    stages = replace(STAGES, triage=triage)

    with pytest.raises(ValueError, match=f"workpiece {WORKPIECE} arrived a second time"):
        await run_workpieces(coordinator(tmp_path, stages), arriving(WORKPIECE, WORKPIECE, "toy-1"), RETRY)
    assert triage.proposals == [PROPOSAL]
    assert exits(stored_log(tmp_path, WORKPIECE)) == ACCEPTED_EXITS
    assert JsonEventStore(tmp_path).load("toy-1") is None
