# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the coordinator with a three-step toy workpiece and controlled stage failures.

``numbers`` writes a resource and ``grader`` the grader, and the ``task`` sink uses both. ``SOURCE``
holds the same program as module text, as an author writing Python would return it, and passes the
restricted loader.

Invariants: validation evidence belongs to the exact task (the build refuses a program other than the
``AUTHORED`` one; a relaunch of a finished workpiece under another proposal, budget, model, research or policy is
refused); the solve rate rests on complete graded evidence (a custom review cannot close a task with an evidence
gap).
"""

import asyncio
import base64
import inspect
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Container, Mapping, Sequence
from dataclasses import replace
from functools import partial
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
from taskforge.defaults import DEFAULT_POLICY
from taskforge.ledger.records import DiscardingLedger
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.client import Completion, FinishReason, ModelUnavailable, StructuredResult, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError
from taskforge.sandbox.factories import MachineHost
from taskforge.spec.controls import ControlKind
from taskforge.spec.draft import assemble, requirements
from taskforge.stage10_proposal.model import (
    REQUIRED_HEADINGS,
    Environment,
    ProposalFormatError,
    SolverMachine,
    TaskProposal,
    parse,
    render,
)
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage20_triage.verdict import TriageDecision, Verdict
from taskforge.stage30_author.author import (
    PROGRAM_FILE,
    AuthorContext,
    AuthoredProgram,
    Revision,
    RevisionKind,
    static_author,
)
from taskforge.stage30_author.loader import SdkExports, load_program_source
from taskforge.stage30_author.program import (
    RLEnvBuildProgram,
    rl_env_build_program,
    rl_env_build_sink,
    rl_env_build_step,
)
from taskforge.stage40_build_with_program.agentic_author import AgenticAuthor, SessionBuild
from taskforge.stage40_build_with_program.context import (
    BuildExpired,
    BuildFailure,
    BuildInfrastructureFailure,
    ExpiredClock,
    RLEnvBuildContext,
)
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause
from taskforge.stage40_build_with_program.outputs import Grader
from taskforge.stage40_build_with_program.program_sdk import load_program
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
from taskforge.stage50_validate.controls import ControlVerdict
from taskforge.stage50_validate.evidence import RewardStats
from taskforge.stage50_validate.outcome import Cause
from taskforge.stage50_validate.round import (
    VALIDATION_FILE,
    ControlResult,
    Lowering,
    LoweringStatus,
    SolverEvidence,
    TrialResult,
    TrialStatus,
    ValidationContext,
    ValidationResult,
    load_validation,
)
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision, Modify, Reject, RejectKind, Retry
from taskforge.stage60_review.rules import BandRules, default_review
from taskforge.stage_coordinator.coordinator import (
    AUTHORED_DIR,
    DECISION_FILE,
    FAILURES_DIR,
    REPAIRED_DIR,
    STORE_DIR,
    StageCoordinator,
    WorkpieceBudget,
    run_workpiece,
)
from taskforge.stage_coordinator.event_log import InMemoryEventLog, JsonEventStore
from taskforge.stage_coordinator.run import InfrastructureRetry, run_workpieces
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge.stage_coordinator.workpiece_machine import EXIT, EventKind, Stage, Terminal, WorkpieceEvent, derive_state
from tests.proposals import proposal_document
from tests.stage40_build_with_program.scripted_author import CONFIG, ScriptedModel, in_process_loader, turn
from tests.stage40_build_with_program.scripted_author import PROGRAM as AGENTIC_SOURCE

WORKPIECE = "toy-0"
SECTION = (
    "The build writes the two numbers six and seven to workspace/numbers.txt, and the solver reads the file, "
    "multiplies them and replies with the product alone inside the answer box as a plain integer. The grader is "
    "an exact match on 42 after stripping whitespace, so the sum 13 or an empty reply earns nothing."
)
PROPOSAL = parse(
    proposal_document(
        proposal_id="toy/0",
        source={"kind": "capability", "ref": "toy", "hash": "0"},
        environment="shellsim",
        verification="simple",
        research_reason="the numbers are in the task's own file",
        build=[
            {"name": "numbers", "description": "the file holding the two numbers"},
            {"name": "grader", "description": "checks the product"},
        ],
        resources=["workspace/numbers.txt"],
        body="".join(f"## {heading}\n{SECTION}\n\n" for heading in REQUIRED_HEADINGS).replace(
            "## Task\n", "## Task\nMultiply the two numbers in /workspace/numbers.txt. ", 1
        ),
    )
)
BUDGET = WorkpieceBudget(max_triage_repairs=0, max_build_revisions=1, max_validation_retries=1, max_modify_rounds=1)
REPAIRING = replace(BUDGET, max_triage_repairs=1)
"""``BUDGET`` for a consumer using the REPAIR surface: one rewrite, then a REPAIR verdict rejects."""
SAMPLING = LLMPolicy()
BAND = BandRules(low=0.25, high=0.75, accepted=frozenset({BandOutcome.IN_BAND}))
SERVICES = RLEnvBuildServices(
    host=MachineHost.LAPTOP,
    factories={Backend.SHELLSIM.value: ShellSimMachineFactory()},
    images=None,
    web_tools=(),
    ledger=DiscardingLedger(),
    reasoning_replay=ReasoningReplay.KEEP,
    agent_limits=AgentLimits(max_turns=1, max_tokens=1024, max_calls=1),
)


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


def passing_result(context: ValidationContext) -> ValidationResult:
    """The round of ``context`` lowered, with four graded solver trials of which two solve the task."""
    trials = tuple(
        TrialResult(str(index), TrialStatus.GRADED, reward, None, f"solver/{index}/attempt-0.json")
        for index, reward in enumerate((1.0, 0.0, 1.0, 0.0))
    )
    stats = RewardStats(graded=4, mean_reward=0.5, solved=2, timed_out=0, candidate_code_errors=0)
    lowering = Lowering(LoweringStatus.LOWERED, "", (), None, "lowered-task")
    return ValidationResult(
        context.workpiece_id,
        context.round,
        lowering,
        SolverEvidence(trials, stats, {}),
        (
            ControlResult(
                "negative", ControlKind.NEGATIVE, ControlVerdict.MET, 0.0, None, "controls/negative/attempt-0.json"
            ),
        ),
        (),
        None,
    )


async def validate(draft: TaskDraft, context: ValidationContext) -> ValidationResult:
    return passing_result(context)


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
    review=partial(default_review, band=BAND),
)
UNUSED_MODEL_DIGEST = build_model_digest(UnusedModel(), SAMPLING, SERVICES)


class Author:
    """Gives each reply in turn and the last from then on: a program, a source to load, or an error to raise.

    It records each call's context, revision and author state.
    """

    def __init__(self, *replies: RLEnvBuildProgram | str | BaseException) -> None:
        self.replies = list(replies)
        self.contexts: list[AuthorContext] = []
        self.revisions: list[Revision | None] = []
        self.states: list[Mapping[str, JsonValue]] = []

    async def __call__(
        self,
        context: AuthorContext,
        proposal: TaskProposal,
        revision: Revision | None,
        author_state: Mapping[str, JsonValue],
    ) -> AuthoredProgram:
        self.contexts.append(context)
        self.revisions.append(revision)
        self.states.append(author_state)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, str):
            return AuthoredProgram(load_toy_program(reply), reply)
        return AuthoredProgram(reply, None)


HOST_FAILURE = BuildInfrastructureFailure("image_pull", "the sandbox image did not pull")
UNAVAILABLE = ModelUnavailable("the hold ran out", ())


class FailsOn:
    """Runs ``then`` and records each call's arguments, raising ``error`` instead on the calls numbered in ``failing``.

    With ``only``, a call that does not pass ``only`` as an argument runs ``then`` unrecorded.
    """

    def __init__(
        self,
        then: Callable[..., Awaitable[Any]],
        error: BaseException = HOST_FAILURE,
        failing: Container[int] = (1,),
        only: str | None = None,
    ) -> None:
        self.then, self.error, self.failing, self.only = then, error, failing, only
        self.calls: list[tuple[Any, ...]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.only is None or self.only in args:
            self.calls.append(args)
            if len(self.calls) in self.failing:
                raise self.error
        return await self.then(*args, **kwargs)


class FlakyStore(JsonEventStore):
    """A JSON event store whose first save of events that ``fails`` accepts finds the storage unreachable."""

    def __init__(self, root: Path, fails: Callable[[tuple[WorkpieceEvent, ...]], bool] = lambda events: True) -> None:
        super().__init__(root)
        self.fails = fails
        self.failed = False

    def save(self, workpiece_id: str, events: tuple[WorkpieceEvent, ...]) -> None:
        if not self.failed and self.fails(events):
            self.failed = True
            raise OSError("the event store is unreachable")
        super().save(workpiece_id, events)


def coordinator(
    root: Path,
    stages: StageFunctions = STAGES,
    budget: WorkpieceBudget = BUDGET,
    digest: str = UNUSED_MODEL_DIGEST,
    research: ResearchRequirement = ResearchRequirement.OFF,
    policy_digest: str = DEFAULT_POLICY.stage_digest,
) -> StageCoordinator:
    return StageCoordinator(
        stages,
        budget,
        root,
        JsonEventStore(root),
        UnusedModel(),
        SAMPLING,
        SERVICES,
        research,
        digest,
        policy_digest,
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
    author = Author(PROGRAM)
    log = InMemoryEventLog()
    assert await run_workpiece(coordinator(tmp_path, replace(STAGES, author=author)), WORKPIECE, PROPOSAL, log) is (
        Terminal.ACCEPTED
    )

    assert exits(log) == ACCEPTED_EXITS
    assert JsonEventStore(tmp_path).load(WORKPIECE) == log.events()
    assert log.events()[1].attrs["author_state"] == "{}"
    assert (author.states, derive_state(log.events()).author_state) == ([{}, {}], {})
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
async def test_a_failed_preflight_holds_before_any_later_stage(tmp_path, proposal, reason):
    assert proposal.header.environment is Environment.SHELLSIM
    log = InMemoryEventLog()
    stages = replace(STAGES, author=no_call, load_program=no_call, build=no_call, validate=no_call, review=no_call)
    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, proposal, log) is Terminal.HELD

    assert exits(log) == [("opened", None), ("triaged", "hold"), ("terminal", "held")]
    triaged, terminal = log.events()[1:]
    assert triaged.attrs["reason"] == terminal.attrs["reason"]
    assert terminal.attrs["reason"].startswith("triage rejected the proposal: ")
    assert reason in terminal.attrs["reason"]


async def test_a_failed_initial_checkpoint_holds_before_execution_and_resumes(tmp_path):
    store = FlakyStore(tmp_path)
    log = InMemoryEventLog()
    with pytest.raises(OSError, match="unreachable"):
        await run_workpiece(replace(coordinator(tmp_path), event_store=store), WORKPIECE, PROPOSAL, log)
    assert derive_state(stored_log(tmp_path, WORKPIECE).events()).terminal is Terminal.HELD
    assert exits(log) == [("opened", None), ("terminal", "held")]

    retried = replace(coordinator(tmp_path), event_store=store)
    assert await run_workpiece(retried, WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert store.load(WORKPIECE) == log.events()
    assert exits(log) == [ACCEPTED_EXITS[0], ("terminal", "held"), *ACCEPTED_EXITS[1:]]


async def test_an_author_returning_source_runs_once_and_the_build_loads_its_program_file(tmp_path):
    author = Author(SOURCE)
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
POLICY = DEFAULT_POLICY.stage_digest


@pytest.mark.parametrize(
    ("proposal", "budget", "digest", "research", "policy_digest", "refusal"),
    [
        (
            replace(PROPOSAL, body=PROPOSAL.body.replace("Multiply", "Add")),
            BUDGET,
            UNUSED_MODEL_DIGEST,
            OFF,
            POLICY,
            "opened for proposal",
        ),
        (PROPOSAL, replace(BUDGET, max_build_revisions=2), UNUSED_MODEL_DIGEST, OFF, POLICY, "opened under budget"),
        (PROPOSAL, BUDGET, "another-model", OFF, POLICY, "opened with model digest"),
        (PROPOSAL, BUDGET, UNUSED_MODEL_DIGEST, ResearchRequirement.ENFORCED, POLICY, "opened with research off"),
        (PROPOSAL, BUDGET, UNUSED_MODEL_DIGEST, OFF, "another-policy", "opened with policy digest"),
    ],
    ids=["proposal", "budget", "model digest", "research", "policy digest"],
)
async def test_a_relaunch_with_another_proposal_budget_model_research_or_policy_is_refused(
    tmp_path, proposal, budget, digest, research, policy_digest, refusal
):
    await run_workpiece(coordinator(tmp_path), WORKPIECE, PROPOSAL, InMemoryEventLog())
    before = tree_bytes(tmp_path)

    relaunch = coordinator(tmp_path, budget=budget, digest=digest, research=research, policy_digest=policy_digest)
    with pytest.raises(ValueError, match=f"{WORKPIECE} was {refusal}"):
        await run_workpiece(relaunch, WORKPIECE, proposal, InMemoryEventLog())
    assert tree_bytes(tmp_path) == before


async def test_a_program_other_than_the_authored_one_fails_the_workpiece_at_build(tmp_path):
    stages = replace(STAGES, author=Author(PROGRAM, FAILING))
    with pytest.raises(ValueError, match="now returns program"):
        await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, InMemoryEventLog())

    state = derive_state(JsonEventStore(tmp_path).load(WORKPIECE) or ())
    assert (state.stage, state.terminal) == (Stage.BUILD, Terminal.HELD)


async def test_a_program_file_changed_on_disk_fails_the_workpiece_at_build(tmp_path):
    stages = replace(STAGES, author=Author(SOURCE), build=FailsOn(run_build))
    with pytest.raises(BuildInfrastructureFailure):
        await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, InMemoryEventLog())
    program_file = tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / AUTHORED_DIR / "0" / PROGRAM_FILE
    program_file.write_text(SOURCE.replace("6 7", "6 8"))

    with pytest.raises(ValueError, match="holds program"):
        await run_workpiece(
            coordinator(tmp_path, replace(stages, author=no_call)), WORKPIECE, PROPOSAL, InMemoryEventLog()
        )
    state = derive_state(JsonEventStore(tmp_path).load(WORKPIECE) or ())
    assert (state.stage, state.terminal) == (Stage.BUILD, Terminal.HELD)


async def test_a_build_failure_costs_a_revision_and_the_next_one_abandons_the_workpiece(tmp_path):
    author = Author(FAILING)
    log = InMemoryEventLog()
    terminal = await run_workpiece(coordinator(tmp_path, replace(STAGES, author=author)), WORKPIECE, PROPOSAL, log)

    assert terminal is Terminal.ABANDONED
    assert exits(log) == [
        ("opened", None),
        ("triaged", "accept"),
        ("authored", "authored"),
        ("build_failed", "revise"),
        ("authored", "authored"),
        ("build_failed", "abandon"),
        ("terminal", "abandoned"),
    ]
    assert log.events()[-1].attrs["kind"] == "budget"
    asked = [None if revision is None else revision.kind for revision in author.revisions]
    assert asked == [None, None, RevisionKind.BUILD_FAILURE, RevisionKind.BUILD_FAILURE]
    revision = author.revisions[-1]
    assert revision is not None
    assert revision.detail.startswith(
        f"program.py:{inspect.getsourcelines(broken.fn)[1] + 2}, in step broken: "
        'raise BuildFailure("the numbers do not multiply to the answer")\n'
        "BuildFailure: the numbers do not multiply to the answer\n\n"
    )


NUMBERS_FILE = 'InlineFile(content_base64=base64.b64encode(b"6 7\\n").decode())'


def program_frame(source: str, text: str, step: str) -> str:
    """The program frame that leads a failure's head, for the line of ``source`` holding ``text`` in ``step``."""
    lines = source.splitlines()
    number = next(number for number, line in enumerate(lines, start=1) if text in line)
    return f"program.py:{number}, in step {step}: {lines[number - 1].strip()}"


async def failure_heads(root: Path, source: str) -> tuple[str, str, str]:
    """Run a workpiece authoring ``source`` to budget exhaustion, and return the head of its failure text.

    It returns the head, the text before the first blank line, from ``build-failures/1.txt`` and from
    the revision detail the second author call received, and the ``failure`` attr of the first
    ``BUILD_FAILED`` event.
    """
    author = Author(source)
    log = InMemoryEventLog()
    assert (
        await run_workpiece(coordinator(root, replace(STAGES, author=author)), WORKPIECE, PROPOSAL, log)
        is Terminal.ABANDONED
    )
    revision = author.revisions[1]
    assert revision is not None and revision.kind is RevisionKind.BUILD_FAILURE
    text = (root / "workpieces" / WORKPIECE / "rounds" / "0" / FAILURES_DIR / "1.txt").read_text()
    assert text == revision.detail
    built = next(event for event in log.events() if event.kind == "build_failed")
    return text.partition("\n\n")[0], revision.detail.partition("\n\n")[0], built.attrs["failure"]


async def test_a_name_error_in_a_step_reports_its_program_line_to_the_author(tmp_path):
    """The loader refuses a name that nothing binds, so the step declares it global and never assigns it."""
    declared = "async def numbers(context: RLEnvBuildContext) -> TaskResource:\n"
    source = SOURCE.replace(NUMBERS_FILE, "InlineFile(content_base64=encoded_numbers)").replace(
        declared, declared + "    global encoded_numbers\n", 1
    )
    expected = (
        program_frame(source, "content_base64=encoded_numbers", "numbers")
        + "\nNameError: name 'encoded_numbers' is not defined"
    )
    assert await failure_heads(tmp_path, source) == (expected,) * 3


async def test_an_os_error_raised_by_a_step_is_the_programs_failure(tmp_path):
    """The restricted loader excludes ``open``, so the step raises FileNotFoundError directly."""
    missing = '    raise FileNotFoundError(2, "No such file or directory", "/workspace/does-not-exist.txt")\n'
    source = SOURCE.replace("    return TaskResource(\n", missing + "    return TaskResource(\n", 1)
    expected = (
        program_frame(source, "does-not-exist", "numbers")
        + "\nFileNotFoundError: [Errno 2] No such file or directory: '/workspace/does-not-exist.txt'"
    )
    assert await failure_heads(tmp_path, source) == (expected,) * 3


async def test_a_library_error_called_from_a_step_still_reports_the_program_line(tmp_path):
    source = SOURCE.replace(NUMBERS_FILE, '"6 7"')
    assert source != SOURCE
    head, detail_head, attr = await failure_heads(tmp_path, source)
    assert head == detail_head == attr
    frame, library, error = head.split("\n")
    assert frame == program_frame(source, "return TaskResource(", "numbers")
    assert library.startswith("raised in library code at pydantic/")
    assert error == "ValidationError: 1 validation error for TaskResource"


async def test_a_step_failing_under_two_dependents_reports_its_own_program_line(tmp_path):
    """Both later steps await ``numbers``; its failure keeps the frame where the step raised it."""
    raising = '    rows = {}["rows"]\n    return TaskResource(\n'
    source = SOURCE.replace("    return TaskResource(\n", raising, 1).replace(
        "async def grader(context: RLEnvBuildContext)",
        "async def grader(context: RLEnvBuildContext, numbers: TaskResource)",
    )
    assert source.count('{}["rows"]') == 1 and "numbers: TaskResource) -> Grader" in source
    head, detail_head, attr = await failure_heads(tmp_path, source)
    assert head == detail_head == attr == program_frame(source, '{}["rows"]', "numbers") + "\nKeyError: 'rows'"


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


CUT = StructuredOutputError(ValueError("the reply was cut at the output limit"), (CUT_REPLY, CUT_REPLY))


async def test_an_author_reply_cut_after_its_repairs_costs_a_revision_and_the_author_samples_again(tmp_path):
    author = Author(CUT, SOURCE)
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


STAGE_ORDER = ("author", "build", "validate")


@pytest.mark.parametrize(
    ("stage", "error"),
    [
        ("author", UNAVAILABLE),
        ("build", HOST_FAILURE),
        ("build", FileNotFoundError("blobs/ab/abcdef is missing")),
        ("build", BuildExpired("ran for more than 1800 seconds without returning", ExpiredClock.STEP, "numbers")),
        ("validate", UNAVAILABLE),
        ("validate", asyncio.CancelledError()),
    ],
    ids=["author model", "build host", "missing file", "expired clock", "validation model", "cancelled"],
)
async def test_a_failure_outside_the_program_holds_at_its_stage_unspent_and_a_relaunch_resumes_there(
    tmp_path, stage, error
):
    stages = replace(STAGES, author=Author(SOURCE))
    failing: dict[str, Any] = {stage: FailsOn(getattr(stages, stage), error)}
    stages = replace(stages, **failing)
    with pytest.raises(type(error), match=re.escape(str(error)) or None):
        await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, InMemoryEventLog())
    state = derive_state(stored_log(tmp_path, WORKPIECE).events())
    assert (state.stage, state.terminal, state.build_revisions) == (stage, Terminal.HELD, 0)
    assert not (tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / FAILURES_DIR).exists()

    earlier = STAGE_ORDER[: STAGE_ORDER.index(stage)]
    log = InMemoryEventLog()
    resumed = replace(stages, **dict.fromkeys(earlier, no_call))
    assert await run_workpiece(coordinator(tmp_path, resumed), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    at = 2 + len(earlier)
    assert exits(log) == [*ACCEPTED_EXITS[:at], ("terminal", "held"), *ACCEPTED_EXITS[at:]]


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
    assert exits(log) == [*ACCEPTED_EXITS[:3], ("terminal", "held"), ("terminal", "held")]
    assert derive_state(log.events()).build_revisions == 0
    assert not (tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / FAILURES_DIR).exists()


async def test_a_failed_checkpoint_after_re_author_resumes_the_recorded_program(tmp_path):
    other = SOURCE.replace("6 7", "6 8")
    build = FailsOn(run_build, BuildFailure("the numbers do not multiply to the answer", "task"))
    first = coordinator(tmp_path, replace(STAGES, author=Author(other), build=build))
    store = FlakyStore(tmp_path, lambda events: events[-1].kind is EventKind.AUTHORED and len(events) > 4)
    with pytest.raises(OSError, match="unreachable"):
        await run_workpiece(replace(first, event_store=store), WORKPIECE, PROPOSAL, InMemoryEventLog())
    authored = tmp_path / "workpieces" / WORKPIECE / "rounds" / "0" / AUTHORED_DIR
    assert (authored / "1" / PROGRAM_FILE).read_text() == other

    log = InMemoryEventLog()
    resumed = replace(STAGES, author=no_call)
    assert await run_workpiece(coordinator(tmp_path, resumed), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert derive_state(log.events()).build_revisions == 1
    assert exits(log)[-4:] == ACCEPTED_EXITS[-4:]
    assert (authored / "0" / PROGRAM_FILE).read_text() == other


async def test_a_modify_re_authors_in_round_one_and_validate_and_review_receive_each_rounds_context_across_a_relaunch(
    tmp_path,
):
    validate_once = FailsOn(validate, UNAVAILABLE, (2,))
    decisions: list[Decision] = [Modify(findings=("the file states the product",)), Accept(band=BandOutcome.IN_BAND)]
    reviewed: list[ValidationResult] = []

    async def review_in_turn(result: ValidationResult) -> Decision:
        reviewed.append(result)
        return decisions.pop(0)

    author = Author(PROGRAM)
    stages = replace(STAGES, author=author, validate=validate_once, review=review_in_turn)
    with pytest.raises(ModelUnavailable):
        await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, InMemoryEventLog())

    log = InMemoryEventLog()
    resumed = replace(stages, author=no_call, build=no_call)
    assert await run_workpiece(coordinator(tmp_path, resumed), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert [(event.kind, event.round) for event in log.events()][5:8] == [("decided", 0), ("authored", 1), ("built", 1)]
    assert author.revisions[2:] == [Revision(kind=RevisionKind.MODIFY, detail="the file states the product")] * 2
    # The author stage and the build stage's call again, as the author returns no source, in each round.
    directory = tmp_path / "workpieces" / WORKPIECE
    assert author.contexts == [AuthorContext(WORKPIECE, 0, directory)] * 2 + [AuthorContext(WORKPIECE, 1, directory)] * 2
    rounds = directory / "rounds"
    assert (rounds / "1" / DRAFT_DIR / TASK_FILE).is_file()
    first, second = (
        ValidationContext(WORKPIECE, number, rounds / str(number), PROPOSAL, str(tmp_path / STORE_DIR), left, 1)
        for number, left in ((0, 1), (1, 0))
    )
    assert [context for _, context in validate_once.calls] == [first, second, second]
    assert load_validation(rounds / "1") == passing_result(second)
    assert reviewed == [passing_result(first), passing_result(second)]
    assert [(event.kind, event.round) for event in log.events()][-4:] == [
        ("terminal", 1),
        ("validated", 1),
        ("decided", 1),
        ("terminal", 1),
    ]


class GradedOnPass:
    """A validate function whose passes before ``graded_from`` leave one solver trial ungraded for a retryable cause
    and whose later passes grade it, recording each context."""

    def __init__(self, graded_from: int) -> None:
        self.graded_from = graded_from
        self.contexts: list[ValidationContext] = []

    async def __call__(self, draft: TaskDraft, context: ValidationContext) -> ValidationResult:
        self.contexts.append(context)
        result = passing_result(context)
        if context.validation_pass >= self.graded_from:
            return result
        assert result.solver is not None
        ungraded = TrialResult("3", TrialStatus.UNGRADED, None, Cause.MODEL_UNAVAILABLE, "solver/3/attempt-1.json")
        stats = RewardStats(graded=3, mean_reward=2 / 3, solved=2, timed_out=0, candidate_code_errors=0)
        solver = SolverEvidence((*result.solver.trials[:3], ungraded), stats, {Cause.MODEL_UNAVAILABLE: 1})
        return replace(result, solver=solver)


@pytest.mark.parametrize("holds", [1, 2])
async def test_holds_resume_validation_keeping_its_passes_without_spending_author_revisions(tmp_path, holds):
    validate_passes = GradedOnPass(graded_from=holds + 1)
    stages = replace(STAGES, validate=validate_passes)
    for _ in range(holds):
        log = InMemoryEventLog()
        assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, log) is Terminal.HELD
        held = derive_state(log.events())
        assert (held.stage, held.round, held.build_revisions, held.retries) == (Stage.VALIDATE, 0, 0, 0)
    resumed = replace(stages, triage=no_call, author=no_call, build=no_call)
    log = InMemoryEventLog()
    assert await run_workpiece(coordinator(tmp_path, resumed), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert [context.validation_pass for context in validate_passes.contexts] == list(range(1, holds + 2))
    assert derive_state(log.events()).round == 0


async def always_modify(result: ValidationResult) -> Decision:
    return Modify(findings=("the file states the product",))


@pytest.mark.parametrize(
    ("max_modify_rounds", "decided"),
    [(0, [(0, "abandon")]), (1, [(0, "modify"), (1, "abandon")])],
    ids=["no rounds", "one round"],
)
async def test_a_modify_past_the_modify_rounds_abandons_and_a_larger_budget_resumes_review_before_a_new_round(
    tmp_path, max_modify_rounds, decided
):
    stages = replace(STAGES, author=Author(PROGRAM), review=always_modify)
    first = coordinator(tmp_path, stages, replace(BUDGET, max_modify_rounds=max_modify_rounds))
    log = InMemoryEventLog()

    assert await run_workpiece(first, WORKPIECE, PROPOSAL, log) is Terminal.ABANDONED
    assert [(event.round, event.attrs[EXIT]) for event in log.events() if event.kind == "decided"] == decided
    closed = log.events()[-1]
    assert (closed.attrs["terminal"], closed.attrs["kind"]) == ("abandoned", RejectKind.BUDGET)
    assert "the file states the product" in closed.attrs["reason"]
    assert derive_state(log.events()).round == max_modify_rounds
    rounds = tmp_path / "workpieces" / WORKPIECE / "rounds"
    old_rounds = [tree_bytes(rounds / str(number)) for number in range(max_modify_rounds + 1)]

    async def review(result: ValidationResult) -> Decision:
        if result.round <= max_modify_rounds:
            return await always_modify(result)
        return await default_review(result, band=BAND)

    budget = replace(BUDGET, max_modify_rounds=max_modify_rounds + 1)
    log = InMemoryEventLog()
    assert await run_workpiece(
        replace(first, stages=replace(STAGES, review=review), budget=budget), WORKPIECE, PROPOSAL, log
    ) is (Terminal.ACCEPTED)
    assert derive_state(log.events()).round == max_modify_rounds + 1
    assert [tree_bytes(rounds / str(number)) for number in range(max_modify_rounds + 1)] == old_rounds


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
    failing = coordinator(tmp_path, replace(stages, author=Author(UNAVAILABLE)), REPAIRING)
    with pytest.raises(ModelUnavailable):
        await run_workpiece(failing, WORKPIECE, MACHINELESS_SHELLSIM, InMemoryEventLog())

    author = Author(PROGRAM)
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
        ("terminal", "held"),
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
    assert repaired.attrs["proposal_file"] == f"{REPAIRED_DIR}/1.json"
    assert (triage.proposals, repair.calls, proposals) == ([MACHINELESS_SHELLSIM, PROPOSAL], 1, [PROPOSAL])
    assert derive_state(log.events()).triage_repairs == 1


async def test_an_invalid_rewrite_holds_with_its_error(tmp_path):
    repair = MachineRepair()
    stages = replace(STAGES, triage=TriageInTurn(TriageDecision.REPAIR), repair=repair, author=no_call)
    log = InMemoryEventLog()

    assert await run_workpiece(coordinator(tmp_path, stages, REPAIRING), WORKPIECE, PROPOSAL, log) is Terminal.HELD
    assert exits(log) == [("opened", None), ("triaged", "hold"), ("terminal", "held")]
    assert log.events()[-1].attrs["reason"] == (
        f"the rewrite for the REPAIR verdict is not a valid proposal: no header fix applies to {PROPOSAL.header.id}"
    )
    assert repair.calls == 1
    assert not (tmp_path / "workpieces" / WORKPIECE / REPAIRED_DIR).exists()


async def test_under_the_default_budget_a_repair_verdict_abandons_without_a_rewrite(tmp_path):
    log = InMemoryEventLog()
    stages = replace(STAGES, triage=TriageInTurn(TriageDecision.REPAIR), repair=no_call)

    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, log) is Terminal.ABANDONED
    assert exits(log) == [("opened", None), ("triaged", "abandon"), ("terminal", "abandoned")]
    assert log.events()[-1].attrs["reason"] == "triage returned REPAIR with 0 of 0 rewrites spent"
    assert not (tmp_path / "workpieces" / WORKPIECE / REPAIRED_DIR).exists()


AUTHOR_STATE: Mapping[str, JsonValue] = {"obligations": ["derive 42 in the grader design"], "checked": 3}
"""Author state a consumer's triage could attach on accept; the package's triage attaches none."""


async def test_a_verdicts_author_state_reaches_every_author_call_through_the_event_log_across_a_relaunch(tmp_path):
    triage = TriageInTurn(TriageDecision.ACCEPT, author_state=AUTHOR_STATE)
    with pytest.raises(ModelUnavailable):
        await run_workpiece(
            coordinator(tmp_path, replace(STAGES, triage=triage, author=Author(UNAVAILABLE))),
            WORKPIECE,
            PROPOSAL,
            InMemoryEventLog(),
        )
    triaged = (JsonEventStore(tmp_path).load(WORKPIECE) or ())[1]
    assert triaged.attrs["author_state"] == content_hash.canonical_json(AUTHOR_STATE)

    author = Author(PROGRAM)
    resumed = replace(
        STAGES, triage=no_call, author=author, build=FailsOn(run_build, BuildFailure("wrong product", "task"))
    )
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


async def test_run_workpieces_fails_a_workpiece_in_place_and_cancels_no_sibling_in_flight(tmp_path):
    waiting: list[str] = []
    siblings_waiting, failed = asyncio.Event(), asyncio.Event()

    async def validate_or_crash(draft: TaskDraft, context: ValidationContext) -> ValidationResult:
        if draft.task.id == "toy-1":
            await siblings_waiting.wait()
            failed.set()
            raise RuntimeError("the validator crashed")
        waiting.append(draft.task.id)
        if len(waiting) == 2:
            siblings_waiting.set()
        await failed.wait()
        return passing_result(context)

    stages = replace(STAGES, validate=validate_or_crash)
    terminals = await run_workpieces(coordinator(tmp_path, stages), arriving("toy-0", "toy-1", "toy-2"), RETRY)

    assert terminals == {"toy-0": Terminal.ACCEPTED, "toy-1": Terminal.HELD, "toy-2": Terminal.ACCEPTED}
    assert exits(stored_log(tmp_path, "toy-1")) == [*ACCEPTED_EXITS[:4], ("terminal", "held")]
    for sibling in ("toy-0", "toy-2"):
        assert exits(stored_log(tmp_path, sibling)) == ACCEPTED_EXITS


@pytest.mark.parametrize(
    ("failures", "terminal", "builds"),
    [(2, Terminal.ACCEPTED, 3), (5, Terminal.ABANDONED, 3)],
    ids=["recovers", "attempts spent"],
)
async def test_run_workpieces_retries_a_host_failure_in_process_while_its_attempts_last(
    tmp_path, failures, terminal, builds
):
    build = FailsOn(run_build, failing=range(1, failures + 1), only="toy-1")
    terminals = await run_workpieces(
        coordinator(tmp_path, replace(STAGES, build=build)), arriving("toy-0", "toy-1"), RETRY
    )

    assert terminals == {"toy-0": Terminal.ACCEPTED, "toy-1": terminal}
    assert len(build.calls) == builds
    log = stored_log(tmp_path, "toy-1")
    assert exits(log).count(("terminal", "held")) == min(failures, RETRY.max_attempts)
    assert derive_state(log.events()).build_revisions == 0


async def test_run_workpieces_retries_an_unavailable_model_at_the_stage_that_raised(tmp_path):
    validate_once = FailsOn(validate, UNAVAILABLE)
    stages = replace(STAGES, validate=validate_once)

    assert await run_workpieces(coordinator(tmp_path, stages), arriving(WORKPIECE), RETRY) == {
        WORKPIECE: Terminal.ACCEPTED
    }
    assert exits(stored_log(tmp_path, WORKPIECE)) == [*ACCEPTED_EXITS[:4], ("terminal", "held"), *ACCEPTED_EXITS[4:]]
    assert len(validate_once.calls) == 2


@pytest.mark.parametrize(
    "error",
    [
        BuildInfrastructureFailure(InfrastructureCause.NO_FACTORY, "this host has no local machine factory"),
        BuildExpired("ran for more than 1800 seconds without returning", ExpiredClock.STEP, "numbers"),
    ],
    ids=["host rejection", "expired clock"],
)
async def test_run_workpieces_does_not_retry_a_failure_another_attempt_would_meet_again(tmp_path, error):
    build = FailsOn(run_build, error, range(1, RETRY.max_attempts + 1))
    stages = replace(STAGES, build=build)

    assert await run_workpieces(coordinator(tmp_path, stages), arriving(WORKPIECE), RETRY) == {WORKPIECE: Terminal.HELD}
    assert len(build.calls) == 1


async def test_run_workpieces_starts_a_workpiece_before_the_next_proposal_arrives_and_returns_once_all_are_terminal(
    tmp_path,
):
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


def agentic_author(model: ScriptedModel, root: Path) -> AgenticAuthor:
    """The agentic author driving ``model``, whose sessions build as the coordinator under ``root`` builds.

    A session's builds run ``run_build`` in process on the coordinator's step store, with its model,
    services, research requirement and model digest, so the build stage serves the submitted program's
    steps from memo hits alone.
    """
    store = str(root / STORE_DIR)

    def builds(context: AuthorContext, proposal: TaskProposal) -> SessionBuild:
        async def builder(source: str, run_dir: Path, *, invalidate: frozenset[str]) -> TaskDraft:
            return await run_build(
                load_program(source),
                proposal,
                context.workpiece_id,
                run_dir,
                store,
                UnusedModel(),
                SAMPLING,
                SERVICES,
                context.round,
                research=ResearchRequirement.OFF,
                invalidate=invalidate,
                model_digest=UNUSED_MODEL_DIGEST,
            )

        return SessionBuild(builder, store, SERVICES, root / "sessions.jsonl", in_process_loader)

    return AgenticAuthor(CONFIG, model, SAMPLING, SERVICES.ledger, builds, DEFAULT_POLICY.author)


async def test_the_coordinator_runs_the_agentic_author_as_its_author_stage_through_build_and_validation(tmp_path):
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": AGENTIC_SOURCE})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "grades 6 * 7"})),
    )
    validate_drafts = FailsOn(validate, failing=())
    stages = replace(STAGES, author=agentic_author(model, tmp_path), load_program=load_program, validate=validate_drafts)
    log = InMemoryEventLog()

    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert exits(log) == ACCEPTED_EXITS and not model.replies
    round_dir = tmp_path / "workpieces" / WORKPIECE / "rounds" / "0"
    assert (round_dir / AUTHORED_DIR / "0" / PROGRAM_FILE).read_text() == AGENTIC_SOURCE
    assert (round_dir / "agentic" / "0" / "workspace" / PROGRAM_FILE).read_text() == AGENTIC_SOURCE
    built = next(event for event in log.events() if event.kind == "built")
    assert built.attrs["hits"] == built.attrs["steps"] == "3", "the build stage reran no step the session built"
    [(draft, _)] = validate_drafts.calls
    assert draft.task.id == WORKPIECE


async def test_a_build_failure_returns_to_the_agentic_session_that_submitted_the_program(tmp_path):
    model = ScriptedModel(
        turn(("write_file", {"path": "program.py", "content": AGENTIC_SOURCE})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "grades 6 * 7"})),
        turn(("run_build", {"path": "program.py"})),
        turn(("submit", {"path": "program.py", "notes": "the grader holds"})),
    )
    failure = BuildFailure("the reference reply does not parse", "grader")
    stages = replace(
        STAGES, author=agentic_author(model, tmp_path), load_program=load_program, build=FailsOn(run_build, failure)
    )
    log = InMemoryEventLog()

    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert exits(log) == [*ACCEPTED_EXITS[:3], ("build_failed", "revise"), *ACCEPTED_EXITS[2:]]
    revision_turn = str(model.requests[3].messages[-1]["content"])
    assert "grader: the reference reply does not parse" in revision_turn
    round_dir = tmp_path / "workpieces" / WORKPIECE / "rounds" / "0"
    session = json.loads((round_dir / "agentic" / "1" / "session.json").read_text())
    assert (session["mode"], session["continued_from"]) == ("same_session", "rounds/0/agentic/0")
    assert (round_dir / AUTHORED_DIR / "1" / PROGRAM_FILE).read_text() == AGENTIC_SOURCE


async def test_author_replies_cut_past_the_budget_abandon_and_a_larger_budget_resumes_keeping_old_files(tmp_path):
    author = Author(CUT, CUT, SOURCE)
    first = coordinator(tmp_path, replace(STAGES, author=author))
    log = InMemoryEventLog()
    assert await run_workpiece(first, WORKPIECE, PROPOSAL, log) is Terminal.ABANDONED
    assert exits(log) == [
        *ACCEPTED_EXITS[:2],
        ("author_failed", "revise"),
        ("author_failed", "abandon"),
        ("terminal", "abandoned"),
    ]
    assert log.events()[-1].attrs["kind"] == RejectKind.BUDGET
    assert [None if revision is None else revision.kind for revision in author.revisions] == [
        None,
        RevisionKind.AUTHOR_FAILURE,
    ]
    assert derive_state(log.events()).stage is Stage.AUTHOR
    before = tree_bytes(tmp_path / "workpieces" / WORKPIECE / "rounds")

    resumed = replace(first, budget=replace(BUDGET, max_build_revisions=3))
    assert await run_workpiece(resumed, WORKPIECE, PROPOSAL, InMemoryEventLog()) is Terminal.ACCEPTED
    after = tree_bytes(tmp_path / "workpieces" / WORKPIECE / "rounds")
    assert all(after[path] == data for path, data in before.items())
    state = derive_state(stored_log(tmp_path, WORKPIECE).events())
    assert state.build_revisions == state.build_revisions_at_resume == 2
    revision = author.revisions[-1]
    assert revision is not None and revision.kind is RevisionKind.AUTHOR_FAILURE


async def test_a_second_repair_verdict_abandons_without_a_second_repair_and_a_relaunch_resumes_the_rewrite(tmp_path):
    triage = TriageInTurn(TriageDecision.REPAIR, TriageDecision.REPAIR, TriageDecision.ACCEPT)
    repair = MachineRepair()
    first = coordinator(tmp_path, replace(STAGES, triage=triage, repair=repair), REPAIRING)
    log = InMemoryEventLog()

    assert await run_workpiece(first, WORKPIECE, MACHINELESS_SHELLSIM, log) is Terminal.ABANDONED
    assert exits(log) == [("opened", None), ("triaged", "repair"), ("triaged", "abandon"), ("terminal", "abandoned")]
    rejected, closed = log.events()[2:]
    assert (rejected.digest, rejected.attrs["decision"], rejected.attrs["repairs"]) == (
        content_hash.digest(PROPOSAL),
        "repair",
        "1",
    )
    assert closed.attrs == {
        "terminal": "abandoned",
        "reason": "triage returned REPAIR with 1 of 1 rewrites spent",
        "kind": RejectKind.BUDGET,
    }
    assert (triage.proposals, repair.calls) == ([MACHINELESS_SHELLSIM, PROPOSAL], 1)

    assert await run_workpiece(first, WORKPIECE, MACHINELESS_SHELLSIM, InMemoryEventLog()) is Terminal.ACCEPTED
    assert triage.proposals[-1] == PROPOSAL
    state = derive_state(stored_log(tmp_path, WORKPIECE).events())
    assert state.triage_repairs == state.triage_repairs_at_resume == 1


async def test_exhausted_validation_retries_resume_without_rebuilding(tmp_path):
    async def retry(result):
        return Retry(Cause.MODEL_UNAVAILABLE, 1)

    first = coordinator(tmp_path, replace(STAGES, review=retry), replace(BUDGET, max_validation_retries=0))
    assert await run_workpiece(first, WORKPIECE, PROPOSAL, InMemoryEventLog()) is Terminal.ABANDONED
    resumed = replace(first, stages=replace(STAGES, author=no_call, build=no_call))
    assert await run_workpiece(resumed, WORKPIECE, PROPOSAL, InMemoryEventLog()) is Terminal.ACCEPTED
    state = derive_state(stored_log(tmp_path, WORKPIECE).events())
    assert state.validation_passes == 2 and state.round == 0


@pytest.mark.parametrize("decision", [Accept(BandOutcome.IN_BAND), Reject(RejectKind.TASK, ("bad task",))])
async def test_custom_review_cannot_close_a_task_with_missing_evidence(tmp_path, decision):
    async def incomplete(draft, context):
        return replace(passing_result(context), controls=())

    async def review(result):
        return decision

    stages = replace(STAGES, validate=incomplete, review=review)
    log = InMemoryEventLog()
    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, log) is Terminal.HELD
    assert derive_state(log.events()).stage is Stage.VALIDATE


async def test_failed_final_checkpoint_preserves_a_conclusive_decision(tmp_path):
    store = FlakyStore(tmp_path, lambda events: events[-1].attrs.get("terminal") == "accepted")
    first = replace(coordinator(tmp_path), event_store=store)
    with pytest.raises(OSError, match="unreachable"):
        await run_workpiece(first, WORKPIECE, PROPOSAL, InMemoryEventLog())
    assert derive_state(stored_log(tmp_path, WORKPIECE).events()).terminal is Terminal.ACCEPTED
    untouched = replace(first, stages=StageFunctions(no_call, no_call, no_call, no_call, no_call, no_call, no_call))
    assert await run_workpiece(untouched, WORKPIECE, PROPOSAL, InMemoryEventLog()) is Terminal.ACCEPTED


async def test_explicit_inner_budget_abandons_without_automatic_retry(tmp_path):
    error = BuildInfrastructureFailure(InfrastructureCause.BUDGET_EXHAUSTED, "model call budget exhausted")
    build = FailsOn(run_build, error)
    first = coordinator(tmp_path, replace(STAGES, build=build))
    assert await run_workpieces(first, arriving(WORKPIECE), RETRY) == {WORKPIECE: Terminal.ABANDONED}
    state = derive_state(stored_log(tmp_path, WORKPIECE).events())
    assert (state.stage, state.terminal, state.build_revisions) == (Stage.BUILD, Terminal.ABANDONED, 0)
    assert await run_workpiece(first, WORKPIECE, PROPOSAL, InMemoryEventLog()) is Terminal.ACCEPTED
