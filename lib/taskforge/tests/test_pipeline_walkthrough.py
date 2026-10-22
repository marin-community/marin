# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the coordinator with a three-step toy workpiece and controlled stage failures.

``numbers`` writes a resource and ``grader`` the grader, and the ``task`` sink uses both. ``SOURCE``
holds the same program as module text, as an author writing Python would return it, and passes the
restricted loader.
"""

import base64
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import pytest
from pydantic import BaseModel
from taskcompendium.grader import verifyit_package
from taskcompendium.models import AnswerType, InlineFile, PlainText, Source, TaskResource, TaskSpec
from verifyit.spec import NumericSpec

from taskforge.llm.client import Completion, FinishReason, ModelUnavailable, StructuredResult, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError
from taskforge.spec.draft import assemble, requirements
from taskforge.stage20_triage.verdict import accept
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
from taskforge.stage40_build_with_program.context import BuildFailure, BuildInfrastructureFailure, RLEnvBuildContext
from taskforge.stage40_build_with_program.outputs import Grader
from taskforge.stage40_build_with_program.run import DRAFT_DIR, TASK_FILE, TaskDraft, run_build
from taskforge.stage40_build_with_program.step_store import StoreCorruption, model_digest
from taskforge.stage60_review.decision import Accept, BandOutcome, Decision, Modify, RejectKind
from taskforge.stage_coordinator.coordinator import (
    AUTHORED_DIR,
    DECISION_FILE,
    FAILURES_DIR,
    STORE_DIR,
    VALIDATION_FILE,
    StageCoordinator,
    WorkpieceBudget,
    run_workpiece,
)
from taskforge.stage_coordinator.event_log import InMemoryEventLog, JsonEventStore
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge.stage_coordinator.workpiece_machine import EXIT, Stage, Terminal, WorkpieceEvent, derive_state

WORKPIECE = "toy-0"
PROPOSAL = {"id": "toy/0", "task": "Multiply the two numbers in /workspace/numbers.txt."}
BUDGET = WorkpieceBudget(max_build_revisions=1, max_validation_retries=1)
SAMPLING = LLMPolicy()
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
async def task(context: RLEnvBuildContext, numbers: TaskResource, grader: Grader, proposal: dict[str, str]) -> TaskSpec:
    return assemble(
        context.workpiece_id,
        proposal["task"],
        AnswerType.TEXT,
        PlainText(),
        grader.package,
        Source(dataset="taskforge", revision="test", row=context.workpiece_id, importer_revision="test"),
        environment=requirements(image=None),
        files=(numbers,),
    )


@rl_env_build_sink
async def broken(context: RLEnvBuildContext, proposal: dict[str, str]) -> TaskSpec:
    raise BuildFailure("the numbers do not multiply to the answer")


PROGRAM = rl_env_build_program(numbers, grader, sink=task)
FAILING = rl_env_build_program(sink=broken)
SOURCE_IMPORTS = """\
import base64

from taskcompendium.grader import verifyit_package
from taskcompendium.models import AnswerType, InlineFile, PlainText, Source, TaskResource, TaskSpec
from verifyit.spec import NumericSpec

from taskforge.spec.draft import assemble, requirements
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
    triage=accept,
    author=static_author(PROGRAM),
    load_program=load_toy_program,
    build=run_build,
    validate=validate,
    review=review,
)
UNUSED_MODEL_DIGEST = model_digest(UnusedModel(), SAMPLING)


class RecordingAuthor:
    """Returns ``programs`` in turn, the last one from then on, and records each context and revision asked for."""

    def __init__(self, *programs: RLEnvBuildProgram):
        self.programs = list(programs)
        self.contexts: list[AuthorContext] = []
        self.revisions: list[Revision | None] = []

    async def __call__(self, context: AuthorContext, proposal: object, revision: Revision | None) -> AuthoredProgram:
        self.contexts.append(context)
        self.revisions.append(revision)
        return AuthoredProgram(self.programs.pop(0) if len(self.programs) > 1 else self.programs[0], None)


class SourceAuthor:
    """Returns ``source`` with the program loaded from it, and records each revision asked for."""

    def __init__(self, source: str = SOURCE) -> None:
        self.source = source
        self.revisions: list[Revision | None] = []

    async def __call__(self, context: AuthorContext, proposal: object, revision: Revision | None) -> AuthoredProgram:
        self.revisions.append(revision)
        return AuthoredProgram(load_toy_program(self.source), self.source)


HOST_FAILURE = BuildInfrastructureFailure("image_pull", "the sandbox image did not pull")


class FailsOnce:
    """A build function raising ``error`` on its first call and running ``run_build`` after."""

    def __init__(self, error: Exception = HOST_FAILURE) -> None:
        self.error = error
        self.calls = 0

    async def __call__(self, *args: Any) -> TaskDraft:
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return await run_build(*args)


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
    root: Path, stages: StageFunctions = STAGES, budget: WorkpieceBudget = BUDGET, digest: str = UNUSED_MODEL_DIGEST
) -> StageCoordinator:
    return StageCoordinator(stages, budget, root, JsonEventStore(root), UnusedModel(), SAMPLING, digest)


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
    untouched = StageFunctions(no_call, no_call, no_call, no_call, no_call, no_call)
    relaunch = await run_workpiece(coordinator(tmp_path, untouched), WORKPIECE, PROPOSAL, InMemoryEventLog())
    assert relaunch is Terminal.ACCEPTED
    assert tree_bytes(tmp_path) == before


async def test_a_retry_after_a_failed_save_of_a_finished_workpiece_writes_its_events(tmp_path):
    store = SaveFailsOnce(tmp_path)
    log = InMemoryEventLog()
    with pytest.raises(OSError, match="unreachable"):
        await run_workpiece(replace(coordinator(tmp_path), event_store=store), WORKPIECE, PROPOSAL, log)
    assert store.load(WORKPIECE) is None

    untouched = StageFunctions(no_call, no_call, no_call, no_call, no_call, no_call)
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


@pytest.mark.parametrize(
    ("proposal", "budget", "digest", "refusal"),
    [
        ({**PROPOSAL, "task": "Add the two numbers."}, BUDGET, UNUSED_MODEL_DIGEST, "opened for proposal"),
        (PROPOSAL, replace(BUDGET, max_build_revisions=2), UNUSED_MODEL_DIGEST, "opened under budget"),
        (PROPOSAL, BUDGET, "another-model", "opened with model digest"),
    ],
    ids=["proposal", "budget", "model digest"],
)
async def test_a_relaunch_with_another_proposal_budget_or_model_digest_is_refused(
    tmp_path, proposal, budget, digest, refusal
):
    await run_workpiece(coordinator(tmp_path), WORKPIECE, PROPOSAL, InMemoryEventLog())
    before = tree_bytes(tmp_path)

    with pytest.raises(ValueError, match=f"{WORKPIECE} was {refusal}"):
        await run_workpiece(coordinator(tmp_path, budget=budget, digest=digest), WORKPIECE, proposal, InMemoryEventLog())
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
    expected = "NameError: name 'encoded_numbers' is not defined at numbers, " + program_line(
        source, "content_base64=encoded_numbers"
    )
    assert await failure_first_lines(tmp_path, source) == (expected,) * 3


async def test_an_os_error_raised_by_a_step_is_the_programs_failure(tmp_path):
    """The restricted loader excludes ``open``, so the step raises FileNotFoundError directly."""
    missing = '    raise FileNotFoundError(2, "No such file or directory", "/workspace/does-not-exist.txt")\n'
    source = SOURCE.replace("    return TaskResource(\n", missing + "    return TaskResource(\n", 1)
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

    async def __call__(self, context: AuthorContext, proposal: object, revision: Revision | None) -> AuthoredProgram:
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


async def unavailable_author(context: AuthorContext, proposal: object, revision: Revision | None) -> AuthoredProgram:
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
    [HOST_FAILURE, FileNotFoundError("blobs/ab/abcdef is missing")],
    ids=["build host", "missing file"],
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


async def builds_then_the_host_fails(*args: Any) -> TaskDraft:
    await run_build(*args)
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


async def fails_in_the_program(*args: Any) -> TaskDraft:
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


async def test_a_modify_decision_re_authors_the_workpiece_in_round_one_with_the_findings_and_context(tmp_path):
    decisions: list[Decision] = [Modify(findings=("the file states the product",)), Accept(band=BandOutcome.IN_BAND)]
    author = RecordingAuthor(PROGRAM)

    async def review_in_turn(summary: object) -> Decision:
        return decisions.pop(0)

    stages = replace(STAGES, author=author, review=review_in_turn)
    log = InMemoryEventLog()

    assert await run_workpiece(coordinator(tmp_path, stages), WORKPIECE, PROPOSAL, log) is Terminal.ACCEPTED
    assert [(event.kind, event.round) for event in log.events()][5:8] == [("decided", 0), ("authored", 1), ("built", 1)]
    assert author.revisions[2:] == [Revision(kind=RevisionKind.MODIFY, detail="the file states the product")] * 2
    # The author stage and the build stage's call again, as the author returns no source, in each round.
    directory = tmp_path / "workpieces" / WORKPIECE
    assert author.contexts == [AuthorContext(WORKPIECE, 0, directory)] * 2 + [AuthorContext(WORKPIECE, 1, directory)] * 2
    assert (tmp_path / "workpieces" / WORKPIECE / "rounds" / "1" / DRAFT_DIR / TASK_FILE).is_file()
