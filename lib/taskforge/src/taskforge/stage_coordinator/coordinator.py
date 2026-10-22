# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a workpiece through its stages with the stage functions a caller passes.

A ``StageCoordinator`` holds the caller's ``StageFunctions``, the workpiece budget, the run root, the
event store, and the model with its sampling policy. ``run_workpiece`` runs a workpiece to a terminal:

1. When the caller's log is empty and the event store holds the workpiece's events, ``run_workpiece``
   restores the log from the store.
2. It opens the workpiece with an ``OPENED`` event, or checks the proposal, the budget and the model
   digest against the existing ``OPENED`` event.
3. It folds the log with ``derive_state``. An accepted or rejected workpiece returns its terminal here,
   before any stage runs or any file is written.
4. It runs the stage in the state, appends the stage's event and folds the event in, until the
   workpiece reaches a terminal.
5. It saves the log to the event store, after every other file of the workpiece.

Each stage runs through an adapter in this module. The adapter calls the caller's function, writes
the round's files, and returns the stage's event.

The triage stage exits accept or reject on the verdict's decision, and a reject records the
verdict's reasons. An accept records the verdict's ``author_state``, a JSON mapping the workpiece
state folds and every author call receives; the preflight's is empty, and nothing in the package
writes one, so it is a surface for a consumer's triage to hand findings or obligations to the
author.

A REPAIR verdict is a surface the preflight or a Taskforge consumer may use, for a deterministic
rewrite such as fixing a proposal header, by setting ``max_triage_repairs``. The edge may later
retry a proposal on the findings of the deterministic preflight, which returns no REPAIR verdict
today. While the budget lasts, the stage calls ``StageFunctions.repair``, writes the rewritten
proposal to ``repaired/<k>.md`` in the workpiece directory, ``k`` counting the rewrites, and exits
repair, so triage judges the rewrite next. A REPAIR verdict past the budget, which is every one
under a budget of zero, or a rewrite raising ``ProposalFormatError``, exits reject. Every stage
after a rewrite runs it, read back from its file and checked against the digest its event records.

The coordinator spends a build revision on:

* a build raising ``BuildFailure`` or any other error, recorded as ``BUILD_FAILED``, unless the
  error is a ``BuildInfrastructureFailure``, a ``BuildExpired``, ``ModelUnavailable``, or an
  ``OSError`` raised outside the program's own frames: a storage fault, a missing blob or memo, or a corrupt one
  (``StoreCorruption``). An ``OSError`` raised in the program, a step opening a file that is not
  there, is the program's and spends a revision like any other error of its code;
* an author raising ``StructuredOutputError``, recorded as ``AUTHOR_FAILED``. The author's
  structured reply stayed invalid through its repairs, as when the output limit cut it, and a new
  call samples again.

For either error the adapter writes the failure text to ``build-failures/<k>.txt`` in the round
directory: a first line locating the failure, then the traceback tail. For a build failure, the
first line gives the step and the program line the failure came from, since the last frames of a
traceback are often library code. The files share a counter, ``WorkpieceState.build_revisions``,
because both errors spend the same budget. Any other exception, from these stages or any other,
ends the run with ``TERMINAL`` failed. The coordinator then saves the log and re-raises, and the
next ``run_workpiece`` on the workpiece resumes at that stage, with the step store serving the steps already
built.

The author stage writes each author call's files to ``authored/<k>/`` in the round directory,
with ``k`` set to the workpiece's build and author failures so far (``WorkpieceState.build_revisions``):

* ``program.json``, the program's manifest;
* ``program.py``, the program's source, when the author returned one.

The build stage reads ``k`` from the same state and loads the program from
``authored/<k>/program.py`` with ``StageFunctions.load_program``. A later author call writes to a
new directory, so a relaunch resuming at build after a process was killed mid-revision still finds
the program its ``AUTHORED`` event records. For a program without source, as ``static_author``
returns, the build stage asks the author again with the revision kept in the state. Such an author
has to return the same program for the same proposal and revision. Either way, the build stage
refuses a program whose digest differs from the ``AUTHORED`` event.

The layout of the run root is in ``lib/taskforge/README.md``.
"""

import inspect
import traceback
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename

from taskforge.content_hash import canonical_json, digest, pretty_json
from taskforge.ledger.records import check_workpiece_id
from taskforge.llm.client import ModelClient, ModelUnavailable
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.structured import StructuredOutputError
from taskforge.stage10_proposal.model import ProposalFormatError, TaskProposal, parse, render
from taskforge.stage20_triage.verdict import TriageDecision
from taskforge.stage30_author.author import (
    PROGRAM_FILE,
    PROGRAM_MANIFEST_FILE,
    Revision,
    RevisionKind,
    program_manifest,
)
from taskforge.stage30_author.program import RLEnvBuildProgram
from taskforge.stage40_build_with_program.context import BuildExpired, BuildFailure, BuildInfrastructureFailure
from taskforge.stage40_build_with_program.infrastructure import exception_chain, storage_fault
from taskforge.stage40_build_with_program.run import DRAFT_DIR, ResearchRequirement, load_draft
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage40_build_with_program.step_store import CacheStatus
from taskforge.stage50_validate.round import ValidationContext, load_validation, write_validation
from taskforge.stage60_review.decision import Accept, Decision, Modify, Reject, RejectKind, Retry
from taskforge.stage60_review.rules import BandRules, ReviewContext
from taskforge.stage_coordinator.event_log import WORKPIECES_DIR, WorkpieceEventLog, WorkpieceEventStore
from taskforge.stage_coordinator.stage_functions import StageFunctions
from taskforge.stage_coordinator.workpiece_machine import (
    AUTHOR_STATE_ATTR,
    EXIT,
    FINAL,
    REASON_ATTR,
    TERMINAL_ATTR,
    TRIAGE_REJECTED,
    EventKind,
    Exit,
    Stage,
    Terminal,
    WorkpieceEvent,
    WorkpieceState,
    derive_state,
    transition,
)

STORE_DIR = "store"
ROUNDS_DIR = "rounds"
AUTHORED_DIR = "authored"
REPAIRED_DIR = "repaired"
FAILURES_DIR = "build-failures"
DECISION_FILE = "decision.json"
BUDGET_DIGEST = "budget_digest"
MODEL_DIGEST = "model_digest"
RESEARCH = "research"
FAILURE_CHARS = 6000
"""The length of the traceback tail kept from a build or author failure for the author's revision."""
ATTR_CHARS = 500
"""The length of the failure's first line kept in a ``BUILD_FAILED`` or ``AUTHOR_FAILED`` event.

The failure file keeps the rest.
"""


@dataclass(frozen=True)
class WorkpieceBudget:
    """The triage rewrites, build revisions, validation retries and MODIFY rounds each workpiece may spend.

    Attributes:
        max_triage_repairs: The REPAIR verdicts a workpiece may answer with a rewrite that triage judges
            again. Zero sends every REPAIR verdict to reject; a consumer using the surface sets one.
        max_build_revisions: The build and author failures a workpiece may revise after. The next failure
            rejects it.
        max_validation_retries: The review ``Retry`` decisions a workpiece may follow before it is abandoned.
        max_modify_rounds: The review ``Modify`` decisions a workpiece may follow into a new round. The next
            one rejects it on its budget; zero rejects on the first.
    """

    max_triage_repairs: int
    max_build_revisions: int
    max_validation_retries: int
    max_modify_rounds: int


@dataclass(frozen=True)
class StageCoordinator:
    """The stage functions, budget and storage shared by a run's workpieces.

    Attributes:
        stages: The caller's function for each stage.
        budget: The budget of every workpiece. ``OPENED`` records its digest.
        band: The band rules every review receives in its ``ReviewContext``.
        root: The run root holding ``workpieces/`` and the step store in ``store/``.
        event_store: The store each workpiece's events are saved to on reaching a terminal.
        model: The model build steps call.
        sampling: The sampling policy build steps pass to ``model``.
        services: The build host's machines, images, web tools, ledger and agent-loop settings, passed
            to every build.
        research: The research requirement of every build (``run_build``'s ``research``). ``OPENED``
            records it.
        model_digest: The model part of every step key, usually
            ``run.build_model_digest(model, sampling, services)``. ``OPENED`` records it.
    """

    stages: StageFunctions
    budget: WorkpieceBudget
    band: BandRules
    root: Path
    event_store: WorkpieceEventStore
    model: ModelClient
    sampling: LLMPolicy
    services: RLEnvBuildServices
    research: ResearchRequirement
    model_digest: str


@dataclass(frozen=True)
class _Workpiece:
    coordinator: StageCoordinator
    workpiece_id: str
    proposal: TaskProposal
    log: WorkpieceEventLog

    @property
    def directory(self) -> Path:
        return self.coordinator.root / WORKPIECES_DIR / self.workpiece_id

    def round_dir(self, round: int) -> Path:  # noqa: A002 - the workpiece's round
        return self.directory / ROUNDS_DIR / str(round)

    def authored_dir(self, state: WorkpieceState) -> Path:
        """The directory of the author call made at ``state``, keyed by the build and author failures so far."""
        return self.round_dir(state.round) / AUTHORED_DIR / str(state.build_revisions)

    def proposal_at(self, state: WorkpieceState) -> TaskProposal:
        """The proposal the stages run at ``state``: the opened one, or the last repair read back from its file.

        Raises:
            ValueError: the repaired proposal's file holds a proposal other than the one its event records.
        """
        if state.triage_repairs == 0:
            return self.proposal
        path = self.directory / _repaired_file(state.triage_repairs)
        proposal = parse(path.read_text())
        if digest(proposal) != state.proposal_digest:
            raise ValueError(f"{path} holds proposal {digest(proposal)}, not the repaired {state.proposal_digest}")
        return proposal


def _repaired_file(repairs: int) -> Path:
    """The path, in the workpiece directory, of the proposal after ``repairs`` triage repairs."""
    return Path(REPAIRED_DIR) / f"{repairs}.md"


async def run_workpiece(
    coordinator: StageCoordinator, workpiece_id: str, proposal: TaskProposal, log: WorkpieceEventLog
) -> Terminal:
    """Run ``proposal`` as workpiece ``workpiece_id`` to a terminal, resuming from ``log`` or the event store.

    A workpiece already ``ACCEPTED`` or ``REJECTED`` returns at once. When the event store lacks that
    final log, as after a failed save and a retry with the same ``log``, ``run_workpiece`` saves it first.
    Otherwise it writes nothing. An ``ABANDONED`` workpiece re-enters validation with its retries reset,
    and a ``FAILED`` one resumes at the stage that raised.

    A REPAIR verdict within ``max_triage_repairs`` has ``StageFunctions.repair`` rewrite the proposal,
    triage judges the rewrite, and every later stage runs it. A REPAIR verdict past the budget rejects
    the workpiece.

    A ``Modify`` decision within ``max_modify_rounds`` sends the workpiece back to the author for the
    next round. One past it rejects the workpiece with ``RejectKind.BUDGET``.

    A build failure, or an author raising ``StructuredOutputError``, spends one of the workpiece's build
    revisions and sends the workpiece back to the author. When the build raises
    ``BuildInfrastructureFailure``, ``BuildExpired``, ``ModelUnavailable`` or an ``OSError`` from
    outside the program's frames, the workpiece fails in place with its revisions unspent, and the next
    ``run_workpiece`` builds the same program again. Such an ``OSError`` is a storage fault, a missing
    blob or memo, or a corrupt one (``StoreCorruption``); one raised by the program's own code is the
    program's.

    Raises:
        ValueError: ``workpiece_id`` is unsafe as a directory name, the workpiece was opened for another
            proposal, budget, model digest or research requirement, the build found a program other than the authored
            one, a repaired proposal's file changed, or the log is inconsistent.
        Exception: whatever a stage raised, after the workpiece is saved as failed.
    """
    check_workpiece_id(workpiece_id)
    stored = coordinator.event_store.load(workpiece_id)
    if not log.events() and stored is not None:
        log.restore(stored)
    workpiece = _Workpiece(coordinator, workpiece_id, proposal, log)
    _open(workpiece)
    state = derive_state(log.events())
    if state.terminal is not None and state.terminal in FINAL:
        if stored != log.events():
            coordinator.event_store.save(workpiece_id, log.events())
        return state.terminal
    try:
        state = await _advance(workpiece, state)
    except Exception as error:
        attrs = {TERMINAL_ATTR: Terminal.FAILED.value, REASON_ATTR: f"{type(error).__name__}: {error}"}
        log.append(WorkpieceEvent(EventKind.TERMINAL, log.events()[-1].round, None, attrs))
        coordinator.event_store.save(workpiece_id, log.events())
        raise
    coordinator.event_store.save(workpiece_id, log.events())
    assert state.terminal is not None
    return state.terminal


async def _advance(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceState:
    while True:
        event = await ADAPTERS[state.stage](workpiece, state)
        state = transition(state, workpiece.log.append(event))
        if state.terminal is not None:
            return state


def _open(workpiece: _Workpiece) -> None:
    proposal_digest = digest(workpiece.proposal)
    coordinator = workpiece.coordinator
    attrs = {
        BUDGET_DIGEST: digest(coordinator.budget),
        MODEL_DIGEST: coordinator.model_digest,
        RESEARCH: str(coordinator.research),
    }
    events = workpiece.log.events()
    if not events:
        workpiece.log.append(WorkpieceEvent(EventKind.OPENED, 0, proposal_digest, attrs))
        return
    first = events[0]
    if first.digest != proposal_digest:
        raise ValueError(f"{workpiece.workpiece_id} was opened for proposal {first.digest}, not {proposal_digest}")
    if first.attrs[BUDGET_DIGEST] != attrs[BUDGET_DIGEST]:
        raise ValueError(
            f"{workpiece.workpiece_id} was opened under budget {first.attrs[BUDGET_DIGEST]}, not {attrs[BUDGET_DIGEST]}"
        )
    if first.attrs[MODEL_DIGEST] != attrs[MODEL_DIGEST]:
        raise ValueError(
            f"{workpiece.workpiece_id} was opened with model digest {first.attrs[MODEL_DIGEST]}, "
            f"not {attrs[MODEL_DIGEST]}"
        )
    if first.attrs[RESEARCH] != attrs[RESEARCH]:
        raise ValueError(
            f"{workpiece.workpiece_id} was opened with research {first.attrs[RESEARCH]}, not {attrs[RESEARCH]}"
        )


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_rename(str(path)) as temp:
        Path(temp).write_bytes(content)


async def _triage(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Triage the workpiece's proposal and exit on the verdict's decision.

    An ACCEPT records the verdict's ``author_state`` as canonical JSON on the event, and a REJECT the
    verdict's reasons. A REPAIR verdict within ``max_triage_repairs`` has ``StageFunctions.repair``
    rewrite the proposal, writes the rewrite to ``repaired/<k>.md``, ``k`` counting the rewrites, and
    exits repair with the rewrite's digest. A REPAIR verdict past the budget, or a rewrite raising
    ``ProposalFormatError``, exits reject with the reason.
    """
    stages = workpiece.coordinator.stages
    proposal = workpiece.proposal_at(state)
    verdict = await stages.triage(proposal)
    spent = {"decision": verdict.decision.value, "repairs": str(state.triage_repairs)}
    if verdict.decision is TriageDecision.ACCEPT:
        attrs = {EXIT: Exit.ACCEPT, **spent, AUTHOR_STATE_ATTR: canonical_json(verdict.author_state)}
        return WorkpieceEvent(EventKind.TRIAGED, state.round, state.proposal_digest, attrs)
    reason = f"{TRIAGE_REJECTED}: {'; '.join(verdict.reasons)}"
    budget = workpiece.coordinator.budget.max_triage_repairs
    if verdict.decision is TriageDecision.REPAIR and state.triage_repairs >= budget:
        reason = f"triage returned REPAIR with {state.triage_repairs} of {budget} rewrites spent"
    elif verdict.decision is TriageDecision.REPAIR:
        try:
            rewrite = await stages.repair(proposal, verdict)
        except ProposalFormatError as error:
            reason = f"the rewrite for the REPAIR verdict is not a valid proposal: {error}"
        else:
            return _repaired(workpiece, state, rewrite)
    attrs = {EXIT: Exit.REJECT, **spent, REASON_ATTR: reason[:ATTR_CHARS]}
    return WorkpieceEvent(EventKind.TRIAGED, state.round, state.proposal_digest, attrs)


def _repaired(workpiece: _Workpiece, state: WorkpieceState, proposal: TaskProposal) -> WorkpieceEvent:
    """Write the rewritten ``proposal`` to ``repaired/<k>.md`` and return the ``TRIAGED`` event exiting repair."""
    repairs = state.triage_repairs + 1
    relative = _repaired_file(repairs)
    _write(workpiece.directory / relative, render(proposal).encode())
    attrs = {
        EXIT: Exit.REPAIR,
        "decision": TriageDecision.REPAIR.value,
        "repairs": str(repairs),
        "triaged_digest": state.proposal_digest,
        "proposal_file": str(relative),
    }
    return WorkpieceEvent(EventKind.TRIAGED, state.round, digest(proposal), attrs)


def _revision(workpiece: _Workpiece, state: WorkpieceState) -> Revision | None:
    """Read what the round's author call revises: nothing, the last build or author failure, or review's findings."""
    match state.revision:
        case RevisionKind.NONE:
            return None
        case RevisionKind.BUILD_FAILURE | RevisionKind.AUTHOR_FAILURE:
            failure = workpiece.round_dir(state.round) / FAILURES_DIR / f"{state.build_revisions}.txt"
            return Revision(kind=state.revision, detail=failure.read_text())
        case RevisionKind.MODIFY:
            path = workpiece.round_dir(state.round - 1) / DECISION_FILE
            decision = _DECISION.validate_json(path.read_bytes())
            assert isinstance(decision, Modify), f"{path} holds {decision}, not a Modify"
            return Revision(kind=state.revision, detail="\n\n".join(decision.findings))


async def _author(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Ask the author for the program and write its manifest and any source to ``authored/<k>/``.

    An author raising ``StructuredOutputError`` returned no valid reply even after its repairs, as
    when the output limit cut the reply. The adapter writes the failure to ``build-failures/`` and
    returns ``AUTHOR_FAILED``, so the next author call samples again with the failure as its
    revision. Every other exception passes through unchanged, and ``run_workpiece`` fails the workpiece in
    place at the author stage.
    """
    try:
        authored = await workpiece.coordinator.stages.author(
            workpiece.proposal_at(state), _revision(workpiece, state), state.author_state
        )
    except StructuredOutputError as error:
        attrs = _failure_attrs(workpiece, state, error, _error_line(error))
        return WorkpieceEvent(EventKind.AUTHOR_FAILED, state.round, state.proposal_digest, attrs)
    directory = workpiece.authored_dir(state)
    _write(directory / PROGRAM_MANIFEST_FILE, pretty_json(program_manifest(authored.program)).encode())
    if authored.source is not None:
        _write(directory / PROGRAM_FILE, authored.source.encode())
    attrs: dict[str, str] = {EXIT: Exit.AUTHORED, "revision": state.revision}
    return WorkpieceEvent(EventKind.AUTHORED, state.round, authored.program.digest, attrs)


async def _authored_program(workpiece: _Workpiece, state: WorkpieceState) -> RLEnvBuildProgram:
    """Load the program from ``authored/<k>/program.py``, or ask the author again when it wrote no source.

    Raises:
        ValueError: the program's digest differs from the ``AUTHORED`` event's.
    """
    source = workpiece.authored_dir(state) / PROGRAM_FILE
    if source.exists():
        program = workpiece.coordinator.stages.load_program(source.read_text())
        found = f"{source} holds program {program.digest}"
    else:
        revision = _revision(workpiece, state)
        program = (
            await workpiece.coordinator.stages.author(workpiece.proposal_at(state), revision, state.author_state)
        ).program
        found = f"the author now returns program {program.digest}"
    if program.digest != state.program_digest:
        raise ValueError(f"{workpiece.workpiece_id}: {found}, not the authored {state.program_digest}")
    return program


async def _build(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Build the authored program, spending a revision only on a failure the author can fix.

    A ``BuildFailure`` or any other exception from the program's own code becomes ``BUILD_FAILED``,
    and the next author call receives its traceback, led by the line from ``_build_failure_line``.
    These errors pass through unchanged, so ``run_workpiece`` fails the workpiece in place and spends no
    revision:

    * ``BuildInfrastructureFailure``, raised when the build host fails;
    * ``BuildExpired``, raised when the build, a step or the grader ran past its clock, which is
      neither the program's fault nor the host's until someone reads the record;
    * ``ModelUnavailable``, raised when the model stays unreachable;
    * ``OSError`` and its subclasses raised outside the program's frames (``infrastructure.storage_fault``): a
      storage fault, a missing blob or memo, or a corrupt one (``StoreCorruption``). An ``OSError``
      raised by the program's own code is a program bug and spends a revision.
    """
    coordinator = workpiece.coordinator
    program = await _authored_program(workpiece, state)
    try:
        draft = await coordinator.stages.build(
            program,
            workpiece.proposal_at(state),
            workpiece.workpiece_id,
            workpiece.round_dir(state.round),
            str(coordinator.root / STORE_DIR),
            coordinator.model,
            coordinator.sampling,
            coordinator.services,
            state.round,
            research=coordinator.research,
            model_digest=coordinator.model_digest,
        )
    except (BuildInfrastructureFailure, BuildExpired, ModelUnavailable):
        raise
    except Exception as error:
        if storage_fault(error, _program_files(program)):
            raise
        attrs = {
            **_failure_attrs(workpiece, state, error, _build_failure_line(error, program)),
            "step": (error.step or "") if isinstance(error, BuildFailure) else "",
        }
        return WorkpieceEvent(EventKind.BUILD_FAILED, state.round, program.digest, attrs)
    steps = draft.provenance.steps
    hits = sum(outcome.status is CacheStatus.HIT for outcome in steps)
    attrs = {EXIT: Exit.BUILT, "program_digest": program.digest, "steps": str(len(steps)), "hits": str(hits)}
    return WorkpieceEvent(EventKind.BUILT, state.round, digest(draft.task), attrs)


def _failure_attrs(workpiece: _Workpiece, state: WorkpieceState, error: Exception, first_line: str) -> dict[str, str]:
    """Write the workpiece's next failure to ``build-failures/<k>.txt``, and return its attrs.

    The file holds ``first_line``, a blank line, then the tail of the traceback. ``k`` is
    ``build_revisions + 1``, counting build and author failures together. The attrs exit revise
    while ``k`` is within ``max_build_revisions`` and reject past it, and keep ``first_line`` as
    ``failure``.
    """
    failures = state.build_revisions + 1
    relative = Path(ROUNDS_DIR) / str(state.round) / FAILURES_DIR / f"{failures}.txt"
    tail = "".join(traceback.format_exception(error))[-FAILURE_CHARS:]
    _write(workpiece.directory / relative, f"{first_line}\n\n{tail}".encode())
    return {
        EXIT: Exit.REVISE if failures <= workpiece.coordinator.budget.max_build_revisions else Exit.REJECT,
        "failure": first_line[:ATTR_CHARS],
        "failure_file": str(relative),
    }


def _error_line(error: BaseException) -> str:
    """Give the error's type and the first line of its message, or the type alone for an empty message."""
    message = error.message if isinstance(error, BuildFailure) else str(error)
    first = message.partition("\n")[0]
    return f"{type(error).__name__}: {first}" if first else type(error).__name__


def _build_failure_line(error: Exception, program: RLEnvBuildProgram) -> str:
    """Locate a build failure in the program and describe it in a line.

    The function finds the program's frames by the file of each step's function. For a program
    from ``load_program_source`` that file is ``<taskforge_program_...>``, the name the loader
    registers in ``linecache``. It then picks a traceback from the error's chain:

    1. the innermost exception whose traceback passes through the program, since the runner
       re-raises a step's ``BuildFailure`` from the original;
    2. otherwise the innermost exception with a traceback.

    In that traceback it takes the step, ``BuildFailure.step`` or else the outermost program frame
    running a step's function, and the deepest program frame from the step's frame on. The line
    reads, for a failure raised in the program, in library code called from it, or with no program
    frame at all::

        NameError: name 'scale' is not defined at numbers, program line 12: return scale * 2
        TypeError: ... in library code at pydantic/main.py:253 (__init__), called from numbers, program line 12: ...
        ValueError: ... outside the program, at taskforge/stage40_build_with_program/run.py:124 (run)
    """
    files = _program_files(program)
    steps = {*program.steps, program.sink.name}
    named = error.step if isinstance(error, BuildFailure) else None
    tracebacks = [traceback.extract_tb(link.__traceback__) for link in reversed(exception_chain(error))]
    passing = [frames for frames in tracebacks if any(frame.filename in files for frame in frames)]
    frames = passing[0] if passing else next((frames for frames in tracebacks if frames), [])
    headline = _error_line(error)
    if not frames:
        return headline
    last = frames[-1]
    library = f"{Path(*Path(last.filename).parts[-2:])}:{last.lineno} ({last.name})"
    inside = [index for index, frame in enumerate(frames) if frame.filename in files]
    wanted = {named} if named is not None else steps
    entries = [index for index in inside if frames[index].name in wanted]
    step = named or (frames[entries[0]].name if entries else None)
    deepest = [index for index in inside if not entries or index >= entries[0]]
    if not deepest:
        return f"{headline} outside the program{f' in step {step}' if step else ''}, at {library}"
    frame = frames[deepest[-1]]
    location = f"{f'{step}, ' if step else ''}program line {frame.lineno}: {(frame.line or '').strip()}"
    if deepest[-1] == len(frames) - 1:
        return f"{headline} at {location}"
    return f"{headline} in library code at {library}, called from {location}"


def _program_files(program: RLEnvBuildProgram) -> set[str]:
    """The files of the program's step functions, as the loader registers them in ``linecache``."""
    return {inspect.getfile(node.fn) for node in (*program.steps.values(), program.sink)}


async def _validate(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Validate the round's draft in the round's ``ValidationContext`` and write the result to ``validation.json``.

    The context carries the MODIFY rounds the workpiece has left (``_modify_rounds_left``).
    """
    coordinator = workpiece.coordinator
    round_dir = workpiece.round_dir(state.round)
    draft = load_draft(round_dir / DRAFT_DIR)
    if digest(draft.task) != state.task_digest:
        raise ValueError(
            f"{workpiece.workpiece_id}: the stored draft has digest {digest(draft.task)}, not {state.task_digest}"
        )
    context = ValidationContext(
        workpiece_id=workpiece.workpiece_id,
        round=state.round,
        round_dir=round_dir,
        proposal=workpiece.proposal_at(state),
        store_root=str(coordinator.root / STORE_DIR),
        modify_rounds_left=_modify_rounds_left(workpiece, state),
    )
    result = await coordinator.stages.validate(draft, context)
    write_validation(round_dir, result)
    attrs = {EXIT: Exit.VALIDATED, "result_digest": digest(result)}
    return WorkpieceEvent(EventKind.VALIDATED, state.round, state.task_digest, attrs)


async def _review(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    """Decide on the result read back from ``validation.json`` and write the decision to ``decision.json``.

    Review receives the round, the MODIFY rounds the workpiece has left and the coordinator's band rules.
    """
    coordinator = workpiece.coordinator
    round_dir = workpiece.round_dir(state.round)
    context = ReviewContext(state.round, _modify_rounds_left(workpiece, state), coordinator.band)
    decision = await coordinator.stages.review(load_validation(round_dir), context)
    _write(round_dir / DECISION_FILE, _DECISION.dump_json(decision, indent=2))
    return WorkpieceEvent(EventKind.DECIDED, state.round, state.task_digest, _decision_attrs(workpiece, state, decision))


def _modify_rounds_left(workpiece: _Workpiece, state: WorkpieceState) -> int:
    """``max_modify_rounds`` less the ``Modify`` decisions the workpiece has followed (``WorkpieceState.round``)."""
    return workpiece.coordinator.budget.max_modify_rounds - state.round


def _decision_attrs(workpiece: _Workpiece, state: WorkpieceState, decision: Decision) -> dict[str, str]:
    """The ``DECIDED`` attrs of ``decision``.

    A ``Retry`` past ``max_validation_retries`` exits abandon. A ``Modify`` once the workpiece's rounds
    have spent ``max_modify_rounds`` (``WorkpieceState.round`` counts the ``Modify`` decisions followed)
    exits reject with ``RejectKind.BUDGET``, its findings in the reasons.
    """
    used = {"retries_used": str(state.retries)}
    match decision:
        case Accept(band=band):
            return {EXIT: Exit.ACCEPT, "band": band, **used}
        case Reject(kind=kind, reasons=reasons):
            return {EXIT: Exit.REJECT, "kind": kind, "reasons": "; ".join(reasons), **used}
        case Retry(cause=cause, count=count):
            spent = state.retries >= workpiece.coordinator.budget.max_validation_retries
            way = Exit.ABANDON if spent else Exit.RETRY
            return {EXIT: way, "cause": cause, "count": str(count), "retries_used": str(state.retries + 1)}
        case Modify(findings=findings):
            budget = workpiece.coordinator.budget.max_modify_rounds
            if state.round >= budget:
                reason = f"review returned Modify with {state.round} of {budget} rounds spent: {'; '.join(findings)}"
                return {EXIT: Exit.REJECT, "kind": RejectKind.BUDGET, "reasons": reason, **used}
            return {EXIT: Exit.MODIFY, "findings": "; ".join(findings), **used}


async def _close(workpiece: _Workpiece, state: WorkpieceState) -> WorkpieceEvent:
    closing = state.closing
    assert closing is not None, f"{workpiece.workpiece_id} is done without a closing"
    attrs = {TERMINAL_ATTR: closing.terminal.value, REASON_ATTR: closing.reason}
    if closing.kind is not None:
        attrs["kind"] = closing.kind.value
    return WorkpieceEvent(EventKind.TERMINAL, state.round, None, attrs)


ADAPTERS: Mapping[Stage, Callable[[_Workpiece, WorkpieceState], Awaitable[WorkpieceEvent]]] = {
    Stage.TRIAGE: _triage,
    Stage.AUTHOR: _author,
    Stage.BUILD: _build,
    Stage.VALIDATE: _validate,
    Stage.REVIEW: _review,
    Stage.DONE: _close,
}
"""The adapter running each stage and returning its event."""

_DECISION: TypeAdapter[Decision] = TypeAdapter(Decision)
