# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The agentic author: agent sessions that load, build and grade a program before they hand it in.

A session works in ``rounds/<n>/agentic/<k>/workspace/`` under the workpiece's directory, which starts with
the standard template as ``program.py``, the proposal, the SDK reference and the template
(``reference/``, read-only) and an empty ``notes.md``. Its tools (``AuthorSession.tools``) read, write
and edit workspace files, run the loader, build the program with the caller's ``ProgramBuilder`` (the
caller's own build stage: same step store, model, policy, services and research requirement, so a
program that built in the session rebuilds there from memo hits alone), grade a candidate against the
last draft the session built, and, when the proposal lists research, search and fetch the web. Every
load of a program runs through the session's ``ProgramLoader``, by default in a child process bounded
by ``LOAD_TIMEOUT`` (``load_program_in_process``), so a module body that never returns cannot hold the
caller's event loop. The program is handed in with ``submit(path)``: the loader runs on the file; a
program that never built in
the session is refused once, with the loader's result and the reason, and accepted when submitted
again; an accepted submission ends the loop through ``run_agent``'s ``stop`` callback.

The model sees a build's failure (the step's message, or for a crash the program's frame that raised it and
the error, then the traceback's tail) and a grade's diagnostics, never a host failure or an expired clock. A
``BuildInfrastructureFailure`` in ``run_build`` or ``run_grade_candidate`` is run again up to
``HOST_RETRIES`` times with a doubling backoff from ``HOST_RETRY_BACKOFF`` inside the tool, at once
re-raised for a cause in ``HOST_REJECTIONS``, and then re-raised like ``ModelUnavailable``: the author call
ends without a record, so it spends no revision and a relaunch hands the workspace to a fresh session. A
``BuildExpired`` in ``run_build`` (a wall, step or grader clock) is re-raised at once, without a retry, and
ends the call the same way, so the driver holds the workpiece as abandoned instead of asking for a rewrite.
An ``OSError`` raised outside the program's frames (``infrastructure.storage_fault``: the step store failed,
or a blob or memo is missing or corrupt) is re-raised at once too, as the coordinator's build stage does,
since no program change repairs the store. A failed build still writes its ``run.json``,
``status: "infrastructure"`` for the host's, ``status: "expired"`` with its ``clock`` for an expiry and
``status: "storage"`` for a storage fault. A loader child the host killed (``LoaderProcessFailed``) raises the same
failure with cause ``BUILD_PROCESS``. A proposal whose solver machine the session's services cannot provide
(``infrastructure.machine_available``: a container on a host without an image builder, a package machine
without the local backend) raises before the session starts, so the author never adapts the task to the
host.

``AgenticAuthor`` is the coordinator's ``Author``: each call runs sessions in the context's round until
one submits, and returns the submitted file's bytes as the program's source, which the coordinator
writes to ``authored/<k>/``. Each session keeps its records in its own ``agentic/<k>/``: ``author.json``
(status, revision, mode, stop reason, wall), ``submit.json``, ``record.json`` for a submission,
``transcript.json`` (every turn: role, tool calls, tool results cut to ``TRANSCRIPT_CHARS``, tokens,
finish) and ``session.json`` (turns, tool calls by tool, tokens, cached share, wall, stop reason,
submissions, in-session builds). ``k`` counts the round's sessions, across the author calls of the
round.

Which session a call runs, from the last session of the round, or of the round before when this round has
none:

* no session yet: a fresh session;
* a last session that ended without a submission, or was interrupted: a fresh session seeded from its
  workspace (``HANDOFF``), with the revision, if any, after the handoff note;
* a last session that submitted, with a build failure to revise: the same session (its saved
  ``conversation.json`` plus the failure as the next user turn, ``turns - max_turns_used`` more turns)
  when it ended by submit, its last prompt held at most ``max_prompt_tokens`` and it used at most
  ``max_turns_used`` turns; otherwise, and for any other revision, a fresh session seeded from its
  workspace, with the revision text as the first user turn after the opening.

Within one call, a session that ends without a submission hands its workspace to one fresh session
(``HANDED_OFF``); if that one ends unsubmitted too, the call raises ``AuthorFailure``, which spends a
revision. A session stopped by the context guard hands off until ``MAX_SESSIONS_PER_CALL`` ran. A
relaunch never resumes a session mid-way: it starts a handoff session from the workspace.

Author turns record under the ledger step ``author`` with ``author_kind``, ``session`` and ``segment``
attributes; every ledger entry of an in-session grade carries ``origin=author_session``.
"""

import asyncio
import inspect
import json
import shutil
import time
import traceback
from collections import Counter
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import JsonValue
from rigging.filesystem.atomic import write_bytes_atomic
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskResource, TextMessage

from taskforge.ledger.records import EntryKind, Ledger, LedgerEntry
from taskforge.llm.agent import AgentRun, AgentStop, AgentTool, AgentTurn, ReasoningReplay, ToolOutcome, run_agent
from taskforge.llm.client import ModelClient, ModelUnavailable, Usage
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger
from taskforge.spec import draft as draft_module
from taskforge.stage10_proposal.model import TaskProposal, render
from taskforge.stage30_author.author import (
    PROGRAM_FILE,
    ROUNDS_DIR,
    AuthorContext,
    AuthoredProgram,
    AuthorFailure,
    Revision,
    RevisionKind,
)
from taskforge.stage30_author.load_process import LoaderProcessFailed, ProgramManifest, load_program_in_process
from taskforge.stage30_author.loader import GRADER_STEP, program_file
from taskforge.stage40_build_with_program.context import (
    BuildExpired,
    BuildFailure,
    BuildInfrastructureFailure,
)
from taskforge.stage40_build_with_program.failure_text import failure_head, failure_text
from taskforge.stage40_build_with_program.glm_author import (
    AUTHOR_STEP,
    RECORD_FILE,
    REVISION_PROMPT,
    author_messages,
    sdk_reference,
)
from taskforge.stage40_build_with_program.infrastructure import (
    HOST_REJECTIONS,
    InfrastructureCause,
    machine_available,
    storage_fault,
)
from taskforge.stage40_build_with_program.outputs import GradedCandidate, Grader
from taskforge.stage40_build_with_program.program_sdk import load_program, sdk_exports
from taskforge.stage40_build_with_program.run import TaskDraft, step_output
from taskforge.stage40_build_with_program.sandbox_effects import grade_candidate_on_host
from taskforge.stage40_build_with_program.sdk import grade_diagnostics, grading_language
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage40_build_with_program.step_store import SDK_VERSION
from taskforge.stage40_build_with_program.template import standard

AUTHOR_KIND = "agentic"
"""The ``author_kind`` this author writes into ``author.json``, ``record.json`` and its ledger entries."""
AGENTIC_DIR = "agentic"
AUTHOR_RECORD = "author.json"

WORKSPACE_DIR = "workspace"
RUNS_DIR = "runs"
REFERENCE_DIR = "reference"
PROPOSAL_PATH = "proposal.md"
NOTES_PATH = "notes.md"
SDK_PATH = f"{REFERENCE_DIR}/sdk.md"
TEMPLATE_PATH = f"{REFERENCE_DIR}/template.py"
CONVERSATION_FILE = "conversation.json"
TRANSCRIPT_FILE = "transcript.json"
SESSION_FILE = "session.json"
SUBMIT_FILE = "submit.json"
RUN_FILE = "run.json"

TOOL_OUTPUT_CHARS = 6000
TOOL_OUTPUT_HEAD = 3500
TOOL_OUTPUT_TAIL = 2000
TRANSCRIPT_CHARS = 4000
WRITE_CHARS = 24000
NOTES_CHARS = 4000
"""The longest ``submit`` notes; a bound of 2,000 refused 32 of 77 first submit calls in an early run."""
READ_LINES = 400
WEB_CALLS = 8
MAX_NUDGES = 2
"""Nudges after replies without a tool call that were not cut; a cut reply gets ``LENGTH_NUDGE`` instead."""
MAX_SESSIONS_PER_CALL = 4
"""Sessions one author call may start; a session stopped by the context guard hands off until this many ran."""
REPLAY = ReasoningReplay.KEEP
LOAD_TIMEOUT = 120.0
"""Seconds the default ``ProgramLoader`` gives a program's module body before it refuses the load."""
HOST_RETRIES = 2
"""Times ``run_build`` and ``run_grade_candidate`` run a build or grade again after the host failed it."""
HOST_RETRY_BACKOFF = 5.0
"""Seconds before the first such retry; each later retry waits twice as long as the one before."""
SESSION_ORIGIN = "author_session"
GRADE_INSTRUCTION = "(instruction)"
"""The instruction the template's grader step grades against."""


LENGTH_NUDGE = (
    "Your reply was cut off at the {limit}-token output limit before it made a tool call, so nothing ran. Take a "
    "smaller step: reason briefly, then call one tool (read one part of a file, or edit one step), and continue "
    "from its result."
)
NUDGE = (
    "The session ends only with submit: a reply without a tool call hands nothing in. Continue with the tools, "
    f"and call submit with the path of the program ({PROGRAM_FILE}) when it is ready."
)
AGENTIC_SECTION = f"""

# This session

You work in a workspace through tools; you do not reply with the program. The workspace holds:
- `{PROGRAM_FILE}`: a copy of the template module. Adapt it to the proposal; it is what you submit.
- `{PROPOSAL_PATH}`: the proposal. `{SDK_PATH}` and `{TEMPLATE_PATH}`: the SDK reference and the template \
(read-only; `read_reference` reads them too).
- `{NOTES_PATH}`: your notes. If this session runs out of turns, a fresh session starts from this workspace \
and reads them.

The loop: adapt `{PROGRAM_FILE}` with `edit_file`, check it with `run_loader`, run it with `run_build`, \
read the failure, fix the step it names, and run it again. The step store reruns only the steps you changed \
and the steps after them, so a rerun costs the edited steps alone. Once it builds, grade a wrong answer or \
two against the built task with `run_grade_candidate`, then `submit` the program. `{SDK_PATH}` lists every \
name, keyword argument and field the program may use: read it instead of probing, since a step that raises \
to print module names, fields or schemas spends a run and shows less than the reference. Never rewrite the whole \
module in one call; edit the step you are fixing (`write_file` takes at most {WRITE_CHARS} characters). The \
session ends only with `submit`: a reply without a tool call hands nothing in. You have {{turns}} turns.
You write the solver instruction yourself as a string in the `instructions` step: it follows the \
instruction rule above (format as requirements, never grader behaviour), and `run_build` reports one that \
names the grading on its `instruction:` line; rewrite it before you submit.
"""
SEED_NOTE = """\
This is a fresh session continuing earlier work. The workspace holds `{program}` as last edited, \
`{notes}` and the files the earlier session left. {submitted}Read them before you change anything.
"""
HANDOFF_PROMPT = "The earlier session ended without a submission ({reason})."
NO_REVISION_PROMPT = "Continue from the earlier session's workspace: finish the program and submit it."
EDIT_AND_SUBMIT = f"Edit `{PROGRAM_FILE}` in the workspace, run it with `run_build`, and submit it."


@dataclass(frozen=True)
class SameSessionBounds:
    """When a downstream build failure re-enters the session that submitted the program.

    A session that ended by submit, whose last prompt held at most ``max_prompt_tokens`` and which used
    at most ``max_turns_used`` turns, receives the failure as its next user turn and continues for
    ``turns - max_turns_used`` more turns; any other revision starts a fresh session from its workspace.
    """

    max_prompt_tokens: int
    max_turns_used: int


@dataclass(frozen=True)
class AgenticAuthorConfig:
    """The budgets of one agentic author session.

    Attributes:
        turns: Model replies one session may take.
        turn_max_tokens: The output budget of every turn.
        session_token_cap: The context guard: the session stops before a turn whose prompt (the last
            turn's prompt and reply) plus ``turn_max_tokens`` would exceed it.
        revision_same_session: When a revision re-enters the submitting session.
        max_continuations: ``LLMPolicy.max_continuations`` of every turn. With 0 a turn cut at
            ``turn_max_tokens`` ends as LENGTH and the session nudges it; with more, one turn can grow to
            (1 + ``max_continuations``) x ``turn_max_tokens`` of replayed reasoning.
    """

    turns: int
    turn_max_tokens: int
    session_token_cap: int
    revision_same_session: SameSessionBounds
    max_continuations: int = 0

    def __post_init__(self) -> None:
        if self.turns < 1:
            raise ValueError(f"turns must be at least 1, got {self.turns}")
        if self.session_token_cap <= self.turn_max_tokens:
            raise ValueError(
                f"session_token_cap ({self.session_token_cap}) must exceed turn_max_tokens ({self.turn_max_tokens})"
            )
        if not 0 <= self.revision_same_session.max_turns_used < self.turns:
            raise ValueError(
                f"revision_same_session.max_turns_used ({self.revision_same_session.max_turns_used}) must lie in "
                f"[0, turns={self.turns})"
            )


class ProgramBuilder(Protocol):
    async def __call__(self, source: str, run_dir: Path, *, invalidate: frozenset[str]) -> TaskDraft:
        """Build ``source`` for the session's workpiece and round, as the caller's own build stage would (same step
        store, model, policy, services and research requirement), leaving ``draft/`` and the build files under
        ``run_dir``.

        Raises what the build raises; the session turns ``BuildFailure`` and other exceptions into text for the
        model, re-raises ``ModelUnavailable``, ``BuildExpired`` and an ``OSError`` raised outside the program's
        frames, and retries a ``BuildInfrastructureFailure`` before re-raising it.
        """
        ...


class ProgramLoader(Protocol):
    async def __call__(self, source: str) -> ProgramManifest:
        """The manifest of ``source``; raises ``ValueError`` with the loader's refusal, and
        ``BuildInfrastructureFailure`` when the host failed the load."""
        ...


async def load_in_child(source: str) -> ProgramManifest:
    """The default ``ProgramLoader``: ``load_program_in_process`` under ``LOAD_TIMEOUT``.

    Raises:
        ValueError: the loader refused ``source``, or its module body ran past ``LOAD_TIMEOUT``.
        BuildInfrastructureFailure: the loader child exited without a result (``BUILD_PROCESS``).
    """
    try:
        return await load_program_in_process(source, sdk_exports(), wall_timeout=LOAD_TIMEOUT)
    except LoaderProcessFailed as error:
        raise BuildInfrastructureFailure(InfrastructureCause.BUILD_PROCESS, str(error)) from error


async def host_retried[T](run: Callable[[], Awaitable[T]]) -> T:
    """``run()``, run again up to ``HOST_RETRIES`` times while the host fails it.

    The first retry waits ``HOST_RETRY_BACKOFF`` seconds and each later one twice as long as the one before.

    Raises:
        BuildInfrastructureFailure: the host failed every run, or failed one with a cause in ``HOST_REJECTIONS``,
            which no retry changes.
    """
    for retry in range(HOST_RETRIES):
        try:
            return await run()
        except BuildInfrastructureFailure as failure:
            if failure.cause in HOST_REJECTIONS:
                raise
        await asyncio.sleep(HOST_RETRY_BACKOFF * 2**retry)
    return await run()


@dataclass(frozen=True)
class SessionBuild:
    """How a session loads, builds and grades its program.

    Attributes:
        builder: Builds a program as the caller's build stage does.
        store_root: The step store the builder uses; ``step_output`` reads the grader from it.
        services: The build's services: ``grade_candidate``'s factories, host, ledger and web tools, and the
            image builder the in-session builds publish with (``None`` when they cannot). ``AgenticAuthor``
            checks the proposal's solver machine against them before the session starts.
        session_ledger: The ledger file in-session builds append to (``<root>/<workpiece_id>.jsonl``); a build's step
            statuses are the ``STEP`` spans appended to it during the build.
        loader: Loads a program for ``run_loader``, ``run_build`` and ``submit``; ``load_in_child`` unless the
            caller bounds the load itself.
    """

    builder: ProgramBuilder
    store_root: str
    services: RLEnvBuildServices
    session_ledger: Path
    loader: ProgramLoader = load_in_child


class AuthorStatus(StrEnum):
    ACCEPTED = "accepted"
    HANDED_OFF = "handed_off"
    """The session ended without a submission and hands its workspace to the call's next session."""
    AUTHOR_FAILED = "author_failed"


class SessionBuilds(Protocol):
    def __call__(self, context: AuthorContext, proposal: TaskProposal, /) -> SessionBuild:
        """How the sessions of ``context``'s workpiece and round load, build and grade ``proposal``'s program."""
        ...


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=1, default=str) + "\n")


def _write_text(path: Path, text: str) -> None:
    write_bytes_atomic(str(path), text.encode())


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _numbered(directory: Path) -> list[int]:
    """The integer-named subdirectories of ``directory``, ascending."""
    if not directory.is_dir():
        return []
    return sorted(int(p.name) for p in directory.iterdir() if p.is_dir() and p.name.isdigit())


def _round_dir(workpiece_dir: Path, round_no: int) -> Path:
    return workpiece_dir / ROUNDS_DIR / str(round_no)


def _traceback_tail(error: BaseException) -> str:
    return "".join(traceback.format_exception(error))[-TOOL_OUTPUT_CHARS:]


def _failed_build(error: Exception, path: str, source: str, steps: Collection[str]) -> tuple[str, str | None, str]:
    """A failed in-session build the host did not fail and no clock ended, as (status, step, text) for ``run.json``
    and the model.

    A ``BuildFailure`` is its step's message. Any other error is a crash, whose text leads with the frame of
    the program at ``path`` that raised it, then the error (``failure_text.failure_head``), above the tail of
    its traceback.
    """
    if isinstance(error, BuildFailure):
        return "build_failure", error.step, error.message
    head = failure_head(error, {program_file(source)}, steps, shown_as=path)
    return "crash", None, failure_text(head, error, TOOL_OUTPUT_CHARS)


class SessionStop(StrEnum):
    SUBMITTED = "submitted"
    MAX_TURNS = "max_turns"
    CONTEXT = "context"
    """The context guard (``session_token_cap``) or the client's ``ContextExhausted`` ended the session."""
    UNSUBMITTED = "unsubmitted"
    """The model kept replying without a tool call after ``MAX_NUDGES`` nudges (cut replies not counted)."""


class SubmitOutcome(StrEnum):
    """How one ``submit`` call ended; every call is recorded in ``submit.json`` and ``session.json``."""

    ACCEPTED = "accepted"
    REFUSED_UNTESTED = "refused_untested"
    """The program loads but never built in the session; the same digest submitted again is accepted."""
    REFUSED_NOTES = "refused_notes"
    """The notes ran past ``NOTES_CHARS``."""
    REFUSED_LOADER = "refused_loader"
    """The loader refused the file; ``error`` holds its message."""
    REFUSED_PATH = "refused_path"
    """The path names no workspace file, or lies outside the workspace."""


class SessionMode(StrEnum):
    FRESH = "fresh"
    REVISION = "revision"
    """A fresh session for a revision, seeded from the previous round's workspace."""
    SAME_SESSION = "same_session"
    HANDOFF = "handoff"
    """A fresh session in the same round, seeded from a session that ended unsubmitted or was interrupted."""


@dataclass(frozen=True)
class TaggedLedger:
    """Adds ``attrs`` to every entry it passes to ``inner``."""

    inner: Ledger
    attrs: Mapping[str, str]

    def record(self, entry: LedgerEntry) -> None:
        self.inner.record(replace(entry, attrs={**entry.attrs, **self.attrs}))


def ledger_steps(path: Path, offset: int) -> list[dict]:
    """The build-step ``STEP`` spans appended to the ledger file ``path`` after byte ``offset``.

    A workpiece has one session at a time, so the entries after ``offset`` are the run just made.
    """
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(offset)
        entries = [LedgerEntry.from_json(json.loads(line)) for line in f if line.strip()]
    # Spans arrive in completion order (independent steps run at once since #10081); a step cancelled because
    # another failed first has cause CancelledError and no cache attribute.
    return [
        {
            "step": e.step,
            "cache": e.attrs.get("cache", "cancelled" if e.cause == "CancelledError" else "failed"),
            "wall": round(e.wall_time, 3),
        }
        for e in entries
        if e.kind is EntryKind.STEP and "tool" not in e.attrs
    ]


class ToolRefusal(Exception):
    """A tool call the session refuses; its message goes back to the model as ``error: ...``."""


def candidate_file(raw: str, content: str) -> TaskResource:
    """The ``TaskResource`` for a candidate's file at ``raw``, a path relative to the machine root.

    Raises:
        ToolRefusal: ``raw`` is absolute, escapes the root or fails ``TaskResource`` validation, with the
            form the task takes, so the author can fix the call.
    """
    relative = PurePosixPath(raw)
    if not raw.strip() or relative.is_absolute() or ".." in relative.parts:
        raise ToolRefusal(
            f"grading refused: {raw!r} is not a path inside the task machine; file paths are relative to the "
            "machine root, for example workspace/report.md"
        )
    try:
        return draft_module.file(raw, content)
    except ValueError as error:
        raise ToolRefusal(f"grading refused: {raw!r} is not a path the task can hold: {error}") from error


@dataclass(frozen=True)
class Submission:
    path: str
    source: bytes
    manifest: ProgramManifest
    notes: str
    tested: bool


def clip(text: str, limit: int = TOOL_OUTPUT_CHARS) -> str:
    """``text`` within ``limit`` characters: the head and the tail around a line saying how much was cut."""
    if len(text) <= limit:
        return text
    dropped = len(text) - TOOL_OUTPUT_HEAD - TOOL_OUTPUT_TAIL
    return f"{text[:TOOL_OUTPUT_HEAD]}\n... [{dropped} characters dropped] ...\n{text[-TOOL_OUTPUT_TAIL:]}"


def _schema(properties: Mapping[str, object], required: Sequence[str]) -> dict[str, object]:
    """A tool's JSON schema. Length limits stay out of it: ``run_agent`` returns a schema violation unclipped,
    and jsonschema's ``maxLength`` message quotes the whole value (a 27k-character file in an early run), so the
    handlers check lengths and refuse with a short message instead."""
    return {"type": "object", "properties": dict(properties), "required": list(required), "additionalProperties": False}


PATH_ARGUMENT = {"type": "string", "description": "A path relative to the workspace, for example program.py."}


def _instruction_words(draft: TaskDraft) -> int:
    return sum(len(e.content.split()) for e in draft.task.context.events if isinstance(e, TextMessage))


def _grading_report(draft: TaskDraft) -> str:
    text = "\n".join(e.content for e in draft.task.context.events if isinstance(e, TextMessage))
    match = grading_language(text)
    return "names no grading" if match is None else f"refers to grading ({match!r}): restate it as a format requirement"


class AuthorSession:
    """The workspace, tools and state of one agentic author session."""

    def __init__(
        self,
        session_dir: Path,
        workpiece_id: str,
        round_no: int,
        proposal: TaskProposal,
        build: SessionBuild,
        session_id: str,
    ):
        self.session_dir = session_dir
        self.workspace = session_dir / WORKSPACE_DIR
        self.runs = session_dir / RUNS_DIR
        self.workpiece_id = workpiece_id
        self.round_no = round_no
        self.proposal = proposal
        self.build = build
        self.ledger = TaggedLedger(build.services.ledger, {"origin": SESSION_ORIGIN, "session": session_id})
        self.services = replace(build.services, ledger=self.ledger)
        self.built: set[str] = set()
        self.refused_untested: set[str] = set()
        self.last_draft: TaskDraft | None = None
        self.submissions: list[dict] = []
        self.submission: Submission | None = None
        self.builds: list[dict] = []
        self.web_calls = 0

    # --- workspace ---------------------------------------------------------------------------

    def prepare(self, seed: Path | None) -> None:
        """Lay out the workspace: a copy of ``seed`` (an earlier session's workspace), or the template."""
        if seed is not None:
            shutil.copytree(seed, self.workspace)
        else:
            _write_text(self.workspace / PROGRAM_FILE, _template_source())
            _write_text(self.workspace / NOTES_PATH, "")
        _write_text(self.workspace / PROPOSAL_PATH, render(self.proposal))
        _write_text(self.workspace / SDK_PATH, sdk_reference())
        _write_text(self.workspace / TEMPLATE_PATH, _template_source())

    def path(self, raw: str, *, write: bool = False) -> Path:
        relative = PurePosixPath(raw)
        if not raw.strip() or relative.is_absolute() or ".." in relative.parts:
            raise ToolRefusal(f"{raw!r} is not a path inside the workspace; give a relative path without '..'")
        if write and relative.parts[0] == REFERENCE_DIR:
            raise ToolRefusal(f"{REFERENCE_DIR}/ is read-only")
        return self.workspace / relative

    def read(self, raw: str) -> str:
        path = self.path(raw)
        if not path.is_file():
            raise ToolRefusal(f"no file {raw!r}; list_files shows the workspace")
        return path.read_text()

    # --- tools -------------------------------------------------------------------------------

    async def read_file(self, arguments: Mapping[str, object]) -> str:
        return _numbered_lines(self.read(str(arguments["path"])), arguments)

    async def read_reference(self, arguments: Mapping[str, object]) -> str:
        path = {"sdk": SDK_PATH, "template": TEMPLATE_PATH, "proposal": PROPOSAL_PATH}[str(arguments["name"])]
        return _numbered_lines(self.read(path), arguments)

    async def write_file(self, arguments: Mapping[str, object]) -> str:
        content = str(arguments["content"])
        if len(content) > WRITE_CHARS:
            raise ToolRefusal(f"content holds {len(content)} characters, more than {WRITE_CHARS}; use edit_file")
        path = self.path(str(arguments["path"]), write=True)
        _write_text(path, content)
        return f"wrote {len(content.encode())} bytes to {arguments['path']}"

    async def edit_file(self, arguments: Mapping[str, object]) -> str:
        raw, old, new = str(arguments["path"]), str(arguments["old_string"]), str(arguments["new_string"])
        text = self.read(raw)
        count = text.count(old) if old else 0
        if count != 1:
            raise ToolRefusal(f"old text {'not found' if count == 0 else f'has {count} matches'} in {raw}")
        _write_text(self.path(raw, write=True), text.replace(old, new))
        return f"replaced 1 occurrence in {raw}"

    async def list_files(self, arguments: Mapping[str, object]) -> str:
        files = sorted(p for p in self.workspace.rglob("*") if p.is_file())
        return "\n".join(f"{p.relative_to(self.workspace)} ({p.stat().st_size} bytes)" for p in files)

    async def run_loader(self, arguments: Mapping[str, object]) -> str:
        try:
            program = await self.build.loader(self.read(str(arguments["path"])))
        except ValueError as error:
            return f"refused by the loader: {error}"
        return f"loads: program {program.digest[:16]}; run order {', '.join(program.order)}"

    async def run_build(self, arguments: Mapping[str, object]) -> str:
        raw = str(arguments["path"])
        source = self.read(raw)
        try:
            program = await self.build.loader(source)
        except ValueError as error:
            return f"refused by the loader: {error}"
        invalidate = frozenset(str(name) for name in _list_argument(arguments, "invalidate"))
        unknown = sorted(invalidate - set(program.steps))
        if unknown:
            raise ToolRefusal(f"invalidate names {unknown}, which are not steps of the program")
        j = len(_numbered(self.runs))
        run_dir = self.runs / str(j)
        ledger = self.build.session_ledger
        offset = ledger.stat().st_size if ledger.exists() else 0
        started = time.monotonic()
        record: dict = {"path": raw, "program_digest": program.digest, "invalidate": sorted(invalidate)}

        async def built() -> tuple[TaskDraft, GradeResult]:
            draft = await self.build.builder(source, run_dir, invalidate=invalidate)
            return draft, await self._check_empty_answer(draft)

        try:
            draft, empty = await host_retried(built)
        except ModelUnavailable:
            raise
        except BuildInfrastructureFailure as error:
            record |= {"status": "infrastructure", "step": None, "wall": time.monotonic() - started}
            record |= {"steps": ledger_steps(ledger, offset), "text": str(error)}
            _write_json(run_dir / RUN_FILE, record)
            self.builds.append(record)
            raise
        except BuildExpired as error:
            record |= {"status": "expired", "clock": error.clock, "step": error.step, "wall": time.monotonic() - started}
            record |= {"steps": ledger_steps(ledger, offset), "text": error.message}
            _write_json(run_dir / RUN_FILE, record)
            self.builds.append(record)
            raise
        except Exception as error:
            if storage_fault(error, {program_file(source)}):
                record |= {"status": "storage", "step": None, "wall": time.monotonic() - started}
                record |= {"steps": ledger_steps(ledger, offset), "text": _traceback_tail(error)}
                _write_json(run_dir / RUN_FILE, record)
                self.builds.append(record)
                raise
            status, step, text = _failed_build(error, raw, source, program.steps)
            record |= {"status": status, "step": step}
            record |= {"wall": time.monotonic() - started, "steps": ledger_steps(ledger, offset), "text": text}
            _write_json(run_dir / RUN_FILE, record)
            self.builds.append(record)
            where = f" in step {step}" if step else ""
            return f"build failed ({status}{where})\nsteps: {_steps_line(record['steps'])}\n\n{text}"
        self.built.add(program.digest)
        self.last_draft = draft
        record |= {"status": "built", "wall": time.monotonic() - started, "steps": ledger_steps(ledger, offset)}
        _write_json(run_dir / RUN_FILE, record)
        self.builds.append(record)
        return (
            f"built: program {program.digest[:16]}\nsteps: {_steps_line(record['steps'])}\n{self._digest(draft, empty)}"
        )

    async def _check_empty_answer(self, draft: TaskDraft) -> GradeResult:
        """Grade the empty answer against the built draft's grader, as validation's negative controls will.

        The empty answer comes back as ``SUBMISSION_FAILURE`` when the submission is refused before
        any grader runs, or ``GRADED`` with whatever an empty candidate earns. Either rejects it,
        unless the reward is above the grader's declared ``rejection_ceiling``: that grader credits
        nothing, and the build fails in the grader step with the grade for the model to fix.

        Raises:
            BuildFailure: the empty answer is graded above the ceiling.
        """
        grader = step_output(draft, self.build.store_root, GRADER_STEP, Grader)
        empty = await self._grade(draft, grader, "", ())
        if empty.status == Outcome.GRADED and (empty.reward or 0.0) > grader.rejection_ceiling:
            raise BuildFailure(
                f"the empty answer is graded with reward {empty.reward}, above the grader's declared "
                f"rejection_ceiling {grader.rejection_ceiling}; a submission with nothing in it must earn at most "
                f"the ceiling\n{grade_diagnostics(empty)}",
                GRADER_STEP,
            )
        return empty

    async def _grade(
        self, draft: TaskDraft, grader: Grader, reply: str, workspace: Sequence[TaskResource]
    ) -> GradeResult:
        """Grade ``reply`` with the ``workspace`` files it leaves against ``draft``'s task, no model called."""
        task = draft.task
        return await grade_candidate_on_host(
            self.services,
            self.workpiece_id,
            task.environment_requirements,
            grader.package,
            task.answer_type,
            task.answer_format,
            GRADE_INSTRUCTION,
            reply,
            files=task.resources.worker,
            workspace=workspace,
            output_paths=task.output_paths,
        )

    async def run_grade_candidate(self, arguments: Mapping[str, object]) -> str:
        """Grade a candidate reply and its files against the last draft; a file path the task cannot hold is refused."""
        draft = self.last_draft
        if draft is None:
            raise ToolRefusal("no draft built in this session yet; run_build the program first")
        grader = step_output(draft, self.build.store_root, GRADER_STEP, Grader)
        files = [candidate_file(str(f["path"]), str(f["content"])) for f in _list_argument(arguments, "files")]
        try:
            grade = await host_retried(lambda: self._grade(draft, grader, str(arguments["reply"]), files))
        except (ModelUnavailable, BuildInfrastructureFailure):
            raise
        except Exception as error:
            return f"grading failed: {_traceback_tail(error)}"
        return f"status {grade.status}, reward {grade.reward}\n{grade_diagnostics(grade)}"

    async def submit(self, arguments: Mapping[str, object]) -> str:
        raw, notes = str(arguments["path"]), str(arguments.get("notes", ""))
        entry: dict = {"path": raw, "notes_chars": len(notes), "loaded": None, "digest": None, "error": None}
        try:
            source = self.read(raw)
            program = await self.build.loader(source)
        except ToolRefusal as refusal:
            self.submissions.append(entry | {"outcome": SubmitOutcome.REFUSED_PATH, "error": str(refusal)})
            raise
        except ValueError as error:
            self.submissions.append(
                entry | {"outcome": SubmitOutcome.REFUSED_LOADER, "loaded": False, "error": str(error)}
            )
            return f"refused by the loader, nothing handed in: {error}"
        entry |= {"loaded": True, "digest": program.digest}
        loaded = f"the loader accepts it (program {program.digest[:16]}; run order {', '.join(program.order)})"
        if len(notes) > NOTES_CHARS:
            self.submissions.append(entry | {"outcome": SubmitOutcome.REFUSED_NOTES})
            raise ToolRefusal(
                f"notes hold {len(notes)} characters, more than {NOTES_CHARS}; nothing handed in ({loaded}). "
                "Shorten the notes and submit again."
            )
        tested = program.digest in self.built
        if not tested and program.digest not in self.refused_untested:
            self.refused_untested.add(program.digest)
            self.submissions.append(entry | {"outcome": SubmitOutcome.REFUSED_UNTESTED})
            return (
                f"not handed in: {loaded}, but this program never built in this session. Run it with run_build "
                "first, or call submit again with the same file to hand it in untested."
            )
        self.submission = Submission(raw, self.path(raw).read_bytes(), program, notes, tested)
        self.submissions.append(entry | {"outcome": SubmitOutcome.ACCEPTED})
        return f"accepted: program {program.digest}{'' if tested else ' (untested)'}. The session ends here."

    def tools(self, web_tools: Sequence[AgentTool]) -> list[AgentTool]:
        """The session's tools; ``web_tools`` (search and fetch) are offered when the proposal lists research."""
        read_window = {
            "start_line": {"type": "integer", "minimum": 1, "description": "First line to show, 1-based (default 1)."},
            "max_lines": {"type": "integer", "minimum": 1, "description": f"Lines to show (default {READ_LINES})."},
        }
        tools = [
            _tool(
                "read_file",
                "Read a workspace file as numbered lines. path: the file; start_line and max_lines: the window.",
                _schema({"path": PATH_ARGUMENT, **read_window}, ["path"]),
                self.read_file,
            ),
            _tool(
                "read_reference",
                "Read a reference document as numbered lines. name: 'sdk' (the build SDK reference), 'template' "
                "(the standard program) or 'proposal'; start_line and max_lines: the window.",
                _schema({"name": {"type": "string", "enum": ["sdk", "template", "proposal"]}, **read_window}, ["name"]),
                self.read_reference,
            ),
            _tool(
                "write_file",
                f"Create or replace a workspace file. path: the file (reference/ is read-only); content: the whole "
                f"file, at most {WRITE_CHARS} characters. Prefer edit_file for changes to program.py.",
                _schema({"path": PATH_ARGUMENT, "content": {"type": "string"}}, ["path", "content"]),
                self.write_file,
            ),
            _tool(
                "edit_file",
                "Replace the one exact occurrence of old_string with new_string in a workspace file. path: the file; "
                "old_string: text that occurs exactly once in it; new_string: its replacement.",
                _schema(
                    {"path": PATH_ARGUMENT, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
                    ["path", "old_string", "new_string"],
                ),
                self.edit_file,
            ),
            _tool(
                "list_files",
                "List the workspace files with their sizes. Takes no arguments.",
                _schema({}, []),
                self.list_files,
            ),
            _tool(
                "run_loader",
                "Load a program file with the build's loader, which checks the module rules, the names it reads and "
                "the step graph. path: the program file. Returns the run order, or the loader's refusal.",
                _schema({"path": PATH_ARGUMENT}, ["path"]),
                self.run_loader,
            ),
            _tool(
                "run_build",
                "Build the task with a program file, exactly as the build stage will: the real model, machines and step "
                "store. Steps whose source and inputs are unchanged load their stored output. path: the program "
                "file; invalidate: optional step names to rerun even when stored. Returns each step's cache status "
                "and wall, then the failure (the step's message, or for a crash the program file, line and source line "
                "that raised it, the error and the traceback's tail) or a digest "
                "of the built task. After a build the library grades the empty answer itself: a reward above the "
                "grader's declared rejection_ceiling fails the build in the grader step.",
                _schema({"path": PATH_ARGUMENT, "invalidate": {"type": "array", "items": {"type": "string"}}}, ["path"]),
                self.run_build,
            ),
            _tool(
                "run_grade_candidate",
                "Grade a candidate answer against the task the last successful run_build built, as validation "
                "would. reply: the candidate's final reply; files: optional files the candidate leaves in the task "
                "machine, each {path, content} with the path relative to the machine root (workspace/report.md, "
                "not /workspace/report.md). Returns status, reward and the grader's diagnostics.",
                _schema(
                    {
                        "reply": {"type": "string"},
                        "files": {
                            "type": "array",
                            "items": _schema(
                                {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]
                            ),
                        },
                    },
                    ["reply"],
                ),
                self.run_grade_candidate,
            ),
            _tool(
                "submit",
                "Hand in the program and end the session. path: the program file (normally program.py); notes: "
                f"at most {NOTES_CHARS} characters on what it builds and any deviation from the proposal. A program "
                "that never built in this session is refused once; submitting it again hands it in untested.",
                _schema({"path": PATH_ARGUMENT, "notes": {"type": "string"}}, ["path", "notes"]),
                self.submit,
            ),
        ]
        if self.proposal.header.research:
            tools += [self._capped(tool) for tool in web_tools]
        return tools

    # --- helpers ------------------------------------------------------------------------------

    def _capped(self, tool: AgentTool) -> AgentTool:
        async def handler(arguments: Mapping[str, object]) -> str:
            if self.web_calls >= WEB_CALLS:
                return f"error: this session's {WEB_CALLS} web calls are spent"
            self.web_calls += 1
            return clip(await tool.handler(arguments))

        return replace(tool, handler=handler)

    def _digest(self, draft: TaskDraft, empty: GradeResult) -> str:
        """What the model reads after a build: the grader, its reference grade, the library's empty-answer grade."""
        grader = step_output(draft, self.build.store_root, GRADER_STEP, Grader)
        reference = next((c for c in grader.prototyped if c.reply == grader.reference_reply), None)
        task = draft.task
        return "\n".join(
            [
                f"grader: {type(grader.package.grader).__name__}",
                f"reference: {_prototyped(reference)}",
                f"empty answer: {empty.status} reward {empty.reward} (graded by the library; ceiling "
                f"{grader.rejection_ceiling})",
                f"instruction: {_instruction_words(draft)} words; {_grading_report(draft)}",
                f"worker files: {[r.path for r in task.resources.worker]}",
            ]
        )


def _prototyped(candidate: GradedCandidate | None) -> str:
    return "not prototyped" if candidate is None else f"{candidate.status} reward {candidate.reward}"


def _template_source() -> str:
    return inspect.getsource(standard)


def _list_argument(arguments: Mapping[str, object], key: str) -> list:
    """An optional array argument; the tool's JSON schema has already checked its type."""
    value = arguments.get(key, [])
    assert isinstance(value, list), f"{key} is not an array"
    return value


def _int_argument(arguments: Mapping[str, object], key: str, default: int) -> int:
    value = arguments.get(key, default)
    assert isinstance(value, int), f"{key} is not an integer"
    return value


def _numbered_lines(text: str, arguments: Mapping[str, object]) -> str:
    lines = text.splitlines()
    start = _int_argument(arguments, "start_line", 1)
    count = _int_argument(arguments, "max_lines", READ_LINES)
    window = lines[start - 1 : start - 1 + count]
    end = start + len(window) - 1
    body = "\n".join(f"{start + i:5d}| {line}" for i, line in enumerate(window))
    return f"lines {start}-{end} of {len(lines)}\n{body}"


def _steps_line(steps: Sequence[Mapping[str, object]]) -> str:
    return ", ".join(f"{s['step']} {str(s['cache']).upper()} {s['wall']}s" for s in steps) or "(none ran)"


def _tool(
    name: str,
    description: str,
    parameters: Mapping[str, object],
    handler: Callable[[Mapping[str, object]], Awaitable[str]],
) -> AgentTool:
    """An ``AgentTool`` whose result is clipped to ``TOOL_OUTPUT_CHARS`` and whose refusals read ``error: ...``."""

    async def run(arguments: Mapping[str, object]) -> str:
        try:
            return clip(await handler(arguments))
        except ToolRefusal as refusal:
            return f"error: {refusal}"

    return AgentTool(name=name, description=description, parameters=parameters, handler=run)


# --- the session loop ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionPlan:
    """How a session starts: its mode, the workspace it is seeded from, its opening conversation and turns."""

    mode: SessionMode
    seed: Path | None
    messages: tuple[Message, ...]
    opening: int
    """How many of ``messages`` the transcript shows as this session's own opening (a same-session revision
    shows only the revision turn)."""
    turns: int
    turns_before: int
    """Turns the continued session had already used (same-session revisions)."""
    continued_from: str | None


@dataclass(frozen=True)
class SessionResult:
    stop: SessionStop
    submission: Submission | None
    turns: tuple[AgentTurn, ...]
    messages: tuple[Message, ...]
    transcript: tuple[dict, ...]
    nudges: int
    length_nudges: int
    wall: float


async def drive(
    session: AuthorSession,
    plan: SessionPlan,
    config: AgenticAuthorConfig,
    client: ModelClient,
    policy: LLMPolicy,
    ledger: Ledger,
    session_id: str,
) -> SessionResult:
    """Run the session's tool loop until submit, the context guard, the turn budget, or unanswered nudges."""
    started = time.monotonic()
    # Continuations off by default: a turn cut at turn_max_tokens comes back as LENGTH and is nudged, so one
    # turn cannot grow to (1 + max_continuations) x turn_max_tokens of replayed reasoning.
    turn_policy = replace(policy, max_tokens=config.turn_max_tokens, max_continuations=config.max_continuations)
    tools = session.tools(session.build.services.web_tools)
    transcript: list[dict] = [_user_entry(m) for m in plan.messages[len(plan.messages) - plan.opening :]]
    conversation: list[Message] = list(plan.messages)
    turns: list[AgentTurn] = []
    guard: list[SessionStop] = []

    def over_cap(turn: AgentTurn) -> bool:
        # A continued turn's usage sums its segments' prompts, so with continuations on this overstates the
        # conversation and the guard trips early.
        usage = turn.completion.usage
        return usage.prompt_tokens + usage.completion_tokens + config.turn_max_tokens > config.session_token_cap

    def stop(turn: AgentTurn) -> bool:
        if session.submission is not None:
            return True
        if over_cap(turn):
            guard.append(SessionStop.CONTEXT)
            return True
        return False

    nudges = 0
    length_nudges = 0
    segment = 0
    while True:
        record = CallLedger(
            TaggedLedger(ledger, {"author_kind": AUTHOR_KIND, "session": session_id, "segment": str(segment)}),
            session.workpiece_id,
            session.round_no,
            AUTHOR_STEP,
        )
        run: AgentRun = await run_agent(
            client, turn_policy, conversation, tools, plan.turns - len(turns), record, REPLAY, stop
        )
        transcript += [_turn_entry(len(turns) + i, t) for i, t in enumerate(run.turns)]
        turns += run.turns
        conversation = list(run.messages)
        segment += 1
        if run.stop is AgentStop.STOPPED:
            outcome = SessionStop.SUBMITTED if session.submission is not None else guard[-1]
            break
        if run.stop is AgentStop.CONTEXT:
            outcome = SessionStop.CONTEXT
            break
        if run.stop is AgentStop.MAX_TURNS or len(turns) >= plan.turns:
            outcome = SessionStop.MAX_TURNS
            break
        # ANSWERED or LENGTH: a reply without a tool call.
        if over_cap(run.turns[-1]):
            outcome = SessionStop.CONTEXT
            break
        if run.stop is AgentStop.LENGTH:
            length_nudges += 1
            nudge: Message = {"role": "user", "content": LENGTH_NUDGE.format(limit=config.turn_max_tokens)}
            conversation.append(nudge)
            transcript.append(_user_entry(nudge))
            continue
        if nudges == MAX_NUDGES:
            outcome = SessionStop.UNSUBMITTED
            break
        nudges += 1
        nudge = {"role": "user", "content": NUDGE}
        conversation.append(nudge)
        transcript.append(_user_entry(nudge))
    return SessionResult(
        outcome,
        session.submission,
        tuple(turns),
        tuple(conversation),
        tuple(transcript),
        nudges,
        length_nudges,
        time.monotonic() - started,
    )


def _user_entry(message: Message) -> dict:
    return {"role": message["role"], "content": str(message.get("content", ""))[:TRANSCRIPT_CHARS]}


def _usage(usage: Usage) -> dict[str, int]:
    return {
        "in": usage.prompt_tokens,
        "out": usage.completion_tokens,
        "reasoning": usage.reasoning_tokens,
        "cached": usage.cached_tokens,
    }


def _turn_entry(index: int, turn: AgentTurn) -> dict:
    completion = turn.completion
    return {
        "turn": index,
        "role": "assistant",
        "content": completion.content[:TRANSCRIPT_CHARS],
        "reasoning_chars": len(completion.reasoning),
        "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments[:TRANSCRIPT_CHARS]} for c in completion.tool_calls
        ],
        "tool_results": [
            {
                "call_id": r.call.id,
                "name": r.call.name,
                "outcome": str(r.outcome),
                "output": r.output[:TRANSCRIPT_CHARS],
                "wall": round(r.wall_time, 3),
            }
            for r in turn.tool_results
        ],
        "tokens": _usage(completion.usage),
        "continuations": completion.continuations,
        "finish": str(completion.finish_reason),
        "wall": round(completion.wall_time, 3),
    }


def session_record(session_id: str, plan: SessionPlan, session: AuthorSession, result: SessionResult) -> dict:
    """``session.json``: the session's turns, tool mix, tokens, cached share, wall, stop reason and builds."""
    usages = [t.completion.usage for t in result.turns]
    total = sum(usages[1:], usages[0]) if usages else Usage(0, 0, 0, 0)
    prompts = [u.prompt_tokens for u in usages]
    first = session.submissions[0] if session.submissions else None
    refusals = Counter(str(s["outcome"]) for s in session.submissions if s["outcome"] != SubmitOutcome.ACCEPTED)
    return {
        "session": session_id,
        "mode": plan.mode,
        "seeded_from": None if plan.seed is None else str(plan.seed),
        "continued_from": plan.continued_from,
        "turns": len(result.turns),
        "turns_total": plan.turns_before + len(result.turns),
        "tool_calls": dict(Counter(c.name for t in result.turns for c in t.completion.tool_calls)),
        "tokens": _usage(total),
        "cached_share": None if total.prompt_tokens == 0 else total.cached_tokens / total.prompt_tokens,
        "last_prompt_tokens": prompts[-1] if prompts else 0,
        "max_prompt_tokens": max(prompts, default=0),
        "max_turn_output": max((u.completion_tokens for u in usages), default=0),
        "turn_outputs": [u.completion_tokens for u in usages],
        "rejected_tool_calls": sum(r.outcome is not ToolOutcome.EXECUTED for t in result.turns for r in t.tool_results),
        "length_turns": sum(t.completion.finish_reason == "length" for t in result.turns),
        "wall": result.wall,
        "tool_wall": sum(r.wall_time for t in result.turns for r in t.tool_results),
        "stop_reason": result.stop,
        "nudges": result.nudges,
        "length_nudges": result.length_nudges,
        "continuations": sum(t.completion.continuations for t in result.turns),
        "submissions": session.submissions,
        "first_submission_loaded": None if first is None else first["loaded"],
        "first_submit_accepted": None if first is None else first["outcome"] == SubmitOutcome.ACCEPTED,
        "submit_refusals": dict(refusals),
        "tested": None if result.submission is None else result.submission.tested,
        "builds": len(session.builds),
        "builds_ok": sum(b["status"] == "built" for b in session.builds),
        "web_calls": session.web_calls,
    }


# --- the author stage ------------------------------------------------------------------------------


def opening_messages(proposal: TaskProposal, turns: int) -> list[Message]:
    """The one-shot author's rules, proposal, SDK reference and template, with the agentic section added."""
    system, user = author_messages(proposal)
    return [{**system, "content": f"{system['content']}{AGENTIC_SECTION.format(turns=turns)}"}, user]


def _revision_text(revision: Revision) -> str:
    return f"{REVISION_PROMPT.format(kind=revision.kind, detail=revision.detail)}\n{EDIT_AND_SUBMIT}"


def _submitted(session_dir: Path) -> bool:
    """Whether the session in ``session_dir`` handed in a program."""
    record = session_dir / AUTHOR_RECORD
    return record.exists() and _read_json(record)["status"] == AuthorStatus.ACCEPTED


def _seed_message(previous: Path, lead: str) -> Message:
    submitted = previous / SUBMIT_FILE
    notes = _read_json(submitted).get("notes", "") if submitted.exists() else ""
    note = SEED_NOTE.format(
        program=PROGRAM_FILE,
        notes=NOTES_PATH,
        submitted=f"Its submission notes: {notes}\n" if notes else "",
    )
    return {"role": "user", "content": f"{note}\n{lead}"}


def plan_session(
    workpiece_dir: Path, round_no: int, proposal: TaskProposal, revision: Revision | None, config: AgenticAuthorConfig
) -> SessionPlan:
    """Which session round ``round_no`` runs next (see the module docstring)."""
    opening = opening_messages(proposal, config.turns)
    sessions = _round_dir(workpiece_dir, round_no) / AGENTIC_DIR
    if _numbered(sessions):
        previous = sessions / str(_numbered(sessions)[-1])
        if not _submitted(previous):
            record = previous / AUTHOR_RECORD
            reason = _read_json(record).get("stop_reason", "unsubmitted") if record.exists() else "interrupted"
            lead = HANDOFF_PROMPT.format(reason=reason)
            if revision is not None:
                lead = f"{lead}\n\n{_revision_text(revision)}"
            seed = _seed_message(previous, lead)
            return SessionPlan(SessionMode.HANDOFF, previous / WORKSPACE_DIR, (*opening, seed), 3, config.turns, 0, None)
    elif round_no == 0:
        return SessionPlan(SessionMode.FRESH, None, tuple(opening), 2, config.turns, 0, None)
    else:
        earlier = _round_dir(workpiece_dir, round_no - 1) / AGENTIC_DIR
        previous = earlier / str(_numbered(earlier)[-1])
    prior = _read_json(previous / SESSION_FILE)
    bounds = config.revision_same_session
    if (
        revision is not None
        and revision.kind is RevisionKind.BUILD_FAILURE
        and prior["stop_reason"] == SessionStop.SUBMITTED
        and prior["last_prompt_tokens"] <= bounds.max_prompt_tokens
        and prior["turns_total"] <= bounds.max_turns_used
    ):
        conversation = json.loads((previous / CONVERSATION_FILE).read_text())
        message: Message = {"role": "user", "content": _revision_text(revision)}
        return SessionPlan(
            SessionMode.SAME_SESSION,
            previous / WORKSPACE_DIR,
            (*conversation, message),
            1,
            config.turns - bounds.max_turns_used,
            prior["turns_total"],
            str(previous.relative_to(workpiece_dir)),
        )
    lead = NO_REVISION_PROMPT if revision is None else _revision_text(revision)
    seed = _seed_message(previous, lead)
    return SessionPlan(SessionMode.REVISION, previous / WORKSPACE_DIR, (*opening, seed), 3, config.turns, 0, None)


@dataclass(frozen=True)
class SessionOutcome:
    """How one session of an author call ended: its ``author.json`` status, the submitted source, the stop."""

    status: AuthorStatus
    source: str | None
    stop: SessionStop


@dataclass(frozen=True)
class AgenticAuthor:
    """The agentic author, an ``Author``: sessions in the workpiece's directory until one submits a program.

    Attributes:
        config: The budgets of each session.
        client: The model the sessions drive.
        policy: The sampling policy of every turn; ``max_tokens`` and ``max_continuations`` come from ``config``.
        ledger: Where the sessions' turns and tool calls are recorded.
        builds: How the sessions of a call build and grade, for the call's workpiece, round and proposal.
    """

    config: AgenticAuthorConfig
    client: ModelClient
    policy: LLMPolicy
    ledger: Ledger
    builds: SessionBuilds

    async def __call__(
        self,
        context: AuthorContext,
        proposal: TaskProposal,
        revision: Revision | None,
        author_state: Mapping[str, JsonValue],
    ) -> AuthoredProgram:
        """Run sessions for ``context``'s round until one submits, and return the submitted program.

        The sessions work in ``rounds/<n>/agentic/`` under ``context.workpiece_dir``. ``author_state``
        does not reach them.

        Raises:
            AuthorFailure: the call's sessions all ended without a submission.
            ModelUnavailable: the model or an in-session build or grade ran out of retries.
            BuildInfrastructureFailure: the host cannot provide the proposal's solver machine
                (``machine_available``), before a session starts or writes a record; or the host failed an
                in-session load, or a build or grade past its retries.
            BuildExpired: an in-session build ran past a wall, step or grader clock.
        """
        build = self.builds(context, proposal)
        machine_available(build.services, proposal)
        index = 0
        while True:
            outcome = await self._session(build, context, proposal, revision, index)
            if outcome.status is AuthorStatus.ACCEPTED:
                assert outcome.source is not None
                return AuthoredProgram(load_program(outcome.source), outcome.source)
            if outcome.status is AuthorStatus.AUTHOR_FAILED:
                raise AuthorFailure(
                    f"{index + 1} agentic sessions of round {context.round} ended without a submission; "
                    f"the last stopped at {outcome.stop}"
                )
            index += 1

    async def _session(
        self,
        build: SessionBuild,
        context: AuthorContext,
        proposal: TaskProposal,
        revision: Revision | None,
        index: int,
    ) -> SessionOutcome:
        """The call's session ``index``: writes ``agentic/<k>/``, ``k`` counting the round's sessions."""
        plan = plan_session(context.workpiece_dir, context.round, proposal, revision, self.config)
        sessions = _round_dir(context.workpiece_dir, context.round) / AGENTIC_DIR
        k = len(_numbered(sessions))
        session_dir = sessions / str(k)
        session_id = f"{context.round}.{k}"
        session = AuthorSession(session_dir, context.workpiece_id, context.round, proposal, build, session_id)
        session.prepare(plan.seed)
        result = await drive(session, plan, self.config, self.client, self.policy, self.ledger, session_id)
        _write_json(session_dir / CONVERSATION_FILE, list(result.messages))
        _write_json(session_dir / TRANSCRIPT_FILE, list(result.transcript))
        _write_json(session_dir / SESSION_FILE, session_record(session_id, plan, session, result))
        record: dict = {
            "author_kind": AUTHOR_KIND,
            "revision": None if revision is None else {"kind": str(revision.kind), "detail": revision.detail},
            "mode": plan.mode,
            "stop_reason": result.stop,
            "wall": result.wall,
        }
        submission = result.submission
        calls = {"calls": session.submissions}
        if submission is None:
            _write_json(session_dir / SUBMIT_FILE, {"accepted": False, **calls})
            handoff = index == 0 or result.stop is SessionStop.CONTEXT
            status = (
                AuthorStatus.HANDED_OFF if handoff and index + 1 < MAX_SESSIONS_PER_CALL else AuthorStatus.AUTHOR_FAILED
            )
            error = f"no submission ({result.stop})"
            _write_json(session_dir / AUTHOR_RECORD, record | {"status": status, "error": error})
            return SessionOutcome(status, None, result.stop)
        _write_json(
            session_dir / SUBMIT_FILE,
            {
                "accepted": True,
                "path": submission.path,
                "digest": submission.manifest.digest,
                "notes": submission.notes,
                "tested": submission.tested,
                "submitted_by": session_id,
                **calls,
            },
        )
        _write_json(
            session_dir / RECORD_FILE,
            {
                "author_kind": AUTHOR_KIND,
                "program_digest": submission.manifest.digest,
                "sdk_version": SDK_VERSION,
                "model": self.client.model,
                "revision": None if revision is None else str(revision.kind),
                "notes": submission.notes,
                "turns": len(result.turns),
            },
        )
        accepted = {"status": AuthorStatus.ACCEPTED, "program_digest": submission.manifest.digest}
        _write_json(session_dir / AUTHOR_RECORD, record | accepted | {"tested": submission.tested})
        return SessionOutcome(AuthorStatus.ACCEPTED, submission.source.decode(), result.stop)
