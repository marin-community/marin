# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-workpiece metrics computed from a workpiece's published snapshot and event log.

``workpiece_metrics`` is pure: callers read ``status.json``, ``workpiece.json`` and ``events.jsonl``
and pass them in. Replayed results add no tokens or latency samples, because the work they stand for
was counted when it first ran.
"""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum

from rolloutengine.models import GradedRollout

from taskforge.effects.events import Note, Resolved, ScopeEntered, ScopeExited, StageCompleted
from taskforge.effects.log import Record, Status
from taskforge.model.llm import Usage
from taskforge.model.review import ValidationResult
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal, Workpiece
from taskforge.resources.env import TIMEOUT_EXIT_CODE, RunCommand, WriteFile
from taskforge.resources.inference import InferenceRequest
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.author import AUTHOR_IMAGE_FALLBACK
from taskforge.stages.rules import solved_trials

RUNNING = "running"
"""The terminal label of a workpiece whose snapshot has no terminal yet."""
GAP_EVENTS = (RunCommand, WriteFile, InferenceRequest)
USAGE_FIELDS = tuple(field.name for field in fields(Usage))


class Origin(StrEnum):
    """Where a workpiece's task came from: a proposal, or a task another source wrote, for repair."""

    PROPOSAL = "proposal"
    SOURCE = "source"


class Role(StrEnum):
    AUTHOR = "author"
    SOLVER = "solver"
    ADVERSARY = "adversary"
    JUDGE = "judge"


@dataclass(frozen=True, slots=True)
class WorkpieceMetrics:
    """One workpiece's outcome, spend and timing.

    Attributes:
        author_sessions: Rounds in which an author session completed. A source task accepted as its
            source wrote it has none.
        author_image_fallbacks: Author sessions that ran in ``author.image`` because the task's own
            environment failed to set up.
        command_timeouts: Author and adversary shell commands that hit their timeout. This and the
            counts above cover the workpiece's whole log, whatever the cutoff.
        started, finished: Wall time of the first and last logged event; ``finished - started`` includes
            time spent waiting on providers and between resumes.
        accepted_at: Wall time of the decision that accepted the workpiece.
        initial_solve_rate: Solver pass rate of the first validated submission; for a source task, the
            task as its source wrote it.
        final_solve_rate: Solver pass rate measured for the submission the workpiece ends with.
        tokens: Token usage by role, from events at or after the caller's cutoff.
        stage_seconds: Durations of stage scopes that ended at or after the cutoff.
        event_gaps: Seconds since the previous event, keyed by request type, for latency proxies.
    """

    id: str
    origin: Origin
    terminal: str
    reasons: tuple[str, ...]
    author_sessions: int
    author_image_fallbacks: int
    command_timeouts: int
    started: float
    finished: float
    accepted_at: float | None
    initial_solve_rate: float | None
    final_solve_rate: float | None
    tokens: Mapping[Role, Usage]
    stage_seconds: Mapping[Stage, tuple[float, ...]]
    event_gaps: Mapping[str, tuple[float, ...]]


def reason_class(reason: str) -> str:
    if reason.startswith("bug:"):
        pattern = r"(?m)^\s*(?:[|+]\s*)*(?:\w+\.)*([A-Z]\w*)(?=:|$)"
        types = re.findall(pattern, reason.removeprefix("bug:").strip())
        types = [name for name in types if name not in {"Traceback", "ExceptionGroup", "BaseExceptionGroup"}]
        return "bug:" + (types[0] if types else "unknown")
    return reason.split(":", 1)[0].split(" <- ", 1)[0].strip()


def solve_rate(result: ValidationResult) -> float | None:
    return solved_trials(result) / len(result.solver) if result.solver else None


def usage_from(raw: Mapping[str, int]) -> Usage:
    """A ``Usage`` from a JSON usage mapping, counting fields it lacks as zero."""
    return Usage(*(raw.get(name, 0) for name in USAGE_FIELDS))


def failure_reasons(status: Status, workpiece: Workpiece) -> tuple[str, ...]:
    """Reason classes of the decision that held or abandoned the workpiece."""
    if status.terminal not in (Terminal.HELD, Terminal.ABANDONED):
        return ()
    last = workpiece.decisions[-1] if workpiece.decisions else None
    return tuple(sorted({reason_class(value) for value in getattr(last, "reasons", ())} or {"unknown"}))


def final_solve_rate(workpiece: Workpiece) -> float | None:
    """The solve rate of the latest validation of the submission the workpiece ends with."""
    submission = workpiece.submission
    if submission is None:
        return None
    final = next((r for r in reversed(workpiece.evidence) if r.task_digest == submission.digest), None)
    return None if final is None else solve_rate(final)


def solver_usage(value: GradedRollout) -> Usage:
    usage = Usage()
    for conversation in value.rollout.conversations:
        for step in conversation.steps:
            recorded = step.turn.usage
            usage += usage_from(step.turn.metadata.get("usage", asdict(recorded) if recorded else {}))
    return usage


@dataclass(frozen=True, slots=True)
class Spend:
    tokens: Mapping[Role, Usage]
    stage_seconds: Mapping[Stage, tuple[float, ...]]
    event_gaps: Mapping[str, tuple[float, ...]]


def spend(records: Sequence[Record], since: float) -> Spend:
    """Tokens by role, stage durations and event gaps from events at or after ``since``.

    Open scopes attribute inference to a role; a ``workpiece`` scope starts a new process, so it
    discards scopes an interrupted process left open.
    """
    tokens: dict[Role, Usage] = defaultdict(Usage)
    stage_seconds: dict[Stage, list[float]] = defaultdict(list)
    event_gaps: dict[str, list[float]] = defaultdict(list)
    scopes: dict[str, list[float]] = defaultdict(list)
    previous = records[0].wall_time
    for record in records:
        event, stamp = record.event, record.wall_time
        if isinstance(event, ScopeEntered):
            if event.name == "workpiece":
                scopes.clear()
            scopes[event.name].append(stamp)
        elif isinstance(event, ScopeExited) and scopes[event.name]:
            start = scopes[event.name].pop()
            if event.name in Stage and stamp >= since:
                stage_seconds[Stage(event.name)].append(max(0, stamp - start))
        elif isinstance(event, Resolved) and not event.replayed and stamp >= since:
            request = event.request
            if isinstance(request, GAP_EVENTS):
                event_gaps[type(request).__name__].append(max(0, stamp - previous))
            if isinstance(request, InferenceRequest):
                adversary = any(v for k, v in scopes.items() if k.startswith("adversary:"))
                role = Role.JUDGE if scopes["judge"] else Role.ADVERSARY if adversary else Role.AUTHOR
                tokens[role] += event.value.usage
            elif isinstance(request, SolverRollout):
                tokens[Role.SOLVER] += solver_usage(event.value)
        previous = stamp
    return Spend(
        dict(tokens),
        {stage: tuple(values) for stage, values in stage_seconds.items()},
        {name: tuple(values) for name, values in event_gaps.items()},
    )


def workpiece_metrics(
    status: Status, workpiece: Workpiece, records: Sequence[Record], *, since: float
) -> WorkpieceMetrics:
    """Summarize one workpiece; events before ``since`` add no tokens, durations or latency samples."""
    if not records:
        raise ValueError(f"Workpiece {workpiece.id} has no events")
    events = [(record.event, record.wall_time) for record in records]
    completed = [(event, stamp) for event, stamp in events if isinstance(event, StageCompleted)]
    accepted = [stamp for event, stamp in completed if event.terminal is Terminal.ACCEPTED]
    used = spend(records, since)
    return WorkpieceMetrics(
        id=workpiece.id,
        origin=Origin.PROPOSAL if workpiece.proposal is not None else Origin.SOURCE,
        terminal=str(status.terminal or RUNNING),
        reasons=failure_reasons(status, workpiece),
        author_sessions=len(
            {e.round for e, _ in completed if e.stage is Stage.AUTHOR and e.terminal is not Terminal.HELD}
        ),
        author_image_fallbacks=sum(
            isinstance(event, Note) and event.msg.startswith(AUTHOR_IMAGE_FALLBACK) for event, _ in events
        ),
        command_timeouts=sum(
            isinstance(event, Resolved)
            and not event.replayed
            and isinstance(event.request, RunCommand)
            and event.value.exit_code == TIMEOUT_EXIT_CODE
            for event, _ in events
        ),
        started=records[0].wall_time,
        finished=records[-1].wall_time,
        accepted_at=accepted[-1] if accepted and status.terminal is Terminal.ACCEPTED else None,
        initial_solve_rate=solve_rate(workpiece.evidence[0]) if workpiece.evidence else None,
        final_solve_rate=final_solve_rate(workpiece),
        tokens=used.tokens,
        stage_seconds=used.stage_seconds,
        event_gaps=used.event_gaps,
    )
