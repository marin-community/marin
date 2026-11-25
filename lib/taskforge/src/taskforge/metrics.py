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


def workpiece_metrics(
    status: Status, workpiece: Workpiece, records: Sequence[Record], *, since: float
) -> WorkpieceMetrics:
    """Summarize one workpiece; events before ``since`` add no tokens, durations or latency samples."""
    if not records:
        raise ValueError(f"Workpiece {workpiece.id} has no events")
    terminal = str(status.terminal or RUNNING)
    reasons: tuple[str, ...] = ()
    if status.terminal in (Terminal.HELD, Terminal.ABANDONED):
        last = workpiece.decisions[-1] if workpiece.decisions else None
        reasons = tuple(sorted({reason_class(value) for value in getattr(last, "reasons", ())} or {"unknown"}))
    tokens: dict[Role, Usage] = defaultdict(Usage)
    stage_seconds: dict[Stage, list[float]] = defaultdict(list)
    event_gaps: dict[str, list[float]] = defaultdict(list)
    scopes: dict[str, list[float]] = defaultdict(list)
    author_rounds: set[int] = set()
    fallbacks = timeouts = 0
    accepted_at = None
    previous = records[0].wall_time
    for record in records:
        event, stamp = record.event, record.wall_time
        if isinstance(event, StageCompleted):
            if event.terminal is Terminal.ACCEPTED:
                accepted_at = stamp
            if event.stage is Stage.AUTHOR and event.terminal is not Terminal.HELD:
                author_rounds.add(event.round)
        elif isinstance(event, Note) and event.msg.startswith(AUTHOR_IMAGE_FALLBACK):
            fallbacks += 1
        elif isinstance(event, ScopeEntered):
            if event.name == "workpiece":
                scopes.clear()
            scopes[event.name].append(stamp)
        elif isinstance(event, ScopeExited) and scopes[event.name]:
            start = scopes[event.name].pop()
            if event.name in Stage and stamp >= since:
                stage_seconds[Stage(event.name)].append(max(0, stamp - start))
        elif isinstance(event, Resolved) and not event.replayed:
            request = event.request
            if isinstance(request, RunCommand) and event.value.exit_code == TIMEOUT_EXIT_CODE:
                timeouts += 1
            if stamp >= since:
                if isinstance(request, GAP_EVENTS):
                    event_gaps[type(request).__name__].append(max(0, stamp - previous))
                if isinstance(request, InferenceRequest):
                    adversary = any(v for k, v in scopes.items() if k.startswith("adversary:"))
                    role = Role.JUDGE if scopes["judge"] else Role.ADVERSARY if adversary else Role.AUTHOR
                    tokens[role] += event.value.usage
                elif isinstance(request, SolverRollout):
                    for conversation in event.value.rollout.conversations:
                        for step in conversation.steps:
                            recorded = step.turn.usage
                            usage = step.turn.metadata.get("usage", asdict(recorded) if recorded else {})
                            tokens[Role.SOLVER] += usage_from(usage)
        previous = stamp
    submission = workpiece.submission
    final = next(
        (
            result
            for result in reversed(workpiece.evidence)
            if submission is not None and result.task_digest == submission.digest
        ),
        None,
    )
    return WorkpieceMetrics(
        id=workpiece.id,
        origin=Origin.PROPOSAL if workpiece.proposal is not None else Origin.SOURCE,
        terminal=terminal,
        reasons=reasons,
        author_sessions=len(author_rounds),
        author_image_fallbacks=fallbacks,
        command_timeouts=timeouts,
        started=records[0].wall_time,
        finished=records[-1].wall_time,
        accepted_at=accepted_at if status.terminal is Terminal.ACCEPTED else None,
        initial_solve_rate=solve_rate(workpiece.evidence[0]) if workpiece.evidence else None,
        final_solve_rate=None if final is None else solve_rate(final),
        tokens=dict(tokens),
        stage_seconds={stage: tuple(values) for stage, values in stage_seconds.items()},
        event_gaps={name: tuple(values) for name, values in event_gaps.items()},
    )
