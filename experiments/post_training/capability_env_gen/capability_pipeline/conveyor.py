"""Item conveyor for the synthesis controller.

Every item a synthesis job owns either moves forward or ends in a visible
terminal state inside that job; nothing waits for an operator relaunch.

* ``STATE_TABLE`` classifies every state ``synthesize_one`` can return as
  terminal, waiting-retryable, or repairable.  A repairable state resolves per
  result: active (re-enter now, the repair loop still has budget and the
  failure is repair-eligible) or terminal (budget exhausted, or an operational
  hold that GLM repair must not touch).  An unclassified state fails closed.
* ``ConveyorScheduler`` keeps at most ``concurrency`` items doing work.  An
  item that returns a waiting state is re-queued with a not-before time and
  holds no slot while it waits.  Each wait kind has a bounded budget; on
  exhaustion the item becomes terminal ``failed``.  Relaunch downtime and time
  queued behind a live publisher are not charged to a wait's window.
* A hold (``wait_hold``: a step's ``retryable: False``, or a transient
  controller exception on a waiting/fresh item) is terminal for this job only:
  the state is kept and a relaunch re-enters the item.
* ``write_status`` stamps every ``status.json`` write (``updated_at``,
  ``state_since``, bounded ``transitions``; ``wait`` while waiting).
* ``ConveyorBoard`` mirrors every item into the run-level ``conveyor.json``;
  ``mark_activity`` records the long step an item has just started.

This module deliberately imports nothing heavy so tests and other modules can
use the stamping helpers without the synthesis controller.
"""

from __future__ import annotations

import heapq
import inspect
import itertools
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .inference import atomic_json, canonical

CONVEYOR_SCHEMA = "capability-conveyor-v1"

TERMINAL = "terminal"
WAITING = "waiting_retryable"
REPAIRABLE = "repairable"
ACTIVE = "active"
UNCLASSIFIED = "unclassified"

# Every state synthesize_one / _synthesize_attempt can return.  The second
# field is the failure stage (terminal and repairable) or the wait kind
# (waiting).  Keep this table exhaustive: a state missing here fails closed.
STATE_TABLE: dict[str, tuple[str, str]] = {
    # Terminal outcomes.
    "quality_accepted": (TERMINAL, "accepted"),
    # A repair decided the admitted design needs readmission: a new proposal
    # (new hash, new item) must come from admission; synthesis cannot move it.
    "pending_readmission": (TERMINAL, "readmission"),
    # Bounded adversary output exhaustion.  Re-running it is an explicit,
    # fresh full adversary-suite revalidation (--retry-adversary), by policy.
    "pending_adversary_retry": (TERMINAL, "adversary_retry"),
    # Job-configuration holds: the environment of a running job does not
    # change, so re-entering inside the job cannot help.  A relaunch with a
    # fixed configuration re-enters them (they are not in the repair set).
    "pending_judge_policy": (TERMINAL, "configuration"),
    "pending_schema_validation": (TERMINAL, "schema_validation"),
    "pending_runtime": (TERMINAL, "configuration"),
    "controls_passed_pending_rollout": (TERMINAL, "configuration"),
    # Repairable: resolved per result from repair eligibility and budget.
    "failed": (REPAIRABLE, "construction"),
    "pending_build_acceptance": (REPAIRABLE, "build_acceptance"),
    "pending_judge_calibration": (REPAIRABLE, "judge_calibration"),
    "pending_solver_adjudication": (REPAIRABLE, "solver_adjudication"),
    "pending_attack_adjudication": (REPAIRABLE, "attack_adjudication"),
    "runtime_controls_passed_pending_adversary": (REPAIRABLE, "adversary"),
    "pending_quality_review": (REPAIRABLE, "quality_review"),
    "pending_repeated_diagnostics": (REPAIRABLE, "repeated_diagnostics"),
    # Waiting-retryable.  pending_build is refined by its session outcomes.
    "pending_build": (WAITING, "builder"),
    "pending_image_capture": (WAITING, "image_capture"),
    "pending_image_publication": (WAITING, "image_publication"),
    "pending_image_cold_pull": (WAITING, "image_cold_pull"),
    "pending_image_migration": (WAITING, "image_migration"),
    "pending_image_review": (WAITING, "image_review"),
    "pending_image_infrastructure": (WAITING, "image_infrastructure"),
    # An ungraded provider/transport/toolchain failure at a runtime gate
    # (runtime controls, judge calibration, repeated diagnostics).  Synthesis
    # re-runs the gate after an automatic health probe, with a durable
    # per-item revalidation cap; see synthesis._infrastructure_hold.
    "pending_runtime_infrastructure": (WAITING, "runtime_infrastructure"),
}

REPAIRABLE_STATES = frozenset(
    state for state, (klass, _) in STATE_TABLE.items() if klass == REPAIRABLE
)

# kind: (window seconds from first_seen, max attempts, backoff seconds).
# Attempts count every call that returned this wait kind, the first included.
DEFAULT_WAIT_BUDGETS: dict[str, tuple[float, int, float]] = {
    # The capture controller owns a per-role attempt budget (image_capture_control);
    # this window is the backstop for multi-role items and its own backoff.
    "image_capture": (14_400, 6, 900),
    # Publication is a separate service; polling is local and cheap.
    "image_publication": (21_600, 200, 120),
    "image_cold_pull": (7_200, 6, 600),
    "image_migration": (7_200, 6, 600),
    "image_review": (3_600, 3, 300),
    # The image-plan reviewer never completed because omp/GLM was unreachable
    # (non-zero exit or timeout): a relay/router outage, not a review verdict,
    # so it is budgeted like builder_process instead of the review's own bound.
    "image_review_transport": (21_600, 12, 900),
    "image_infrastructure": (7_200, 6, 900),
    # omp exited non-zero (relay/router outage, crash): retry after a pause.
    "builder_process": (21_600, 12, 900),
    # A session stopped without a handoff but may continue from its transcript.
    "builder_continuation": (86_400, 3, 60),
    # synthesize_one raised something other than SynthesisError.
    "controller_exception": (7_200, 3, 300),
    # Runtime-gate infrastructure holds.  Each call probes provider health
    # (cheap) and re-runs the gate only when healthy; synthesis separately caps
    # gate re-runs per item across relaunches (CAPABILITY_RUNTIME_INFRA_MAX_RETRIES).
    # The window, not the attempt count, is the real bound during an outage.
    "runtime_infrastructure": (43_200, 16, 900),
}

# An image_publication wait whose poll reports ``awaiting_publisher`` with a
# fresh publisher heartbeat is queued behind a live publisher, not stuck: that
# time is not charged to the window (attempts still are), up to this total.
PUBLISHER_BACKLOG_CAP_SECONDS = 172_800
# Mirrors publication_exchange.HEARTBEAT_STALE_SECONDS (not imported: this
# module stays dependency-free).
PUBLISHER_FRESH_SECONDS = 300

TRANSITION_LIMIT = 100
REASON_LIMIT = 240

_FAILED_STAGE_PREFIXES = (
    ("invalid task bundle:", "task_bundle"),
    ("TaskCompendium validation/lowering failed:", "lowering"),
    ("runtime controls failed:", "runtime_controls"),
    ("runtime infrastructure retries exhausted", "runtime_infrastructure"),
    ("duplicate generated TaskSpec id:", "duplicate_task_id"),
    ("unclassified_state:", "unclassified_state"),
    ("controller_exception:", "controller_exception"),
    ("conveyor_error:", "conveyor"),
    ("conveyor_call_cap_exhausted:", "conveyor"),
)
_WAIT_FAILURE_PREFIXES = ("wait_budget_exhausted:",)


class ConveyorConfigError(ValueError):
    pass


def _log(event: dict[str, Any], *, stderr: bool = False) -> None:
    print(canonical(event), flush=True)
    if stderr:
        print(canonical(event), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class WaitBudget:
    kind: str
    window_seconds: float
    max_attempts: int
    backoff_seconds: float


def _env_number(environ: Mapping[str, str], name: str, default: float, *, integer: bool = False) -> float:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as error:
        raise ConveyorConfigError(f"{name} must be a {'whole' if integer else ''} number: {raw!r}") from error
    if not math.isfinite(value) or value < 0:
        raise ConveyorConfigError(f"{name} must be finite and non-negative: {raw!r}")
    return value


@dataclass(frozen=True)
class ConveyorConfig:
    budgets: dict[str, WaitBudget]
    max_calls_per_item: int = 500
    max_active_reentries: int = 6
    heartbeat_seconds: float = 60.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ConveyorConfig:
        environ = os.environ if environ is None else environ
        budgets = {}
        for kind, (window, attempts, backoff) in DEFAULT_WAIT_BUDGETS.items():
            prefix = f"CAPABILITY_WAIT_{kind.upper()}_"
            max_attempts = int(_env_number(environ, prefix + "ATTEMPTS", attempts, integer=True))
            if max_attempts < 1:
                raise ConveyorConfigError(f"{prefix}ATTEMPTS must be at least 1")
            budgets[kind] = WaitBudget(
                kind,
                _env_number(environ, prefix + "SECONDS", window),
                max_attempts,
                _env_number(environ, prefix + "BACKOFF_SECONDS", backoff),
            )
        max_calls = int(_env_number(environ, "CAPABILITY_CONVEYOR_MAX_CALLS_PER_ITEM", 500, integer=True))
        reentries = int(_env_number(environ, "CAPABILITY_CONVEYOR_MAX_ACTIVE_REENTRIES", 6, integer=True))
        heartbeat = _env_number(environ, "CAPABILITY_CONVEYOR_HEARTBEAT_SECONDS", 60.0)
        if max_calls < 1:
            raise ConveyorConfigError("CAPABILITY_CONVEYOR_MAX_CALLS_PER_ITEM must be at least 1")
        if heartbeat <= 0:
            raise ConveyorConfigError("CAPABILITY_CONVEYOR_HEARTBEAT_SECONDS must be positive")
        return cls(budgets, max_calls, reentries, heartbeat)

    def describe(self) -> dict[str, Any]:
        return {
            "wait_budgets": {
                kind: {
                    "window_seconds": budget.window_seconds,
                    "max_attempts": budget.max_attempts,
                    "backoff_seconds": budget.backoff_seconds,
                }
                for kind, budget in self.budgets.items()
            },
            "max_calls_per_item": self.max_calls_per_item,
            "max_active_reentries": self.max_active_reentries,
            "heartbeat_seconds": self.heartbeat_seconds,
        }


# ---------------------------------------------------------------------------
# Classification


@dataclass(frozen=True)
class Classification:
    klass: str
    stage: str
    reason: str


def image_failure_stage(value: Any) -> str:
    """Normalise an image step name to ``image_<step>`` for failure counts."""
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "unknown").lower()).strip("_") or "unknown"
    return text if text.startswith("image") else "image_" + text


def _first_issue(result: Mapping[str, Any]) -> str:
    issues = result.get("issues")
    return str(issues[0]) if isinstance(issues, list) and issues else ""


ISSUE_LIMIT = 200


def issue_head(result: Mapping[str, Any]) -> str | None:
    """The first issue as one line for ``conveyor.json`` (a traceback keeps its gate and final line)."""
    lines = [line.strip() for line in _first_issue(result).splitlines() if line.strip()]
    if not lines:
        return None
    head = lines[0]
    if len(lines) > 1 and any("Traceback (most recent call last)" in line for line in lines):
        head = f"{head.split(': ', 1)[0]}: {lines[-1]}"
    return head[:ISSUE_LIMIT]


def failed_stage(result: Mapping[str, Any]) -> str:
    """Best stage attribution for a ``failed`` result."""
    head = _first_issue(result)
    exhausted = result.get("wait_exhausted")
    if head.startswith(_WAIT_FAILURE_PREFIXES) and isinstance(exhausted, dict) and isinstance(exhausted.get("kind"), str):
        return exhausted["kind"]
    images = result.get("custom_images")
    if isinstance(images, dict) and images.get("state") == "failed_terminal":
        return image_failure_stage(images.get("failure_stage"))
    for prefix, stage in _FAILED_STAGE_PREFIXES:
        if head.startswith(prefix):
            return stage
    return "construction"


def _session_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    sessions = result.get("sessions")
    return [row for row in sessions if isinstance(row, dict)] if isinstance(sessions, list) else []


def classify_result(
    result: Mapping[str, Any], *, repair_actionable: Callable[[Mapping[str, Any]], bool]
) -> Classification:
    state = result.get("state")
    entry = STATE_TABLE.get(state) if isinstance(state, str) else None
    if entry is None:
        return Classification(UNCLASSIFIED, "unclassified_state", f"unclassified_state:{state}")
    klass, stage = entry
    if klass == TERMINAL:
        return Classification(TERMINAL, stage, "accepted" if state == "quality_accepted" else f"terminal:{state}")
    if klass == REPAIRABLE:
        if state == "failed":
            stage = failed_stage(result)
        if result.get("terminal_disposition") == "rejected":
            return Classification(TERMINAL, stage, "repair_budget_exhausted")
        budget = result.get("repair_budget")
        exhausted = isinstance(budget, dict) and budget.get("exhausted") is True
        if not exhausted and repair_actionable(result):
            return Classification(ACTIVE, stage, "construction_repair_pending")
        return Classification(TERMINAL, stage, f"operational_hold:{state}")
    if state == "pending_build":
        stopped = [row for row in _session_rows(result) if row.get("status") == "continuation_required"]
        reasons = {row.get("continuation_reason") for row in stopped}
        if "agent_process_failed" in reasons:
            return Classification(WAITING, "builder_process", "builder_session:agent_process_failed")
        if stopped:
            reason = min(str(value) for value in reasons) if reasons else "continuation_required"
            return Classification(WAITING, "builder_continuation", f"builder_session:{reason}")
        rows = _session_rows(result)
        if rows and all(row.get("status") == "complete" for row in rows) and _first_issue(result).startswith(
            "missing final bundle files"
        ):
            # Every session handed off; re-entry would reproduce the same gap.
            return Classification(TERMINAL, "build", "missing_final_bundle_files")
        return Classification(UNCLASSIFIED, "unclassified_state", "unclassified_state:pending_build(unrecognised sessions)")
    if state == "pending_image_review":
        images = result.get("custom_images")
        if isinstance(images, dict) and images.get("failure_class") == "transport":
            return Classification(WAITING, "image_review_transport", "waiting:image_review_transport")
    return Classification(WAITING, stage, f"waiting:{state}")


# ---------------------------------------------------------------------------
# Status stamping


def transition_reason(result: Mapping[str, Any]) -> str | None:
    head = _first_issue(result)
    if not head:
        images = result.get("custom_images")
        if isinstance(images, dict) and images.get("reason"):
            head = str(images["reason"])
    if not head:
        return None
    lines = head.strip().splitlines()
    return (lines[0] if lines else head)[:REASON_LIMIT]


def _number(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def read_status(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def stamp_status(
    result: dict[str, Any], prior: Mapping[str, Any] | None, *, now: float, intermediate: bool = True
) -> dict[str, Any]:
    """Add timing and a bounded transition history to ``result`` in place.

    ``prior`` is the status currently on disk (the history source of truth);
    when it is absent (an archive moved it) the result's own history is used.
    Intermediate writes (inside ``synthesize_one``) drop the settled
    ``conveyor`` record, keep the prior ``wait`` while the state is unchanged
    (so a crash mid-call keeps the wait budget) and drop it on a change.  The
    scheduler's settling writes (``intermediate=False``) set both explicitly.
    """
    state = result.get("state")
    history_source: Mapping[str, Any] = {}
    for source in (prior, result):
        if isinstance(source, Mapping) and isinstance(source.get("transitions"), list):
            history_source = source
            break
    history = [dict(entry) for entry in history_source.get("transitions", []) if isinstance(entry, dict)]
    if isinstance(prior, Mapping):
        last_state = prior.get("state")
        since_source: Mapping[str, Any] = prior
    else:
        last_state = history[-1].get("state") if history else None
        since_source = history_source or result
    changed = state != last_state or not history
    if changed:
        history.append({"state": state, "at": now, "reason": transition_reason(result)})
        state_since = now
    else:
        state_since = _number(since_source.get("state_since"))
        if state_since is None:
            state_since = _number(history[-1].get("at")) or now
    result["transitions"] = history[-TRANSITION_LIMIT:]
    result["state_since"] = state_since
    result["updated_at"] = now
    if intermediate:
        result.pop("conveyor", None)
        if changed:
            result.pop("wait", None)
        elif "wait" not in result and isinstance(prior, Mapping) and isinstance(prior.get("wait"), dict):
            result["wait"] = dict(prior["wait"])
    return result


def write_status(
    item_root: Path, result: dict[str, Any], *, intermediate: bool = True, now: float | None = None
) -> dict[str, Any]:
    """The one writer of ``items/<item>/status.json`` in synthesis."""
    item_root = Path(item_root)
    path = item_root / "status.json"
    stamp_status(result, read_status(path), now=time.time() if now is None else now, intermediate=intermediate)
    atomic_json(path, result)
    board = board_for(item_root)
    if board is not None:
        board.observe_status(item_root.name, result)
    return result


# ---------------------------------------------------------------------------
# Run-level board and activity markers

_BOARDS: dict[str, ConveyorBoard] = {}
_BOARDS_LOCK = threading.Lock()


def _root_key(path: Path) -> str:
    return os.path.realpath(path)


def board_for(item_root: Path) -> ConveyorBoard | None:
    item_root = Path(item_root)
    if item_root.parent.name != "items":
        return None
    with _BOARDS_LOCK:
        if not _BOARDS:
            return None
        return _BOARDS.get(_root_key(item_root.parent.parent))


def mark_activity(item_root: Path, step: str, **detail: Any) -> dict[str, Any] | None:
    """Record the long step an item just started (only inside a conveyor run).

    Writes ``items/<item>/activity.json`` and the item's ``activity`` in
    ``conveyor.json``.  Never raises: visibility must not break construction.
    """
    try:
        item_root = Path(item_root)
        board = board_for(item_root)
        if board is None:
            return None
        activity = {"step": step, "since": time.time()}
        activity.update({key: value for key, value in detail.items() if value is not None})
        atomic_json(item_root / "activity.json", activity)
        board.observe_activity(item_root.name, activity)
        return activity
    except Exception as error:  # noqa: BLE001 -- markers are best-effort
        _log({"event": "activity_marker_failed", "item": str(item_root), "step": step, "error": repr(error)}, stderr=True)
        return None


class ActivityAgent:
    """Delegate to an agent, marking ``step`` whenever a session is invoked."""

    def __init__(self, agent: Any, item_root: Path, step: str) -> None:
        self._agent, self._item_root, self._step = agent, Path(item_root), step

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        mark_activity(self._item_root, self._step)
        return self._agent.invoke(*args, **kwargs)


_IMAGE_COMMAND_STEPS = {
    "capture_generic_task_image.py": "image_capture",
    "probe_generic_task_image.py": "image_cold_pull",
}


def activity_command_runner(function: Callable[..., Any], item_root: Path) -> Callable[..., Any] | None:
    """Wrap ``function``'s default ``command_runner`` with activity markers.

    Returns ``None`` when ``function`` takes no such keyword with a callable
    default, so a changed image controller signature is never broken.
    """
    try:
        parameter = inspect.signature(function).parameters.get("command_runner")
    except (TypeError, ValueError):
        return None
    if parameter is None or parameter.default is inspect.Parameter.empty or not callable(parameter.default):
        return None
    base = parameter.default

    def runner(command: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            parts = [str(part) for part in command]
            name = Path(parts[0]).name if parts else ""
            role = parts[parts.index("--role") + 1] if "--role" in parts[:-1] else None
            mark_activity(item_root, _IMAGE_COMMAND_STEPS.get(name, "image_command"), command=name, role=role)
        except Exception:  # noqa: BLE001, S110 -- never let a marker change the command
            pass
        return base(command, *args, **kwargs)

    return runner


def _atomic_write_json(path: Path, payload: Any) -> None:
    temp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


class ConveyorBoard:
    """Run-level ``conveyor.json``: one row per item, rewritten atomically."""

    def __init__(self, root: Path, entries: list[tuple[str, str]], *, concurrency: int, config: ConveyorConfig) -> None:
        self.root = Path(root)
        self.path = self.root / "conveyor.json"
        self._lock = threading.Lock()
        self._names = {name: key for key, name in entries}
        self._items: dict[str, dict[str, Any]] = {
            key: {
                "item": name,
                "state": None,
                "state_since": None,
                "class": "queued",
                "activity": None,
                "wait": None,
                "wait_hold": None,
                "failure_stage": None,
                "issue": None,
                "calls": 0,
                "updated_at": None,
            }
            for key, name in entries
        }
        self._meta = {
            "schema_version": CONVEYOR_SCHEMA,
            "started_at": time.time(),
            "concurrency": concurrency,
            "config": config.describe(),
        }
        self._last_write = 0.0

    def register(self) -> None:
        with _BOARDS_LOCK:
            _BOARDS[_root_key(self.root)] = self

    def unregister(self) -> None:
        with _BOARDS_LOCK:
            if _BOARDS.get(_root_key(self.root)) is self:
                del _BOARDS[_root_key(self.root)]

    def _row(self, name: str) -> dict[str, Any] | None:
        key = self._names.get(name)
        return self._items.get(key) if key is not None else None

    def observe_status(self, name: str, status: Mapping[str, Any]) -> None:
        with self._lock:
            row = self._row(name)
            if row is None:
                return
            row.update(
                state=status.get("state"),
                state_since=status.get("state_since"),
                wait=status.get("wait") if isinstance(status.get("wait"), dict) else None,
                # Ops tools read held items (terminal for this job, re-entered on
                # relaunch) from this row: a held row is not a finished item.
                wait_hold=status.get("wait_hold") if isinstance(status.get("wait_hold"), dict) else None,
                failure_stage=status.get("failure_stage") if isinstance(status.get("failure_stage"), str) else None,
                issue=issue_head(status),
                updated_at=status.get("updated_at"),
            )
            # ``class`` is set only by the scheduler (dispatch and settle).
            self._write_locked()

    def observe_activity(self, name: str, activity: Mapping[str, Any]) -> None:
        with self._lock:
            row = self._row(name)
            if row is None:
                return
            row["activity"] = dict(activity)
            self._write_locked()

    def update(self, name: str, *, flush: bool = True, **fields: Any) -> None:
        with self._lock:
            row = self._row(name)
            if row is None:
                return
            row.update(fields)
            if flush:
                self._write_locked()

    def heartbeat(self, interval: float) -> None:
        with self._lock:
            if time.time() - self._last_write >= interval:
                self._write_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._payload_locked()

    def _payload_locked(self) -> dict[str, Any]:
        counts = Counter(row["class"] for row in self._items.values())
        return {
            **self._meta,
            "updated_at": time.time(),
            "counts": dict(counts),
            "items": {key: dict(row) for key, row in self._items.items()},
        }

    def _write_locked(self) -> None:
        try:
            _atomic_write_json(self.path, self._payload_locked())
            self._last_write = time.time()
        except (OSError, TypeError, ValueError) as error:
            _log({"event": "conveyor_write_failed", "path": str(self.path), "error": repr(error)}, stderr=True)


# ---------------------------------------------------------------------------
# Scheduler


@dataclass
class ConveyorEntry:
    key: str
    item_root: Path
    proposal_hash: str
    call: Callable[[], dict[str, Any]]


@dataclass
class _Track:
    index: int
    entry: ConveyorEntry
    wait: dict[str, Any] | None = None
    calls: int = 0
    exceptions: int = 0
    reentries: int = 0
    result: dict[str, Any] | None = None


@dataclass
class ConveyorOutcome:
    results: list[dict[str, Any]]
    failures: dict[str, dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)


class ConveyorScheduler:
    """Run every entry to a terminal state with at most ``concurrency`` active.

    ``classify(result, entry)`` returns a :class:`Classification`;
    ``retry_exception(error)`` says whether an exception from ``entry.call`` is
    worth another attempt (deterministic controller errors are not).
    """

    def __init__(
        self,
        root: Path,
        entries: list[ConveyorEntry],
        concurrency: int,
        *,
        classify: Callable[[Mapping[str, Any], ConveyorEntry], Classification],
        retry_exception: Callable[[BaseException], bool],
        config: ConveyorConfig,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if concurrency < 1:
            raise ConveyorConfigError("concurrency must be at least 1")
        self.root = Path(root)
        self.entries = list(entries)
        self.concurrency = concurrency
        self.classify = classify
        self.retry_exception = retry_exception
        self.config = config
        self.clock = clock
        self.failures: dict[str, dict[str, Any]] = {}
        self._failures_lock = threading.Lock()
        self._idle = threading.Event()
        self.board = ConveyorBoard(
            self.root,
            [(entry.key, entry.item_root.name) for entry in self.entries],
            concurrency=concurrency,
            config=config,
        )
        self.stats: Counter[str] = Counter()
        self._stats_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def _seed(self, track: _Track) -> None:
        status = read_status(track.entry.item_root / "status.json")
        if status is None:
            return
        self.board.update(
            track.entry.item_root.name,
            flush=False,  # one write after every row is seeded
            state=status.get("state"),
            state_since=_number(status.get("state_since")),
            failure_stage=status.get("failure_stage") if isinstance(status.get("failure_stage"), str) else None,
            wait_hold=status.get("wait_hold") if isinstance(status.get("wait_hold"), dict) else None,
            updated_at=_number(status.get("updated_at")),
        )
        wait = status.get("wait")
        entry = STATE_TABLE.get(status.get("state")) if isinstance(status.get("state"), str) else None
        if (
            isinstance(wait, dict)
            and entry is not None
            and entry[0] == WAITING
            and isinstance(wait.get("kind"), str)
            and _number(wait.get("first_seen")) is not None
            and type(wait.get("attempts")) is int
        ):
            # A relaunch keeps the attempt count but not the downtime: the time
            # between the prior job's last poll and now is credited to the
            # window, so a wait is never exhausted by an outage it never polled.
            track.wait = _carry_wait(wait, now=self.clock())

    def run(self) -> ConveyorOutcome:
        tracks = [_Track(index, entry) for index, entry in enumerate(self.entries)]
        sequence = itertools.count()
        self.board.register()
        try:
            for track in tracks:
                self._seed(track)
            self.board.heartbeat(0)
            fresh = deque(tracks)
            delayed: list[tuple[float, int, int]] = []
            running: dict[Future, _Track] = {}
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                while fresh or delayed or running:
                    now = self.clock()
                    while len(running) < self.concurrency:
                        if delayed and delayed[0][0] <= now:
                            track = tracks[heapq.heappop(delayed)[2]]
                        elif fresh:
                            track = fresh.popleft()
                        else:
                            break
                        self.board.update(track.entry.item_root.name, **{"class": "running"})
                        running[pool.submit(self._work, track)] = track
                    if not running:
                        # Every open item is waiting: no slot is held.
                        delay = max(0.0, delayed[0][0] - self.clock())
                        self._idle.wait(min(delay, self.config.heartbeat_seconds))
                        self.board.heartbeat(self.config.heartbeat_seconds)
                        continue
                    timeout = self.config.heartbeat_seconds
                    if delayed and len(running) < self.concurrency:
                        # A free slot and a future not-before: wake for it.  With
                        # every slot busy only a completion can free one, so a due
                        # retry must not turn this wait into a spin.
                        timeout = min(timeout, max(0.0, delayed[0][0] - self.clock()))
                    done, _ = wait_futures(tuple(running), timeout=timeout, return_when=FIRST_COMPLETED)
                    self.board.heartbeat(self.config.heartbeat_seconds)
                    for future in done:
                        track = running.pop(future)
                        not_before = future.result()
                        if not_before is not None:
                            heapq.heappush(delayed, (not_before, next(sequence), track.index))
        finally:
            self.board.heartbeat(0)
            self.board.unregister()
        results = [track.result for track in tracks]
        missing = [track.entry.key for track in tracks if track.result is None]
        if missing:  # _work always settles; this would be a scheduler bug.
            raise RuntimeError(f"conveyor finished with unsettled items: {missing}")
        with self._stats_lock:
            summary = {"calls": sum(track.calls for track in tracks), **dict(self.stats)}
        return ConveyorOutcome(results, dict(self.failures), summary)

    # -- one call ----------------------------------------------------------

    def _count(self, name: str) -> None:
        with self._stats_lock:
            self.stats[name] += 1

    def _work(self, track: _Track) -> float | None:
        """Run one call and settle it.  Returns a not-before time, or None if terminal."""
        try:
            mark_activity(track.entry.item_root, "dispatched", call=track.calls + 1)
            try:
                result = track.entry.call()
                if not isinstance(result, dict):
                    raise TypeError(f"item call returned {type(result).__name__}, not a status object")
            except Exception as error:  # noqa: BLE001 -- per-item isolation
                return self._settle_exception(track, error)
            return self._settle_result(track, result)
        except Exception as error:  # noqa: BLE001 -- a conveyor bug fails the item closed, loudly
            return self._settle_internal_error(track, error)

    def _finish(self, track: _Track, result: dict[str, Any], klass: str, reason: str, *, now: float) -> None:
        entry = track.entry
        result.setdefault("key", entry.key)
        result.setdefault("proposal_hash", entry.proposal_hash)
        result.setdefault("item_root", str(entry.item_root))
        result["conveyor"] = {"class": klass, "reason": reason, "calls": track.calls}
        write_status(entry.item_root, result, intermediate=False, now=now)
        board_class = {TERMINAL: "terminal", WAITING: "waiting", ACTIVE: "queued"}.get(klass, klass)
        wait = result.get("wait")
        activity = (
            {"step": "terminal", "since": now}
            if klass == TERMINAL
            else {"step": "waiting", "since": now, "kind": wait.get("kind"), "next_attempt_at": wait.get("next_attempt_at")}
            if klass == WAITING and isinstance(wait, dict)
            else {"step": "queued", "since": now}
        )
        try:
            atomic_json(entry.item_root / "activity.json", activity)
        except OSError:
            pass
        self.board.update(entry.item_root.name, **{"class": board_class, "activity": activity, "calls": track.calls})
        if klass == TERMINAL:
            track.result = result

    def _terminal(
        self, track: _Track, result: dict[str, Any], stage: str, reason: str, *, now: float, keep_wait: bool = False
    ) -> None:
        if not keep_wait:
            result.pop("wait", None)
        if result.get("state") == "quality_accepted":
            result.pop("failure_stage", None)
        else:
            result["failure_stage"] = stage
        track.wait = None
        self._finish(track, result, TERMINAL, reason, now=now)
        self._count("terminal")
        _log(
            {
                "event": "complete",
                "item": track.entry.key,
                "state": result.get("state"),
                "failure_stage": result.get("failure_stage"),
                "reason": reason,
                "calls": track.calls,
            }
        )

    def _fail(
        self,
        track: _Track,
        result: dict[str, Any],
        *,
        tag: str,
        stage: str,
        now: float,
        extra: dict[str, Any] | None = None,
        issue: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Convert ``result`` into a terminal ``failed`` whose first issue says why."""
        prior_state = result.get("state")
        issues = result.get("issues") if isinstance(result.get("issues"), list) else []
        reason = reason or f"{tag}:{prior_state}"
        result.update(extra or {})
        result["state"] = "failed"
        result["issues"] = [issue or reason, *[str(value) for value in issues]]
        self._terminal(track, result, stage, reason, now=now)

    def _settle_result(self, track: _Track, result: dict[str, Any]) -> float | None:
        now = self.clock()
        track.calls += 1
        entry = track.entry
        # A hold from an earlier job is over once the item has been re-entered;
        # _settle_wait sets a fresh one when the step holds again.
        result.pop("wait_hold", None)
        classification = self.classify(result, entry)
        state = result.get("state")
        if classification.klass == UNCLASSIFIED:
            _log(
                {
                    "event": "unclassified_state",
                    "item": entry.key,
                    "state": state,
                    "reason": classification.reason,
                    "action": "failed_closed",
                },
                stderr=True,
            )
            self._count("unclassified")
            self._fail(
                track,
                result,
                tag="unclassified_state",
                stage="unclassified_state",
                now=now,
                extra={"unclassified_state": state},
            )
            return None
        if classification.klass == TERMINAL:
            self._terminal(track, result, classification.stage, classification.reason, now=now)
            return None
        if track.calls >= self.config.max_calls_per_item:
            self._fail(track, result, tag="conveyor_call_cap_exhausted", stage="conveyor", now=now)
            return None
        if classification.klass == ACTIVE:
            track.wait = None
            track.reentries += 1
            if track.reentries > self.config.max_active_reentries:
                self._terminal(track, result, classification.stage, "active_reentry_cap_exhausted", now=now)
                return None
            result.pop("wait", None)
            self._finish(track, result, ACTIVE, classification.reason, now=now)
            self._count("requeued")
            _log({"event": "requeued", "item": entry.key, "state": state, "reason": classification.reason})
            return now
        return self._settle_wait(track, result, classification, now=now)

    def _settle_wait(self, track: _Track, result: dict[str, Any], classification: Classification, *, now: float) -> float | None:
        kind = classification.stage
        state = result.get("state")
        budget = self.config.budgets.get(kind)
        if budget is None:  # table and budgets disagree: a bug, fail closed
            self._fail(track, result, tag="unclassified_state", stage="unclassified_state", now=now, extra={"unclassified_state": state})
            return None
        hints = _image_hints(result) if isinstance(state, str) and state.startswith("pending_image_") else {}
        prior = track.wait if track.wait and track.wait.get("kind") == kind else None
        first_seen = _number(prior.get("first_seen")) if prior else None
        first_seen = now if first_seen is None else first_seen
        attempts = (prior.get("attempts", 0) if prior else 0) + 1
        # A retryable step that reports its own attempt budget (per role, and it
        # returns failed_terminal when spent) owns the attempt count; the
        # conveyor then enforces only its window and the per-item call cap.
        step_owned = hints.get("retryable") is True and "max_attempts" in hints
        if not step_owned:
            attempts = max(attempts, hints.get("attempts", 0))
        backoff = hints.get("backoff_seconds", budget.backoff_seconds)
        # Uncharged time: relaunch downtime (credited in _seed) and time spent
        # queued behind a live publisher.
        downtime = (_number(prior.get("downtime_seconds")) or 0.0) if prior else 0.0
        backlog = (_number(prior.get("publisher_backlog_seconds")) or 0.0) if prior else 0.0
        last_seen = _number(prior.get("last_seen")) if prior else None
        if prior and last_seen is not None and _publisher_live(kind, result):
            backlog += max(0.0, now - last_seen)
        window = budget.window_seconds
        step_timeout = hints.get("step_timeout_seconds")
        if step_timeout is not None:
            # One step can legitimately run for its whole timeout: the window
            # must fit at least that plus the backoffs between attempts, or a
            # slow retry is killed while the step still has budget.
            step_attempts = hints["max_attempts"] if step_owned else budget.max_attempts
            window = max(window, step_timeout + max(0, step_attempts - 1) * backoff)
        charged = window + min(backlog, max(0.0, PUBLISHER_BACKLOG_CAP_SECONDS - window))
        deadline = first_seen + downtime + charged
        wait = {
            "kind": kind,
            "state": state,
            "attempts": attempts,
            "max_attempts": None if step_owned else budget.max_attempts,
            "first_seen": first_seen,
            "last_seen": now,
            "deadline": deadline,
            "window_seconds": window,
            "reason": transition_reason(result),
        }
        if downtime:
            wait["downtime_seconds"] = downtime
        if backlog:
            wait["publisher_backlog_seconds"] = backlog
        for hint, name in (
            ("attempts", "step_attempts"),
            ("max_attempts", "step_max_attempts"),
            ("packet_sha", "packet_sha"),
            ("step_timeout_seconds", "step_timeout_seconds"),
        ):
            if hint in hints:
                wait[name] = hints[hint]
        if hints.get("retryable") is False:
            # The step says: not again in this job (a harness hold, a disabled
            # queue, spent publisher retries).  Terminal for this job, but the
            # state is kept, so a relaunch re-enters it without file surgery.
            track.wait = None
            self._count("wait_holds")
            images = result.get("custom_images") if isinstance(result.get("custom_images"), dict) else {}
            reason = f"not_retryable:{images.get('reason') or state}"
            result["wait_hold"] = {
                **wait,
                "held_at": now,
                "blocked_until": images.get("blocked_until") or "job_relaunch",
            }
            self._terminal(track, result, kind, reason, now=now)
            return None
        if (not step_owned and attempts >= budget.max_attempts) or now >= deadline:
            track.wait = None
            self._count("wait_budget_exhausted")
            cause = "deadline" if now >= deadline else "attempts"
            self._fail(
                track,
                result,
                tag="wait_budget_exhausted",
                stage=kind,
                now=now,
                extra={"wait_exhausted": {**wait, "exhausted_at": now, "cause": cause}},
            )
            return None
        next_attempt = min(now + backoff, deadline)
        wait.update(next_attempt_at=next_attempt, backoff_seconds=backoff)
        result["wait"] = wait
        track.wait = wait
        self._finish(track, result, WAITING, classification.reason, now=now)
        self._count("waits")
        _log(
            {
                "event": "waiting",
                "item": track.entry.key,
                "state": state,
                "kind": kind,
                "attempt": attempts,
                "max_attempts": wait["max_attempts"],
                "step_attempts": wait.get("step_attempts"),
                "next_attempt_at": next_attempt,
                "deadline": deadline,
            }
        )
        return next_attempt

    def _record_failure(self, track: _Track, error: BaseException) -> None:
        with self._failures_lock:
            self.failures[track.entry.key] = {"error_type": type(error).__name__, "error": str(error)}

    def _settle_exception(self, track: _Track, error: Exception) -> float | None:
        now = self.clock()
        track.calls += 1
        track.exceptions += 1
        budget = self.config.budgets["controller_exception"]
        transient = bool(self.retry_exception(error))
        retry = (
            transient
            and track.exceptions < budget.max_attempts
            and track.calls < self.config.max_calls_per_item
        )
        _log(
            {
                "event": "failed",
                "item": track.entry.key,
                "error_type": type(error).__name__,
                "error": str(error),
                "attempt": track.exceptions,
                "retry": retry,
            }
        )
        if retry:
            next_attempt = now + budget.backoff_seconds
            wait = {
                "kind": "controller_exception",
                "attempts": track.exceptions,
                "max_attempts": budget.max_attempts,
                "first_seen": now,
                "next_attempt_at": next_attempt,
                "deadline": now + budget.window_seconds,
                "reason": f"{type(error).__name__}: {error}"[:REASON_LIMIT],
            }
            activity = {"step": "waiting", "since": now, "kind": "controller_exception", "next_attempt_at": next_attempt}
            self.board.update(track.entry.item_root.name, **{"class": "waiting", "wait": wait, "activity": activity})
            self._count("exception_retries")
            return next_attempt
        self._record_failure(track, error)
        self._count("controller_exceptions")
        record = {
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
            "attempts": track.exceptions,
            "at": now,
            "transient": transient,
        }
        if transient:
            # The job exits non-zero (synthesize) so Iris retries it and the
            # supervisor relaunches: a transient fault must not end the item.
            self._count("controller_exception_holds")
        self._terminal_from_disk(
            track, record, now=now, tag="controller_exception", stage="controller_exception", hold=transient
        )
        return None

    def _terminal_from_disk(
        self, track: _Track, record: dict[str, Any], *, now: float, tag: str, stage: str, hold: bool = False
    ) -> None:
        """End an item whose call produced no result, preserving its status.

        A terminal state is kept and annotated.  When ``hold`` (the error is
        transient) a waiting, repairable, absent or mid-attempt state is kept
        with a ``wait_hold``: the item is terminal for this job only and a
        relaunch re-enters it.  Otherwise (a deterministic error) a repairable
        state is kept and annotated and anything else becomes terminal
        ``failed``.
        """
        entry = track.entry
        prior = read_status(entry.item_root / "status.json")
        result = dict(prior) if prior is not None else {
            "key": entry.key,
            "proposal_hash": entry.proposal_hash,
            "sessions": [],
            "state": None,
            "issues": [],
            "item_root": str(entry.item_root),
        }
        result[tag] = record
        state = result.get("state")
        table = STATE_TABLE.get(state) if isinstance(state, str) else None
        reason = f"{tag}:{record['error_type']}"
        if table is not None and (table[0] == TERMINAL or (table[0] == REPAIRABLE and not hold)):
            # Already at a resting state: keep it and its issues; annotate only.
            # (A repairable state hit by a transient error is held below: a
            # relaunch must re-enter it, e.g. to continue its repair loop.)
            self._terminal(track, result, stage, reason, now=now)
            return
        lines = f"{record['error_type']}: {record['error']}".splitlines() or [record["error_type"]]
        issue = f"{tag}:{lines[0][:REASON_LIMIT]}"
        if hold:
            wait = result.get("wait") if isinstance(result.get("wait"), dict) else None
            if state is None and not result.get("issues"):
                result["issues"] = [issue]
            result["wait_hold"] = {
                "kind": tag,
                "state": state,
                "attempts": record.get("attempts"),
                "max_attempts": self.config.budgets[tag].max_attempts if tag in self.config.budgets else None,
                "held_at": now,
                "blocked_until": "job_relaunch",
                "reason": issue,
            }
            self._count("wait_holds")
            # Keep a waiting state's own wait budget so a relaunch carries its
            # attempts (and _seed credits the downtime) instead of restarting it.
            self._terminal(track, result, stage, f"{tag}_hold:{record['error_type']}", now=now, keep_wait=wait is not None)
            return
        self._fail(track, result, tag=tag, stage=stage, now=now, issue=issue, reason=reason)

    def _settle_internal_error(self, track: _Track, error: Exception) -> None:
        now = self.clock()
        _log(
            {"event": "conveyor_error", "item": track.entry.key, "error_type": type(error).__name__, "error": str(error), "action": "failed_closed"},
            stderr=True,
        )
        self._record_failure(track, error)
        self._count("conveyor_errors")
        record = {"error_type": type(error).__name__, "error": str(error)[:4000], "at": now}
        try:
            self._terminal_from_disk(track, record, now=now, tag="conveyor_error", stage="conveyor")
        except Exception as second:  # noqa: BLE001 -- keep an in-memory terminal row
            _log({"event": "conveyor_error", "item": track.entry.key, "error": repr(second), "action": "memory_only"}, stderr=True)
            track.result = {
                "key": track.entry.key,
                "proposal_hash": track.entry.proposal_hash,
                "item_root": str(track.entry.item_root),
                "state": "failed",
                "issues": [f"conveyor_error:{type(error).__name__}: {error}"[:REASON_LIMIT]],
                "failure_stage": "conveyor",
            }


def _image_hints(result: Mapping[str, Any]) -> dict[str, Any]:
    images = result.get("custom_images")
    if not isinstance(images, dict):
        return {}
    hints: dict[str, Any] = {}
    if isinstance(images.get("retryable"), bool):
        hints["retryable"] = images["retryable"]
    attempts = images.get("attempts")
    if type(attempts) is int and attempts >= 0:
        hints["attempts"] = attempts
    maximum = images.get("max_attempts")
    if type(maximum) is int and maximum >= 1:
        hints["max_attempts"] = maximum
    backoff = _number(images.get("backoff_seconds"))
    if backoff is not None and backoff >= 0:
        hints["backoff_seconds"] = backoff
    packet = images.get("packet_sha")
    if isinstance(packet, str) and packet:
        hints["packet_sha"] = packet
    timeout = _number(images.get("step_timeout_seconds"))
    if timeout is not None and timeout > 0:
        hints["step_timeout_seconds"] = timeout
    return hints


def _publisher_live(kind: str, result: Mapping[str, Any]) -> bool:
    """Positive evidence that an image_publication wait is queued behind a live publisher."""
    if kind != "image_publication":
        return False
    images = result.get("custom_images")
    if not isinstance(images, dict) or images.get("reason") != "awaiting_publisher":
        return False
    age = _number(images.get("publisher_heartbeat_age_seconds"))
    return age is not None and 0 <= age <= PUBLISHER_FRESH_SECONDS


def _carry_wait(wait: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    """A prior job's wait, re-based so the relaunch gap is not charged to the window.

    Attempts carry over unchanged.  The gap since the prior job's last poll
    accumulates in ``downtime_seconds``; a wait with no ``last_seen`` (older
    status) cannot measure its gap, so its window restarts from now.
    """
    carried = dict(wait)
    last_seen = _number(carried.get("last_seen"))
    if last_seen is None:
        carried["first_seen"] = now
        carried.pop("downtime_seconds", None)
        carried["window_restarted_at"] = now
    else:
        gap = max(0.0, now - last_seen)
        carried["downtime_seconds"] = (_number(carried.get("downtime_seconds")) or 0.0) + gap
    # Charging resumes from here (the publisher-backlog credit measures from it).
    carried["last_seen"] = now
    return carried


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Terminal accounting for report.json."""
    failures = Counter(
        str(result.get("failure_stage") or "unattributed")
        for result in results
        if result.get("state") != "quality_accepted"
    )
    classes = Counter(
        (result.get("conveyor") or {}).get("class", "unknown") if isinstance(result.get("conveyor"), dict) else "unknown"
        for result in results
    )
    return {
        "terminal_items": sum(classes.get(name, 0) for name in (TERMINAL,)),
        "terminal_failures_by_stage": dict(sorted(failures.items())),
        "wait_exhausted_items": sum(isinstance(result.get("wait_exhausted"), dict) and result.get("state") == "failed" for result in results),
        "wait_hold_items": sum(
            isinstance(result.get("wait_hold"), dict) and result.get("state") == (result.get("wait_hold") or {}).get("state")
            for result in results
        ),
        "unclassified_items": sum("unclassified_state" in result for result in results),
    }
