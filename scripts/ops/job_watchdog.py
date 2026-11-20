#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Code-driven job watchdog: monitor tracked work, raise alerts, render a dashboard.

The watchdog replaces LLM-based fleet monitoring with a plain Python tick. Each tick it
probes every registry entry (Iris task states, storage prefixes, status-artifact JSON
fields, or the newest Finelog task-log line), records a state row, and raises
deduplicated alerts. Delivery is pluggable; the default sinks are

- an append-only JSONL alert log (one record per new or changed alert) that the
  coordinator's stall watch reads every tick, and
- a macOS desktop notification for every alert of severity >= warning.

The watchdog is strictly read-only toward Iris, k8s and object storage: it never
submits, restarts, deletes or mutates anything.

Usage:

    uv run python scripts/ops/job_watchdog.py --once \\
        --registry /Users/benfeuer/Documents/experiments/active/job-watchdog/registry.yaml \\
        --state-dir /Users/benfeuer/Documents/experiments/active/job-watchdog/state \\
        --dashboard /Users/benfeuer/Documents/experiments/active/job-watchdog/dashboard.html

    uv run python scripts/ops/job_watchdog.py --loop --interval-minutes 5

Registry schema (YAML):

    coordinator: <session id of the coordinating agent>   # recorded on every alert row
    defaults:
      stall_threshold_minutes: 90   # NO_PROGRESS beyond this
      gated_threshold_minutes: 60   # Kueue-gated beyond this
      cooldown_minutes: 360         # re-alert an unchanged condition after this
    entries:
      - id: difficulty-v17a9
        owner: <rhizome session id of the owning workstream>
        description: Difficulty-v2 round v17a9 training run
        expected_terminal: SUCCEEDED          # optional
        iris: {cluster: <cluster>, job: <job id>, job_is_prefix: false}  # optional lifecycle source
        stall_threshold_minutes: 120          # optional, overrides defaults
        preempt_alert_count: 2                # optional, default 2
        probe:
          type: iris_task_states   # Iris task-state histogram (+ job lifecycle state)
            cluster, job, job_is_prefix
          type: storage_objects    # object count + newest mtime under a prefix
            url: s3://bucket/prefix
          type: json_field         # a field of a JSON status artifact
            location: s3://bucket/path/status.json, field: dotted.path, terminal_value: sealed
          type: log_regex          # regex over the newest task log line (Finelog)
            cluster, job, pattern, failure_pattern, log_contains, namespace: marin, log_window_minutes: 360
"""

import argparse
import dataclasses
import fcntl
import html
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import yaml
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3

logger = logging.getLogger("job_watchdog")

WATCHDOG_DIR = Path.home() / "Documents" / "experiments" / "active" / "job-watchdog"
DEFAULT_REGISTRY_PATH = WATCHDOG_DIR / "registry.yaml"
DEFAULT_STATE_DIR = WATCHDOG_DIR / "state"
DEFAULT_DASHBOARD_PATH = WATCHDOG_DIR / "dashboard.html"
DEFAULT_ALERTS_PATH = WATCHDOG_DIR / "alerts.jsonl"

CLUSTER_CONFIG_DIRS = (Path.home() / ".config" / "marin" / "clusters", Path("lib/iris/config"))

LOG_WINDOW_MINUTES = 360
PREEMPT_ALERT_COUNT_DEFAULT = 2
GATE_REASON_PATTERN = re.compile(r"gate|quota|kueue", re.IGNORECASE)
JSONL_MAX_LINE_BYTES = 1_048_576

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

SEVERITY_BY_KIND: dict[str, str] = {
    "JOB_FAILED": "critical",
    "JOB_KILLED": "critical",
    "JOB_PREEMPTED": "warning",
    "NO_PROGRESS": "warning",
    "GATED": "warning",
    "SUCCEEDED": "info",
    "PROBE_ERROR": "warning",
}

# Iris job states that keep a job alive; everything else is terminal or gated.
IRIS_ACTIVE_STATES = frozenset({"pending", "building", "running"})
# Iris job states that mean the work failed outright.
IRIS_FAILED_STATES = frozenset({"failed", "worker_failed"})


class EntryState(StrEnum):
    """Effective state of one tracked entry after a tick."""

    RUNNING = "RUNNING"
    NO_PROGRESS = "NO_PROGRESS"
    GATED = "GATED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    KILLED = "KILLED"
    UNKNOWN = "UNKNOWN"


class ProbeKind(StrEnum):
    IRIS_TASK_STATES = "iris_task_states"
    STORAGE_OBJECTS = "storage_objects"
    JSON_FIELD = "json_field"
    LOG_REGEX = "log_regex"


class RegistryError(ValueError):
    """The registry file is malformed or carries unknown fields."""


class ProbeError(RuntimeError):
    """A probe could not observe its target this tick."""


@dataclasses.dataclass(frozen=True)
class IrisJobRef:
    cluster: str
    job: str
    job_is_prefix: bool = False


@dataclasses.dataclass(frozen=True)
class IrisTaskStatesProbe:
    cluster: str
    job: str
    job_is_prefix: bool = False


@dataclasses.dataclass(frozen=True)
class StorageObjectsProbe:
    url: str


@dataclasses.dataclass(frozen=True)
class JsonFieldProbe:
    location: str
    field: str
    terminal_value: str | None = None


@dataclasses.dataclass(frozen=True)
class LogRegexProbe:
    cluster: str
    job: str
    job_is_prefix: bool = False
    pattern: str = ""
    failure_pattern: str | None = None
    log_contains: str | None = None
    namespace: str = "marin"
    log_window_minutes: int = LOG_WINDOW_MINUTES


@dataclasses.dataclass(frozen=True)
class RegistryEntry:
    id: str
    owner: str
    description: str
    probe: IrisTaskStatesProbe | StorageObjectsProbe | JsonFieldProbe | LogRegexProbe
    iris: IrisJobRef | None
    stall_threshold: timedelta
    gated_threshold: timedelta
    cooldown: timedelta
    preempt_alert_count: int
    expected_terminal: EntryState | None
    expected_replacement: IrisJobRef | None = None


@dataclasses.dataclass(frozen=True)
class Registry:
    coordinator: str
    entries: tuple[RegistryEntry, ...]
    defaults: dict[str, timedelta]


@dataclasses.dataclass(frozen=True)
class IrisJobView:
    """What one Iris job looked like this tick."""

    job_id: str
    state: str
    preemptions: int
    task_counts: dict[str, int]
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class LogLine:
    """The newest Finelog log line observed for one job."""

    job_key: str
    data: str
    seq: int
    cluster: str
    epoch_ms: int


@dataclasses.dataclass
class FleetSnapshot:
    """Read-only fleet state gathered once per tick and shared by all entries."""

    iris: dict[tuple[str, str], IrisJobView] = dataclasses.field(default_factory=dict)
    logs: dict[str, LogLine] = dataclasses.field(default_factory=dict)
    kueue: dict[str, list[dict[str, Any]]] = dataclasses.field(default_factory=dict)
    filtered_logs: dict[tuple[str, int], dict[str, LogLine]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Observation:
    """What one probe saw during one tick.

    A None signature means "no fresh progress signal this tick": the caller keeps the
    previous signature, progress value and progress-change time instead of resetting
    the stall clock.
    """

    state: EntryState
    progress: str
    signature: str | None
    detail: str
    # Cumulative Iris preemption counter for this entry, when the probe reads one;
    # None for probes that cannot observe preemptions.
    preemptions: int | None = None


@dataclasses.dataclass(frozen=True)
class Alert:
    key: str
    kind: str
    severity: str
    entry_id: str
    state: str
    message: str
    evidence: dict[str, Any]


@dataclasses.dataclass
class EntryRecord:
    """Persisted rolling state of one entry across ticks."""

    state: EntryState | None = None
    progress_signature: str | None = None
    progress: str = ""
    detail: str = ""
    progress_changed_at: datetime | None = None
    gated_since: datetime | None = None
    preempted_seen: int = 0
    preempted_since_alert: int = 0


@dataclasses.dataclass
class AlertRecord:
    """Dedup bookkeeping for one alert key."""

    last_state: str | None = None
    last_sent_at: datetime | None = None
    count: int = 0


@dataclasses.dataclass
class WatchdogState:
    entries: dict[str, EntryRecord] = dataclasses.field(default_factory=dict)
    alerts: dict[str, AlertRecord] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EntryStatus:
    """One state row per entry, written each tick and rendered on the dashboard."""

    id: str
    owner: str
    description: str
    state: EntryState
    progress: str
    since_progress: str
    alerts: tuple[str, ...]
    detail: str
    probe_ok: bool
    ticked_at: datetime


@dataclasses.dataclass(frozen=True)
class TickResult:
    ticked_at: datetime
    statuses: tuple[EntryStatus, ...]
    fired: tuple[Alert, ...]


class AlertSink(Protocol):
    """Delivery boundary for alerts; sinks never see an alert twice per fire."""

    def deliver(self, alert: Alert) -> None: ...


class FanOutSink:
    """Delivers each fired alert to every configured sink."""

    def __init__(self, sinks: tuple[AlertSink, ...]) -> None:
        self.sinks = sinks

    def deliver(self, alert: Alert) -> None:
        failures = []
        for sink in self.sinks:
            try:
                sink.deliver(alert)
            except Exception as e:
                logger.warning("sink %s failed for %s: %s", type(sink).__name__, alert.key, e)
                failures.append(type(sink).__name__)
        if len(failures) == len(self.sinks):
            raise RuntimeError(f"every sink failed for {alert.key}: {failures}")


class JsonlAlertSink:
    """Append-only JSONL alert log; one record per fired alert."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def deliver(self, alert: Alert) -> None:
        now = datetime.now(UTC)
        record = {
            "time_utc": now.isoformat(timespec="seconds"),
            "time_local": now.astimezone().isoformat(timespec="seconds"),
            "key": alert.key,
            "kind": alert.kind,
            "severity": alert.severity,
            "entry": alert.entry_id,
            "state": alert.state,
            "evidence": alert.evidence,
            "message": alert.message,
        }
        line = json.dumps(record, separators=(",", ":"))
        if len(line) >= JSONL_MAX_LINE_BYTES:
            raise ValueError(f"alert record exceeds {JSONL_MAX_LINE_BYTES} bytes: {alert.key}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class DesktopNotificationSink:
    """macOS desktop notification for every alert at or above a severity."""

    def __init__(self, min_severity: str = "warning") -> None:
        self.min_severity = SEVERITY_ORDER[min_severity]

    def deliver(self, alert: Alert) -> None:
        if SEVERITY_ORDER.get(alert.severity, 0) < self.min_severity:
            return
        title = f"job-watchdog: {alert.kind} {alert.entry_id}"
        body = alert.message.splitlines()
        body_text = " · ".join(part.strip() for part in body[1:3])[:200]
        script = f'display notification "{_applescript_escape(body_text)}" with title "{_applescript_escape(title)}"'
        result = subprocess.run(
            ["/usr/bin/osascript", "-e", script], capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"osascript failed: {result.stderr.strip()[:200]}")


def _applescript_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def parse_duration_minutes(value: Any, field: str) -> timedelta:
    minutes = value if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None
    if minutes is None:
        raise RegistryError(f"{field} must be a non-negative number of minutes, got {value!r}")
    return timedelta(minutes=float(minutes))


def _require_str(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{context}: {key} must be a non-empty string")
    return value.strip()


def parse_probe(raw: Any, entry_id: str) -> IrisTaskStatesProbe | StorageObjectsProbe | JsonFieldProbe | LogRegexProbe:
    if not isinstance(raw, dict):
        raise RegistryError(f"{entry_id}: probe must be a mapping")
    kind = raw.get("type")
    known = {k.value for k in ProbeKind}
    if kind not in known:
        raise RegistryError(f"{entry_id}: probe.type must be one of {sorted(known)}, got {kind!r}")
    context = f"{entry_id}.probe"
    if kind == ProbeKind.IRIS_TASK_STATES:
        return IrisTaskStatesProbe(
            cluster=_require_str(raw, "cluster", context),
            job=_require_str(raw, "job", context),
            job_is_prefix=bool(raw.get("job_is_prefix", False)),
        )
    if kind == ProbeKind.STORAGE_OBJECTS:
        return StorageObjectsProbe(url=_require_str(raw, "url", context))
    if kind == ProbeKind.JSON_FIELD:
        terminal_value = raw.get("terminal_value")
        if terminal_value is not None and not isinstance(terminal_value, str):
            raise RegistryError(f"{context}: terminal_value must be a string when present")
        return JsonFieldProbe(
            location=_require_str(raw, "location", context),
            field=_require_str(raw, "field", context),
            terminal_value=terminal_value,
        )
    pattern = str(raw.get("pattern", "") or "")
    try:
        re.compile(pattern)
    except re.error as e:
        raise RegistryError(f"{context}: pattern does not compile: {e}") from e
    failure_pattern = raw.get("failure_pattern")
    if failure_pattern is not None:
        if not isinstance(failure_pattern, str):
            raise RegistryError(f"{context}: failure_pattern must be a string when present")
        try:
            re.compile(failure_pattern)
        except re.error as e:
            raise RegistryError(f"{context}: failure_pattern does not compile: {e}") from e
    log_contains = raw.get("log_contains")
    if log_contains is not None and not isinstance(log_contains, str):
        raise RegistryError(f"{context}: log_contains must be a string when present")
    return LogRegexProbe(
        cluster=_require_str(raw, "cluster", context),
        job=_require_str(raw, "job", context),
        job_is_prefix=bool(raw.get("job_is_prefix", False)),
        pattern=pattern,
        failure_pattern=failure_pattern,
        log_contains=log_contains,
        namespace=str(raw.get("namespace", "marin")),
        log_window_minutes=int(raw.get("log_window_minutes", LOG_WINDOW_MINUTES)),
    )


def parse_registry(raw: dict[str, Any]) -> Registry:
    if not isinstance(raw, dict):
        raise RegistryError("registry must be a YAML mapping")
    coordinator = _require_str(raw, "coordinator", "registry")
    defaults_raw = raw.get("defaults") or {}
    if not isinstance(defaults_raw, dict):
        raise RegistryError("registry.defaults must be a mapping")
    defaults = {
        "stall_threshold": parse_duration_minutes(
            defaults_raw.get("stall_threshold_minutes", 90), "defaults.stall_threshold_minutes"
        ),
        "gated_threshold": parse_duration_minutes(
            defaults_raw.get("gated_threshold_minutes", 60), "defaults.gated_threshold_minutes"
        ),
        "cooldown": parse_duration_minutes(defaults_raw.get("cooldown_minutes", 360), "defaults.cooldown_minutes"),
    }
    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise RegistryError("registry.entries must be a non-empty list")
    entries: list[RegistryEntry] = []
    seen: set[str] = set()
    for item in entries_raw:
        if not isinstance(item, dict):
            raise RegistryError("each registry entry must be a mapping")
        entry_id = _require_str(item, "id", "entry")
        if entry_id in seen:
            raise RegistryError(f"duplicate entry id {entry_id!r}")
        seen.add(entry_id)
        context = f"entry {entry_id}"
        iris_raw = item.get("iris")
        iris = None
        if iris_raw is not None:
            if not isinstance(iris_raw, dict):
                raise RegistryError(f"{context}: iris must be a mapping")
            iris = IrisJobRef(
                cluster=_require_str(iris_raw, "cluster", f"{context}.iris"),
                job=_require_str(iris_raw, "job", f"{context}.iris"),
                job_is_prefix=bool(iris_raw.get("job_is_prefix", False)),
            )
        expected = item.get("expected_terminal")
        expected_state = None
        if expected is not None:
            try:
                expected_state = EntryState(str(expected))
            except ValueError as e:
                raise RegistryError(f"{context}: unknown expected_terminal {expected!r}") from e
        replacement_raw = item.get("expected_replacement")
        replacement = None
        if replacement_raw is not None:
            if not isinstance(replacement_raw, dict):
                raise RegistryError(f"{context}: expected_replacement must be a mapping with cluster and job")
            replacement = IrisJobRef(
                cluster=_require_str(replacement_raw, "cluster", f"{context}.expected_replacement"),
                job=_require_str(replacement_raw, "job", f"{context}.expected_replacement"),
                job_is_prefix=bool(replacement_raw.get("job_is_prefix", False)),
            )
        entries.append(
            RegistryEntry(
                id=entry_id,
                owner=_require_str(item, "owner", context),
                description=str(item.get("description", "")),
                probe=parse_probe(item.get("probe"), entry_id),
                iris=iris,
                stall_threshold=parse_duration_minutes(
                    item.get("stall_threshold_minutes", defaults_raw.get("stall_threshold_minutes", 90)),
                    f"{context}.stall_threshold_minutes",
                ),
                gated_threshold=parse_duration_minutes(
                    item.get("gated_threshold_minutes", defaults_raw.get("gated_threshold_minutes", 60)),
                    f"{context}.gated_threshold_minutes",
                ),
                cooldown=parse_duration_minutes(
                    item.get("cooldown_minutes", defaults_raw.get("cooldown_minutes", 360)),
                    f"{context}.cooldown_minutes",
                ),
                preempt_alert_count=int(item.get("preempt_alert_count", PREEMPT_ALERT_COUNT_DEFAULT)),
                expected_terminal=expected_state,
                expected_replacement=replacement,
            )
        )
    return Registry(coordinator=coordinator, entries=tuple(entries), defaults=defaults)


def load_registry(path: Path) -> Registry:
    return parse_registry(yaml.safe_load(path.read_text(encoding="utf-8")))


def parse_timestamp(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


def format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def load_state(path: Path) -> WatchdogState:
    if not path.exists():
        return WatchdogState()
    raw = json.loads(path.read_text(encoding="utf-8"))
    state = WatchdogState()
    for entry_id, record in raw.get("entries", {}).items():
        state.entries[entry_id] = EntryRecord(
            state=EntryState(record["state"]) if record.get("state") else None,
            progress_signature=record.get("progress_signature"),
            progress=record.get("progress", ""),
            detail=record.get("detail", ""),
            progress_changed_at=parse_timestamp(record.get("progress_changed_at")),
            gated_since=parse_timestamp(record.get("gated_since")),
            preempted_seen=int(record.get("preempted_seen", 0)),
            preempted_since_alert=int(record.get("preempted_since_alert", 0)),
        )
    for key, record in raw.get("alerts", {}).items():
        state.alerts[key] = AlertRecord(
            last_state=record.get("last_state"),
            last_sent_at=parse_timestamp(record.get("last_sent_at")),
            count=int(record.get("count", 0)),
        )
    return state


def save_state(path: Path, state: WatchdogState) -> None:
    raw = {
        "entries": {
            entry_id: {
                "state": record.state.value if record.state else None,
                "progress_signature": record.progress_signature,
                "progress": record.progress,
                "detail": record.detail,
                "progress_changed_at": format_timestamp(record.progress_changed_at),
                "gated_since": format_timestamp(record.gated_since),
                "preempted_seen": record.preempted_seen,
                "preempted_since_alert": record.preempted_since_alert,
            }
            for entry_id, record in state.entries.items()
        },
        "alerts": {
            key: {
                "last_state": record.last_state,
                "last_sent_at": format_timestamp(record.last_sent_at),
                "count": record.count,
            }
            for key, record in state.alerts.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def humanize_duration(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 90:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 36:
        return f"{hours}h{minutes % 60:02d}m"
    return f"{hours // 24}d{hours % 24}h"


def effective_state(entry: RegistryEntry, record: EntryRecord, observation: Observation, now: datetime) -> EntryState:
    """Collapse an observation into the effective state row, applying the stall threshold."""
    if observation.state == EntryState.UNKNOWN:
        return EntryState.UNKNOWN
    if observation.state != EntryState.RUNNING:
        return observation.state
    baseline = record.progress_changed_at
    if baseline is None:
        return EntryState.RUNNING
    if now - baseline > entry.stall_threshold:
        return EntryState.NO_PROGRESS
    return EntryState.RUNNING


def compose_message(
    entry: RegistryEntry,
    alert_kind: str,
    state: EntryState,
    record: EntryRecord,
    now: datetime,
    replacement_state: str | None = None,
) -> str:
    since = "unknown"
    if record.progress_changed_at is not None:
        since = humanize_duration(now - record.progress_changed_at)
    lines = [
        f"[job-watchdog] {alert_kind} {entry.id} ({severity_for_kind(alert_kind, entry, replacement_state)})",
        f"state: {state.value} | progress: {record.progress or 'n/a'} | last progress: {since} ago",
    ]
    if record.detail:
        lines.append(record.detail)
    if entry.expected_replacement is not None:
        replacement = entry.expected_replacement.job
        state_text = replacement_state or "not in fleet list"
        lines.append(f"expected replacement {replacement}: {state_text}")
    suffix = f" | expected terminal: {entry.expected_terminal.value}" if entry.expected_terminal else ""
    lines.append(f"owner: {entry.owner}{suffix}")
    return "\n".join(lines)


def severity_for_kind(alert_kind: str, entry: RegistryEntry, replacement_state: str | None) -> str:
    """An owner-initiated replacement that is already alive downgrades a death to info."""
    default = SEVERITY_BY_KIND[alert_kind]
    if (
        alert_kind in ("JOB_FAILED", "JOB_KILLED")
        and entry.expected_replacement is not None
        and replacement_state in IRIS_ACTIVE_STATES
    ):
        return "info"
    return default


def alert_kind_for(
    entry: RegistryEntry, state: EntryState, record: EntryRecord, probe_ok: bool, now: datetime
) -> str | None:
    """The alert kind this tick's condition raises, or None when healthy."""
    if not probe_ok:
        return "PROBE_ERROR"
    if state == EntryState.FAILED:
        return "JOB_FAILED"
    if state == EntryState.KILLED:
        return "JOB_KILLED"
    if record.preempted_since_alert >= entry.preempt_alert_count:
        return "JOB_PREEMPTED"
    if state == EntryState.NO_PROGRESS:
        return "NO_PROGRESS"
    if (
        state == EntryState.GATED
        and record.gated_since is not None
        and now - record.gated_since >= entry.gated_threshold
    ):
        return "GATED"
    if state == EntryState.SUCCEEDED:
        return "SUCCEEDED"
    return None


def open_alert_kinds(entry: RegistryEntry, state: EntryState, record: EntryRecord, probe_ok: bool) -> tuple[str, ...]:
    """Conditions that currently hold, independent of alert dedup."""
    kinds: list[str] = []
    if not probe_ok:
        kinds.append("PROBE_ERROR")
    if state == EntryState.FAILED:
        kinds.append("JOB_FAILED")
    if state == EntryState.KILLED:
        kinds.append("JOB_KILLED")
    if state in (EntryState.NO_PROGRESS, EntryState.GATED, EntryState.SUCCEEDED):
        kinds.append(state.value)
    if record.preempted_since_alert >= entry.preempt_alert_count:
        kinds.append("JOB_PREEMPTED")
    return tuple(kinds)


def evaluate_alert(
    entry: RegistryEntry,
    alert_kind: str,
    previous_state: EntryState | None,
    state: EntryState,
    record: EntryRecord,
    book: AlertRecord,
    now: datetime,
    replacement_state: str | None = None,
) -> Alert | None:
    """Return the alert to fire for this tick, or None.

    Dedup rule: an alert fires when the key has never fired, the effective state
    changed since the previous tick (a failed job that was restarted and failed again
    is a new occurrence), or the cooldown elapsed on an unchanged condition.
    """
    key = f"{entry.id}:{alert_kind}"
    never_fired = book.last_sent_at is None
    changed = previous_state is not None and previous_state != state
    cooled = book.last_sent_at is not None and now - book.last_sent_at >= entry.cooldown
    if not (never_fired or changed or cooled):
        return None
    message = compose_message(entry, alert_kind, state, record, now, replacement_state)
    evidence = {
        "progress": record.progress,
        "since_progress": humanize_duration(now - record.progress_changed_at) if record.progress_changed_at else None,
        "detail": record.detail,
        "owner": entry.owner,
        "expected_terminal": entry.expected_terminal.value if entry.expected_terminal else None,
        "replacement_state": replacement_state,
    }
    return Alert(
        key=key,
        kind=alert_kind,
        severity=severity_for_kind(alert_kind, entry, replacement_state),
        entry_id=entry.id,
        state=state.value,
        message=message,
        evidence=evidence,
    )


def render_dashboard(statuses: Iterable[EntryStatus], ticked_at: datetime) -> str:
    rows = []
    for status in statuses:
        badge_class = {"RUNNING": "ok", "SUCCEEDED": "info", "UNKNOWN": "warn"}.get(status.state.value, "bad")
        alerts = html.escape(", ".join(status.alerts)) if status.alerts else "—"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(status.id)}</code></td>"
            f"<td>{html.escape(status.description)}</td>"
            f'<td><span class="badge {badge_class}">{html.escape(status.state.value)}</span></td>'
            f"<td>{html.escape(status.progress or '—')}</td>"
            f"<td>{html.escape(status.since_progress)}</td>"
            f"<td><code>{html.escape(status.owner)}</code></td>"
            f"<td>{alerts}</td>"
            "</tr>"
        )
    footer = (
        f"<footer>tick {html.escape(ticked_at.isoformat(timespec='seconds'))} · "
        "alerts land in alerts.jsonl and on desktop</footer>"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>job-watchdog</title><style>"
        "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:2rem;color:#1a1a1a}"
        "h1{font-size:1.2rem}table{border-collapse:collapse;width:100%;font-size:.9rem}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;vertical-align:top}"
        "th{background:#f4f4f4}.badge{padding:.1rem .5rem;border-radius:.6rem;font-size:.8rem}"
        ".ok{background:#d6f5d6}.info{background:#d6e6f5}.warn{background:#f5e9c2}.bad{background:#f5cfcf}"
        "footer{margin-top:1rem;color:#666;font-size:.8rem}"
        "</style></head><body>"
        "<h1>job-watchdog fleet state</h1>"
        "<table><thead><tr><th>id</th><th>description</th><th>state</th><th>progress</th>"
        "<th>since progress</th><th>owner</th><th>alerts</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"{footer}"
        "</body></html>\n"
    )


def write_dashboard(path: Path, statuses: Iterable[EntryStatus], ticked_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_dashboard(statuses, ticked_at)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".html") as f:
        f.write(content)
        tmp = Path(f.name)
    os.replace(tmp, path)


def append_history(path: Path, statuses: Iterable[EntryStatus]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for status in statuses:
            row = {
                "ticked_at": status.ticked_at.isoformat(),
                "id": status.id,
                "state": status.state.value,
                "progress": status.progress,
                "since_progress": status.since_progress,
                "alerts": list(status.alerts),
                "detail": status.detail,
                "probe_ok": status.probe_ok,
            }
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def run_readonly(command: list[str], timeout_seconds: int = 180) -> str:
    """Run a read-only external probe command and return stdout."""
    logger.debug("probe command: %s", command)
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=True)
    return result.stdout


def run_tick(
    registry: Registry,
    state: WatchdogState,
    probe_runner,
    sink: AlertSink,
    now: datetime,
    resolve_replacement=None,
) -> TickResult:
    """One watchdog tick: probe every entry, update rolling state, fire deduplicated alerts.

    resolve_replacement(entry) returns the expected replacement job's current Iris
    state (or None when unknown); used to soften owner-initiated replacement deaths.
    """
    statuses: list[EntryStatus] = []
    fired: list[Alert] = []
    for entry in registry.entries:
        record = state.entries.setdefault(entry.id, EntryRecord())
        previous_state = record.state
        try:
            observation = probe_runner(entry, now)
            probe_ok = True
        except Exception as e:
            logger.warning("probe failed for %s: %s", entry.id, e)
            observation = Observation(
                state=EntryState.UNKNOWN,
                progress=record.progress,
                signature="probe-error",
                detail=f"probe failed: {e}"[:400],
            )
            probe_ok = False
        if probe_ok:
            if observation.signature is not None:
                if record.progress_signature is None or record.progress_signature != observation.signature:
                    record.progress_signature = observation.signature
                    record.progress = observation.progress
                    record.progress_changed_at = now
            elif record.progress_signature is None and record.progress_changed_at is None:
                # Entry has never produced a progress signal; start the stall clock.
                record.progress = observation.progress
                record.progress_changed_at = now
        record.detail = observation.detail
        if observation.preemptions is not None:
            record.preempted_since_alert += max(0, observation.preemptions - record.preempted_seen)
            record.preempted_seen = observation.preemptions
        current_state = effective_state(entry, record, observation, now)
        if current_state == EntryState.GATED and record.gated_since is None:
            record.gated_since = now
        if current_state != EntryState.GATED:
            record.gated_since = None
        record.state = current_state
        alert_kind = alert_kind_for(entry, current_state, record, probe_ok, now)
        if alert_kind is not None:
            key = f"{entry.id}:{alert_kind}"
            book = state.alerts.setdefault(key, AlertRecord())
            alert = evaluate_alert(
                entry,
                alert_kind,
                previous_state,
                current_state,
                record,
                book,
                now,
                replacement_state=resolve_replacement(entry) if entry.expected_replacement is not None else None,
            )
            if alert is not None:
                fired.append(alert)
                book.last_state = current_state.value
                book.last_sent_at = now
                book.count += 1
                if alert.kind == "JOB_PREEMPTED":
                    record.preempted_since_alert = 0
                try:
                    sink.deliver(alert)
                except Exception as e:
                    logger.error("alert %s could not be delivered: %s", alert.key, e)
        statuses.append(
            EntryStatus(
                id=entry.id,
                owner=entry.owner,
                description=entry.description,
                state=current_state,
                progress=record.progress,
                since_progress=(
                    humanize_duration(now - record.progress_changed_at) if record.progress_changed_at else "—"
                ),
                alerts=open_alert_kinds(entry, current_state, record, probe_ok),
                detail=record.detail,
                probe_ok=probe_ok,
                ticked_at=now,
            )
        )
    return TickResult(ticked_at=now, statuses=tuple(statuses), fired=tuple(fired))


def collect_fleet_snapshot(registry: Registry, now: datetime) -> FleetSnapshot:
    """Gather the fleet state once per tick: Iris job lists per tracked prefix, the
    newest Finelog line per tracked job, and Kueue workloads where something is pending.

    Batch probes keep a tick at a handful of subprocess calls regardless of fleet size.
    """
    snapshot = FleetSnapshot()
    prefixes_by_cluster: dict[str, set[str]] = {}
    log_patterns: list[str] = []
    for entry in registry.entries:
        if isinstance(entry.probe, IrisTaskStatesProbe):
            continue  # describes its own job directly; never in the shared snapshot
        refs = [entry.iris] if entry.iris is not None else []
        if isinstance(entry.probe, LogRegexProbe):
            refs.append(
                IrisJobRef(cluster=entry.probe.cluster, job=entry.probe.job, job_is_prefix=entry.probe.job_is_prefix)
            )
        for ref in refs:
            prefixes_by_cluster.setdefault(ref.cluster, set()).add(ref.job)
            log_patterns.append(_log_pattern_for(ref))
        if entry.expected_replacement is not None:
            prefixes_by_cluster.setdefault(entry.expected_replacement.cluster, set()).add(entry.expected_replacement.job)
    for cluster, prefixes in sorted(prefixes_by_cluster.items()):
        for prefix in sorted(prefixes):
            # A tight per-prefix limit: the fleet list is most-recent-first, so one
            # wide /benfeuer/ listing would drop every job older than the newest few
            # hundred rows.
            try:
                listing = run_readonly(
                    [
                        "uv",
                        "run",
                        "--no-sync",
                        "iris",
                        f"--cluster={cluster}",
                        "job",
                        "list",
                        "--prefix",
                        prefix,
                        "--limit",
                        "50",
                    ]
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # Entries on this cluster degrade to UNKNOWN and raise PROBE_ERROR,
                # which is the honest signal; do not kill the whole tick.
                logger.warning("iris job list failed for %s prefix %s: %s", cluster, prefix, e)
                continue
            for view in parse_iris_job_list(listing):
                snapshot.iris[(cluster, view.job_id)] = view
        if any(view.state in ("pending", "unschedulable") for (clu, _), view in snapshot.iris.items() if clu == cluster):
            snapshot.kueue[cluster] = kueue_workloads(cluster)
    if log_patterns:
        try:
            snapshot.logs = newest_finelog_lines(sorted(set(log_patterns)), now)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Stale log data is safer than a dead tick: missing lines keep entries'
            # previous signatures, so stall clocks are not reset by a hub hiccup.
            logger.warning("finelog batch query failed; keeping previous log state: %s", e)
    filters = {
        (entry.probe.log_contains, entry.probe.log_window_minutes)
        for entry in registry.entries
        if isinstance(entry.probe, LogRegexProbe) and entry.probe.log_contains
    }
    for text, window in sorted(filters):
        patterns = [
            _log_pattern_for(IrisJobRef(entry.probe.cluster, entry.probe.job, entry.probe.job_is_prefix))
            for entry in registry.entries
            if isinstance(entry.probe, LogRegexProbe)
            and entry.probe.log_contains == text
            and entry.probe.log_window_minutes == window
        ]
        try:
            snapshot.filtered_logs[(text, window)] = newest_finelog_lines(
                sorted(set(patterns)), now, window_minutes=window, contains=text
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            logger.warning("finelog filtered query failed for %r: %s", text, error)
    return snapshot


def _log_pattern_for(ref: IrisJobRef) -> str:
    if "'" in ref.job:
        raise ProbeError(f"job pattern {ref.job!r} contains a quote; cannot build a Finelog key filter")
    return f"{ref.job}%" if ref.job_is_prefix else f"{ref.job}/%"


def newest_finelog_lines(
    patterns: list[str], now: datetime, window_minutes: int = LOG_WINDOW_MINUTES, *, contains: str | None = None
) -> dict[str, LogLine]:
    """Newest Finelog line per tracked pattern, in one query.

    Each pattern gets its own bounded subquery joined by UNION ALL, so a flood of log
    lines from one job cannot push a quiet job's newest line out of the result. A
    pattern LIMITs to one row: an exact job yields that job's newest line, and a
    prefix yields the single newest line under the prefix across matching jobs.
    ``contains`` filters message content before selecting the newest line, so
    later progress cannot hide an earlier escalation.
    """
    since_ms = int(now.timestamp() * 1000) - int(window_minutes * 60 * 1000)
    columns = 'key, seq, epoch_ms, "data", "cluster"'
    content_filter = "" if contains is None else " AND contains(data, '" + contains.replace("'", "''") + "')"
    subqueries = " UNION ALL ".join(
        f"(SELECT {columns} FROM \"log\" WHERE key LIKE '{pattern}' AND epoch_ms > {since_ms} {content_filter} "
        "ORDER BY epoch_ms DESC, seq DESC LIMIT 1)"
        for pattern in patterns
    )
    output = run_readonly(
        ["uv", "run", "--no-sync", "finelog", "query", "marin", "--format", "jsonl", subqueries],
        timeout_seconds=180,
    )
    newest: dict[str, LogLine] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        row = json.loads(stripped)
        job_key = _job_key_of(row.get("key", ""))
        if job_key is None:
            continue
        candidate = LogLine(
            job_key=job_key,
            data=str(row.get("data", "")),
            seq=int(row.get("seq", 0)),
            cluster=str(row.get("cluster", "")),
            epoch_ms=int(row.get("epoch_ms", 0)),
        )
        if job_key not in newest or (candidate.epoch_ms, candidate.seq) > (
            newest[job_key].epoch_ms,
            newest[job_key].seq,
        ):
            newest[job_key] = candidate
    return newest


def _job_key_of(key: str) -> str | None:
    """'/benfeuer/<job>/<task>:<attempt>' -> '/benfeuer/<job>'; None when unmatched."""
    parts = key.strip("/").split("/")
    if len(parts) < 2:
        return None
    return "/" + "/".join(parts[:2])


def snapshot_probe_runner(snapshot: FleetSnapshot):
    """Build the per-entry probe runner over a shared FleetSnapshot."""

    def iris_view_for(ref: IrisJobRef) -> IrisJobView:
        if not ref.job_is_prefix:
            view = snapshot.iris.get((ref.cluster, ref.job))
            if view is None:
                raise ProbeError(f"Iris job {ref.job!r} not in the fleet list of {ref.cluster}")
            return view
        matches = [
            view
            for (cluster, _), view in snapshot.iris.items()
            if cluster == ref.cluster and view.job_id.startswith(ref.job)
        ]
        if not matches:
            raise ProbeError(f"no Iris jobs match prefix {ref.job!r} on cluster {ref.cluster}")
        return matches[0]

    def runner(entry: RegistryEntry, now: datetime) -> Observation:
        if isinstance(entry.probe, IrisTaskStatesProbe):
            return probe_iris_task_states(entry.probe)
        cluster = None
        iris_view = None
        if entry.iris is not None:
            cluster = entry.iris.cluster
            iris_view = iris_view_for(entry.iris)
        elif isinstance(entry.probe, LogRegexProbe):
            cluster = entry.probe.cluster
            iris_view = iris_view_for(
                IrisJobRef(cluster=entry.probe.cluster, job=entry.probe.job, job_is_prefix=entry.probe.job_is_prefix)
            )
        lifecycle = None
        if iris_view is not None:
            state, detail = iris_lifecycle_observation(iris_view, cluster or "", snapshot.kueue.get(cluster or "", []))
            lifecycle = (state, detail, iris_view.preemptions)
        if isinstance(entry.probe, StorageObjectsProbe):
            return probe_storage_objects(entry.probe, lifecycle)
        if isinstance(entry.probe, JsonFieldProbe):
            return probe_json_field(entry.probe, lifecycle)
        if isinstance(entry.probe, LogRegexProbe):
            filter_key = (entry.probe.log_contains, entry.probe.log_window_minutes)
            if entry.probe.log_contains and filter_key not in snapshot.filtered_logs:
                raise ProbeError(f"Finelog query for {entry.probe.log_contains!r} failed")
            log_snapshot = (
                dataclasses.replace(snapshot, logs=snapshot.filtered_logs[filter_key])
                if entry.probe.log_contains
                else snapshot
            )
            return probe_log_regex(entry.probe, _resolve_log_line(log_snapshot, entry.probe), lifecycle)
        raise ProbeError(f"unsupported probe type for {entry.id}")

    return runner


def _resolve_log_line(snapshot: FleetSnapshot, probe: LogRegexProbe) -> LogLine | None:
    if not probe.job_is_prefix:
        return snapshot.logs.get(probe.job)
    matches = [line for job_key, line in snapshot.logs.items() if job_key.startswith(probe.job)]
    if not matches:
        return None
    return max(matches, key=lambda line: (line.epoch_ms, line.seq))


def replacement_state_resolver(snapshot: FleetSnapshot):
    """Resolve an entry's expected replacement job to its current Iris state."""

    def resolve(entry: RegistryEntry) -> str | None:
        ref = entry.expected_replacement
        if ref is None:
            return None
        if not ref.job_is_prefix:
            view = snapshot.iris.get((ref.cluster, ref.job))
            return view.state if view else None
        matches = [
            view
            for (cluster, _), view in snapshot.iris.items()
            if cluster == ref.cluster and view.job_id.startswith(ref.job)
        ]
        return matches[0].state if matches else None

    return resolve


def parse_iris_job_describe(output: str) -> IrisJobView:
    """Parse `iris job describe` output into an IrisJobView."""
    state_match = re.search(r"^State:\s+(\S+)\s+exit=\S+\s+failures=\d+\s+preemptions=(\d+)", output, re.MULTILINE)
    if state_match is None:
        raise ProbeError("job describe output did not carry a State line")
    job_match = re.search(r"^Job:\s+(\S+)", output, re.MULTILINE)
    counts: dict[str, int] = {}
    tasks_match = re.search(r"^Tasks:\s+(\d+)/(\d+)\s+completed(.*)$", output, re.MULTILINE)
    if tasks_match is not None:
        for token, count in re.findall(r"(\w+)=(\d+)", tasks_match.group(3)):
            counts[token] = int(count)
        counts["completed"] = int(tasks_match.group(1))
    return IrisJobView(
        job_id=job_match.group(1) if job_match else "",
        state=state_match.group(1),
        preemptions=int(state_match.group(2)),
        task_counts=counts,
    )


def parse_iris_job_list(output: str) -> list[IrisJobView]:
    """Parse `iris job list` rows (job ids start with '/')."""
    views = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("/"):
            continue
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 2:
            continue
        reason = " ".join(parts[3:]).strip() if len(parts) > 3 else ""
        views.append(IrisJobView(job_id=parts[0], state=parts[1], preemptions=0, task_counts={}, reason=reason))
    return views


def newest_iris_job_id(cluster: str, prefix: str) -> str:
    """The most recently submitted Iris job id matching a prefix (read-only)."""
    listing = run_readonly(
        ["uv", "run", "--no-sync", "iris", f"--cluster={cluster}", "job", "list", "--prefix", prefix, "--limit", "20"]
    )
    views = parse_iris_job_list(listing)
    if not views:
        raise ProbeError(f"no Iris jobs match prefix {prefix!r} on cluster {cluster}")
    return views[0].job_id


def describe_iris_job(cluster: str, job: str, job_is_prefix: bool) -> IrisJobView:
    """Read one Iris job's state, task histogram and preemption counter (read-only)."""
    base = ["uv", "run", "--no-sync", "iris", f"--cluster={cluster}"]
    if job_is_prefix:
        job = newest_iris_job_id(cluster, job)
    try:
        return parse_iris_job_describe(run_readonly([*base, "job", "describe", job]))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        listing = run_readonly([*base, "job", "list", "--prefix", job, "--limit", "20"])
        views = [view for view in parse_iris_job_list(listing) if view.job_id == job]
        if not views:
            raise ProbeError(f"Iris job {job!r} not found on cluster {cluster}") from None
        return views[0]


@dataclasses.dataclass(frozen=True)
class K8sClusterSettings:
    """Kubernetes access settings resolved from one cluster's config file."""

    namespace: str
    kubeconfig: str
    context: str


def k8s_cluster_settings(cluster: str) -> K8sClusterSettings | None:
    """Kubernetes settings for a cluster, or None when not Kubernetes-backed."""
    for directory in CLUSTER_CONFIG_DIRS:
        path = directory / f"{cluster}.yaml"
        if not path.exists():
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        provider = config.get("kubernetes_provider") or {}
        platform = (config.get("platform") or {}).get("coreweave") or {}
        namespace = provider.get("namespace")
        context = platform.get("kube_context")
        kubeconfig = platform.get("kubeconfig_path")
        if not (namespace and context and kubeconfig):
            return None
        return K8sClusterSettings(namespace=namespace, kubeconfig=str(Path(kubeconfig).expanduser()), context=context)
    return None


def kueue_workloads(cluster: str) -> list[dict[str, Any]]:
    """Kueue workload condition lists for a cluster (read-only; best effort)."""
    settings = k8s_cluster_settings(cluster)
    if settings is None:
        return []
    command = [
        "kubectl",
        "--kubeconfig",
        settings.kubeconfig,
        "--context",
        settings.context,
        "-n",
        settings.namespace,
        "get",
        "workloads.kueue.x-k8s.io",
        "-o",
        "json",
    ]
    try:
        raw = run_readonly(command, timeout_seconds=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("kueue workload lookup failed for %s: %s", cluster, e)
        return []
    return json.loads(raw).get("items", [])


def kueue_gate_view(cluster: str, job_id: str, workloads: list[dict[str, Any]] | None = None) -> tuple[bool, str]:
    """(gated, condition message) from the Kueue workloads matching one job (read-only)."""
    if workloads is None:
        workloads = kueue_workloads(cluster)
    slug = job_id.rsplit("/", 1)[-1]
    for workload in workloads:
        if slug not in workload.get("metadata", {}).get("name", ""):
            continue
        for condition in workload.get("status", {}).get("conditions", []):
            kind, status = condition.get("type"), condition.get("status")
            message = condition.get("message", "")
            if kind == "QuotaReserved" and status != "True":
                return True, f"Kueue QuotaReserved={status} reason={condition.get('reason')}: {message}"
            if kind == "Admitted" and status == "False":
                return True, f"Kueue Admitted=False reason={condition.get('reason')}: {message}"
    return False, ""


def iris_state_to_entry_state(view: IrisJobView) -> EntryState:
    if view.state in IRIS_FAILED_STATES:
        return EntryState.FAILED
    if view.state == "killed":
        return EntryState.KILLED
    if view.state == "succeeded":
        return EntryState.SUCCEEDED
    return EntryState.RUNNING


def iris_lifecycle_observation(
    view: IrisJobView, cluster: str, kueue: list[dict[str, Any]] | None = None
) -> tuple[EntryState, str]:
    """(state, diagnostic) from an Iris job view, with Kueue gating resolved."""
    state = iris_state_to_entry_state(view)
    counts = _format_counts(view.task_counts)
    preemptions = f" preemptions={view.preemptions}" if view.preemptions else ""
    detail = f"iris {view.job_id}: state={view.state} tasks={counts}{preemptions}"
    if view.state == "unschedulable":
        return EntryState.GATED, f"{detail}; unschedulable: {view.reason or 'no reason reported'}"
    if view.state == "pending":
        kueue_gated, kueue_message = kueue_gate_view(cluster, view.job_id, kueue)
        reason = kueue_message or view.reason
        if kueue_gated or GATE_REASON_PATTERN.search(reason):
            return EntryState.GATED, f"{detail}; gated: {reason or 'no reason reported'}"
    return state, detail


def _format_counts(counts: dict[str, int]) -> str:
    return ",".join(f"{name}={count}" for name, count in sorted(counts.items())) or "none"


def probe_iris_task_states(probe: IrisTaskStatesProbe) -> Observation:
    view = describe_iris_job(probe.cluster, probe.job, probe.job_is_prefix)
    state, detail = iris_lifecycle_observation(view, probe.cluster)
    counts = _format_counts(view.task_counts)
    signature = f"{view.job_id}|{view.state}|preempt={view.preemptions}|{counts}"
    return Observation(state=state, progress=counts, signature=signature, detail=detail, preemptions=view.preemptions)


def _detail_mtime(details: dict[str, Any]) -> datetime | None:
    for key in ("LastModified", "last_modified", "updated", "timeModified", "mtime", "ModificationTime"):
        value = details.get(key)
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def probe_storage_objects(probe: StorageObjectsProbe, lifecycle: tuple[EntryState, str, int] | None) -> Observation:
    configure_coreweave_s3()
    filesystem, path = url_to_fs(probe.url)
    details = filesystem.find(path, detail=True)
    files = [entry for entry in details.values() if entry.get("type") != "directory"]
    count = len(files)
    mtimes = [mtime for mtime in (_detail_mtime(entry) for entry in files) if mtime is not None]
    newest = max(mtimes) if mtimes else None
    progress = f"{count} objects, newest {newest.isoformat(timespec='seconds') if newest else '—'}"
    signature = f"{count}|{newest.isoformat() if newest else 'none'}"
    detail = f"{probe.url}: {progress}"
    state = EntryState.RUNNING
    preemptions = None
    if lifecycle is not None:
        state, iris_detail, preemptions = lifecycle[0], lifecycle[1], lifecycle[2] or None
        detail = f"{detail}; {iris_detail}"
    return Observation(state=state, progress=progress, signature=signature, detail=detail, preemptions=preemptions)


def probe_json_field(probe: JsonFieldProbe, lifecycle: tuple[EntryState, str, int] | None) -> Observation:
    configure_coreweave_s3()
    with open_url(probe.location, "r") as f:
        document = json.loads(f.read())
    value: Any = document
    for part in probe.field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ProbeError(f"field {probe.field!r} missing in {probe.location}")
        value = value[part]
    value_text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    state = (
        EntryState.SUCCEEDED
        if probe.terminal_value is not None and value_text == probe.terminal_value
        else EntryState.RUNNING
    )
    progress = f"{probe.field}={value_text[:200]}"
    detail = f"{probe.location}: {progress}"
    preemptions = None
    if lifecycle is not None:
        state, iris_detail, preemptions = lifecycle[0], lifecycle[1], lifecycle[2] or None
        detail = f"{detail}; {iris_detail}"
    return Observation(state=state, progress=progress, signature=progress, detail=detail, preemptions=preemptions)


def probe_log_regex(
    probe: LogRegexProbe, log_line: LogLine | None, lifecycle: tuple[EntryState, str, int] | None
) -> Observation:
    """Regex over the newest Finelog task-log line.

    The progress signature is the Finelog sequence of the newest line, so any new log
    line counts as progress; the regex match is the human-readable progress value.
    """
    if log_line is None:
        # No recent log line says nothing about liveness: the Iris lifecycle still decides
        # whether the job is running, gated or terminal.
        detail = f"finelog: no lines for {probe.job} in the last {probe.log_window_minutes}m"
        state = EntryState.RUNNING
        preemptions = None
        if lifecycle is not None:
            state, iris_detail, preemptions = lifecycle[0], lifecycle[1], lifecycle[2] or None
            detail = f"{detail}; {iris_detail}"
        return Observation(
            state=state,
            progress="(no log lines)",
            signature=None,
            detail=detail,
            preemptions=preemptions,
        )
    job_id = log_line.job_key
    data = log_line.data
    pattern = re.compile(probe.pattern) if probe.pattern else None
    failure_pattern = re.compile(probe.failure_pattern) if probe.failure_pattern else None
    state = EntryState.RUNNING
    if failure_pattern is not None and failure_pattern.search(data):
        state = EntryState.FAILED
    match = pattern.search(data) if pattern else None
    matched = match.group(1) if match and match.groups() else (match.group(0) if match else "")
    progress = matched or data[:160]
    signature = f"{job_id}|{log_line.seq}"
    detail = f"finelog {job_id} ({log_line.cluster}): newest line: {data[:200]}"
    preemptions = None
    if lifecycle is not None:
        iris_state, iris_detail, preemptions = lifecycle[0], lifecycle[1], lifecycle[2] or None
        detail = f"{detail}; {iris_detail}"
        if state == EntryState.RUNNING:
            state = iris_state
    return Observation(state=state, progress=progress, signature=signature, detail=detail, preemptions=preemptions)


def acquire_tick_lock(state_dir: Path) -> int | None:
    """Non-blocking exclusive lock so launchd and manual runs never overlap."""
    state_dir.mkdir(parents=True, exist_ok=True)
    handle = os.open(state_dir / "tick.lock", os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(handle)
        return None
    return handle


def tick_once(registry: Registry, state_dir: Path, dashboard_path: Path, alerts_path: Path) -> TickResult:
    lock = acquire_tick_lock(state_dir)
    if lock is None:
        logger.warning("another tick is already running; skipping")
        raise SystemExit(0)
    try:
        state = load_state(state_dir / "state.json")
        now = datetime.now(UTC)
        snapshot = collect_fleet_snapshot(registry, now)
        sink = FanOutSink((JsonlAlertSink(alerts_path), DesktopNotificationSink()))
        runner = snapshot_probe_runner(snapshot)
        result = run_tick(registry, state, runner, sink, now, resolve_replacement=replacement_state_resolver(snapshot))
        save_state(state_dir / "state.json", state)
        append_history(state_dir / "history.jsonl", result.statuses)
        write_dashboard(dashboard_path, result.statuses, result.ticked_at)
        return result
    finally:
        os.close(lock)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="run a single tick and exit")
    mode.add_argument("--loop", action="store_true", help="run ticks forever, sleeping between them")
    parser.add_argument("--interval-minutes", type=float, default=5.0, help="loop mode: sleep between ticks")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="registry YAML path")
    parser.add_argument(
        "--state-dir", type=Path, default=DEFAULT_STATE_DIR, help="directory for state.json/history.jsonl"
    )
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD_PATH, help="dashboard.html output path")
    parser.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS_PATH, help="append-only alerts.jsonl path")
    parser.add_argument("--verbose", action="store_true", help="debug logging including probe commands")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    registry = load_registry(args.registry)

    if args.once:
        result = tick_once(registry, args.state_dir, args.dashboard, args.alerts)
        for status in result.statuses:
            logger.info(
                "%s: state=%s progress=%s alerts=%s",
                status.id,
                status.state.value,
                status.progress,
                ",".join(status.alerts) or "none",
            )
        logger.info("tick complete: %d entries, %d alerts fired", len(result.statuses), len(result.fired))
        return
    while True:
        try:
            tick_once(registry, args.state_dir, args.dashboard, args.alerts)
        except SystemExit:
            raise
        except Exception:
            logger.exception("tick failed; continuing loop")
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
