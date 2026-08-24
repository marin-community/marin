# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build bounded, advisory operator context for hero-run Loom alerts.

Context collection fails open so an unavailable evidence source never blocks
the alert, and callers must treat the result as a first pass rather than a
diagnosis.
"""

import asyncio
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import pyarrow as pa
from config import HERO_NOTIFICATION, HERO_OPERATOR_BEHAVIOR, OPERATOR_BEHAVIOR_LABEL
from hero_runs import as_utc, hero_run_id, root_job_for, sql_epoch_ms, sql_timestamp
from rigging.redaction import REDACTED_VALUE, is_sensitive_key_name, redact_string
from vllm_observability import sql_string

CONTEXT_LOOKBACK = timedelta(minutes=70)
LOG_ANCHOR_RADIUS = timedelta(minutes=2)
MAX_RECENT_EXECUTIONS = 4
MAX_LOG_ANCHORS = 6
MAX_LOG_ROWS_PER_ANCHOR = 100
MAX_LOG_EXCERPTS_PER_ANCHOR = 4
MAX_LOG_EXCERPTS = 20
MAX_EVENT_GROUPS = 20
MAX_STATE_ROWS = 12
MAX_CONTEXT_BYTES = 24_000
LOW_FANOUT_MAX_TASKS = 4
LOW_FANOUT_FRACTION_DENOMINATOR = 3

_CONTEXT_METRICS = (
    "phase",
    "step",
    "progress_time_seconds",
    "train_loss",
    "grad_norm_total",
    "optim_skipped_step",
    "moe_drop_fraction",
    "train_router_routing_entropy_mean",
    "train_router_bias_max",
    "train_router_bias_min",
    "throughput_tokens_per_second",
    "throughput_mfu",
    "memory_peak_gib",
    "eval_paloma_macro_loss",
)
_SAFE_RUN = re.compile(r"hero-[A-Za-z0-9_.-]+")
_SAFE_JOB = re.compile(r"/[A-Za-z0-9_.@/-]+")
_RANK_PREFIX = re.compile(r"^\[rank\d+\]\s*")
_GLOG_PREFIX = re.compile(r"\b[DIWEF]\d{4}\s+\d\d:\d\d:\d\d\.\d+\s+\d+\s+")
_PYTHON_LOG_PREFIX = re.compile(r"^\d{4}-\d\d-\d\d\s+\d\d:\d\d:\d\d,\d+\s+(?:DEBUG|INFO|WARNING|ERROR)\s+")
_LOCAL_RANK = re.compile(r"\blocal rank \d+\b", re.IGNORECASE)
_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
_LONG_NUMBER = re.compile(r"\b\d{4,}\b")
_ASSIGNMENT = re.compile(r"(?P<prefix>\b(?P<key>[A-Za-z][A-Za-z0-9_-]*)\s*[:=]\s*)[^\s,;]+")
_BEARER_CREDENTIAL = re.compile(r"(?i)(?P<prefix>\bBearer\s+)[A-Za-z0-9._~+/-]+=*")
_URL_CREDENTIALS = re.compile(r"(?P<scheme>https?://)[^\s/@:]+:[^\s/@]+@")


class MetricSource(Protocol):
    def query(self, sql: str, *, max_rows: int) -> pa.Table: ...


@dataclass(frozen=True)
class HeroAlertIdentity:
    cluster: str
    run_id: str
    root_jobs: tuple[str, ...]


def hero_alert_identity(alerts: Sequence[Mapping[str, object]]) -> HeroAlertIdentity:
    """Extract one logical run identity from a Grafana hero-run group."""
    clusters: set[str] = set()
    runs: set[str] = set()
    jobs: set[str] = set()
    for alert in alerts:
        labels = alert.get("labels")
        if not isinstance(labels, Mapping):
            raise ValueError("hero alert has no labels")
        notification = str(labels.get("notification", ""))
        behavior = str(labels.get(OPERATOR_BEHAVIOR_LABEL, ""))
        cluster = str(labels.get("cluster", ""))
        run_id = str(labels.get("run", ""))
        root_job = str(labels.get("job", ""))
        if (
            notification != HERO_NOTIFICATION
            or behavior != HERO_OPERATOR_BEHAVIOR
            or not cluster
            or not run_id
            or not root_job
        ):
            raise ValueError("hero alerts require hero notification and operator behavior, cluster, run, and job labels")
        if (
            len(cluster) > 128
            or len(run_id) > 128
            or len(root_job) > 512
            or not _SAFE_RUN.fullmatch(run_id)
            or not _SAFE_JOB.fullmatch(root_job)
        ):
            raise ValueError("hero alert carries an invalid run or job identity")
        if hero_run_id(root_job) != run_id:
            raise ValueError("hero alert job and run labels disagree")
        clusters.add(cluster)
        runs.add(run_id)
        jobs.add(root_job)
    if len(clusters) != 1 or len(runs) != 1 or not jobs:
        raise ValueError("hero alert group spans more than one cluster or run")
    return HeroAlertIdentity(cluster=clusters.pop(), run_id=runs.pop(), root_jobs=tuple(sorted(jobs)))


def telemetry_context_query(identity: HeroAlertIdentity, now: datetime) -> str:
    """Newest values and execution bounds for the run's recent attempts."""
    start = sql_epoch_ms(now - CONTEXT_LOOKBACK)
    end = sql_epoch_ms(now)
    metrics = ", ".join(sql_string(name) for name in _CONTEXT_METRICS)
    return (
        "WITH scoped AS ("
        "SELECT execution_uid, name, value, timestamp_ms, seq "
        'FROM "telemetry_v1" '
        f"WHERE cluster = {sql_string(identity.cluster)} AND run_id = {sql_string(identity.run_id)} "
        "AND service = 'levanter' AND process_index = '0' AND execution_uid IS NOT NULL "
        f"AND name IN ({metrics}) AND timestamp_ms >= {start} AND timestamp_ms < {end}"
        "), execution_bounds AS ("
        "SELECT execution_uid, MIN(timestamp_ms) AS first_ms, MAX(timestamp_ms) AS last_ms "
        "FROM scoped GROUP BY execution_uid"
        "), recent_executions AS ("
        "SELECT execution_uid, first_ms, last_ms, ROW_NUMBER() OVER (ORDER BY last_ms DESC) AS execution_rank "
        "FROM execution_bounds"
        "), ranked AS ("
        "SELECT scoped.execution_uid, scoped.name, scoped.value, scoped.timestamp_ms, scoped.seq, "
        "recent_executions.first_ms, recent_executions.last_ms, recent_executions.execution_rank, "
        "ROW_NUMBER() OVER (PARTITION BY scoped.execution_uid, scoped.name "
        "ORDER BY scoped.timestamp_ms DESC, scoped.seq DESC) AS metric_rank "
        "FROM scoped JOIN recent_executions ON scoped.execution_uid = recent_executions.execution_uid "
        f"WHERE recent_executions.execution_rank <= {MAX_RECENT_EXECUTIONS}"
        ") "
        "SELECT execution_uid, name, value, to_timestamp_millis(timestamp_ms) AS observed_at, "
        "to_timestamp_millis(first_ms) AS execution_first_at, "
        "to_timestamp_millis(last_ms) AS execution_last_at, execution_rank "
        "FROM ranked WHERE metric_rank = 1 ORDER BY execution_rank, name"
    )


def task_state_context_query(identity: HeroAlertIdentity, now: datetime) -> str:
    jobs = ", ".join(sql_string(job) for job in identity.root_jobs)
    return (
        "WITH ranked AS ("
        "SELECT root_job_id, ts, running, ROW_NUMBER() OVER ("
        "PARTITION BY root_job_id ORDER BY ts DESC) AS rn "
        'FROM "iris.task_state" '
        f"WHERE cluster = {sql_string(identity.cluster)} AND root_job_id IN ({jobs}) "
        f"AND ts >= TIMESTAMP '{sql_timestamp(now - CONTEXT_LOOKBACK)}' "
        f"AND ts < TIMESTAMP '{sql_timestamp(now)}'"
        ") SELECT root_job_id AS job, ts AS observed_at, running FROM ranked "
        f"WHERE rn <= 6 ORDER BY observed_at DESC LIMIT {MAX_STATE_ROWS}"
    )


def task_event_context_query(identity: HeroAlertIdentity, now: datetime) -> str:
    task_predicate = " OR ".join(f"task_id LIKE {sql_string(job + '/%')}" for job in identity.root_jobs)
    return (
        "SELECT reason, type, source, MIN(ts) AS first_at, MAX(ts) AS last_at, "
        "COUNT(*) AS event_count, COUNT(DISTINCT task_id) AS affected_tasks, "
        "MIN(task_id) AS sample_task, MIN(message) AS sample_message "
        'FROM "iris.task_event" '
        f"WHERE cluster = {sql_string(identity.cluster)} AND ({task_predicate}) "
        f"AND ts >= TIMESTAMP '{sql_timestamp(now - CONTEXT_LOOKBACK)}' "
        f"AND ts < TIMESTAMP '{sql_timestamp(now)}' "
        "GROUP BY reason, type, source ORDER BY last_at DESC "
        f"LIMIT {MAX_EVENT_GROUPS}"
    )


def log_context_query(identity: HeroAlertIdentity, anchors: Sequence[datetime]) -> str:
    """Rare, low-fanout warning/error lines near evidence-derived anchors."""
    values = ", ".join(
        f"(TIMESTAMP '{sql_timestamp(anchor)}', {round(anchor.timestamp() * 1000)})" for anchor in anchors
    )
    key_predicate = " OR ".join(f"key LIKE {sql_string(job + '/%')}" for job in identity.root_jobs)
    radius_ms = round(LOG_ANCHOR_RADIUS.total_seconds() * 1000)
    return (
        "WITH anchors(anchor_at, anchor_ms) AS (VALUES "
        f"{values}"
        "), scoped AS ("
        "SELECT anchors.anchor_at, anchors.anchor_ms, logs.key, logs.source, logs.data, logs.epoch_ms, logs.level "
        'FROM "log" AS logs CROSS JOIN anchors '
        f"WHERE logs.cluster = {sql_string(identity.cluster)} AND ({key_predicate}) "
        "AND (logs.level >= 3 OR logs.source = 'error') "
        f"AND logs.epoch_ms >= anchors.anchor_ms - {radius_ms} "
        f"AND logs.epoch_ms < anchors.anchor_ms + {radius_ms}"
        "), cardinality AS ("
        "SELECT anchor_at, COUNT(DISTINCT key) AS total_task_attempts FROM scoped GROUP BY anchor_at"
        "), grouped AS ("
        "SELECT anchor_at, anchor_ms, source, level, data, COUNT(*) AS occurrences, "
        "COUNT(DISTINCT key) AS task_attempts, MIN(key) AS sample_key, MIN(epoch_ms) AS first_ms "
        "FROM scoped GROUP BY anchor_at, anchor_ms, source, level, data"
        "), ranked AS ("
        "SELECT grouped.*, cardinality.total_task_attempts, "
        "ROW_NUMBER() OVER (PARTITION BY grouped.anchor_at ORDER BY grouped.task_attempts, "
        "ABS(grouped.first_ms - grouped.anchor_ms), grouped.occurrences, grouped.first_ms) AS candidate_rank "
        "FROM grouped JOIN cardinality ON grouped.anchor_at = cardinality.anchor_at "
        f"WHERE cardinality.total_task_attempts <= {LOW_FANOUT_MAX_TASKS} "
        f"OR grouped.task_attempts * {LOW_FANOUT_FRACTION_DENOMINATOR} "
        "<= cardinality.total_task_attempts"
        ") "
        "SELECT anchor_at, to_timestamp_millis(first_ms) AS observed_at, source, level, data AS message, "
        "occurrences, task_attempts, total_task_attempts, sample_key "
        f"FROM ranked WHERE candidate_rank <= {MAX_LOG_ROWS_PER_ANCHOR} "
        "ORDER BY anchor_at DESC, candidate_rank"
    )


def normalize_log_message(message: str) -> str:
    """Collapse volatile rank/time/address fields for generic deduplication."""
    normalized = _RANK_PREFIX.sub("", message)
    normalized = _GLOG_PREFIX.sub("<glog> ", normalized)
    normalized = _PYTHON_LOG_PREFIX.sub("<log> ", normalized)
    normalized = _LOCAL_RANK.sub("local rank <n>", normalized)
    normalized = _UUID.sub("<uuid>", normalized)
    normalized = _HEX.sub("<hex>", normalized)
    normalized = _LONG_NUMBER.sub("<n>", normalized)
    return " ".join(normalized.split())


def root_job_from_execution_uid(execution_uid: str) -> str | None:
    """Recover the coordinator root carried in Iris's telemetry execution UID."""
    if not execution_uid.startswith("iris:") or ":attempt:" not in execution_uid:
        return None
    task_id = execution_uid.removeprefix("iris:").rsplit(":attempt:", 1)[0]
    root_job = root_job_for(task_id)
    return root_job if root_job is not None and _SAFE_JOB.fullmatch(root_job) else None


def _text(value: object, limit: int) -> str:
    result = str(value or "").replace("\x00", "")
    return result if len(result) <= limit else f"{result[: limit - 1]}…"


def _log_text(value: object, limit: int) -> str:
    """Redact free-form output using shared secret classification plus log syntax."""
    result = str(value or "").replace("\x00", "")
    result = _ASSIGNMENT.sub(
        lambda match: (
            f"{match.group('prefix')}{REDACTED_VALUE}" if is_sensitive_key_name(match.group("key")) else match.group(0)
        ),
        result,
    )
    result = _BEARER_CREDENTIAL.sub(rf"\g<prefix>{REDACTED_VALUE}", result)
    result = _URL_CREDENTIALS.sub(rf"\g<scheme>{REDACTED_VALUE}@", result)
    result = redact_string(result)
    return result if len(result) <= limit else f"{result[: limit - 1]}…"


def _iso(value: object) -> str:
    if not isinstance(value, datetime):
        return _text(value, 80)
    return as_utc(value).isoformat()


def select_log_evidence(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Dedupe generic low-fanout candidates while reserving space per anchor."""
    selected: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    per_anchor: dict[str, int] = {}
    for row in rows:
        task_attempts = int(row.get("task_attempts") or 0)
        total_task_attempts = int(row.get("total_task_attempts") or 0)
        if (
            total_task_attempts > LOW_FANOUT_MAX_TASKS
            and task_attempts * LOW_FANOUT_FRACTION_DENOMINATOR > total_task_attempts
        ):
            continue
        anchor = _iso(row.get("anchor_at"))
        message = _log_text(row.get("message"), 700)
        template = normalize_log_message(message)
        identity = (anchor, template)
        if not template or identity in seen or per_anchor.get(anchor, 0) >= MAX_LOG_EXCERPTS_PER_ANCHOR:
            continue
        seen.add(identity)
        per_anchor[anchor] = per_anchor.get(anchor, 0) + 1
        selected.append(
            {
                "anchorAt": anchor,
                "observedAt": _iso(row.get("observed_at")),
                "source": _text(row.get("source"), 40),
                "level": int(row.get("level") or 0),
                "message": message,
                "occurrences": int(row.get("occurrences") or 0),
                "taskAttempts": task_attempts,
                "totalTaskAttempts": total_task_attempts,
                "sampleKey": _text(row.get("sample_key"), 300),
            }
        )
        if len(selected) >= MAX_LOG_EXCERPTS:
            break
    return selected


@dataclass(frozen=True)
class _ExecutionContext:
    executions: list[dict[str, object]]
    anchors: list[datetime]
    root_jobs: set[str]


@dataclass(frozen=True)
class _EventContext:
    events: list[dict[str, object]]
    anchors: list[datetime]


@dataclass(frozen=True)
class _QueryResult:
    rows: list[dict[str, object]]
    error: str | None


def _execution_context(rows: Sequence[Mapping[str, object]], run_id: str) -> _ExecutionContext:
    """Project telemetry into attempts, log anchors, and discovered Iris roots."""
    executions: dict[str, dict[str, object]] = {}
    anchors: list[datetime] = []
    root_jobs: set[str] = set()
    for row in rows:
        execution_uid = str(row.get("execution_uid") or "")
        if not execution_uid:
            continue
        root_job = root_job_from_execution_uid(execution_uid)
        if root_job is not None and hero_run_id(root_job) == run_id:
            root_jobs.add(root_job)
        execution = executions.setdefault(
            execution_uid,
            {
                "executionUid": _text(execution_uid, 400),
                "rank": int(row.get("execution_rank") or 0),
                "firstObservedAt": _iso(row.get("execution_first_at")),
                "lastObservedAt": _iso(row.get("execution_last_at")),
                "latestMetrics": {},
            },
        )
        value = row.get("value")
        if isinstance(value, int | float) and math.isfinite(float(value)):
            execution["latestMetrics"][str(row.get("name"))] = {
                "value": float(value),
                "observedAt": _iso(row.get("observed_at")),
            }
        last_at = row.get("execution_last_at")
        if isinstance(last_at, datetime):
            anchors.append(as_utc(last_at))
    ordered = sorted(executions.values(), key=lambda item: int(item["rank"]))
    return _ExecutionContext(executions=ordered, anchors=anchors, root_jobs=root_jobs)


def _event_context(rows: Sequence[Mapping[str, object]]) -> _EventContext:
    """Project Iris event groups and retain their end times as log anchors."""
    events: list[dict[str, object]] = []
    anchors: list[datetime] = []
    for row in rows[:MAX_EVENT_GROUPS]:
        events.append(
            {
                "reason": _text(row.get("reason"), 120),
                "type": _text(row.get("type"), 40),
                "source": _text(row.get("source"), 80),
                "firstAt": _iso(row.get("first_at")),
                "lastAt": _iso(row.get("last_at")),
                "eventCount": int(row.get("event_count") or 0),
                "affectedTasks": int(row.get("affected_tasks") or 0),
                "sampleTask": _text(row.get("sample_task"), 300),
                "sampleMessage": _log_text(row.get("sample_message"), 500),
            }
        )
        last_at = row.get("last_at")
        if isinstance(last_at, datetime):
            anchors.append(as_utc(last_at))
    return _EventContext(events=events, anchors=anchors)


def _dedupe_anchors(values: Sequence[datetime]) -> list[datetime]:
    selected: list[datetime] = []
    for value in values:
        if all(abs((value - existing).total_seconds()) >= 30 for existing in selected):
            selected.append(value)
        if len(selected) >= MAX_LOG_ANCHORS:
            break
    return selected


def _fit_context(context: dict[str, object]) -> dict[str, object]:
    """Enforce the prompt budget by dropping lowest-priority evidence tails."""
    omitted = {"logExcerpts": 0, "taskEvents": 0, "taskStates": 0}
    context["budgetOmissions"] = omitted
    sections = (
        (context["logEvidence"]["excerpts"], "logExcerpts"),
        (context["taskEvents"], "taskEvents"),
        (context["taskStates"], "taskStates"),
    )
    while len(json.dumps(context, sort_keys=True).encode()) > MAX_CONTEXT_BYTES:
        for values, name in sections:
            if values:
                values.pop()
                omitted[name] += 1
                break
        else:
            break
    if not any(omitted.values()):
        del context["budgetOmissions"]
    return context


class HeroAlertContextAssembler:
    """Query the federation hub without making alert delivery depend on it."""

    def __init__(self, source: MetricSource, *, max_rows: int) -> None:
        self._source = source
        self._max_rows = max_rows

    async def _query(self, name: str, sql: str, max_rows: int) -> _QueryResult:
        """Return rows or an error-class sentinel so other evidence remains usable."""
        try:
            table = await asyncio.to_thread(self._source.query, sql, max_rows=min(max_rows, self._max_rows))
            return _QueryResult(rows=table.to_pylist(), error=None)
        except Exception as err:
            return _QueryResult(rows=[], error=f"{name}: {type(err).__name__}")

    async def assemble(self, alerts: list[Mapping[str, object]]) -> Mapping[str, object]:
        """Build versioned, bounded context, marking individual query failures partial."""
        now = datetime.now(UTC)
        identity = hero_alert_identity(alerts)
        telemetry_result, state_result, event_result = await asyncio.gather(
            self._query("telemetry", telemetry_context_query(identity, now), 500),
            self._query("task_state", task_state_context_query(identity, now), MAX_STATE_ROWS),
            self._query("task_event", task_event_context_query(identity, now), MAX_EVENT_GROUPS),
        )
        execution_context = _execution_context(telemetry_result.rows, identity.run_id)
        evidence_identity = HeroAlertIdentity(
            cluster=identity.cluster,
            run_id=identity.run_id,
            root_jobs=tuple(sorted({*identity.root_jobs, *execution_context.root_jobs})),
        )
        state_rows = state_result.rows
        state_error = state_result.error
        event_rows = event_result.rows
        event_error = event_result.error
        if evidence_identity.root_jobs != identity.root_jobs:
            expanded_state, expanded_events = await asyncio.gather(
                self._query("task_state", task_state_context_query(evidence_identity, now), MAX_STATE_ROWS),
                self._query("task_event", task_event_context_query(evidence_identity, now), MAX_EVENT_GROUPS),
            )
            state_error = expanded_state.error
            event_error = expanded_events.error
            if state_error is None:
                state_rows = expanded_state.rows
            if event_error is None:
                event_rows = expanded_events.rows
        event_context = _event_context(event_rows)
        anchors = _dedupe_anchors([*execution_context.anchors, *event_context.anchors, now])
        log_result = await self._query(
            "logs",
            log_context_query(evidence_identity, anchors),
            MAX_LOG_ANCHORS * MAX_LOG_ROWS_PER_ANCHOR,
        )
        errors = [error for error in (telemetry_result.error, state_error, event_error, log_result.error) if error]
        context: dict[str, object] = {
            "schemaVersion": 1,
            "status": "partial" if errors else "complete",
            "assembledAt": now.isoformat(),
            "scope": {
                "cluster": identity.cluster,
                "run": identity.run_id,
                "alertRootJobs": list(identity.root_jobs),
                "evidenceRootJobs": list(evidence_identity.root_jobs),
                "lookbackMinutes": round(CONTEXT_LOOKBACK.total_seconds() / 60),
            },
            "recentExecutions": execution_context.executions,
            "taskStates": [
                {
                    "job": _text(row.get("job"), 300),
                    "observedAt": _iso(row.get("observed_at")),
                    "running": int(row.get("running") or 0),
                }
                for row in state_rows[:MAX_STATE_ROWS]
            ],
            "taskEvents": event_context.events,
            "logEvidence": {
                "strategy": (
                    "warning/error output near recent execution and task-event boundaries, retaining messages "
                    "localized to a minority of task attempts and deduplicating volatile rank/time/address fields"
                ),
                "anchors": [_iso(anchor) for anchor in anchors],
                "excerpts": select_log_evidence(log_result.rows),
            },
            "collectionErrors": errors,
            "caveat": (
                "This is a bounded first pass, not an exhaustive log search or a diagnosis. A delegated child "
                "session should gather additional current evidence as needed."
            ),
        }
        return _fit_context(context)
