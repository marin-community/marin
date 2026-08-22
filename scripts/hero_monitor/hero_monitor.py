#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hero-run Pushover monitor.

Polls finelog (the same store Grafana's training-run dashboard reads) for the hero
training run's telemetry and sends Pushover notifications:

  - EMERGENCY (priority 2, repeats until acked): run down / telemetry gone / stalled
    past the page threshold / non-finite loss.
  - HIGH (priority 1): loss spike (Grafana's 6-sigma rule), loss jump > 1, grad-norm
    breach, repeated skipped optimizer steps.
  - NORMAL (priority 0): token-drop fraction, router entropy/bias, throughput sag,
    eval regression, new Iris job or task attempt, or an empty-window digest.
  - LOW (priority -1): a healthy status digest every DIGEST_MINUTES.

Run it from a persistent box that has the marin checkout and a cached Marin IAP
credential (run `uv run iris login` once, or provide ambient service-account ADC):

    cd <marin repo> && uv run scripts/hero_monitor/hero_monitor.py

PUSHOVER_APP_TOKEN and PUSHOVER_USER_KEY enable delivery; without them the monitor
logs a dry run. HERO_RUN_ID, MARIN_REPO, FINELOG_NAME, POLL_SECONDS,
DIGEST_MINUTES, HERO_MON_STATE, and COOLDOWN_SECONDS override runtime defaults.
The default poll and digest intervals are two and ten minutes. The monitor requires
only the Python stdlib; finelog is queried by shelling out to `uv run finelog query`.

State (alert cooldowns, last digest, baselines) is kept in a JSON file so the
monitor survives restarts without re-firing everything.
"""

import json
import math
import os
import signal
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

# ----------------------------------------------------------------------------- CONFIG

DEFAULT_RUN_ID = "hero-12d8b6f0-dee637"
UNKNOWN_CLUSTER = "unknown"
TELEMETRY_LOOKBACK_MINUTES = 65
HEALTH_WINDOW = 15 * 60
JOB_HISTORY_LIMIT = 50


@dataclass(frozen=True)
class MonitorConfig:
    """Runtime inputs resolved once by the monitor entry point."""

    pushover_app_token: str
    pushover_user_key: str
    run_id: str
    marin_repo: str
    finelog_name: str
    poll_interval: int
    digest_interval: int
    state_file: str
    cooldown: int

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> "MonitorConfig":
        """Resolve the command-line process configuration from environment variables."""
        return cls(
            pushover_app_token=environ.get("PUSHOVER_APP_TOKEN", ""),
            pushover_user_key=environ.get("PUSHOVER_USER_KEY", ""),
            run_id=environ.get("HERO_RUN_ID", DEFAULT_RUN_ID),
            marin_repo=environ.get("MARIN_REPO", os.path.expanduser("~/marin")),
            finelog_name=environ.get("FINELOG_NAME", "marin"),
            poll_interval=int(environ.get("POLL_SECONDS", "120")),
            digest_interval=int(environ.get("DIGEST_MINUTES", "10")) * 60,
            state_file=environ.get("HERO_MON_STATE", os.path.expanduser("~/.hero_pushover_state.json")),
            cooldown=int(environ.get("COOLDOWN_SECONDS", "1800")),
        )


@dataclass(frozen=True)
class AlertThresholds:
    """Alert thresholds from the Grafana contracts and hero on-call policy."""

    telemetry_gone: int = 600
    stall_page: int = 900
    initializing_page: int = 2700
    task_state_fresh: int = 90
    task_state_degraded: int = 300
    loss_jump: float = 1.0
    sigma_factor: float = 6.0
    min_rise: float = 0.05
    grad_norm_max: float = 2.0
    stall_warn: int = 300
    drop_fraction_max: float = 0.05
    router_entropy_min: float = 5.92
    router_bias_max: float = 400.0
    tokens_per_second_min: float = 2.0e6
    mfu_min: float = 15.0


THRESHOLDS = AlertThresholds()

# finelog metric names (normalized: non-alphanumerics -> "_"). All verified live.
METRICS = [
    "phase",
    "step",
    "run_progress",
    "progress_time_seconds",
    "train_loss",
    "grad_norm_total",
    "throughput_tokens_per_second",
    "throughput_mfu",
    "throughput_duration",
    "moe_drop_fraction",
    "moe_sender_drop_fraction",
    "moe_receiver_drop_fraction",
    "train_router_routing_entropy_mean",
    "train_router_bias_max",
    "train_router_bias_min",
    "memory_peak_gib",
    "memory_limit_gib",
    "optim_skipped_step",
    "eval_paloma_macro_loss",
]

FINAL_STEP = 390_251  # d6144 stop step (schedule == stop for the hero rung)
STATE_VERSION = 6


@dataclass(frozen=True)
class ExecutionSnapshot:
    """The current Iris job plus its independently retried task attempts."""

    cluster: str
    job_id: str
    selected_execution_uid: str
    phase_observed_since: float
    attempts: dict[str, int]


@dataclass(frozen=True)
class TaskStateSnapshot:
    """One root row from Iris's active-task rollup."""

    root_job_id: str
    age_seconds: float
    pending: int
    assigned: int
    building: int
    running: int

    @property
    def active_count(self):
        return self.pending + self.assigned + self.building + self.running


@dataclass(frozen=True)
class AlertStats:
    """Raw-step loss windows plus the exact count of skipped optimizer steps."""

    baseline_samples: int
    baseline_loss: float | None
    baseline_stddev: float | None
    baseline_floor: float | None
    recent_samples: int
    recent_loss: float | None
    recent_floor: float | None
    recent_peak: float | None
    recent_skips: float


@dataclass(frozen=True)
class RetryEvent:
    """Iris controller's durable decision to retry or gang-requeue an attempt."""

    key: str
    task_id: str
    attempt_id: int
    observed_at: float


@dataclass
class MonitorState:
    """Persistent cooldown, execution, and notification state."""

    state_version: int = STATE_VERSION
    last_fired: dict[str, float] = field(default_factory=dict)
    last_digest: float = 0
    last_job_id: str | None = None
    seen_job_ids: list[str] = field(default_factory=list)
    attempts_by_job: dict[str, dict[str, int]] = field(default_factory=dict)
    retry_event_keys_by_job: dict[str, list[str]] = field(default_factory=dict)
    notified_attempts_by_job: dict[str, dict[str, int]] = field(default_factory=dict)
    last_eval_paloma: float | None = None
    restart_count: int = 0
    query_failures: int = 0
    run_finished: bool = False


# ----------------------------------------------------------------------------- helpers


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_state(config: MonitorConfig) -> MonitorState:
    """Load state, migrating the deployed monitor's earlier JSON schemas."""
    try:
        with open(config.state_file) as state_stream:
            raw: Any = json.load(state_stream)
    except FileNotFoundError:
        return MonitorState()
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot load monitor state {config.state_file}") from error
    if not isinstance(raw, dict):
        raise RuntimeError(f"monitor state {config.state_file} is not a JSON object")

    state_version = int(raw.get("state_version", 1))
    # Version 1 compared one arbitrarily selected execution_uid. An execution_uid
    # identifies one task attempt, so a healthy 176-task gang appeared to "restart"
    # whenever a different task emitted the newest row. Its counts are unusable.
    if state_version < 2:
        raw["last_fired"] = {
            key: value for key, value in raw.get("last_fired", {}).items() if not key.startswith("attempt_")
        }
        raw.pop("last_execution_uid", None)
        raw["last_job_id"] = None
        raw["seen_job_ids"] = []
        raw["restart_count"] = 0

    # Version 2 tracked only job changes. Preserve its valid job-level state and
    # silently establish per-task attempt baselines on the first v3 poll.
    if state_version < 3:
        raw["attempts_by_job"] = {}

    if state_version < 4:
        raw["run_finished"] = False

    # Version 5 uses controller-authored retry/requeue events rather than
    # inferring retries solely from application telemetry. Establish a baseline.
    if state_version < 5:
        raw["retry_event_keys_by_job"] = {}

    # Version 6 cross-checks retained controller events with observed phase
    # attempt increments, without notifying twice when both describe one retry.
    if state_version < 6:
        raw["notified_attempts_by_job"] = {}

    def nested_attempts(name: str) -> dict[str, dict[str, int]]:
        return {
            str(job_id): {str(task_id): int(attempt_id) for task_id, attempt_id in attempts.items()}
            for job_id, attempts in raw.get(name, {}).items()
        }

    return MonitorState(
        last_fired={str(key): float(value) for key, value in raw.get("last_fired", {}).items()},
        last_digest=float(raw.get("last_digest", 0)),
        last_job_id=raw.get("last_job_id"),
        seen_job_ids=[str(job_id) for job_id in raw.get("seen_job_ids", [])],
        attempts_by_job=nested_attempts("attempts_by_job"),
        retry_event_keys_by_job={
            str(job_id): [str(key) for key in keys] for job_id, keys in raw.get("retry_event_keys_by_job", {}).items()
        },
        notified_attempts_by_job=nested_attempts("notified_attempts_by_job"),
        last_eval_paloma=(float(raw["last_eval_paloma"]) if raw.get("last_eval_paloma") is not None else None),
        restart_count=int(raw.get("restart_count", 0)),
        query_failures=int(raw.get("query_failures", 0)),
        run_finished=bool(raw.get("run_finished", False)),
    )


def save_state(config: MonitorConfig, state: MonitorState) -> None:
    """Atomically persist the monitor state."""
    temporary_path = config.state_file + ".tmp"
    with open(temporary_path, "w") as state_stream:
        json.dump(asdict(state), state_stream)
    os.replace(temporary_path, config.state_file)


def pushover(config: MonitorConfig, title: str, message: str, priority: int = 0) -> bool:
    """Send one Pushover notification, returning whether delivery succeeded."""
    if not config.pushover_app_token or not config.pushover_user_key:
        log(f"DRY-RUN pushover p{priority}: {title}: {message}")
        return True
    data = {
        "token": config.pushover_app_token,
        "user": config.pushover_user_key,
        "title": title,
        "message": message[:1024],
        "priority": priority,
    }
    if priority == 2:
        data["retry"] = 60  # re-alert every 60s
        data["expire"] = 3600  # for up to an hour or until acked
    try:
        req = urllib.request.Request(
            "https://api.pushover.net/1/messages.json",
            data=urllib.parse.urlencode(data).encode(),
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            ok = resp.status == 200
    except Exception as e:
        log(f"pushover send failed: {e}")
        return False
    log(f"pushover p{priority} sent: {title}")
    return ok


def fire(
    config: MonitorConfig,
    state: MonitorState,
    key: str,
    title: str,
    message: str,
    priority: int,
) -> bool:
    """Fire an alert unless cooled down; false only when delivery failed."""
    now = time.time()
    last = state.last_fired.get(key, 0)
    if now - last < config.cooldown:
        return True
    if pushover(config, title, message, priority):
        state.last_fired[key] = now
        return True
    return False


def finelog_query(config: MonitorConfig, sql: str) -> list[dict[str, Any]]:
    """Run SQL against finelog; returns list of dict rows. Raises on failure."""
    out = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "finelog",
            "query",
            config.finelog_name,
            sql,
            "--format",
            "json",
            "--timeout",
            "30",
        ],
        cwd=config.marin_repo,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if out.returncode != 0:
        raise RuntimeError(f"finelog query failed: {out.stderr.strip()[-500:]}")
    # stderr may carry the harmless "missing scopes email" warning; stdout is JSON.
    return json.loads(out.stdout)


def sql_literal(value):
    """Quote a value for the small, read-only SQL strings sent to DataFusion."""
    return "'" + str(value).replace("'", "''") + "'"


def root_job_id(job_id):
    """Return Iris's /user/root-job identity, or None for a malformed job ID."""
    parts = job_id.split("/")
    if len(parts) < 3 or parts[0] != "" or not parts[1] or not parts[2]:
        return None
    return f"/{parts[1]}/{parts[2]}"


def timestamp_seconds(value: str | int | float) -> float:
    """Normalize Finelog JSON timestamps, which may be ISO text or epoch millis."""
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    return float(value) / 1000.0


def fetch_telemetry(
    config: MonitorConfig, execution: ExecutionSnapshot | None = None
) -> dict[str, list[tuple[int, float]]]:
    """Pull recent run scalars aggregated into 30-second buckets.

    (telemetry rows are per-process — 704 of them — so raw rows are far too many).
    Returns {name: [(ts_ms, value), ...]} sorted by time."""
    names = ",".join(sql_literal(n) for n in METRICS)
    identity_filter = ""
    if execution is not None:
        # Status belongs to one attempt. Scalar tracker metrics are emitted only by
        # global process zero, which need not be the attempt chosen by newest phase.
        status_names = ",".join(sql_literal(name) for name in ("phase", "step", "run_progress", "progress_time_seconds"))
        identity_filter = (
            f"AND COALESCE(NULLIF(cluster,''),{sql_literal(UNKNOWN_CLUSTER)})="
            f"{sql_literal(execution.cluster)} "
            f"AND job_id={sql_literal(execution.job_id)} "
            f"AND (name NOT IN ({status_names}) OR "
            f"execution_uid={sql_literal(execution.selected_execution_uid)}) "
        )
    sql = (
        "SELECT name, "
        "CAST(EXTRACT(EPOCH FROM date_bin(INTERVAL '30 seconds', "
        "to_timestamp_millis(timestamp_ms)))*1000 AS BIGINT) AS bucket_ms, "
        "AVG(value) AS avg_v, MIN(value) AS min_v, MAX(value) AS max_v "
        'FROM "telemetry_v1" '
        f"WHERE service='levanter' AND run_id={sql_literal(config.run_id)} "
        f"AND name IN ({names}) "
        f"{identity_filter}"
        f"AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL "
        f"'{TELEMETRY_LOOKBACK_MINUTES} minutes')*1000 AS BIGINT) "
        "GROUP BY name, bucket_ms ORDER BY bucket_ms"
    )
    take_max = {
        "step",
        "phase",
        "progress_time_seconds",
        "run_progress",
        "memory_peak_gib",
        "memory_limit_gib",
        "train_router_bias_max",
    }
    take_min = {"train_router_bias_min"}
    series = {}
    for row in finelog_query(config, sql):
        if row["name"] in take_max:
            v = row.get("max_v")
        elif row["name"] in take_min:
            v = row.get("min_v")
        else:
            v = row.get("avg_v")
        if v is None:
            continue
        series.setdefault(row["name"], []).append((row["bucket_ms"], float(v)))
    return series


def parse_execution_uid(execution_uid):
    """Return (task_id, attempt_id) for an Iris execution UID, if recognized."""
    prefix = "iris:"
    separator = ":attempt:"
    if not execution_uid.startswith(prefix):
        return None
    task_id, found, attempt_text = execution_uid[len(prefix) :].rpartition(separator)
    if not found or not task_id:
        return None
    try:
        attempt_id = int(attempt_text)
    except ValueError:
        return None
    if attempt_id < 0:
        return None
    return task_id, attempt_id


def fetch_execution_snapshot(config: MonitorConfig) -> ExecutionSnapshot | None:
    """Return the freshest training execution and max attempt per task.

    Iris keeps job_id stable while independently retrying tasks. execution_uid is
    ``iris:{task_id}:attempt:{attempt_id}``, so collecting every UID distinguishes
    a real attempt increment from normal telemetry arriving from different tasks.
    Levanter emits ``phase`` immediately and every minute, so it is the canonical
    attempt heartbeat; ``step`` is absent while an attempt initializes.
    """
    sql = (
        f"WITH phase AS (SELECT COALESCE(NULLIF(cluster,''),{sql_literal(UNKNOWN_CLUSTER)}) "
        "AS origin_cluster, "
        'job_id, execution_uid, timestamp_ms, seq FROM "telemetry_v1" '
        f"WHERE service='levanter' AND run_id={sql_literal(config.run_id)} AND name='phase' "
        "AND job_id IS NOT NULL AND execution_uid IS NOT NULL "
        f"AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL "
        f"'{TELEMETRY_LOOKBACK_MINUTES} minutes')*1000 AS BIGINT)) "
        ", ranked AS (SELECT origin_cluster, job_id, execution_uid, timestamp_ms, seq, "
        "ROW_NUMBER() OVER (PARTITION BY origin_cluster, job_id "
        "ORDER BY timestamp_ms DESC, seq DESC) AS rn FROM phase) "
        ", bounds AS (SELECT origin_cluster, job_id, execution_uid, "
        "MIN(timestamp_ms) AS first_ts, MAX(timestamp_ms) AS last_ts FROM phase "
        "GROUP BY origin_cluster, job_id, execution_uid) "
        "SELECT bounds.origin_cluster, bounds.job_id, bounds.execution_uid, "
        "bounds.first_ts, bounds.last_ts, ranked.timestamp_ms AS selected_ts, "
        "ranked.seq AS selected_seq, ranked.execution_uid AS selected_execution_uid "
        "FROM bounds JOIN ranked ON bounds.origin_cluster=ranked.origin_cluster "
        "AND bounds.job_id=ranked.job_id WHERE ranked.rn=1 "
        "ORDER BY selected_ts DESC, selected_seq DESC, bounds.execution_uid DESC"
    )
    rows = finelog_query(config, sql)
    parsed_rows = []
    for row in rows:
        parsed = parse_execution_uid(row["execution_uid"])
        if parsed is not None:
            parsed_rows.append((row, parsed))
    if not parsed_rows:
        return None

    latest_row, _ = max(
        parsed_rows,
        key=lambda item: (
            float(item[0]["selected_ts"]),
            int(item[0]["selected_seq"]),
            item[0]["origin_cluster"],
            item[0]["job_id"],
            item[0]["execution_uid"],
        ),
    )
    latest_job_id = latest_row["job_id"]
    selected_execution_uid = latest_row["selected_execution_uid"]
    attempts = {}
    selected_first_ts = None
    for row, parsed in parsed_rows:
        if row["origin_cluster"] != latest_row["origin_cluster"] or row["job_id"] != latest_job_id:
            continue
        task_id, attempt_id = parsed
        attempts[task_id] = max(attempt_id, attempts.get(task_id, -1))
        if row["execution_uid"] == selected_execution_uid:
            selected_first_ts = float(row["first_ts"])
    if selected_first_ts is None:
        return None
    return ExecutionSnapshot(
        cluster=latest_row["origin_cluster"],
        job_id=latest_job_id,
        selected_execution_uid=selected_execution_uid,
        phase_observed_since=selected_first_ts / 1000.0,
        attempts=attempts,
    )


def fetch_task_state(config: MonitorConfig, execution: ExecutionSnapshot | None = None) -> TaskStateSnapshot | None:
    """Return the latest matching Iris active-task aggregate.

    When an execution is available, match its exact root and cluster. Otherwise,
    match the newest root whose ID includes this monitor's run ID.
    """
    root = root_job_id(execution.job_id) if execution is not None else None
    root_filter = (
        f"root_job_id={sql_literal(root)}"
        if root is not None
        else f"root_job_id LIKE {sql_literal('%' + config.run_id + '%')}"
    )
    cluster_filter = (
        f"AND COALESCE(NULLIF(cluster,''),{sql_literal(UNKNOWN_CLUSTER)})=" f"{sql_literal(execution.cluster)} "
        if execution is not None
        else ""
    )
    sql = (
        'SELECT root_job_id, ts, pending, assigned, building, running FROM "iris.task_state" '
        f"WHERE {root_filter} "
        f"{cluster_filter}"
        "AND ts >= now() - INTERVAL '15 minutes' ORDER BY ts DESC LIMIT 1"
    )
    rows = finelog_query(config, sql)
    if not rows:
        return None
    row = rows[0]
    return TaskStateSnapshot(
        root_job_id=row["root_job_id"],
        age_seconds=max(0.0, time.time() - timestamp_seconds(row["ts"])),
        pending=int(row.get("pending") or 0),
        assigned=int(row.get("assigned") or 0),
        building=int(row.get("building") or 0),
        running=int(row.get("running") or 0),
    )


def fetch_retry_events(config: MonitorConfig, execution: ExecutionSnapshot | None) -> list[RetryEvent]:
    """Return Iris's retained controller decisions to retry/requeue tasks."""
    if execution is None:
        return []
    task_prefix = execution.job_id.rstrip("/") + "/%"
    sql = (
        'SELECT task_id, attempt_id, attempt_uid, ts FROM "iris.task_event" '
        f"WHERE task_id LIKE {sql_literal(task_prefix)} "
        f"AND COALESCE(NULLIF(cluster,''),{sql_literal(UNKNOWN_CLUSTER)})="
        f"{sql_literal(execution.cluster)} "
        "AND reason IN ('TaskRetryScheduled','CoscheduledSiblingRequeued') "
        "AND ts >= now() - INTERVAL '7 days' "
        "ORDER BY ts, task_id, attempt_id"
    )
    events = []
    for row in finelog_query(config, sql):
        observed_at = timestamp_seconds(row["ts"])
        attempt_uid = row.get("attempt_uid")
        key = f"uid:{attempt_uid}" if attempt_uid else f"legacy:{row['task_id']}:{int(row['attempt_id'])}:{observed_at}"
        events.append(
            RetryEvent(
                key=key,
                task_id=row["task_id"],
                attempt_id=int(row["attempt_id"]),
                observed_at=observed_at,
            )
        )
    return events


def fetch_alert_stats(config: MonitorConfig, execution: ExecutionSnapshot | None = None) -> AlertStats | None:
    """Fetch raw loss and skipped-step windows for the matching execution."""
    run = sql_literal(config.run_id)
    cluster_filter = (
        f"AND COALESCE(NULLIF(cluster,''),{sql_literal(UNKNOWN_CLUSTER)})=" f"{sql_literal(execution.cluster)} "
        if execution is not None
        else ""
    )
    recent_ms = "CAST(EXTRACT(EPOCH FROM now() - INTERVAL '5 minutes')*1000 AS BIGINT)"
    skips_ms = "CAST(EXTRACT(EPOCH FROM now() - INTERVAL '15 minutes')*1000 AS BIGINT)"
    sql = (
        'WITH samples AS (SELECT name, value, timestamp_ms FROM "telemetry_v1" '
        f"WHERE service='levanter' AND run_id={run} "
        f"{cluster_filter}"
        "AND name IN ('train_loss','optim_skipped_step') "
        "AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL '60 minutes')*1000 AS BIGINT)) "
        "SELECT "
        f"SUM(CASE WHEN name='train_loss' AND timestamp_ms < {recent_ms} THEN 1 ELSE 0 END) "
        "AS baseline_samples, "
        f"AVG(CASE WHEN name='train_loss' AND timestamp_ms < {recent_ms} THEN value END) "
        "AS baseline_loss, "
        f"STDDEV(CASE WHEN name='train_loss' AND timestamp_ms < {recent_ms} THEN value END) "
        "AS baseline_stddev, "
        f"MIN(CASE WHEN name='train_loss' AND timestamp_ms < {recent_ms} THEN value END) "
        "AS baseline_floor, "
        f"SUM(CASE WHEN name='train_loss' AND timestamp_ms >= {recent_ms} THEN 1 ELSE 0 END) "
        "AS recent_samples, "
        f"AVG(CASE WHEN name='train_loss' AND timestamp_ms >= {recent_ms} THEN value END) "
        "AS recent_loss, "
        f"MIN(CASE WHEN name='train_loss' AND timestamp_ms >= {recent_ms} THEN value END) "
        "AS recent_floor, "
        f"MAX(CASE WHEN name='train_loss' AND timestamp_ms >= {recent_ms} THEN value END) "
        "AS recent_peak, "
        f"SUM(CASE WHEN name='optim_skipped_step' AND timestamp_ms >= {skips_ms} "
        "THEN value ELSE 0 END) AS recent_skips FROM samples"
    )
    rows = finelog_query(config, sql)
    if not rows:
        return None
    row = rows[0]

    def number(name):
        value = row.get(name)
        return float(value) if value is not None else None

    return AlertStats(
        baseline_samples=int(row.get("baseline_samples") or 0),
        baseline_loss=number("baseline_loss"),
        baseline_stddev=number("baseline_stddev"),
        baseline_floor=number("baseline_floor"),
        recent_samples=int(row.get("recent_samples") or 0),
        recent_loss=number("recent_loss"),
        recent_floor=number("recent_floor"),
        recent_peak=number("recent_peak"),
        recent_skips=float(row.get("recent_skips") or 0),
    )


def latest_value(series: dict[str, list[tuple[int, float]]], name: str) -> float | None:
    """Return the newest value for a metric, or None when it is absent."""
    pts = series.get(name)
    return pts[-1][1] if pts else None


def latest_timestamp(series: dict[str, list[tuple[int, float]]], name: str) -> float | None:
    """Return the newest timestamp for a metric in epoch seconds, if present."""
    pts = series.get(name)
    return pts[-1][0] / 1000.0 if pts else None


# ----------------------------------------------------------------------------- checks


def check_liveness_and_stall(
    config: MonitorConfig,
    series: dict[str, list[tuple[int, float]]],
    task_state: TaskStateSnapshot | None,
    state: MonitorState,
    execution: ExecutionSnapshot | None = None,
) -> bool:
    """Apply the liveness contract and return whether fresh-data checks may run."""
    now = time.time()
    run = config.run_id

    # Keep phase/progress semantics aligned with
    # docs/ops/training-stall-alert-contract.md.
    phase_value = latest_value(series, "phase")
    phase = int(phase_value) if phase_value is not None else None
    step = latest_value(series, "step") or 0
    if phase == 2:
        # Telemetry deliberately stops after Levanter marks the tracker finished.
        # Remember that terminal state so aging data is not later called an outage.
        state.run_finished = True
        return False
    if phase in (0, 1) or step > 0:
        state.run_finished = False

    all_ts = [p[0] for pts in series.values() for p in pts]
    status_ts = latest_timestamp(series, "phase")
    sample_age = now - status_ts if status_ts is not None else now - max(all_ts) / 1000.0 if all_ts else math.inf

    if state.run_finished and not all_ts:
        return False

    if sample_age > THRESHOLDS.telemetry_gone:
        task_is_fresh = task_state is not None and task_state.age_seconds <= THRESHOLDS.task_state_fresh
        if task_state is None:
            detail = " Iris has no active-root row in the 15-minute query window."
        elif not task_is_fresh:
            detail = f" Iris's last active-root row is stale ({task_state.age_seconds:.0f}s old)."
        else:
            detail = (
                f" Iris still reports {task_state.active_count} active tasks "
                f"({task_state.running} running, {task_state.pending} pending, "
                f"{task_state.assigned} assigned, {task_state.building} building); "
                "this points to the telemetry path, not a confirmed job exit."
            )
        fire(
            config,
            state,
            "run_down",
            f"HERO DOWN? {run}",
            f"No telemetry for {sample_age / 60:.1f} min.{detail}",
            priority=2,
        )
        return False

    if task_state is None or task_state.age_seconds > THRESHOLDS.task_state_degraded:
        detail = (
            "no active-root row in 15 minutes"
            if task_state is None
            else f"last root row is {task_state.age_seconds / 60:.1f} minutes old"
        )
        fire(
            config,
            state,
            "task_state_stale",
            f"hero Iris state stale: {run}",
            f"Levanter phase telemetry is fresh, but iris.task_state has {detail}. "
            "The run is still emitting; investigate the controller/state telemetry path.",
            priority=0,
        )

    progress_timestamp = latest_value(series, "progress_time_seconds")
    attempt_age = max(0.0, now - execution.phase_observed_since) if execution is not None else 0.0
    is_training = phase == 1 or step > 0
    since_last_step = now - progress_timestamp if progress_timestamp and progress_timestamp > 0 else attempt_age
    if not is_training and attempt_age >= THRESHOLDS.initializing_page:
        fire(
            config,
            state,
            "initializing_stale",
            f"HERO INIT STALLED {run}",
            f"Current Iris task attempt has remained in Levanter initialization for "
            f"{attempt_age / 60:.1f} min (45 min threshold).",
            priority=2,
        )
    elif is_training and since_last_step >= THRESHOLDS.stall_page:
        fire(
            config,
            state,
            "stalled",
            f"HERO STALLED {run}",
            f"No completed step for {since_last_step / 60:.1f} min "
            f"(step {step:.0f}). Telemetry still flowing — live wedge or "
            "restart catch-up. Check `iris job describe` and cuMemAllocAsync in logs.",
            priority=2,
        )
    elif is_training and since_last_step >= THRESHOLDS.stall_warn:
        fire(
            config,
            state,
            "stall_warn",
            f"hero slow: {run}",
            f"{since_last_step / 60:.1f} min since last step.",
            priority=0,
        )
    return True


def check_loss_and_optimizer(
    config: MonitorConfig,
    series: dict[str, list[tuple[int, float]]],
    state: MonitorState,
    alert_stats: AlertStats | None,
) -> None:
    """Apply the raw-sample loss contract and optimizer safety checks."""
    run = config.run_id
    # Keep this raw-window rule aligned with docs/ops/training-loss-spike-alert.md.
    if alert_stats is not None and alert_stats.baseline_samples >= 20 and alert_stats.recent_samples >= 5:
        diverged = any(
            value is not None and not math.isfinite(value)
            for value in (alert_stats.recent_loss, alert_stats.recent_peak)
        )
        if diverged:
            fire(
                config,
                state,
                "loss_nan",
                f"HERO NON-FINITE LOSS {run}",
                "train_loss is NaN/Inf. crash_on_nan should kill the run imminently.",
                priority=2,
            )
        elif (
            alert_stats.recent_floor is not None
            and alert_stats.baseline_loss is not None
            and math.isfinite(alert_stats.baseline_loss)
        ):
            stddev = alert_stats.baseline_stddev
            spread = stddev if stddev is not None and math.isfinite(stddev) else 0.0
            threshold = alert_stats.baseline_loss + max(THRESHOLDS.min_rise, THRESHOLDS.sigma_factor * spread)
            if alert_stats.recent_floor > threshold:
                fire(
                    config,
                    state,
                    "loss_spike",
                    f"hero LOSS SPIKE {run}",
                    f"min(recent 5m)={alert_stats.recent_floor:.4f} > threshold "
                    f"{threshold:.4f}. Same raw-sample rule as Grafana TrainingLossSpike.",
                    priority=1,
                )

        if (
            alert_stats.recent_floor is not None
            and alert_stats.baseline_floor is not None
            and math.isfinite(alert_stats.recent_floor)
            and math.isfinite(alert_stats.baseline_floor)
            and alert_stats.recent_floor - alert_stats.baseline_floor > THRESHOLDS.loss_jump
        ):
            fire(
                config,
                state,
                "loss_jump",
                f"hero loss jump {run}",
                f"Loss floor jumped "
                f"{alert_stats.recent_floor - alert_stats.baseline_floor:+.2f} "
                "between the trailing baseline and recent five minutes.",
                priority=1,
            )

    gradient_norm = latest_value(series, "grad_norm_total")
    if gradient_norm is not None and gradient_norm > THRESHOLDS.grad_norm_max:
        fire(
            config,
            state,
            "grad_norm",
            f"hero grad norm {run}",
            f"grad_norm_total={gradient_norm:.3f} > {THRESHOLDS.grad_norm_max} "
            "(no grad clipping is configured on this run).",
            priority=1,
        )

    recent_skips = alert_stats.recent_skips if alert_stats is not None else 0
    if recent_skips >= 3:
        fire(
            config,
            state,
            "skipped_steps",
            f"hero skipped steps {run}",
            f"{recent_skips:.0f} optimizer steps skipped in 15 min.",
            priority=1,
        )


def check_training_health(
    config: MonitorConfig,
    series: dict[str, list[tuple[int, float]]],
    state: MonitorState,
) -> None:
    """Apply hero-specific routing, throughput, and evaluation checks."""
    now = time.time()
    run = config.run_id
    drop = latest_value(series, "moe_drop_fraction")
    if drop is not None and drop > THRESHOLDS.drop_fraction_max:
        fire(
            config,
            state,
            "drop_fraction",
            f"hero token drops {run}",
            f"moe_drop_fraction={drop:.3f} > {THRESHOLDS.drop_fraction_max}.",
            priority=0,
        )

    routing_entropy = latest_value(series, "train_router_routing_entropy_mean")
    if routing_entropy is not None and routing_entropy < THRESHOLDS.router_entropy_min:
        fire(
            config,
            state,
            "router_entropy",
            f"hero router entropy {run}",
            f"routing_entropy_mean={routing_entropy:.4f} < {THRESHOLDS.router_entropy_min} "
            "(uniform is 5.951; falling entropy = expert collapse).",
            priority=0,
        )

    bias_hi = latest_value(series, "train_router_bias_max")
    bias_lo = latest_value(series, "train_router_bias_min")
    bias_mag = (
        max(abs(b) for b in (bias_hi, bias_lo) if b is not None)
        if (bias_hi is not None or bias_lo is not None)
        else None
    )
    if bias_mag is not None and bias_mag > THRESHOLDS.router_bias_max:
        fire(
            config,
            state,
            "router_bias",
            f"hero router bias {run}",
            f"router bias magnitude {bias_mag:.1f} > {THRESHOLDS.router_bias_max}.",
            priority=0,
        )

    median_throughput = recent_median(series, "throughput_tokens_per_second", now)
    if median_throughput is not None:
        if median_throughput < THRESHOLDS.tokens_per_second_min:
            fire(
                config,
                state,
                "throughput",
                f"hero throughput {run}",
                f"median tokens/s over 15 min = {median_throughput / 1e6:.2f}M "
                f"(healthy ~2.5M, threshold {THRESHOLDS.tokens_per_second_min / 1e6:.1f}M).",
                priority=0,
            )

    median_mfu = recent_median(series, "throughput_mfu", now)
    if median_mfu is not None:
        if median_mfu < THRESHOLDS.mfu_min:
            fire(
                config,
                state,
                "mfu",
                f"hero MFU low {run}",
                f"median MFU over 15 min = {median_mfu:.1f}% " f"(threshold {THRESHOLDS.mfu_min:.1f}%).",
                priority=0,
            )

    evaluation_loss = latest_value(series, "eval_paloma_macro_loss")
    if evaluation_loss is not None:
        prev = state.last_eval_paloma
        if prev is not None and evaluation_loss > prev:
            fire(
                config,
                state,
                "eval_regress",
                f"hero eval regression {run}",
                f"paloma macro loss {prev:.4f} -> {evaluation_loss:.4f} (worse).",
                priority=0,
            )
        if prev != evaluation_loss:
            state.last_eval_paloma = evaluation_loss


def recent_median(series: dict[str, list[tuple[int, float]]], name: str, now: float) -> float | None:
    """Return a metric's upper median over the health window when well sampled."""
    values = [value for timestamp, value in series.get(name, []) if now - timestamp / 1000.0 <= HEALTH_WINDOW]
    if len(values) < 10:
        return None
    return sorted(values)[len(values) // 2]


def check(
    config: MonitorConfig,
    series: dict[str, list[tuple[int, float]]],
    task_state: TaskStateSnapshot | None,
    state: MonitorState,
    execution: ExecutionSnapshot | None = None,
    alert_stats: AlertStats | None = None,
) -> None:
    """Evaluate one coherent poll of run, Iris, and alert-window state."""
    if not check_liveness_and_stall(config, series, task_state, state, execution):
        return
    check_loss_and_optimizer(config, series, state, alert_stats)
    check_training_health(config, series, state)


def check_restart(
    config: MonitorConfig,
    state: MonitorState,
    snapshot: ExecutionSnapshot | None,
    retry_events: list[RetryEvent],
) -> None:
    if snapshot is None:
        return
    job_id = snapshot.job_id
    current_attempts = snapshot.attempts

    last_job_id = state.last_job_id
    seen_job_ids = state.seen_job_ids
    attempts_by_job = state.attempts_by_job
    previous_attempts = attempts_by_job.get(job_id)
    retry_keys_by_job = state.retry_event_keys_by_job
    previous_retry_keys = retry_keys_by_job.get(job_id)
    notified_by_job = state.notified_attempts_by_job
    previous_notified = notified_by_job.get(job_id)
    is_new_job = last_job_id is not None and job_id != last_job_id and job_id not in seen_job_ids

    if is_new_job:
        # During a handoff, old and new jobs can briefly emit telemetry at the same
        # time. Do not call a previously observed job another restart if it becomes
        # the most recent emitter again.
        next_restart_count = state.restart_count + 1
        delivered = fire(
            config,
            state,
            f"job_{job_id}",
            "hero new attempt",
            f"New Iris training job: {job_id} (restart #{next_restart_count} "
            "since monitor start). Expect W&B silence while it re-does steps "
            "below the high-water mark; watch tqdm rate in iris logs instead.",
            priority=0,
        )
        if not delivered:
            return
        state.restart_count = next_restart_count

    # Iris emits these events from the controller transaction that changes a
    # failed or gang-bounced task back to PENDING. The phase attempt ids provide
    # an independent fallback because task-event publication is best-effort.
    current_retry_keys = {event.key for event in retry_events}
    notified = dict(previous_notified or {})
    controller_events = []
    if previous_retry_keys is not None:
        for event in retry_events:
            if event.key in previous_retry_keys:
                continue
            next_attempt = event.attempt_id + 1
            if next_attempt > notified.get(event.task_id, -1):
                controller_events.append(event)
            notified[event.task_id] = max(next_attempt, notified.get(event.task_id, -1))
    else:
        # First v6 observation is a silent baseline, including any retained event
        # that predates this monitor process.
        for event in retry_events:
            notified[event.task_id] = max(event.attempt_id + 1, notified.get(event.task_id, -1))

    phase_fallbacks = []
    if previous_attempts is not None and previous_notified is not None:
        for task_id, attempt_id in current_attempts.items():
            if attempt_id > previous_attempts.get(task_id, 0) and attempt_id > notified.get(task_id, -1):
                phase_fallbacks.append((task_id, attempt_id))
                notified[task_id] = attempt_id

    if controller_events or phase_fallbacks:
        controller_events.sort(key=lambda event: (event.observed_at, event.task_id))
        phase_fallbacks.sort()
        examples = [
            f"{event.task_id.removeprefix(job_id + '/')} attempt {event.attempt_id}->" f"{event.attempt_id + 1}"
            for event in controller_events[:3]
        ]
        remaining_examples = max(0, 3 - len(examples))
        examples.extend(
            f"{task_id.removeprefix(job_id + '/')} ->attempt {attempt_id} (phase fallback)"
            for task_id, attempt_id in phase_fallbacks[:remaining_examples]
        )
        event_signature = (
            controller_events[-1].key
            if controller_events
            else f"phase:{phase_fallbacks[-1][0]}:{phase_fallbacks[-1][1]}"
        )
        source_detail = f"{len(controller_events)} controller retry/requeue event(s)" if controller_events else ""
        if phase_fallbacks:
            if source_detail:
                source_detail += " and "
            source_detail += f"{len(phase_fallbacks)} phase attempt increment(s) without a " "retained matching event"
        next_restart_count = state.restart_count + 1
        delivered = fire(
            config,
            state,
            f"task_retry_{job_id}_{event_signature}",
            "hero task retry scheduled",
            f"Iris shows {source_detail} within {job_id}: {', '.join(examples)}. "
            f"Retry #{next_restart_count} observed since monitor start.",
            priority=0,
        )
        if not delivered:
            return
        state.restart_count = next_restart_count

    retry_keys_by_job[job_id] = sorted(current_retry_keys)[-10_000:]
    if previous_notified is None:
        for task_id, attempt_id in current_attempts.items():
            notified[task_id] = max(attempt_id, notified.get(task_id, -1))
    notified_by_job[job_id] = notified

    merged_attempts = dict(previous_attempts or {})
    for task_id, attempt_id in current_attempts.items():
        merged_attempts[task_id] = max(attempt_id, merged_attempts.get(task_id, -1))
    attempts_by_job[job_id] = merged_attempts

    if job_id not in seen_job_ids:
        seen_job_ids.append(job_id)
        if len(seen_job_ids) > JOB_HISTORY_LIMIT:
            expired_job_ids = seen_job_ids[:-JOB_HISTORY_LIMIT]
            del seen_job_ids[:-JOB_HISTORY_LIMIT]
            for expired_job_id in expired_job_ids:
                attempts_by_job.pop(expired_job_id, None)
                retry_keys_by_job.pop(expired_job_id, None)
                notified_by_job.pop(expired_job_id, None)
    state.last_job_id = job_id


def digest(
    config: MonitorConfig,
    series: dict[str, list[tuple[int, float]]],
    task_state: TaskStateSnapshot | None,
    state: MonitorState,
) -> None:
    now = time.time()
    if now - state.last_digest < config.digest_interval:
        return
    step = latest_value(series, "step")
    loss = latest_value(series, "train_loss")
    tps = latest_value(series, "throughput_tokens_per_second")
    mfu = latest_value(series, "throughput_mfu")
    prog = latest_value(series, "run_progress")
    drop = latest_value(series, "moe_drop_fraction")
    gradient_norm = latest_value(series, "grad_norm_total")
    routing_entropy = latest_value(series, "train_router_routing_entropy_mean")
    progress_timestamp = latest_value(series, "progress_time_seconds")
    last_step_status = f"{(now - progress_timestamp) / 60:.1f}m ago" if progress_timestamp else "?"
    loss_text = f"{loss:.3f}" if loss is not None else "?"
    tasks = (
        f"{task_state.running}/{task_state.active_count} running/active"
        if task_state is not None and task_state.age_seconds <= THRESHOLDS.task_state_fresh
        else "state stale"
    )
    msg = (
        f"step {step:.0f}/{FINAL_STEP} ({(prog or 0) * 100:.2f}%) | loss {loss_text} | "
        f"{(tps or 0) / 1e6:.2f}M tok/s, MFU {mfu or 0:.1f}% | last step {last_step_status} | "
        f"tasks {tasks} | drop {drop or 0:.3f} | grad {gradient_norm or 0:.2f} | "
        f"entropy {routing_entropy or 0:.3f} | "
        f"restarts {state.restart_count}"
        if step is not None
        else "no telemetry in window"
    )
    priority = -1 if step is not None else 0
    if pushover(config, "hero status", msg, priority=priority):
        state.last_digest = now


# ----------------------------------------------------------------------------- main


def main():
    config = MonitorConfig.from_environment(os.environ)
    state = load_state(config)
    log(
        f"monitoring {config.run_id} every {config.poll_interval}s "
        f"(digest every {config.digest_interval / 60:g}m); state in {config.state_file}"
    )
    if not config.pushover_app_token or not config.pushover_user_key:
        log("WARNING: PUSHOVER_APP_TOKEN/PUSHOVER_USER_KEY unset — running in dry-run mode")
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while running:
        try:
            execution = fetch_execution_snapshot(config)
            series = fetch_telemetry(config, execution)
            task_state = fetch_task_state(config, execution)
            retry_events = fetch_retry_events(config, execution)
            alert_stats = fetch_alert_stats(config, execution)
            check(config, series, task_state, state, execution, alert_stats)
            check_restart(config, state, execution, retry_events)
            digest(config, series, task_state, state)
            state.query_failures = 0
        except Exception as e:
            state.query_failures = state.query_failures + 1
            log(f"poll failed ({state.query_failures} consecutive): {e}")
            if state.query_failures == 3:
                fire(
                    config,
                    state,
                    "monitor_degraded",
                    "hero MONITOR degraded",
                    f"3 consecutive query failures ({e}). This is 'monitor stale', not "
                    "necessarily 'run unhealthy' — check IAP credential "
                    "(`uv run iris login`) and network.",
                    priority=0,
                )
        save_state(config, state)
        for _ in range(config.poll_interval):
            if not running:
                break
            time.sleep(1)
    log("stopped")


if __name__ == "__main__":
    main()
