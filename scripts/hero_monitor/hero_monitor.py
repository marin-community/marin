#!/usr/bin/env python3
"""Hero-run Pushover monitor.

Polls finelog (the same store Grafana's training-run dashboard reads) for the hero
training run's telemetry and sends Pushover notifications:

  - EMERGENCY (priority 2, repeats until acked): run down / telemetry gone / stalled
    past the page threshold / non-finite loss.
  - HIGH (priority 1): loss spike (Grafana's 6-sigma rule), loss jump > 1, grad-norm
    breach, repeated skipped optimizer steps.
  - NORMAL (priority 0): oncall-draft "things to watch for" — token-drop fraction,
    router entropy/bias, throughput sag, eval regression, new Iris job or task attempt.
  - LOW (priority -1): a status digest every DIGEST_MINUTES.

Run it from a persistent box that has the marin checkout and a cached Marin IAP
credential (run `uv run iris login` once, or provide ambient service-account ADC):

    cd <marin repo> && uv run scripts/hero_monitor/hero_monitor.py

Configuration is via environment variables (see CONFIG below). Requires only the
Python stdlib; finelog is queried by shelling out to `uv run finelog query`.

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
from dataclasses import dataclass
from datetime import datetime, timezone

# ----------------------------------------------------------------------------- CONFIG

CONFIG = {
    # Required: create an app at https://pushover.net/apps/build
    "PUSHOVER_APP_TOKEN": os.environ.get("PUSHOVER_APP_TOKEN", ""),
    "PUSHOVER_USER_KEY": os.environ.get("PUSHOVER_USER_KEY", ""),
    # The Levanter run id (= W&B display name = finelog run_id).
    "RUN_ID": os.environ.get("HERO_RUN_ID", "hero-12d8b6f0-dee637"),
    # Marin checkout; `uv run finelog` resolves its env from here.
    "MARIN_REPO": os.environ.get("MARIN_REPO", os.path.expanduser("~/marin")),
    "FINELOG_NAME": os.environ.get("FINELOG_NAME", "marin"),
    "POLL_SECONDS": int(os.environ.get("POLL_SECONDS", "120")),
    "DIGEST_MINUTES": int(os.environ.get("DIGEST_MINUTES", "10")),
    "STATE_FILE": os.environ.get(
        "HERO_MON_STATE", os.path.expanduser("~/.hero_pushover_state.json")
    ),
    # Re-fire suppression per alert key, seconds.
    "COOLDOWN_SECONDS": int(os.environ.get("COOLDOWN_SECONDS", "1800")),
}

# Thresholds. Sources: oncall draft, Grafana panel/rule constants, manage-hero-run skill.
THRESHOLDS = {
    # EMERGENCY
    "telemetry_gone_s": 600,  # no sample at all in 10 min (samples normally ~30s apart)
    "stall_page_s": 900,  # since_last_step red threshold on the dashboard / 15 min stall rule
    "initializing_page_s": 2700,  # Grafana training-stall contract: 45 min in init
    "task_state_fresh_s": 90,  # Grafana eligibility: 3 missed 30s root rollups
    "task_state_degraded_s": 300,  # debounce before alerting on state-path staleness
    # HIGH
    "loss_jump": 1.0,  # oncall draft: loss jump by over 1
    "sigma_factor": 6.0,  # Grafana loss-spike rule
    "min_rise": 0.05,  # Grafana loss-spike floor
    "grad_norm_max": 2.0,  # oncall draft: total grad norm climbs past 2
    # NORMAL
    "stall_warn_s": 300,  # dashboard orange threshold
    "drop_fraction_max": 0.05,  # oncall draft: token dropping exceeds 5%
    "router_entropy_min": 5.92,  # oncall draft
    "router_bias_max": 400.0,  # oncall draft: router bias norm beyond 400
    "tokens_per_s_min": 2.0e6,  # healthy ~2.5e6; sustained < 2.0e6 is a >20% sag
    "mfu_min": 15.0,  # healthy ~21
}

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
    execution_started_at: float
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
    def active(self):
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

# ----------------------------------------------------------------------------- helpers


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_state():
    try:
        with open(CONFIG["STATE_FILE"]) as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {
            "state_version": STATE_VERSION,
            "last_fired": {},
            "last_digest": 0,
            "last_job_id": None,
            "seen_job_ids": [],
            "attempts_by_job": {},
            "last_eval_paloma": None,
            "restart_count": 0,
            "query_failures": 0,
        }

    state_version = state.get("state_version", 1)
    # Version 1 compared one arbitrarily selected execution_uid. An execution_uid
    # identifies one task attempt, so a healthy 176-task gang appeared to "restart"
    # whenever a different task emitted the newest row. Its counts are unusable.
    if state_version < 2:
        state["last_fired"] = {
            key: value
            for key, value in state.get("last_fired", {}).items()
            if not key.startswith("attempt_")
        }
        state.pop("last_execution_uid", None)
        state["last_job_id"] = None
        state["seen_job_ids"] = []
        state["restart_count"] = 0

    # Version 2 tracked only job changes. Preserve its valid job-level state and
    # silently establish per-task attempt baselines on the first v3 poll.
    if state_version < 3:
        state["attempts_by_job"] = {}

    if state_version < 4:
        state["run_finished"] = False

    # Version 5 uses controller-authored retry/requeue events rather than
    # inferring retries solely from application telemetry. Establish a baseline.
    if state_version < 5:
        state["retry_event_keys_by_job"] = {}

    # Version 6 cross-checks retained controller events with observed phase
    # attempt increments, without notifying twice when both describe one retry.
    if state_version < 6:
        state["notified_attempts_by_job"] = {}

    state["state_version"] = STATE_VERSION

    state.setdefault("last_fired", {})
    state.setdefault("last_digest", 0)
    state.setdefault("last_job_id", None)
    state.setdefault("seen_job_ids", [])
    state.setdefault("attempts_by_job", {})
    state.setdefault("last_eval_paloma", None)
    state.setdefault("restart_count", 0)
    state.setdefault("query_failures", 0)
    state.setdefault("run_finished", False)
    state.setdefault("retry_event_keys_by_job", {})
    state.setdefault("notified_attempts_by_job", {})
    return state


def save_state(state):
    tmp = CONFIG["STATE_FILE"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CONFIG["STATE_FILE"])


def pushover(title, message, priority=0):
    """Send one Pushover notification. Never raises."""
    if not CONFIG["PUSHOVER_APP_TOKEN"] or not CONFIG["PUSHOVER_USER_KEY"]:
        log(f"DRY-RUN pushover p{priority}: {title}: {message}")
        return True
    data = {
        "token": CONFIG["PUSHOVER_APP_TOKEN"],
        "user": CONFIG["PUSHOVER_USER_KEY"],
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
    except Exception as e:  # noqa: BLE001 - monitor must not die on notify failure
        log(f"pushover send failed: {e}")
        return False
    log(f"pushover p{priority} sent: {title}")
    return ok


def fire(state, key, title, message, priority):
    """Fire an alert unless cooled down; false only when delivery failed."""
    now = time.time()
    last = state["last_fired"].get(key, 0)
    if now - last < CONFIG["COOLDOWN_SECONDS"]:
        return True
    if pushover(title, message, priority):
        state["last_fired"][key] = now
        return True
    return False


def finelog_query(sql):
    """Run SQL against finelog; returns list of dict rows. Raises on failure."""
    out = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "finelog",
            "query",
            CONFIG["FINELOG_NAME"],
            sql,
            "--format",
            "json",
            "--timeout",
            "30",
        ],
        cwd=CONFIG["MARIN_REPO"],
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
    """Match Iris JobName.root_job: /user/root-job for a canonical job id."""
    parts = job_id.split("/")
    if len(parts) < 3 or parts[0] != "" or not parts[1] or not parts[2]:
        return None
    return f"/{parts[1]}/{parts[2]}"


def timestamp_seconds(value):
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return float(value) / 1000.0


def fetch_telemetry(execution=None):
    """Pull the last 65 min of the run's scalars, aggregated into 30s buckets
    (telemetry rows are per-process — 704 of them — so raw rows are far too many).
    Returns {name: [(ts_ms, value), ...]} sorted by time."""
    names = ",".join(sql_literal(n) for n in METRICS)
    identity_filter = ""
    if execution is not None:
        # Status belongs to one attempt. Scalar tracker metrics are emitted only by
        # global process zero, which need not be the attempt chosen by newest phase.
        status_names = ",".join(
            sql_literal(name)
            for name in ("phase", "step", "run_progress", "progress_time_seconds")
        )
        identity_filter = (
            "AND COALESCE(NULLIF(cluster,''),'unknown')="
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
        f"WHERE service='levanter' AND run_id={sql_literal(CONFIG['RUN_ID'])} "
        f"AND name IN ({names}) "
        f"{identity_filter}"
        "AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL '65 minutes')*1000 AS BIGINT) "
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
    for row in finelog_query(sql):
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


def fetch_execution_snapshot():
    """Return the freshest training execution and max attempt per task.

    Iris keeps job_id stable while independently retrying tasks. execution_uid is
    ``iris:{task_id}:attempt:{attempt_id}``, so collecting every UID distinguishes
    a real attempt increment from normal telemetry arriving from different tasks.
    Levanter emits ``phase`` immediately and every minute, so it is the canonical
    attempt heartbeat; ``step`` is absent while an attempt initializes.
    """
    sql = (
        "WITH phase AS (SELECT COALESCE(NULLIF(cluster,''),'unknown') AS origin_cluster, "
        "job_id, execution_uid, timestamp_ms, seq FROM \"telemetry_v1\" "
        f"WHERE service='levanter' AND run_id={sql_literal(CONFIG['RUN_ID'])} AND name='phase' "
        "AND job_id IS NOT NULL AND execution_uid IS NOT NULL "
        "AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL '65 minutes')*1000 AS BIGINT)) "
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
    rows = finelog_query(sql)
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
        if (
            row["origin_cluster"] != latest_row["origin_cluster"]
            or row["job_id"] != latest_job_id
        ):
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
        execution_started_at=selected_first_ts / 1000.0,
        attempts=attempts,
    )


def fetch_task_state(execution=None):
    """Return Iris's latest active-task aggregate for the exact current root."""
    root = root_job_id(execution.job_id) if execution is not None else None
    root_filter = (
        f"root_job_id={sql_literal(root)}"
        if root is not None
        else f"root_job_id LIKE {sql_literal('%' + CONFIG['RUN_ID'] + '%')}"
    )
    cluster_filter = (
        "AND COALESCE(NULLIF(cluster,''),'unknown')="
        f"{sql_literal(execution.cluster)} "
        if execution is not None
        else ""
    )
    sql = (
        'SELECT root_job_id, ts, pending, assigned, building, running FROM "iris.task_state" '
        f"WHERE {root_filter} "
        f"{cluster_filter}"
        "AND ts >= now() - INTERVAL '15 minutes' ORDER BY ts DESC LIMIT 1"
    )
    rows = finelog_query(sql)
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


def fetch_retry_events(execution):
    """Return Iris's retained controller decisions to retry/requeue tasks."""
    if execution is None:
        return []
    task_prefix = execution.job_id.rstrip("/") + "/%"
    sql = (
        "SELECT task_id, attempt_id, attempt_uid, ts FROM \"iris.task_event\" "
        f"WHERE task_id LIKE {sql_literal(task_prefix)} "
        "AND COALESCE(NULLIF(cluster,''),'unknown')="
        f"{sql_literal(execution.cluster)} "
        "AND reason IN ('TaskRetryScheduled','CoscheduledSiblingRequeued') "
        "AND ts >= now() - INTERVAL '7 days' "
        "ORDER BY ts, task_id, attempt_id"
    )
    events = []
    for row in finelog_query(sql):
        observed_at = timestamp_seconds(row["ts"])
        attempt_uid = row.get("attempt_uid")
        key = (
            f"uid:{attempt_uid}"
            if attempt_uid
            else f"legacy:{row['task_id']}:{int(row['attempt_id'])}:{observed_at}"
        )
        events.append(
            RetryEvent(
                key=key,
                task_id=row["task_id"],
                attempt_id=int(row["attempt_id"]),
                observed_at=observed_at,
            )
        )
    return events


def fetch_alert_stats(execution=None):
    """Fetch raw (not bucket-averaged) windows used by source alert rules."""
    run = sql_literal(CONFIG["RUN_ID"])
    cluster_filter = (
        "AND COALESCE(NULLIF(cluster,''),'unknown')="
        f"{sql_literal(execution.cluster)} "
        if execution is not None
        else ""
    )
    recent_ms = (
        "CAST(EXTRACT(EPOCH FROM now() - INTERVAL '5 minutes')*1000 AS BIGINT)"
    )
    skips_ms = (
        "CAST(EXTRACT(EPOCH FROM now() - INTERVAL '15 minutes')*1000 AS BIGINT)"
    )
    sql = (
        "WITH samples AS (SELECT name, value, timestamp_ms FROM \"telemetry_v1\" "
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
    rows = finelog_query(sql)
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


def latest(series, name):
    pts = series.get(name)
    return pts[-1][1] if pts else None


def latest_ts(series, name):
    pts = series.get(name)
    return pts[-1][0] / 1000.0 if pts else None


# ----------------------------------------------------------------------------- checks


def check(series, task_state, state, execution=None, alert_stats=None):
    now = time.time()
    run = CONFIG["RUN_ID"]

    # --- liveness ------------------------------------------------------------
    phase_value = latest(series, "phase")
    phase = int(phase_value) if phase_value is not None else None
    step = latest(series, "step") or 0
    if phase == 2:
        # Telemetry deliberately stops after Levanter marks the tracker finished.
        # Remember that terminal state so aging data is not later called an outage.
        state["run_finished"] = True
        return
    if phase in (0, 1) or step > 0:
        state["run_finished"] = False

    all_ts = [p[0] for pts in series.values() for p in pts]
    status_ts = latest_ts(series, "phase")
    sample_age = (
        now - status_ts
        if status_ts is not None
        else now - max(all_ts) / 1000.0
        if all_ts
        else math.inf
    )

    if state.get("run_finished") and not all_ts:
        return

    if sample_age > THRESHOLDS["telemetry_gone_s"]:
        task_is_fresh = (
            task_state is not None
            and task_state.age_seconds <= THRESHOLDS["task_state_fresh_s"]
        )
        if task_state is None:
            detail = " Iris has no active-root row in the 15-minute query window."
        elif not task_is_fresh:
            detail = (
                f" Iris's last active-root row is stale ({task_state.age_seconds:.0f}s old)."
            )
        else:
            detail = (
                f" Iris still reports {task_state.active} active tasks "
                f"({task_state.running} running, {task_state.pending} pending, "
                f"{task_state.assigned} assigned, {task_state.building} building); "
                "this points to the telemetry path, not a confirmed job exit."
            )
        fire(
            state,
            "run_down",
            f"HERO DOWN? {run}",
            f"No telemetry for {sample_age / 60:.1f} min.{detail}",
            priority=2,
        )
        return  # nothing else is meaningful without fresh data

    if (
        task_state is None
        or task_state.age_seconds > THRESHOLDS["task_state_degraded_s"]
    ):
        detail = (
            "no active-root row in 15 minutes"
            if task_state is None
            else f"last root row is {task_state.age_seconds / 60:.1f} minutes old"
        )
        fire(
            state,
            "task_state_stale",
            f"hero Iris state stale: {run}",
            f"Levanter phase telemetry is fresh, but iris.task_state has {detail}. "
            "The run is still emitting; investigate the controller/state telemetry path.",
            priority=0,
        )

    prog_t = latest(series, "progress_time_seconds")
    attempt_age = (
        max(0.0, now - execution.execution_started_at)
        if execution is not None
        else 0.0
    )
    is_training = phase == 1 or step > 0
    since_last_step = now - prog_t if prog_t and prog_t > 0 else attempt_age
    if not is_training and attempt_age >= THRESHOLDS["initializing_page_s"]:
        fire(
            state,
            "initializing_stale",
            f"HERO INIT STALLED {run}",
            f"Current Iris task attempt has remained in Levanter initialization for "
            f"{attempt_age / 60:.1f} min (45 min threshold).",
            priority=2,
        )
    elif is_training and since_last_step >= THRESHOLDS["stall_page_s"]:
        fire(
            state,
            "stalled",
            f"HERO STALLED {run}",
            f"No completed step for {since_last_step / 60:.1f} min "
            f"(step {step:.0f}). Telemetry still flowing — live wedge or "
            "restart catch-up. Check `iris job describe` and cuMemAllocAsync in logs.",
            priority=2,
        )
    elif is_training and since_last_step >= THRESHOLDS["stall_warn_s"]:
        fire(
            state,
            "stall_warn",
            f"hero slow: {run}",
            f"{since_last_step / 60:.1f} min since last step.",
            priority=0,
        )

    # --- loss ---------------------------------------------------------------
    if (
        alert_stats is not None
        and alert_stats.baseline_samples >= 20
        and alert_stats.recent_samples >= 5
    ):
        diverged = any(
            value is not None and not math.isfinite(value)
            for value in (alert_stats.recent_loss, alert_stats.recent_peak)
        )
        if diverged:
            fire(
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
            threshold = alert_stats.baseline_loss + max(
                THRESHOLDS["min_rise"], THRESHOLDS["sigma_factor"] * spread
            )
            if alert_stats.recent_floor > threshold:
                fire(
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
            and alert_stats.recent_floor - alert_stats.baseline_floor
            > THRESHOLDS["loss_jump"]
        ):
            fire(
                state,
                "loss_jump",
                f"hero loss jump {run}",
                f"Loss floor jumped "
                f"{alert_stats.recent_floor - alert_stats.baseline_floor:+.2f} "
                "between the trailing baseline and recent five minutes.",
                priority=1,
            )

    gn = latest(series, "grad_norm_total")
    if gn is not None and gn > THRESHOLDS["grad_norm_max"]:
        fire(
            state,
            "grad_norm",
            f"hero grad norm {run}",
            f"grad_norm_total={gn:.3f} > {THRESHOLDS['grad_norm_max']} "
            "(no grad clipping is configured on this run).",
            priority=1,
        )

    recent_skips = alert_stats.recent_skips if alert_stats is not None else 0
    if recent_skips >= 3:
        fire(
            state,
            "skipped_steps",
            f"hero skipped steps {run}",
            f"{recent_skips:.0f} optimizer steps skipped in 15 min.",
            priority=1,
        )

    # --- oncall-draft watch items ------------------------------------------
    drop = latest(series, "moe_drop_fraction")
    if drop is not None and drop > THRESHOLDS["drop_fraction_max"]:
        fire(
            state,
            "drop_fraction",
            f"hero token drops {run}",
            f"moe_drop_fraction={drop:.3f} > {THRESHOLDS['drop_fraction_max']}.",
            priority=0,
        )

    ent = latest(series, "train_router_routing_entropy_mean")
    if ent is not None and ent < THRESHOLDS["router_entropy_min"]:
        fire(
            state,
            "router_entropy",
            f"hero router entropy {run}",
            f"routing_entropy_mean={ent:.4f} < {THRESHOLDS['router_entropy_min']} "
            "(uniform is 5.951; falling entropy = expert collapse).",
            priority=0,
        )

    bias_hi = latest(series, "train_router_bias_max")
    bias_lo = latest(series, "train_router_bias_min")
    bias_mag = (
        max(abs(b) for b in (bias_hi, bias_lo) if b is not None)
        if (bias_hi is not None or bias_lo is not None)
        else None
    )
    if bias_mag is not None and bias_mag > THRESHOLDS["router_bias_max"]:
        fire(
            state,
            "router_bias",
            f"hero router bias {run}",
            f"router bias magnitude {bias_mag:.1f} > {THRESHOLDS['router_bias_max']}.",
            priority=0,
        )

    tps_pts = [
        v
        for t, v in series.get("throughput_tokens_per_second", [])
        if now - t / 1000.0 <= 900
    ]
    if len(tps_pts) >= 10:
        med = sorted(tps_pts)[len(tps_pts) // 2]
        if med < THRESHOLDS["tokens_per_s_min"]:
            fire(
                state,
                "throughput",
                f"hero throughput {run}",
                f"median tokens/s over 15 min = {med / 1e6:.2f}M "
                f"(healthy ~2.5M, threshold {THRESHOLDS['tokens_per_s_min'] / 1e6:.1f}M).",
                priority=0,
            )

    mfu_pts = [
        v
        for t, v in series.get("throughput_mfu", [])
        if now - t / 1000.0 <= 900
    ]
    if len(mfu_pts) >= 10:
        median_mfu = sorted(mfu_pts)[len(mfu_pts) // 2]
        if median_mfu < THRESHOLDS["mfu_min"]:
            fire(
                state,
                "mfu",
                f"hero MFU low {run}",
                f"median MFU over 15 min = {median_mfu:.1f}% "
                f"(threshold {THRESHOLDS['mfu_min']:.1f}%).",
                priority=0,
            )

    ev = latest(series, "eval_paloma_macro_loss")
    if ev is not None:
        prev = state.get("last_eval_paloma")
        if prev is not None and ev > prev:
            fire(
                state,
                "eval_regress",
                f"hero eval regression {run}",
                f"paloma macro loss {prev:.4f} -> {ev:.4f} (worse).",
                priority=0,
            )
        if prev != ev:
            state["last_eval_paloma"] = ev


def check_restart(state, snapshot, retry_events):
    if snapshot is None:
        return
    job_id = snapshot.job_id
    current_attempts = snapshot.attempts

    last_job_id = state.get("last_job_id")
    seen_job_ids = state.setdefault("seen_job_ids", [])
    attempts_by_job = state.setdefault("attempts_by_job", {})
    previous_attempts = attempts_by_job.get(job_id)
    retry_keys_by_job = state.setdefault("retry_event_keys_by_job", {})
    previous_retry_keys = retry_keys_by_job.get(job_id)
    notified_by_job = state.setdefault("notified_attempts_by_job", {})
    previous_notified = notified_by_job.get(job_id)
    is_new_job = last_job_id is not None and job_id != last_job_id and job_id not in seen_job_ids

    if is_new_job:
        # During a handoff, old and new jobs can briefly emit telemetry at the same
        # time. Do not call a previously observed job another restart if it becomes
        # the most recent emitter again.
        next_restart_count = state.get("restart_count", 0) + 1
        delivered = fire(
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
        state["restart_count"] = next_restart_count

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
            notified[event.task_id] = max(
                next_attempt, notified.get(event.task_id, -1)
            )
    else:
        # First v6 observation is a silent baseline, including any retained event
        # that predates this monitor process.
        for event in retry_events:
            notified[event.task_id] = max(
                event.attempt_id + 1, notified.get(event.task_id, -1)
            )

    phase_fallbacks = []
    if previous_attempts is not None and previous_notified is not None:
        for task_id, attempt_id in current_attempts.items():
            if (
                attempt_id > previous_attempts.get(task_id, 0)
                and attempt_id > notified.get(task_id, -1)
            ):
                phase_fallbacks.append((task_id, attempt_id))
                notified[task_id] = attempt_id

    if controller_events or phase_fallbacks:
        controller_events.sort(key=lambda event: (event.observed_at, event.task_id))
        phase_fallbacks.sort()
        examples = [
            f"{event.task_id.removeprefix(job_id + '/')} attempt {event.attempt_id}->"
            f"{event.attempt_id + 1}"
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
        source_detail = (
            f"{len(controller_events)} controller retry/requeue event(s)"
            if controller_events
            else ""
        )
        if phase_fallbacks:
            if source_detail:
                source_detail += " and "
            source_detail += (
                f"{len(phase_fallbacks)} phase attempt increment(s) without a "
                "retained matching event"
            )
        next_restart_count = state.get("restart_count", 0) + 1
        delivered = fire(
            state,
            f"task_retry_{job_id}_{event_signature}",
            "hero task retry scheduled",
            f"Iris shows {source_detail} within {job_id}: {', '.join(examples)}. "
            f"Retry #{next_restart_count} observed since monitor start.",
            priority=0,
        )
        if not delivered:
            return
        state["restart_count"] = next_restart_count

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
        if len(seen_job_ids) > 50:
            expired_job_ids = seen_job_ids[:-50]
            del seen_job_ids[:-50]
            for expired_job_id in expired_job_ids:
                attempts_by_job.pop(expired_job_id, None)
                retry_keys_by_job.pop(expired_job_id, None)
                notified_by_job.pop(expired_job_id, None)
    state["last_job_id"] = job_id


def digest(series, task_state, state):
    now = time.time()
    if now - state.get("last_digest", 0) < CONFIG["DIGEST_MINUTES"] * 60:
        return
    step = latest(series, "step")
    loss = latest(series, "train_loss")
    tps = latest(series, "throughput_tokens_per_second")
    mfu = latest(series, "throughput_mfu")
    prog = latest(series, "run_progress")
    drop = latest(series, "moe_drop_fraction")
    gn = latest(series, "grad_norm_total")
    ent = latest(series, "train_router_routing_entropy_mean")
    prog_t = latest(series, "progress_time_seconds")
    sls = f"{(now - prog_t) / 60:.1f}m ago" if prog_t else "?"
    loss_text = f"{loss:.3f}" if loss is not None else "?"
    tasks = (
        f"{task_state.running}/{task_state.active} running/active"
        if task_state is not None
        and task_state.age_seconds <= THRESHOLDS["task_state_fresh_s"]
        else "state stale"
    )
    msg = (
        f"step {step:.0f}/{FINAL_STEP} ({(prog or 0) * 100:.2f}%) | loss {loss_text} | "
        f"{(tps or 0) / 1e6:.2f}M tok/s, MFU {mfu or 0:.1f}% | last step {sls} | "
        f"tasks {tasks} | drop {drop or 0:.3f} | grad {gn or 0:.2f} | ent {ent or 0:.3f} | "
        f"restarts {state.get('restart_count', 0)}"
        if step is not None
        else "no telemetry in window"
    )
    priority = -1 if step is not None else 0
    if pushover("hero status", msg, priority=priority):
        state["last_digest"] = now


# ----------------------------------------------------------------------------- main


def main():
    state = load_state()
    log(
        f"monitoring {CONFIG['RUN_ID']} every {CONFIG['POLL_SECONDS']}s "
        f"(digest every {CONFIG['DIGEST_MINUTES']}m); state in {CONFIG['STATE_FILE']}"
    )
    if not CONFIG["PUSHOVER_APP_TOKEN"] or not CONFIG["PUSHOVER_USER_KEY"]:
        log(
            "WARNING: PUSHOVER_APP_TOKEN/PUSHOVER_USER_KEY unset — running in dry-run mode"
        )
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while running:
        try:
            execution = fetch_execution_snapshot()
            series = fetch_telemetry(execution)
            task_state = fetch_task_state(execution)
            retry_events = fetch_retry_events(execution)
            alert_stats = fetch_alert_stats(execution)
            check(series, task_state, state, execution, alert_stats)
            check_restart(state, execution, retry_events)
            digest(series, task_state, state)
            state["query_failures"] = 0
        except Exception as e:  # noqa: BLE001 - distinguish monitor failure from run failure
            state["query_failures"] = state.get("query_failures", 0) + 1
            log(f"poll failed ({state['query_failures']} consecutive): {e}")
            if state["query_failures"] == 3:
                fire(
                    state,
                    "monitor_degraded",
                    "hero MONITOR degraded",
                    f"3 consecutive query failures ({e}). This is 'monitor stale', not "
                    "necessarily 'run unhealthy' — check IAP credential "
                    "(`uv run iris login`) and network.",
                    priority=0,
                )
        save_state(state)
        for _ in range(CONFIG["POLL_SECONDS"]):
            if not running:
                break
            time.sleep(1)
    log("stopped")


if __name__ == "__main__":
    main()
