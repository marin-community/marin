# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded finelog queries and alert projections for hero-run health.

Beyond the progress and loss rules: the telemetry path itself, the optimizer, MoE
routing, throughput, evaluation, and Iris retries. One `telemetry_v1` scan per
bridge cache interval feeds all three projections. The telemetry and optimizer
ones page; the health one announces in Slack without opening a triage session.
See docs/ops/hero-run-health-alerts.md.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

import pyarrow as pa
from hero_runs import (
    HERO_ROOT_PATTERNS,
    PHASE_METRIC,
    TASK_STATE_FRESHNESS,
    TELEMETRY_GONE_AGE,
    TRAINING_PHASE,
    as_number,
    as_utc,
    hero_run_id,
    root_job_for,
    run_id_predicate,
    sql_epoch_ms,
    sql_timestamp,
)
from loss_spikes import LossWindows, loss_spike_reason, windows_by_run

# Past this the rollup no longer enrolls a run, so the rules that read it go blind.
IRIS_STATE_STALE_AGE = timedelta(minutes=5)

LOSS_JUMP = 1.0
GRAD_NORM_MAX = 2.0
SKIPPED_STEPS_MAX = 3
# The dashboard band is 7%, above the intermittent 5% spikes a healthy MoE run shows.
DROP_FRACTION_MAX = 0.07
# Uniform routing over 384 experts is 5.951; falling entropy is expert collapse.
ROUTER_ENTROPY_MIN = 5.92
ROUTER_BIAS_MAX = 400.0
TOKENS_PER_SECOND_MIN = 2.0e6
MFU_MIN = 15.0

_GRAD_NORM = "grad_norm_total"
_SKIPPED_STEP = "optim_skipped_step"
_DROP_FRACTION = "moe_drop_fraction"
_ROUTER_ENTROPY = "train_router_routing_entropy_mean"
_ROUTER_BIAS_MAX = "train_router_bias_max"
_ROUTER_BIAS_MIN = "train_router_bias_min"
_TOKENS_PER_SECOND = "throughput_tokens_per_second"
_MFU = "throughput_mfu"
_EVAL_LOSS = "eval_paloma_macro_loss"

_SIGNAL_METRICS = (
    PHASE_METRIC,
    _GRAD_NORM,
    _SKIPPED_STEP,
    _DROP_FRACTION,
    _ROUTER_ENTROPY,
    _ROUTER_BIAS_MAX,
    _ROUTER_BIAS_MIN,
    _TOKENS_PER_SECOND,
    _MFU,
    _EVAL_LOSS,
)
# The floor each throughput check compares its window against.
_FLOORS = {_TOKENS_PER_SECOND: TOKENS_PER_SECOND_MIN, _MFU: MFU_MIN}

_SIGNAL_LOOKBACK = timedelta(minutes=65)
# The phase heartbeat and the evaluations need their own window: an outage is
# measured from the last heartbeat however long ago it was, and evaluations are
# hours apart. Both are one row at a time, so the wider scan stays cheap.
_LIVENESS_LOOKBACK = timedelta(hours=24)
HEALTH_WINDOW = timedelta(minutes=15)
# Below this a median says more about sampling than about the run.
_MIN_HEALTH_SAMPLES = 10
_LATEST_FRESHNESS = HEALTH_WINDOW
_EVAL_FRESHNESS = timedelta(minutes=30)
RETRY_WINDOW = timedelta(minutes=15)

_RETRY_REASONS = ("TaskRetryScheduled", "CoscheduledSiblingRequeued")


@dataclass(frozen=True)
class WatchedRun:
    """One hero run watched from the Iris side, the telemetry side, or both."""

    cluster: str
    root_job: str
    run_id: str
    iris_running: bool
    iris_state_age: timedelta | None


@dataclass(frozen=True)
class MetricSignal:
    """One metric's newest sample and the reductions the health window needs."""

    latest: float
    observed_at: datetime
    previous: float | None
    recent_samples: int
    recent_total: float
    recent_below_floor: int


@dataclass(frozen=True)
class RunSignals:
    """One run's metrics and the execution every reduction covers."""

    execution_uid: str
    metrics: dict[str, MetricSignal]


Signals = dict[tuple[str, str], RunSignals]


def signal_query(now: datetime, runs: tuple[WatchedRun, ...]) -> str:
    """Return the newest sample and health-window reductions per run and metric.

    Everything reduces over one execution: the newest attempt process zero
    reports. A retry keeps the run ID and takes a new `execution_uid`, so
    partitioning on the run alone would sum one attempt's skipped steps into the
    next. Process zero because Levanter publishes tracker metrics only from it.
    """
    run_predicate = run_id_predicate(runs)
    signal_since = sql_epoch_ms(now - _SIGNAL_LOOKBACK)
    liveness_since = sql_epoch_ms(now - _LIVENESS_LOOKBACK)
    health_since = sql_epoch_ms(now - HEALTH_WINDOW)
    end = sql_epoch_ms(now)
    metric_names = ", ".join(f"'{name}'" for name in _SIGNAL_METRICS)
    below_floor = " OR ".join(f"(name = '{name}' AND value < {floor})" for name, floor in _FLOORS.items())
    return (
        "WITH attempts AS ("
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS origin_cluster, run_id, execution_uid, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY COALESCE(NULLIF(cluster,''),'unknown'), run_id "
        "ORDER BY timestamp_ms DESC, seq DESC"
        ") AS rn "
        'FROM "telemetry_v1" '
        f"WHERE service = 'levanter' AND name = '{PHASE_METRIC}' AND process_index = '0' "
        f"AND {run_predicate} AND execution_uid IS NOT NULL "
        f"AND timestamp_ms >= {liveness_since} AND timestamp_ms < {end}"
        "), execution AS ("
        "SELECT origin_cluster, run_id, execution_uid FROM attempts WHERE rn = 1"
        "), samples AS ("
        "SELECT execution.origin_cluster, execution.run_id, execution.execution_uid, "
        "telemetry.name, telemetry.value, telemetry.timestamp_ms, telemetry.seq "
        'FROM "telemetry_v1" AS telemetry JOIN execution '
        "ON COALESCE(NULLIF(telemetry.cluster,''),'unknown') = execution.origin_cluster "
        "AND telemetry.run_id = execution.run_id "
        "AND telemetry.execution_uid = execution.execution_uid "
        f"WHERE telemetry.service = 'levanter' AND telemetry.name IN ({metric_names}) "
        f"AND telemetry.timestamp_ms >= {liveness_since} AND telemetry.timestamp_ms < {end} "
        f"AND (telemetry.name IN ('{PHASE_METRIC}', '{_EVAL_LOSS}') "
        f"OR telemetry.timestamp_ms >= {signal_since})"
        "), ranked AS ("
        "SELECT origin_cluster, run_id, execution_uid, name, value, timestamp_ms, "
        "ROW_NUMBER() OVER ("
        "PARTITION BY origin_cluster, run_id, name ORDER BY timestamp_ms DESC, seq DESC"
        ") AS rn FROM samples"
        "), newest AS ("
        "SELECT origin_cluster, run_id, execution_uid, name, "
        "MAX(CASE WHEN rn = 1 THEN value END) AS latest_value, "
        "MAX(CASE WHEN rn = 1 THEN timestamp_ms END) AS latest_at, "
        "MAX(CASE WHEN rn = 2 THEN value END) AS previous_value "
        "FROM ranked WHERE rn <= 2 GROUP BY origin_cluster, run_id, execution_uid, name"
        "), health_window AS ("
        "SELECT origin_cluster, run_id, name, COUNT(*) AS recent_samples, "
        "SUM(value) AS recent_total, "
        f"SUM(CASE WHEN {below_floor} THEN 1 ELSE 0 END) AS recent_below_floor "
        f"FROM samples WHERE timestamp_ms >= {health_since} "
        "GROUP BY origin_cluster, run_id, name"
        ") "
        "SELECT newest.origin_cluster AS cluster, newest.run_id, newest.execution_uid, newest.name, "
        "newest.latest_value, to_timestamp_millis(newest.latest_at) AS observed_at, "
        "newest.previous_value, "
        "COALESCE(health_window.recent_samples, 0) AS recent_samples, "
        "COALESCE(health_window.recent_total, 0) AS recent_total, "
        "COALESCE(health_window.recent_below_floor, 0) AS recent_below_floor "
        "FROM newest LEFT JOIN health_window "
        "ON health_window.origin_cluster = newest.origin_cluster "
        "AND health_window.run_id = newest.run_id AND health_window.name = newest.name"
    )


def retry_event_query(now: datetime) -> str:
    """Return the controller's recent decisions to retry or requeue a hero task."""
    start = sql_timestamp(now - RETRY_WINDOW)
    end = sql_timestamp(now)
    reasons = ", ".join(f"'{reason}'" for reason in _RETRY_REASONS)
    hero_tasks = " OR ".join(f"task_id LIKE '{pattern}/%'" for pattern in HERO_ROOT_PATTERNS)
    return (
        "SELECT COALESCE(NULLIF(cluster,''),'unknown') AS cluster, task_id "
        'FROM "iris.task_event" '
        f"WHERE reason IN ({reasons}) AND ({hero_tasks}) "
        f"AND ts >= TIMESTAMP '{start}' AND ts < TIMESTAMP '{end}'"
    )


def watched_runs(task_states: pa.Table, phase_runs: pa.Table, now: datetime) -> tuple[WatchedRun, ...]:
    """Return every hero run the Iris rollup or Levanter telemetry still reports.

    Watching the union means an outage on one path does not silence the checks
    the other path can still answer.
    """
    now = as_utc(now)
    states: dict[tuple[str, str], tuple[timedelta, bool]] = {}
    for row in task_states.to_pylist():
        root_job = str(row["job"])
        if hero_run_id(root_job) is None:
            continue
        age = now - as_utc(row["state_at"])
        states[(str(row["cluster"]), root_job)] = (age, int(row["running"] or 0) > 0)

    enrolled = [key for key, (age, running) in states.items() if running and age <= TASK_STATE_FRESHNESS]
    for row in phase_runs.to_pylist():
        root_job = root_job_for(str(row["telemetry_job"]))
        if root_job is None:
            continue
        key = (str(row["cluster"]), root_job)
        if key not in enrolled:
            enrolled.append(key)

    runs = []
    for cluster, root_job in enrolled:
        run_id = hero_run_id(root_job)
        if run_id is None:
            continue
        age, running = states.get((cluster, root_job), (None, False))
        runs.append(
            WatchedRun(
                cluster=cluster,
                root_job=root_job,
                run_id=run_id,
                iris_running=running and age is not None and age <= TASK_STATE_FRESHNESS,
                iris_state_age=age,
            )
        )
    return tuple(runs)


def signals_by_run(signal_rows: pa.Table) -> Signals:
    """Fold signal rows into one metric map per run."""
    signals: Signals = {}
    for row in signal_rows.to_pylist():
        latest = as_number(row["latest_value"])
        if latest is None:
            continue
        key = (str(row["cluster"]), str(row["run_id"]))
        run = signals.setdefault(key, RunSignals(execution_uid=str(row["execution_uid"]), metrics={}))
        run.metrics[str(row["name"])] = MetricSignal(
            latest=latest,
            observed_at=as_utc(row["observed_at"]),
            previous=as_number(row["previous_value"]),
            recent_samples=int(row["recent_samples"] or 0),
            recent_total=float(row["recent_total"] or 0.0),
            recent_below_floor=int(row["recent_below_floor"] or 0),
        )
    return signals


def selected_executions(signals: Signals) -> tuple[str, ...]:
    """The execution UIDs the signal scan selected, for a query that must match them."""
    return tuple(sorted({run.execution_uid for run in signals.values()}))


def _fresh(signal: MetricSignal | None, now: datetime, within: timedelta) -> MetricSignal | None:
    """Return the signal only when its newest sample still describes the run now."""
    if signal is None or not isfinite(signal.latest) or now - signal.observed_at > within:
        return None
    return signal


def _row(run: WatchedRun | None, reason: str, value: int) -> dict:
    if run is None:
        return {"cluster": "fleet", "job": "", "run": "", "reason": reason, "value": value}
    return {"cluster": run.cluster, "job": run.root_job, "run": run.run_id, "reason": reason, "value": value}


def _project(runs: tuple[WatchedRun, ...], reasons_for: Callable[[WatchedRun], list[str]]) -> list[dict]:
    """Return one row per firing reason, a healthy row per quiet run, and a fleet row.

    The explicit zeros clear a resolved instance and keep an empty fleet out of NoData.
    """
    rows = []
    for run in runs:
        firing = reasons_for(run)
        rows.extend(_row(run, reason, 1) for reason in firing)
        if not firing:
            rows.append(_row(run, "healthy", 0))
    return rows or [_row(None, "healthy", 0)]


def telemetry_alert_rows(runs: tuple[WatchedRun, ...], signals: Signals, now: datetime) -> list[dict]:
    """Project runs that published telemetry and then went silent.

    The reason separates the two causes an operator acts on differently: Iris
    still counting the tasks points at the telemetry path or a wedged process,
    while Iris no longer counting them points at a job that exited.
    """
    now = as_utc(now)

    def reasons_for(run: WatchedRun) -> list[str]:
        phase = _metrics(signals, run).get(PHASE_METRIC)
        # Only a run whose last word was that it is training. A finished tracker
        # ended on purpose, and one that never left initialization is
        # TrainingProgressStalled's, which allows the full startup budget. That
        # last case is most of the traffic: `hero-` names a smoke test as often
        # as a production run, and a smoke test that dies in restore is not an
        # incident.
        if phase is None or not isfinite(phase.latest) or int(phase.latest) != TRAINING_PHASE:
            return []
        if now - phase.observed_at <= TELEMETRY_GONE_AGE:
            return []
        return ["telemetry_gone" if run.iris_running else "run_down"]

    return _project(runs, reasons_for)


def _metrics(signals: Signals, run: WatchedRun) -> dict[str, MetricSignal]:
    found = signals.get((run.cluster, run.run_id))
    return found.metrics if found is not None else {}


def _is_training(metrics: dict[str, MetricSignal], now: datetime) -> bool:
    """True while the run's own telemetry is fresh and reports the training phase.

    An initializing attempt has published none of these metrics, a finished one
    leaves its last samples behind, and a silent one is TrainingTelemetryGone's.
    """
    phase = _fresh(metrics.get(PHASE_METRIC), now, TELEMETRY_GONE_AGE)
    return phase is not None and int(phase.latest) == TRAINING_PHASE


def optimizer_alert_rows(
    runs: tuple[WatchedRun, ...], signals: Signals, loss_windows: pa.Table, now: datetime
) -> list[dict]:
    """Project the optimizer signals that precede a divergence the loss rule catches later."""
    now = as_utc(now)
    windows = windows_by_run(loss_windows)

    def reasons_for(run: WatchedRun) -> list[str]:
        metrics = _metrics(signals, run)
        if not _is_training(metrics, now):
            return []
        reasons = []
        if _loss_jumped(windows.get((run.cluster, run.run_id))):
            reasons.append("loss_jump")
        grad_norm = _fresh(metrics.get(_GRAD_NORM), now, _LATEST_FRESHNESS)
        if grad_norm is not None and grad_norm.latest > GRAD_NORM_MAX:
            reasons.append("grad_norm_high")
        skipped = metrics.get(_SKIPPED_STEP)
        if skipped is not None and skipped.recent_total >= SKIPPED_STEPS_MAX:
            reasons.append("steps_skipped")
        return reasons

    return _project(runs, reasons_for)


def _loss_jumped(windows: LossWindows | None) -> bool:
    """True when the recent loss floor sits a whole unit above the trailing floor.

    A rise the six-sigma band already caught is TrainingLossSpike's. What is left
    is the level shift a wide trailing spread hides.
    """
    if windows is None or loss_spike_reason(windows) != ("healthy", 0):
        return False
    if windows.baseline_floor is None or windows.recent_floor is None:
        return False
    if not isfinite(windows.baseline_floor) or not isfinite(windows.recent_floor):
        return False
    return windows.recent_floor - windows.baseline_floor > LOSS_JUMP


def health_alert_rows(
    runs: tuple[WatchedRun, ...], signals: Signals, retry_events: pa.Table, now: datetime
) -> list[dict]:
    """Project the routing, throughput, evaluation, and Iris signals an operator reads."""
    now = as_utc(now)
    retries = _retries_by_root(retry_events)

    def reasons_for(run: WatchedRun) -> list[str]:
        metrics = _metrics(signals, run)
        if not _is_training(metrics, now):
            return []
        reasons = []
        drops = _fresh(metrics.get(_DROP_FRACTION), now, _LATEST_FRESHNESS)
        if drops is not None and drops.latest > DROP_FRACTION_MAX:
            reasons.append("token_drops")
        entropy = _fresh(metrics.get(_ROUTER_ENTROPY), now, _LATEST_FRESHNESS)
        if entropy is not None and entropy.latest < ROUTER_ENTROPY_MIN:
            reasons.append("router_entropy")
        if _router_bias_magnitude(metrics, now) > ROUTER_BIAS_MAX:
            reasons.append("router_bias")
        if _mostly_below_floor(metrics.get(_TOKENS_PER_SECOND)):
            reasons.append("throughput_low")
        if _mostly_below_floor(metrics.get(_MFU)):
            reasons.append("mfu_low")
        if _evaluation_regressed(metrics.get(_EVAL_LOSS), now):
            reasons.append("eval_regressed")
        # No row at all means a controller that publishes no rollup (the GCE
        # clusters), not a rollup that broke.
        if run.iris_state_age is not None and run.iris_state_age > IRIS_STATE_STALE_AGE:
            reasons.append("iris_state_stale")
        if retries.get((run.cluster, run.root_job), 0) > 0:
            reasons.append("task_retried")
        return reasons

    return _project(runs, reasons_for)


def _retries_by_root(retry_events: pa.Table) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in retry_events.to_pylist():
        root_job = root_job_for(str(row["task_id"]))
        if root_job is None:
            continue
        key = (str(row["cluster"]), root_job)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _router_bias_magnitude(metrics: dict[str, MetricSignal], now: datetime) -> float:
    """Return how far the router's per-expert bias has travelled from zero."""
    bounds = [
        signal.latest
        for name in (_ROUTER_BIAS_MAX, _ROUTER_BIAS_MIN)
        if (signal := _fresh(metrics.get(name), now, _LATEST_FRESHNESS)) is not None
    ]
    return max((abs(bound) for bound in bounds), default=0.0)


def _mostly_below_floor(signal: MetricSignal | None) -> bool:
    """True when most of the health window sat below the metric's floor.

    A count rather than a mean, so one restart step at zero cannot drag the
    window under.
    """
    if signal is None or signal.recent_samples < _MIN_HEALTH_SAMPLES:
        return False
    return signal.recent_below_floor * 2 > signal.recent_samples


def _evaluation_regressed(signal: MetricSignal | None, now: datetime) -> bool:
    """True when the newest evaluation is worse than the one before it."""
    signal = _fresh(signal, now, _EVAL_FRESHNESS)
    if signal is None or signal.previous is None or not isfinite(signal.previous):
        return False
    return signal.latest > signal.previous
