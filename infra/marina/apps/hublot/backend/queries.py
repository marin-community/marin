# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded reads of SkyRL telemetry and driver timing facts."""

import re
from dataclasses import dataclass

from .query_contract import (
    COUNTER_MARGIN_MS,
    CUMULATIVE_NAMES,
    CUMULATIVE_PAIRS,
    DETAIL_NAMES,
    DETAIL_SIGNALS,
    DISPATCH_WAITS,
    DRIVER_MARKERS,
    OVERVIEW_NAMES,
    OVERVIEW_SIGNALS,
    SNAPSHOT_NAMES,
    SPAN_LOOKAHEAD_MS,
)

NAMESPACE = "telemetry_v1.marinskyrl"
CHOICE_LIMIT = 8
DRIVER_LIMIT = 64
PAGE_ROWS = 24_000

_CLUSTER = "coalesce(nullif(cluster, ''), 'local')"
_JOB = "coalesce(job_id, '')"
_UID = r"iris:[a-zA-Z0-9_.-]+"
_UID_PATTERN = re.compile(_UID)
_RUN_LINK = re.compile(rf"([a-z0-9-]+)@({_UID})(?:@(.*))?")
_SERVICE_FILTER = "service = 'marinskyrl'"
CONFIG_EVENT = "async_run_configuration"
CAPACITY_EVENT = "rollout_capacity"
GPU_METRICS = {
    "async/performance/configured_inference_gpus": "inferenceGpus",
    "async/performance/configured_policy_gpus": "policyGpus",
}
GRADING_METRICS = {"eval/all/timing/grading/mean", "eval/all/outcome/graded", "generate/outcome/graded"}
_CONTROLLER = "json_get(attributes_json, 'role') = 'controller'"
_START = f"name = 'lifecycle' AND {_CONTROLLER} AND json_get(body_json, 'state') = 'started'"
_END = f"name = 'terminal' AND {_CONTROLLER}"


@dataclass(frozen=True)
class Window:
    start: int
    end: int

    def as_dict(self) -> dict[str, int]:
        return {"from": self.start, "to": self.end}


@dataclass(frozen=True)
class Execution:
    cluster: str
    uid: str
    job: str

    @property
    def link_id(self) -> str:
        return f"{self.cluster}@{self.uid}@{self.job}"

    @property
    def predicate(self) -> str:
        cluster_column = _CLUSTER if self.cluster == "local" else "cluster"
        job_column = "job_id" if self.job else _JOB
        return (
            f"{cluster_column} = {_literal(self.cluster)} AND execution_uid = {_literal(self.uid)}"
            f" AND {job_column} = {_literal(self.job)}"
        )

    @classmethod
    def from_link_id(cls, run: str) -> "Execution | None":
        if match := _RUN_LINK.fullmatch(run):
            cluster, uid, job = match.groups()
            if job is not None:
                return cls(cluster, uid, job)
        return None


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_predicate(run: str) -> str:
    if match := _RUN_LINK.fullmatch(run):
        cluster, uid, job = match.groups()
        if job is not None:
            return Execution(cluster, uid, job).predicate
        return f"{_CLUSTER} = {_literal(cluster)} AND execution_uid = {_literal(uid)}"
    if _UID_PATTERN.fullmatch(run):
        return f"execution_uid = {_literal(run)}"
    return f"(run_id = {_literal(run)} OR job_id = {_literal(run)})"


def is_execution_id(run: str) -> bool:
    return _UID_PATTERN.fullmatch(run) is not None or _RUN_LINK.fullmatch(run) is not None


def _execution_bounds_sql(run: str, search: Window, predicate: str) -> str:
    return f"""
SELECT {_CLUSTER} AS cluster, execution_uid, {_JOB} AS job_id, min(run_id) AS run_id,
       min(timestamp_ms) AS first_ms, max(timestamp_ms) AS last_ms,
       min(CASE WHEN {_START} THEN timestamp_ms END) AS start_ms,
       max(CASE WHEN {_END} THEN timestamp_ms END) AS end_ms
FROM "{NAMESPACE}"
WHERE timestamp_ms >= {search.start} AND timestamp_ms < {search.end} AND {_SERVICE_FILTER}
  AND execution_uid IS NOT NULL AND execution_uid <> ''
  AND ({_run_predicate(run)})
  AND {predicate}
GROUP BY {_CLUSTER}, execution_uid, {_JOB}
ORDER BY last_ms DESC LIMIT {CHOICE_LIMIT + 1}"""


def lifecycle_sql(run: str, search: Window) -> str:
    return _execution_bounds_sql(run, search, f"name IN ('lifecycle', 'terminal') AND (({_START}) OR ({_END}))")


def activity_sql(run: str, search: Window) -> str:
    markers = (
        CONFIG_EVENT,
        CAPACITY_EVENT,
        "async_phase_window",
        "rollout_queue_depth",
        "weight_sync_completed",
        "work_completed",
    )
    names = _names(markers)
    return _execution_bounds_sql(
        run, search, f"name IN ('lifecycle', 'terminal', {names}) AND (name IN ({names}) OR ({_START}) OR ({_END}))"
    )


def telemetry_sql(
    execution: Execution, visible: Window, *, card: str | None, limit: int, after: tuple[int, int] | None
) -> str:
    start = visible.start - COUNTER_MARGIN_MS
    end = visible.end + COUNTER_MARGIN_MS
    if card is None:
        metadata_names = _names((CONFIG_EVENT, CAPACITY_EVENT))
        names = _names((*OVERVIEW_NAMES, CONFIG_EVENT, CAPACITY_EVENT))
        samples = f"""name IN ({names})
  AND (name IN ('async_phase_window', 'rollout_call', 'weight_sync_completed') OR ({DISPATCH_WAITS})
       OR (timestamp_ms < {end} AND (name IN ('rollout_groups', {metadata_names}) OR ({OVERVIEW_SIGNALS})
           OR (name = 'training_metric_value'
               AND json_get(attributes_json, 'metric') IN ({_names(tuple(GPU_METRICS))})))))"""
        end = visible.end + SPAN_LOOKAHEAD_MS
    else:
        samples = f"({DETAIL_SIGNALS[card]})"
    cursor = ""
    if after is not None:
        start, seq = after
        cursor = f"AND (timestamp_ms > {start} OR (timestamp_ms = {start} AND seq > {seq}))"
    return f"""
SELECT cluster, service, run_id, execution_uid, job_id, region, node_name, process_index, seq,
       timestamp_ms, name, value, body_json, attributes_json, resource_attributes_json
FROM "{NAMESPACE}"
WHERE timestamp_ms >= {start} AND timestamp_ms < {end} AND {_SERVICE_FILTER}
  {cursor}
  AND ({execution.predicate})
  AND {samples}
ORDER BY timestamp_ms, seq LIMIT {limit}"""


def _names(names: tuple[str, ...]) -> str:
    return ", ".join(map(_literal, names))


def detail_summary_sql(execution: Execution, window: Window, card: str, bucket_ms: int) -> str:
    """Roll up fleet metrics at a bucket width of at least two counter margins."""
    names = DETAIL_NAMES[card]
    counters = _names(tuple(name for name in names if name in CUMULATIVE_NAMES))
    snapshot_names = tuple(name for name in names if name in SNAPSHOT_NAMES)
    identity = "region, node_name, process_index, resource_attributes_json"
    readings = f"""
WITH readings AS (
    SELECT timestamp_ms, name, value, seq, {identity}, attributes_json
    FROM "{NAMESPACE}"
    WHERE timestamp_ms >= {window.start - COUNTER_MARGIN_MS}
      AND timestamp_ms < {window.end + COUNTER_MARGIN_MS}
      AND {_SERVICE_FILTER} AND ({execution.predicate})
      AND name IN ({_names(names)}) AND ({DETAIL_SIGNALS[card]}) AND value IS NOT NULL
)"""

    snapshots = ""
    if snapshot_names:
        snapshots = f"""
SELECT bucket_start, name, reason, SUM(engine_mean) AS value,
       CAST(NULL AS DOUBLE) AS requests, COUNT(*) AS engines
FROM (
    SELECT timestamp_ms - (timestamp_ms - {window.start}) % {bucket_ms} AS bucket_start,
           name, json_get(attributes_json, 'reason') AS reason, AVG(value) AS engine_mean
    FROM readings
    WHERE timestamp_ms >= {window.start} AND timestamp_ms < {window.end}
      AND name IN ({_names(snapshot_names)})
    GROUP BY bucket_start, name, reason, {identity},
             json_get(attributes_json, 'engine'), json_get(attributes_json, 'engine_index'),
             json_get(attributes_json, 'model_name')
) AS engine_snapshots
GROUP BY bucket_start, name, reason"""

    if not counters:
        return f"{readings}\n{snapshots}\nORDER BY bucket_start, name, reason"

    means = {name: count for name, count in CUMULATIVE_PAIRS.items() if name in names}
    name = "name"
    value = "FIRST_VALUE(value ORDER BY seq DESC)"
    requests = "CAST(NULL AS DOUBLE)"
    needs_pair = "TRUE"
    if means:
        count_names = _names(tuple(means.values()))
        pairs = " ".join(f"WHEN {_literal(count)} THEN {_literal(total)}" for total, count in means.items())
        name = f"CASE name {pairs} ELSE name END"
        requests = f"{value} FILTER (WHERE name IN ({count_names}))"
        value += f" FILTER (WHERE name NOT IN ({count_names}))"
        needs_pair = f"(metric NOT IN ({_names(tuple(means))}) OR requests >= previous_requests)"

    return f"""
{readings}, counter_samples AS (
    SELECT {name} AS metric, timestamp_ms, {identity}, attributes_json,
           {value} AS value, {requests} AS requests
    FROM readings WHERE name IN ({counters})
    GROUP BY metric, timestamp_ms, {identity}, attributes_json
), ordered AS (
    SELECT metric, timestamp_ms, value, requests,
           LAG(timestamp_ms) OVER engine AS previous_ms,
           LAG(value) OVER engine AS previous_value,
           LAG(requests) OVER engine AS previous_requests
    FROM counter_samples
    WINDOW engine AS (PARTITION BY metric, {identity}, attributes_json ORDER BY timestamp_ms)
), intervals AS (
    SELECT metric, GREATEST(previous_ms, {window.start}) AS interval_start,
           LEAST(timestamp_ms, {window.end}) AS interval_end,
           (value - previous_value) / (timestamp_ms - previous_ms) AS units_per_ms,
           (requests - previous_requests) / (timestamp_ms - previous_ms) AS requests_per_ms
    FROM ordered
    WHERE timestamp_ms - previous_ms BETWEEN 1 AND {COUNTER_MARGIN_MS}
      AND timestamp_ms > {window.start} AND previous_ms < {window.end}
      AND value >= previous_value AND {needs_pair}
), bucket_overlaps AS (
    SELECT metric, units_per_ms, requests_per_ms, interval_start, interval_end,
           interval_start - (interval_start - {window.start}) % {bucket_ms} + bucket_offset * {bucket_ms} AS bucket_start
    FROM intervals CROSS JOIN (VALUES (0), (1)) AS offsets(bucket_offset)
), contributions AS (
    SELECT *, LEAST(interval_end, bucket_start + {bucket_ms}) - GREATEST(interval_start, bucket_start) AS overlap
    FROM bucket_overlaps
)
SELECT bucket_start, metric AS name, CAST(NULL AS VARCHAR) AS reason,
       SUM(units_per_ms * overlap) AS value,
       SUM(requests_per_ms * overlap) AS requests, CAST(NULL AS BIGINT) AS engines
FROM contributions WHERE overlap > 0
GROUP BY bucket_start, metric
{"UNION ALL" + snapshots if snapshots else ""}
ORDER BY bucket_start, name, reason"""


def sampled_summary_sql(execution: Execution, windows: list[Window], card: str, slice_ms: int) -> str:
    """Give each slice its own time predicate so Finelog can skip the intervening data."""
    return " UNION ALL ".join(
        f"SELECT * FROM ({detail_summary_sql(execution, window, card, slice_ms)}) AS sample_{index}"
        for index, window in enumerate(windows)
    )


def driver_sql(execution: Execution, search: Window) -> str:
    return f"""
SELECT key, epoch_ms, seq, substr(data, 1, 360) AS line
FROM "log"
WHERE epoch_ms >= {search.start} AND epoch_ms <= {search.end} AND cluster = {_literal(execution.cluster)}
  AND prefix(key, {_literal(execution.job + '/0:')})
  AND ({DRIVER_MARKERS})
ORDER BY epoch_ms, seq LIMIT {DRIVER_LIMIT}"""
