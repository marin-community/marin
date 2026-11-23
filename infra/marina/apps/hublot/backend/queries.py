# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded reads of SkyRL telemetry and driver timing facts."""

import re
from dataclasses import dataclass

from .query_contract import COUNTER_MARGIN_MS, DISPATCH_WAITS, DRIVER_MARKERS, REGULAR_SIGNALS, SPAN_LOOKAHEAD_MS

NAMESPACE = "telemetry_v1.marinskyrl"
CHOICE_LIMIT = 8
DRIVER_LIMIT = 64
PAGE_ROWS = 24_000

_CLUSTER = "coalesce(nullif(cluster, ''), 'local')"
_JOB = "coalesce(job_id, '')"
_UID = r"iris:[a-zA-Z0-9_.-]+"
_SERVICE_FILTER = "service = 'marinskyrl'"
_CONFIG_EVENT = "'async_run_configuration'"
_CAPACITY_EVENT = "'rollout_capacity'"
_WAIT = "name = 'async_phase_window' AND json_get(attributes_json, 'phase') = 'rollout_wait'"
_GRADING = """name = 'training_metric_value' AND json_get(attributes_json, 'metric') IN
('eval/all/timing/grading/mean', 'eval/all/outcome/graded', 'generate/outcome/graded')"""
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
        return (
            f"{_CLUSTER} = {_literal(self.cluster)} AND execution_uid = {_literal(self.uid)}"
            f" AND {_JOB} = {_literal(self.job)}"
        )


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_predicate(run: str) -> str:
    if match := re.fullmatch(rf"([a-z0-9-]+)@({_UID})(?:@(.*))?", run):
        cluster, uid, job = match.groups()
        if job is not None:
            return Execution(cluster, uid, job).predicate
        return f"{_CLUSTER} = {_literal(cluster)} AND execution_uid = {_literal(uid)}"
    if re.fullmatch(_UID, run):
        return f"execution_uid = {_literal(run)}"
    return f"(run_id = {_literal(run)} OR job_id = {_literal(run)})"


def _gpu_sample_predicate(role: str) -> str:
    metric = _literal(f"async/performance/configured_{role}_gpus")
    return f"name = 'training_metric_value' AND json_get(attributes_json, 'metric') = {metric}"


def resolution_sql(run: str, search: Window) -> str:
    inference_gpus, policy_gpus = _gpu_sample_predicate("inference"), _gpu_sample_predicate("policy")
    # A bounded aggregate returns just enough metadata to choose and open one execution.
    return f"""
SELECT {_CLUSTER} AS cluster, execution_uid, {_JOB} AS job_id, min(run_id) AS run_id,
       min(timestamp_ms) AS first_ms, max(timestamp_ms) AS last_ms,
       max(CASE WHEN {_WAIT} THEN timestamp_ms END) AS wait_ms,
       count(DISTINCT CASE WHEN name = {_CONFIG_EVENT} THEN body_json END) AS configurations,
       max(CASE WHEN name = {_CONFIG_EVENT} THEN body_json END) AS configuration,
       max(CASE WHEN {_GRADING} THEN timestamp_ms END) AS grading_ms,
       min(CASE WHEN {_START} THEN timestamp_ms END) AS start_ms,
       max(CASE WHEN {_END} THEN timestamp_ms END) AS end_ms,
       min(CASE WHEN {inference_gpus} THEN value END) AS inference_gpu_min,
       max(CASE WHEN {inference_gpus} THEN value END) AS inference_gpu_max,
       min(CASE WHEN {policy_gpus} THEN value END) AS policy_gpu_min,
       max(CASE WHEN {policy_gpus} THEN value END) AS policy_gpu_max,
       min(CASE WHEN name = {_CAPACITY_EVENT} THEN value END) AS buffer_capacity_min,
       max(CASE WHEN name = {_CAPACITY_EVENT} THEN value END) AS buffer_capacity_max
FROM "{NAMESPACE}"
WHERE timestamp_ms >= {search.start} AND timestamp_ms < {search.end} AND {_SERVICE_FILTER}
  AND execution_uid IS NOT NULL AND execution_uid <> ''
  AND ({_run_predicate(run)})
  AND (name IN ('async_phase_window', {_CONFIG_EVENT}, 'generation_tokens_total',
                'rollout_queue_depth', {_CAPACITY_EVENT})
       OR ({_GRADING}) OR ({_START}) OR ({_END}) OR ({inference_gpus}) OR ({policy_gpus}))
GROUP BY {_CLUSTER}, execution_uid, {_JOB}
ORDER BY last_ms DESC LIMIT {CHOICE_LIMIT + 1}"""


def telemetry_sql(execution: Execution, visible: Window, *, limit: int, after: tuple[int, int] | None) -> str:
    start = visible.start - COUNTER_MARGIN_MS
    end = visible.end + SPAN_LOOKAHEAD_MS
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
  AND (name IN ('async_phase_window', 'rollout_call', 'weight_sync_completed') OR ({DISPATCH_WAITS})
       OR (timestamp_ms < {visible.end + COUNTER_MARGIN_MS} AND (name = 'rollout_groups' OR ({REGULAR_SIGNALS}))))
ORDER BY timestamp_ms, seq LIMIT {limit}"""


def driver_sql(execution: Execution, search: Window) -> str:
    return f"""
SELECT key, epoch_ms, seq, substr(data, 1, 360) AS line
FROM "log"
WHERE epoch_ms >= {search.start} AND epoch_ms <= {search.end} AND cluster = {_literal(execution.cluster)}
  AND prefix(key, {_literal(execution.job + '/0:')})
  AND ({DRIVER_MARKERS})
ORDER BY epoch_ms, seq LIMIT {DRIVER_LIMIT}"""
