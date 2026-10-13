# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog-backed API for the Zephyr Marina app.

Finelog is the stats endpoint named by ``ZEPHYR_FINELOG_URL`` (with
``ZEPHYR_IAP_CLUSTER`` for IAP credentials), or else the ``finelog-marin`` VM
found through GCE and reached by internal IP. The app only reads Finelog.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from connectrpc.errors import ConnectError
from fastapi import FastAPI, HTTPException, Query
from finelog.client import LogClient
from finelog.types import is_retryable_error
from marina.apps import RegisteredApi, Services, registered_api
from marina.discovery import resolve_internal_ip
from rigging.connect import IapAuth
from rigging.credentials import iap_provider_for

REDUCER_PAGE_SIZE = 20
EXECUTION_LIMIT = 100
EXECUTION_NAMESPACE = "zephyr.execution"
STAGE_LOOKBACK_SECONDS = 60


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def time_predicate(start: datetime) -> str:
    return f"ts >= TIMESTAMP {sql_string(start.isoformat())} AND ts <= now()"


def executions_sql(*, since: datetime, root_job: str | None, limit: int) -> str:
    where = [time_predicate(since)]
    if root_job:
        where.append(f"root_job_id = {sql_string(root_job)}")
    return f"""SELECT execution_id, root_job_id, coordinator_job_id, ts, input_shards, stages_json
FROM "{EXECUTION_NAMESPACE}"
WHERE {" AND ".join(where)}
QUALIFY ROW_NUMBER() OVER (PARTITION BY execution_id ORDER BY ts DESC, seq DESC) = 1
ORDER BY ts DESC LIMIT {int(limit)}"""


def execution_sql(execution_id: str) -> str:
    return f"""SELECT execution_id, root_job_id, coordinator_job_id, ts, input_shards, stages_json
FROM "{EXECUTION_NAMESPACE}"
WHERE execution_id = {sql_string(execution_id)}
QUALIFY ROW_NUMBER() OVER (PARTITION BY execution_id ORDER BY ts DESC, seq DESC) = 1"""


def stage_stats_sql(execution_id: str, start: datetime) -> str:
    return f"""SELECT stage_name, status, elapsed, items, total_shards, mem_peak_bytes_max
FROM "zephyr.stage" WHERE execution_id = {sql_string(execution_id)} AND {time_predicate(start)}
QUALIFY ROW_NUMBER() OVER (PARTITION BY stage_name ORDER BY ts DESC, seq DESC) = 1"""


def shuffle_snapshots_sql(execution_id: str, stage: str, start: datetime) -> str:
    """Keep each target's highest attempt, preferring measurements over placeholders."""
    return f"""WITH snapshots AS (
SELECT *, ROW_NUMBER() OVER (
PARTITION BY target_shard ORDER BY attempt DESC, (input_rows IS NOT NULL) DESC, ts DESC, seq DESC
) AS sample_rank
FROM "zephyr.shuffle" WHERE execution_id = {sql_string(execution_id)}
AND stage_name = {sql_string(stage)} AND {time_predicate(start)}
)"""


def reducer_stats_sql(execution_id: str, stage: str, start: datetime, page: int) -> str:
    return f"""{shuffle_snapshots_sql(execution_id, stage, start)}
SELECT target_shard, input_rows, payload_bytes, num_sources, attempt FROM snapshots
WHERE sample_rank = 1 ORDER BY payload_bytes DESC NULLS LAST, target_shard
LIMIT {REDUCER_PAGE_SIZE} OFFSET {int(page) * REDUCER_PAGE_SIZE}"""


def reducer_task_stats_sql(execution_id: str, stage: str, start: datetime, page: int) -> str:
    return f"""WITH targets AS ({reducer_stats_sql(execution_id, stage, start, page)}),
task_states AS (
SELECT shard_idx, status FROM "zephyr.worker"
WHERE execution_id = {sql_string(execution_id)} AND stage_name = {sql_string(stage)} AND {time_predicate(start)}
AND shard_idx IN (SELECT target_shard FROM targets)
QUALIFY ROW_NUMBER() OVER (PARTITION BY shard_idx ORDER BY ts DESC, seq DESC) = 1
)
SELECT targets.*, task_states.status AS task_status FROM targets
LEFT JOIN task_states ON targets.target_shard = task_states.shard_idx
ORDER BY payload_bytes DESC NULLS LAST, target_shard"""


def shuffle_summary_sql(execution_id: str, stage: str, start: datetime) -> str:
    return f"""{shuffle_snapshots_sql(execution_id, stage, start)}
SELECT COUNT(*) AS persisted_targets, COUNT(input_rows) AS observed_targets,
MAX(num_targets) AS expected_targets, MEDIAN(CAST(input_rows AS DOUBLE)) AS median_rows,
MAX(input_rows) AS max_rows, MEDIAN(CAST(payload_bytes AS DOUBLE)) AS median_bytes,
MAX(payload_bytes) AS max_bytes FROM snapshots WHERE sample_rank = 1"""


logger = logging.getLogger(__name__)

PROJECT = os.environ.get("ZEPHYR_GCP_PROJECT", "hai-gcp-models")
ZONE = os.environ.get("ZEPHYR_GCP_ZONE", "us-central1-a")
FINELOG_FILTER = "name = finelog-marin"
FINELOG_PORT = 10001
QUERY_TIMEOUT_MS = 12_000
MAX_ROWS = 20_000
NAMESPACE_CACHE_TTL = 60.0


@dataclass(frozen=True)
class ExecutionPlanStage:
    stage_name: str
    label: str
    stage_type: str
    has_reduce: bool
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    root_job_id: str
    coordinator_job_id: str
    ts: int
    input_shards: int
    stages: tuple[ExecutionPlanStage, ...]
    plan_error: str | None = None


class _Finelog:
    """Process-wide Finelog client with GCE-backed endpoint resolution."""

    def __init__(self, *, url: str | None, iap_cluster: str | None) -> None:
        self._url = url
        self._lock = threading.Lock()
        self._source = ""
        self._namespaces: list[str] = []
        self._namespaces_at = 0.0
        interceptors = tuple(IapAuth(iap_provider_for(iap_cluster)).interceptors()) if iap_cluster else ()
        self._client = LogClient.connect(
            url or FINELOG_FILTER,
            resolver=self._resolve,
            timeout_ms=QUERY_TIMEOUT_MS,
            interceptors=interceptors,
        )

    @property
    def source(self) -> str:
        return self._source

    def _resolve(self, _endpoint: str) -> str:
        url = self._url
        if url:
            source = url
            address = url
        else:
            ip = resolve_internal_ip(PROJECT, ZONE, FINELOG_FILTER, timeout=5.0)
            source = f"finelog-marin internal {ip}:{FINELOG_PORT}"
            address = f"http://{ip}:{FINELOG_PORT}"
        with self._lock:
            self._source = source
        return address

    def _call[Result](self, operation: Callable[[LogClient], Result]) -> Result:
        """Retry one idempotent read after LogClient invalidates its transport."""
        for attempt in range(2):
            try:
                return operation(self._client)
            except Exception as error:
                cause = error.__cause__
                retryable = (
                    isinstance(error, (ConnectionError, OSError, TimeoutError))
                    or (isinstance(error, ConnectError) and is_retryable_error(error))
                    or (isinstance(cause, ConnectError) and is_retryable_error(cause))
                )
                if attempt == 1 or not retryable:
                    raise
        raise AssertionError("unreachable")

    def query(self, sql: str) -> list[dict[str, Any]]:
        table = self._call(lambda client: client.query(sql, max_rows=MAX_ROWS))
        return [_json_row(row) for row in table.to_pylist()]

    def namespaces(self) -> list[str]:
        with self._lock:
            if time.monotonic() - self._namespaces_at < NAMESPACE_CACHE_TTL:
                return list(self._namespaces)
        infos = self._call(lambda client: client.list_namespaces())
        names = sorted(info.namespace for info in infos)
        with self._lock:
            self._namespaces = names
            self._namespaces_at = time.monotonic()
        return list(names)


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            stamp = value if value.tzinfo else value.replace(tzinfo=UTC)
            out[key] = int(stamp.timestamp() * 1000)
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            out[key] = None
        else:
            out[key] = value
    return out


def _execution_plan_stage(value: object) -> ExecutionPlanStage:
    if not isinstance(value, dict):
        raise ValueError("stage is not an object")
    stage_name = value.get("stage_name")
    label = value.get("label")
    stage_type = value.get("stage_type")
    has_reduce = value.get("has_reduce")
    dependencies = value.get("dependencies")
    if not isinstance(stage_name, str) or not isinstance(label, str) or not isinstance(stage_type, str):
        raise ValueError("stage names, labels, and types must be strings")
    if not isinstance(has_reduce, bool):
        raise ValueError("stage has_reduce must be a boolean")
    if not isinstance(dependencies, list) or not all(isinstance(dependency, str) for dependency in dependencies):
        raise ValueError("stage dependencies must be a list of strings")
    return ExecutionPlanStage(stage_name, label, stage_type, has_reduce, tuple(dependencies))


def _execution_record(row: dict[str, Any]) -> ExecutionRecord:
    stages_json = row.get("stages_json") or "[]"
    plan_error = None
    try:
        stage_values = json.loads(stages_json)
        if not isinstance(stage_values, list):
            raise ValueError("stages_json is not a list")
        stages = tuple(_execution_plan_stage(value) for value in stage_values)
    except (TypeError, ValueError) as error:
        plan_error = f"Cannot read execution plan: {error}"
        stages = ()
    return ExecutionRecord(
        execution_id=row["execution_id"],
        root_job_id=row["root_job_id"],
        coordinator_job_id=row["coordinator_job_id"],
        ts=row["ts"],
        input_shards=row["input_shards"],
        stages=stages,
        plan_error=plan_error,
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _execution_window_start(record: ExecutionRecord) -> datetime:
    return datetime.fromtimestamp(record.ts / 1000, UTC) - timedelta(seconds=STAGE_LOOKBACK_SECONDS)


def create_api(_services: Services) -> RegisteredApi:
    api = FastAPI()
    finelog = _Finelog(
        url=os.environ.get("ZEPHYR_FINELOG_URL"),
        iap_cluster=os.environ.get("ZEPHYR_IAP_CLUSTER"),
    )

    def unavailable(error: Exception) -> HTTPException:
        logger.warning("Finelog request failed: %s", error)
        return HTTPException(status_code=503, detail=f"Finelog unavailable: {type(error).__name__}: {error}")

    def run(sql: str) -> list[dict[str, Any]]:
        try:
            return finelog.query(sql)
        except Exception as error:
            raise unavailable(error) from error

    def run_namespaces() -> list[str]:
        try:
            return finelog.namespaces()
        except Exception as error:
            raise unavailable(error) from error

    def plan_or_404(execution_id: str) -> ExecutionRecord:
        rows = run(execution_sql(execution_id)) if EXECUTION_NAMESPACE in run_namespaces() else []
        if not rows:
            raise HTTPException(status_code=404, detail=f"No plan record for execution {execution_id}")
        return _execution_record(rows[0])

    @api.get("/health")
    def health() -> dict[str, Any]:
        names = run_namespaces()
        return {"finelog": finelog.source, "namespaces": names, "plan_records": EXECUTION_NAMESPACE in names}

    @api.get("/executions")
    def executions(
        days: int = Query(14, ge=1, le=90),
        limit: int = Query(EXECUTION_LIMIT, ge=1, le=500),
        root_job: str | None = None,
    ) -> list[ExecutionRecord]:
        if EXECUTION_NAMESPACE not in run_namespaces():
            return []
        since = _now() - timedelta(days=days)
        rows = run(executions_sql(since=since, root_job=root_job or None, limit=limit))
        return [_execution_record(row) for row in rows]

    @api.get("/executions/{execution_id}")
    def execution(execution_id: str) -> ExecutionRecord:
        return plan_or_404(execution_id)

    @api.get("/executions/{execution_id}/stages")
    def stages(execution_id: str) -> list[dict[str, Any]]:
        plan = plan_or_404(execution_id)
        return run(stage_stats_sql(execution_id, _execution_window_start(plan)))

    @api.get("/executions/{execution_id}/stages/{stage}/summary")
    def summary(execution_id: str, stage: str) -> dict[str, Any]:
        plan = plan_or_404(execution_id)
        rows = run(shuffle_summary_sql(execution_id, stage, _execution_window_start(plan)))
        return rows[0] if rows else {"persisted_targets": 0, "observed_targets": 0, "expected_targets": None}

    @api.get("/executions/{execution_id}/stages/{stage}/reducers")
    def reducers(execution_id: str, stage: str, page: int = Query(0, ge=0)) -> list[dict[str, Any]]:
        plan = plan_or_404(execution_id)
        start = _execution_window_start(plan)
        builder = reducer_task_stats_sql if "zephyr.worker" in run_namespaces() else reducer_stats_sql
        return run(builder(execution_id, stage, start, page))

    return registered_api(api)
