# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog-backed API for the Zephyr executions applet.

Finelog is the stats endpoint named by ``ZEPHYR_APPLET_FINELOG_URL`` (with
``ZEPHYR_APPLET_IAP_CLUSTER`` for IAP credentials), or else the
``finelog-marin`` VM found through GCE and reached by internal IP. The applet
only reads; it never writes to Finelog or to its own schema.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import closing
from datetime import UTC, datetime, timedelta
from typing import Any

from connectrpc.errors import ConnectError
from fastapi import FastAPI, HTTPException, Query
from finelog.client import LogClient
from google.cloud import compute_v1
from marina.applets import AppletServices
from rigging.connect import IapAuth
from rigging.credentials import iap_provider_for

from .queries import (
    EXECUTION_LIMIT,
    STAGE_LOOKBACK_SECONDS,
    execution_sql,
    executions_sql,
    reducer_stats_sql,
    reducer_task_stats_sql,
    shuffle_summary_sql,
    stage_stats_sql,
)

logger = logging.getLogger(__name__)

PROJECT = os.environ.get("ZEPHYR_APPLET_GCP_PROJECT", "hai-gcp-models")
ZONE = os.environ.get("ZEPHYR_APPLET_GCP_ZONE", "us-central1-a")
FINELOG_FILTER = "name = finelog-marin"
FINELOG_PORT = 10001
ADDRESS_CACHE_TTL = 300.0
QUERY_TIMEOUT_MS = 12_000
MAX_ROWS = 20_000
NAMESPACE_CACHE_TTL = 60.0


class _Finelog:
    """Cached Finelog address; one short-lived client per call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._address: str | tuple[str, int] | None = None
        self._interceptors: tuple[Any, ...] = ()
        self._source = ""
        self._resolved_at = 0.0
        self._namespaces: list[str] = []
        self._namespaces_at = 0.0

    @property
    def source(self) -> str:
        return self._source

    def _resolve_explicit(self) -> bool:
        url = os.environ.get("ZEPHYR_APPLET_FINELOG_URL")
        if not url:
            return False
        cluster = os.environ.get("ZEPHYR_APPLET_IAP_CLUSTER")
        self._interceptors = tuple(IapAuth(iap_provider_for(cluster)).interceptors()) if cluster else ()
        self._address = url
        self._source = f"{url} ({'IAP ' + cluster if cluster else 'direct'})"
        return True

    def _resolve_internal(self) -> None:
        request = compute_v1.ListInstancesRequest(project=PROJECT, zone=ZONE, filter=FINELOG_FILTER)
        for instance in compute_v1.InstancesClient().list(request=request, timeout=5.0):
            for interface in instance.network_interfaces:
                if interface.network_i_p:
                    self._address = (interface.network_i_p, FINELOG_PORT)
                    self._interceptors = ()
                    self._source = f"{instance.name} internal {interface.network_i_p}:{FINELOG_PORT}"
                    return
        raise RuntimeError(f"no VM with an internal IP for filter {FINELOG_FILTER!r} in {ZONE}")

    def _connect(self) -> LogClient:
        with self._lock:
            if self._address is None or time.monotonic() - self._resolved_at >= ADDRESS_CACHE_TTL:
                if not self._resolve_explicit():
                    self._resolve_internal()
                self._resolved_at = time.monotonic()
            address, interceptors = self._address, self._interceptors
        assert address is not None
        return LogClient.connect(address, timeout_ms=QUERY_TIMEOUT_MS, interceptors=interceptors)

    def invalidate(self) -> None:
        with self._lock:
            self._address = None

    def _call(self, operation: Callable[[LogClient], Any]) -> Any:
        """Run one operation on a fresh client, re-resolving once after a transport failure."""
        for attempt in range(2):
            try:
                with closing(self._connect()) as client:
                    return operation(client)
            except ConnectError:
                self.invalidate()
                if attempt == 1:
                    raise
        raise AssertionError("unreachable")

    def query(self, sql: str) -> list[dict[str, Any]]:
        table = self._call(lambda client: client.query(sql, max_rows=MAX_ROWS))
        return [_json_row(row) for row in table.to_pylist()]

    def namespaces(self) -> list[str]:
        with self._lock:
            if time.monotonic() - self._namespaces_at < NAMESPACE_CACHE_TTL:
                return list(self._namespaces)
        infos: Iterable[Any] = self._call(lambda client: client.list_namespaces())
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


def _plan_row(row: dict[str, Any]) -> dict[str, Any]:
    stages_json = row.pop("stages_json", None) or "[]"
    try:
        stages = json.loads(stages_json)
        if not isinstance(stages, list):
            raise ValueError("stages_json is not a list")
    except ValueError as error:
        row["plan_error"] = f"Cannot read execution plan: {error}"
        stages = []
    row["stages"] = stages
    return row


def _now() -> datetime:
    return datetime.now(UTC)


def _stage_start(plan: dict[str, Any]) -> datetime:
    return datetime.fromtimestamp(plan["ts"] / 1000, UTC) - timedelta(seconds=STAGE_LOOKBACK_SECONDS)


def create_api(services: AppletServices | None = None) -> FastAPI:
    api = FastAPI()
    finelog = _Finelog()

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

    def plan_or_404(execution_id: str) -> dict[str, Any]:
        rows = run(execution_sql(execution_id)) if "zephyr.execution" in run_namespaces() else []
        if not rows:
            raise HTTPException(status_code=404, detail=f"No plan record for execution {execution_id}")
        return _plan_row(rows[0])

    @api.get("/health")
    def health() -> dict[str, Any]:
        names = run_namespaces()
        return {"finelog": finelog.source, "namespaces": names, "plan_records": "zephyr.execution" in names}

    @api.get("/executions")
    def executions(
        days: int = Query(14, ge=1, le=90),
        limit: int = Query(EXECUTION_LIMIT, ge=1, le=500),
        root_job: str | None = None,
    ) -> list[dict[str, Any]]:
        if "zephyr.execution" not in run_namespaces():
            return []
        since = _now() - timedelta(days=days)
        rows = run(executions_sql(since=since, root_job=root_job or None, limit=limit))
        return [_plan_row(row) for row in rows]

    @api.get("/executions/{execution_id}")
    def execution(execution_id: str) -> dict[str, Any]:
        return plan_or_404(execution_id)

    @api.get("/executions/{execution_id}/stages")
    def stages(execution_id: str) -> list[dict[str, Any]]:
        plan = plan_or_404(execution_id)
        return run(stage_stats_sql(execution_id, _stage_start(plan)))

    @api.get("/executions/{execution_id}/stages/{stage}/summary")
    def summary(execution_id: str, stage: str) -> dict[str, Any]:
        plan = plan_or_404(execution_id)
        rows = run(shuffle_summary_sql(execution_id, stage, _stage_start(plan)))
        return rows[0] if rows else {"persisted_targets": 0, "observed_targets": 0, "expected_targets": None}

    @api.get("/executions/{execution_id}/stages/{stage}/reducers")
    def reducers(execution_id: str, stage: str, page: int = Query(0, ge=0)) -> list[dict[str, Any]]:
        plan = plan_or_404(execution_id)
        start = _stage_start(plan)
        builder = reducer_task_stats_sql if "zephyr.worker" in run_namespaces() else reducer_stats_sql
        return run(builder(execution_id, stage, start, page))

    return api
