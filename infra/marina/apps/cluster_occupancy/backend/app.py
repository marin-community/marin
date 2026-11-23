# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""API for the Cluster Occupancy Marina app.

Two read-only sources, one map per cluster:

- **Now** — every cluster's controller database answers read-only SELECTs over
  the Iris raw-query RPC (``live.py``), so the current moment is available
  even for clusters whose controller does not publish mirror rows.
- **History** — the Finelog ``marin`` hub's ``iris.controller_db__*`` mirrors.
  Each mirror row is a snapshot: queries take the latest row per key by
  ``ts``. Placement comes from ``task_attempts.node_name``, grouped by the
  NVLink domain (rack) when the node reports one.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from connectrpc.errors import ConnectError
from fastapi import FastAPI, HTTPException, Query
from finelog.client import LogClient
from finelog.types import is_retryable_error
from marina.apps import RegisteredApi, Services, registered_api
from marina.discovery import resolve_internal_ip
from rigging.connect import IapAuth
from rigging.credentials import iap_provider_for

from . import live

logger = logging.getLogger(__name__)

PROJECT = os.environ.get("OCCUPANCY_GCP_PROJECT", "hai-gcp-models")
ZONE = os.environ.get("OCCUPANCY_GCP_ZONE", "us-central1-a")
FINELOG_FILTER = "name = finelog-marin"
FINELOG_PORT = 10001
QUERY_TIMEOUT_MS = 20_000
MAX_ROWS = 50_000

JOBS = "iris.controller_db__jobs"
JOB_CONFIG = "iris.controller_db__job_config"
TASK_ATTEMPTS = "iris.controller_db__task_attempts"

# The clusters the map always shows, whether or not the hub mirrors them yet.
KNOWN_CLUSTERS = ("cw-us-east-08a", "cw-us-east-02a", "cw-rno2a")

WINDOW_HOURS = 1
STEP_MS = 5 * 60 * 1000
# The mirror writes every minute or so; a window starting well past the last
# snapshot means forwarding stalled, and an idle-looking map there would lie.
STALE_MS = 30 * 60 * 1000
COVERAGE_TTL_SECONDS = 60.0
INVENTORY_TTL_SECONDS = 300.0

# An idle-looking map for a cluster with no mirror data would lie, so every
# cluster gets an explicit state instead of an empty node list.
STATE_MIRRORED = "mirrored"
STATE_NO_DATA = "no_data"
BEFORE_COVERAGE = "before_coverage"
AFTER_COVERAGE = "after_coverage"
NOT_MIRRORED = "not_mirrored"


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def window_timestamp(epoch_ms: int) -> str:
    """A UTC ``TIMESTAMP`` literal for a Finelog ``timestamp_ms`` comparison."""
    return f"TIMESTAMP '{datetime.fromtimestamp(epoch_ms / 1000, UTC).strftime('%Y-%m-%d %H:%M:%S')}'"


JOBS_TABLE_SQL = f"""
SELECT cluster, job_id, root_job_id, user_id, name, num_tasks, started_at, finished_at,
       row_number() OVER (PARTITION BY cluster, job_id ORDER BY ts DESC, seq DESC) AS rn
FROM "{JOBS}" WHERE cluster LIKE 'cw-%'
"""

CONFIG_TABLE_SQL = f"""
SELECT cluster, job_id, priority_band, accelerator_type, accelerators_per_task,
       row_number() OVER (PARTITION BY cluster, job_id ORDER BY ts DESC, seq DESC) AS rn
FROM "{JOB_CONFIG}" WHERE cluster LIKE 'cw-%'
"""

ATTEMPTS_TABLE_SQL = f"""
SELECT cluster, job_id, task_id, node_name, started_at, finished_at, topology_json,
       row_number() OVER (PARTITION BY cluster, job_id, task_id ORDER BY ts DESC, seq DESC) AS rn
FROM "{TASK_ATTEMPTS}" WHERE cluster LIKE 'cw-%' AND coalesce(node_name, '') <> ''
"""


def jobs_sql(start_ms: int, end_ms: int) -> str:
    """Jobs that held the cluster at any point in the window, one row per job.

    Window membership follows the reference query: started before the window
    ends and not finished before it starts. The per-job latest snapshot is
    picked before the window filter so a finished job's older running-looking
    snapshot cannot survive.
    """
    return f"""WITH jobs AS ({JOBS_TABLE_SQL}),
cfg AS ({CONFIG_TABLE_SQL})
SELECT j.cluster, j.job_id, j.root_job_id, j.user_id, j.name,
       j.started_at, j.finished_at,
       c.priority_band, c.accelerator_type, c.accelerators_per_task,
       j.num_tasks * coalesce(c.accelerators_per_task, 0) AS gpus
FROM jobs j JOIN cfg c ON c.cluster = j.cluster AND c.job_id = j.job_id AND c.rn = 1
WHERE j.rn = 1 AND j.started_at IS NOT NULL
  AND j.started_at < {window_timestamp(end_ms)}
  AND (j.finished_at IS NULL OR j.finished_at >= {window_timestamp(start_ms)})
ORDER BY gpus DESC"""


def placements_sql(start_ms: int, end_ms: int) -> str:
    """One row per task placement in the window, for jobs the window holds."""
    return f"""WITH jobs AS ({JOBS_TABLE_SQL}),
att AS ({ATTEMPTS_TABLE_SQL})
SELECT a.cluster, a.job_id, a.node_name, a.started_at, a.finished_at
FROM att a JOIN jobs j ON j.cluster = a.cluster AND j.job_id = a.job_id AND j.rn = 1
WHERE a.rn = 1 AND a.started_at IS NOT NULL
  AND a.started_at < {window_timestamp(end_ms)}
  AND (a.finished_at IS NULL OR a.finished_at >= {window_timestamp(start_ms)})"""


def root_names_sql(identities: list[tuple[str, str]]) -> str:
    """Current names for the root jobs behind the window's jobs.

    A root job may have finished before the window while a child still runs, so
    this lookup cannot reuse the window filter.
    """
    pairs = " OR ".join(
        f"(cluster = {sql_string(cluster)} AND job_id = {sql_string(job_id)})" for cluster, job_id in identities
    )
    return f"""WITH jobs AS ({JOBS_TABLE_SQL})
SELECT cluster, job_id, name FROM jobs WHERE rn = 1 AND ({pairs})"""


def coverage_sql() -> str:
    """Mirror coverage per cluster: the span of snapshot times the hub holds."""
    return f"""SELECT cluster, min(ts) AS first_ts, max(ts) AS last_ts, count(*) AS rows
FROM "{JOBS}" WHERE cluster LIKE 'cw-%' AND ts IS NOT NULL
GROUP BY cluster"""


def inventory_sql() -> str:
    """Every node the mirror has seen holding a GPU task.

    ``capacity_gpus`` is the most GPUs seen held at once on the node, the only
    capacity the controller data can support. Rack grouping is the node's
    NVLink domain, reported by CoreWeave on GB200 nodes.
    """
    return f"""WITH att AS ({ATTEMPTS_TABLE_SQL}),
cfg AS ({CONFIG_TABLE_SQL}),
gpu AS (
  SELECT a.cluster, a.node_name, a.started_at, a.finished_at,
         c.accelerator_type, c.accelerators_per_task
  FROM att a JOIN cfg c ON c.cluster = a.cluster AND c.job_id = a.job_id AND c.rn = 1
  WHERE a.rn = 1 AND c.accelerators_per_task > 0
),
events AS (
  SELECT cluster, node_name, started_at AS t, accelerators_per_task AS delta FROM gpu
  UNION ALL
  SELECT cluster, node_name, finished_at, -accelerators_per_task FROM gpu WHERE finished_at IS NOT NULL
),
held AS (
  SELECT cluster, node_name, t,
         sum(delta) OVER (
           PARTITION BY cluster, node_name ORDER BY t, delta DESC
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS concurrent
  FROM events
),
racks AS (
  SELECT cluster, node_name,
         max(nullif(json_get(topology_json, 'ds.coreweave.com/nvlink.domain'), '')) AS rack
  FROM att GROUP BY cluster, node_name
),
kinds AS (
  SELECT cluster, node_name, max(accelerator_type) AS gpu_type FROM gpu GROUP BY cluster, node_name
)
SELECT h.cluster, h.node_name, coalesce(r.rack, '') AS rack, coalesce(k.gpu_type, '') AS gpu_type,
       max(h.concurrent) AS capacity_gpus
FROM held h
JOIN racks r ON r.cluster = h.cluster AND r.node_name = h.node_name
JOIN kinds k ON k.cluster = h.cluster AND k.node_name = h.node_name
GROUP BY h.cluster, h.node_name, r.rack, k.gpu_type"""


class _Finelog:
    """Process-wide Finelog client with GCE-backed endpoint resolution."""

    def __init__(self, *, url: str | None, iap_cluster: str | None) -> None:
        self._url = url
        self._lock = threading.Lock()
        self._source = url or FINELOG_FILTER
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
        if self._url:
            address = self._url
        else:
            ip = resolve_internal_ip(PROJECT, ZONE, FINELOG_FILTER, timeout=5.0)
            address = f"http://{ip}:{FINELOG_PORT}"
        with self._lock:
            self._source = f"finelog-marin {address}" if not self._url else self._url
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
            if time.monotonic() - self._namespaces_at < COVERAGE_TTL_SECONDS:
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


@dataclass(frozen=True)
class Holder:
    """What one root job holds on one node during the window."""

    user_id: str | None
    job_id: str | None
    root_job_id: str | None
    root_name: str | None
    priority_band: str | None
    accelerator_type: str | None
    gpus: int
    started_ms: int | None
    finished_ms: int | None


@dataclass(frozen=True)
class NodeView:
    node_name: str
    rack: str
    gpu_type: str
    capacity_gpus: int
    holders: list[dict[str, Any]]


class _Caches:
    """Coverage and node inventory, refreshed on short TTLs."""

    def __init__(self, finelog: _Finelog) -> None:
        self._finelog = finelog
        self._lock = threading.Lock()
        self._coverage: tuple[float, list[dict[str, Any]]] | None = None
        self._inventory: tuple[float, list[dict[str, Any]]] | None = None

    def _cached(self, entry: tuple[float, list[dict[str, Any]]] | None, ttl: float) -> list[dict[str, Any]] | None:
        if entry is None or time.monotonic() - entry[0] >= ttl:
            return None
        return entry[1]

    def coverage(self) -> list[dict[str, Any]]:
        cached = self._cached(self._coverage, COVERAGE_TTL_SECONDS)
        if cached is not None:
            return cached
        rows = self._finelog.query(coverage_sql())
        with self._lock:
            self._coverage = (time.monotonic(), rows)
        return rows

    def inventory(self) -> list[dict[str, Any]]:
        cached = self._cached(self._inventory, INVENTORY_TTL_SECONDS)
        if cached is not None:
            return cached
        rows = self._finelog.query(inventory_sql())
        with self._lock:
            self._inventory = (time.monotonic(), rows)
        return rows


def _cluster_state(coverage: dict[str, Any] | None, start_ms: int, end_ms: int) -> tuple[str, str | None]:
    if coverage is None:
        return STATE_NO_DATA, NOT_MIRRORED
    first_ms, last_ms = coverage["first_ts"], coverage["last_ts"]
    if end_ms <= first_ms:
        return STATE_NO_DATA, BEFORE_COVERAGE
    if start_ms > last_ms + STALE_MS:
        return STATE_NO_DATA, AFTER_COVERAGE
    return STATE_MIRRORED, None


def _overlap_seconds(start_ms: int, end_ms: int, job_started: int, job_finished: int | None) -> float:
    lo = max(start_ms, job_started)
    hi = min(end_ms, job_finished) if job_finished is not None else end_ms
    return max(0.0, (hi - lo) / 1000.0)


def _node_holders(
    placements: list[dict[str, Any]],
    jobs_by_id: dict[tuple[str, str], dict[str, Any]],
    root_names: dict[tuple[str, str], str],
    start_ms: int,
) -> dict[tuple[str, str], list[Holder]]:
    """One holder per (cluster, node, job): its tasks on that node, in GPUs."""
    counted: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in placements:
        finished = row.get("finished_at")
        if finished is not None and finished <= start_ms:
            continue  # the placement ended when the window began: it held nothing inside
        key = (row["cluster"], row["job_id"], row["node_name"])
        entry = counted.setdefault(key, {"tasks": 0, "started_ms": None, "finished_ms": None, "running": False})
        entry["tasks"] += 1
        started = row.get("started_at")
        if started is not None and (entry["started_ms"] is None or started < entry["started_ms"]):
            entry["started_ms"] = started
        if finished is None:
            entry["running"] = True
            entry["finished_ms"] = None
        elif not entry["running"]:
            entry["finished_ms"] = (
                finished if entry["finished_ms"] is None else max(entry["finished_ms"], finished)
            )

    holders: dict[tuple[str, str], list[Holder]] = {}
    for (cluster, job_id, node_name), entry in counted.items():
        job = jobs_by_id.get((cluster, job_id))
        if job is None:
            continue
        gpus = entry["tasks"] * (job.get("accelerators_per_task") or 0)
        if gpus <= 0:
            continue  # a task without accelerators holds no part of the GPU map
        holder = Holder(
            user_id=job.get("user_id"),
            job_id=job_id,
            root_job_id=job.get("root_job_id"),
            root_name=root_names.get((cluster, job.get("root_job_id") or job_id), job.get("name")),
            priority_band=job.get("priority_band"),
            accelerator_type=job.get("accelerator_type"),
            gpus=gpus,
            started_ms=entry["started_ms"],
            finished_ms=entry["finished_ms"],
        )
        holders.setdefault((cluster, node_name), []).append(holder)
    return holders


def _legend(jobs: list[dict[str, Any]], start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    """GPU-hours per user across the visible window, following the reference query."""
    per_user: dict[str, float] = {}
    for job in jobs:
        gpus = job.get("gpus") or 0
        if gpus <= 0:
            continue
        user = job.get("user_id") or ""
        hours = gpus * _overlap_seconds(start_ms, end_ms, job["started_at"], job.get("finished_at")) / 3600.0
        per_user[user] = per_user.get(user, 0.0) + hours
    return [
        {"user_id": user, "gpu_hours": hours}
        for user, hours in (
            (user, round(hours, 2)) for user, hours in sorted(per_user.items(), key=lambda item: (-item[1], item[0]))
        )
        if hours > 0
    ]


def create_api(_services: Services) -> RegisteredApi:
    api = FastAPI()
    finelog = _Finelog(
        url=os.environ.get("OCCUPANCY_FINELOG_URL"),
        iap_cluster=os.environ.get("OCCUPANCY_IAP_CLUSTER"),
    )
    caches = _Caches(finelog)

    def unavailable(error: Exception) -> HTTPException:
        logger.warning("Finelog request failed: %s", error)
        return HTTPException(status_code=503, detail=f"Finelog unavailable: {type(error).__name__}: {error}")

    def run(sql: str) -> list[dict[str, Any]]:
        try:
            return finelog.query(sql)
        except Exception as error:
            raise unavailable(error) from error

    def namespaces() -> list[str]:
        try:
            return finelog.namespaces()
        except Exception as error:
            raise unavailable(error) from error

    @api.get("/health")
    def health() -> dict[str, Any]:
        names = namespaces()
        return {
            "finelog": finelog.source,
            "namespaces_present": [ns for ns in (JOBS, JOB_CONFIG, TASK_ATTEMPTS) if ns in names],
        }

    @api.get("/coverage")
    def coverage() -> dict[str, Any]:
        """Mirror coverage per cluster and the slider range it implies."""
        rows = caches.coverage() if JOBS in namespaces() else []
        by_cluster = {row["cluster"]: row for row in rows}
        clusters = []
        for cluster in KNOWN_CLUSTERS:
            row = by_cluster.get(cluster)
            clusters.append(
                {
                    "cluster": cluster,
                    "first_ms": row["first_ts"] if row else None,
                    "last_ms": row["last_ts"] if row else None,
                }
            )
        for cluster, row in sorted(by_cluster.items()):
            if cluster not in KNOWN_CLUSTERS:
                clusters.append({"cluster": cluster, "first_ms": row["first_ts"], "last_ms": row["last_ts"]})
        starts = [c["first_ms"] for c in clusters if c["first_ms"] is not None]
        ends = [c["last_ms"] for c in clusters if c["last_ms"] is not None]
        earliest = min(starts) // STEP_MS * STEP_MS if starts else None
        latest = max(ends) if ends else None
        return {
            "window_hours": WINDOW_HOURS,
            "step_ms": STEP_MS,
            "earliest_ms": earliest,
            "latest_ms": latest,
            "now_ms": int(time.time() * 1000),
            "clusters": clusters,
        }

    @api.get("/now")
    def now_view() -> dict[str, Any]:
        """The current moment, read live from every cluster's controller.

        Covers clusters the mirror does not reach (cw-rno2a): its controller
        answers the same read-only SELECTs, so "now" is real even though its
        history is unavailable.
        """
        as_of_ms = int(time.time() * 1000)
        clusters: list[dict[str, Any]] = []
        rows: list[dict[str, object]] = []
        for cluster in KNOWN_CLUSTERS:
            try:
                cluster_rows = live.running_tasks(cluster)
            except Exception as error:
                logger.warning("Live read failed for %s: %s", cluster, error)
                clusters.append(
                    {
                        "cluster": cluster,
                        "state": "unavailable",
                        "reason": "controller_error",
                        "as_of_ms": as_of_ms,
                        "nodes": [],
                    }
                )
                continue
            rows.extend(cluster_rows)
            clusters.append(
                {
                    "cluster": cluster,
                    "state": "live",
                    "reason": None,
                    "as_of_ms": as_of_ms,
                    "nodes": live.node_views(cluster_rows),
                }
            )
        return {"as_of_ms": as_of_ms, "clusters": clusters, "legend": live.legend(rows)}

    @api.get("/occupancy")
    def occupancy(start_ms: int = Query(...)) -> dict[str, Any]:
        """The map for the window [start_ms, start_ms + window hours)."""
        if start_ms <= 0 or start_ms > (1 << 45):
            raise HTTPException(status_code=422, detail="start_ms must be a positive epoch-milliseconds value")
        end_ms = start_ms + WINDOW_HOURS * 3600 * 1000

        jobs = run(jobs_sql(start_ms, end_ms)) if JOBS in namespaces() else []
        placements = run(placements_sql(start_ms, end_ms)) if TASK_ATTEMPTS in namespaces() else []
        jobs_by_id = {(row["cluster"], row["job_id"]): row for row in jobs}
        root_identities = sorted(
            {
                (row["cluster"], row["root_job_id"] or row["job_id"])
                for row in jobs
                if row.get("root_job_id") and row["root_job_id"] != row["job_id"]
            }
        )
        root_names: dict[tuple[str, str], str] = {}
        if root_identities:
            root_names = {
                (row["cluster"], row["job_id"]): row["name"] for row in run(root_names_sql(root_identities))
            }

        inventory = caches.inventory() if TASK_ATTEMPTS in namespaces() else []
        holders = _node_holders(placements, jobs_by_id, root_names, start_ms)
        coverage_rows = caches.coverage() if JOBS in namespaces() else []
        by_cluster_coverage = {row["cluster"]: row for row in coverage_rows}

        clusters: list[dict[str, Any]] = []
        seen = set()
        for cluster in (*KNOWN_CLUSTERS, *(c for c in sorted(by_cluster_coverage) if c not in KNOWN_CLUSTERS)):
            if cluster in seen:
                continue
            seen.add(cluster)
            coverage_row = by_cluster_coverage.get(cluster)
            state, reason = _cluster_state(coverage_row, start_ms, end_ms)
            nodes = []
            if state == STATE_MIRRORED:
                for node in inventory:
                    if node["cluster"] != cluster:
                        continue
                    node_holders = sorted(
                        (asdict(h) for h in holders.get((cluster, node["node_name"]), [])),
                        key=lambda holder: (-holder["gpus"], holder["root_name"] or "", holder["job_id"] or ""),
                    )
                    nodes.append(
                        asdict(
                            NodeView(
                                node_name=node["node_name"],
                                rack=node["rack"],
                                gpu_type=node["gpu_type"],
                                capacity_gpus=node["capacity_gpus"],
                                holders=node_holders,
                            )
                        )
                    )
                nodes.sort(key=lambda node: (node["rack"] == "", node["rack"], node["node_name"]))
            clusters.append(
                {
                    "cluster": cluster,
                    "state": state,
                    "reason": reason,
                    "first_ms": coverage_row["first_ts"] if coverage_row else None,
                    "last_ms": coverage_row["last_ts"] if coverage_row else None,
                    "nodes": nodes,
                }
            )
        return {
            "window": {"start_ms": start_ms, "end_ms": end_ms, "hours": WINDOW_HOURS},
            "clusters": clusters,
            "legend": _legend(jobs, start_ms, end_ms),
        }

    return registered_api(api)
