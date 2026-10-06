# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Point-in-time GPU cohorts with explicit gaps in retained allocation history.

Running attempt intervals recover a subset of allocated requests. They do not
establish historical pod binding, cleanup, or Kubernetes allocatable capacity.
Consequently historical points are incomplete and never manufacture idle.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from typing import Protocol

import pyarrow as pa
from cache import TtlCache
from config import K8S_CLUSTERS
from errors import UpstreamError
from finelog.errors import QueryResultTooLargeError
from finelog_source import MetricSource

DAY_MS = 86_400_000
MAX_WINDOW_MS = 7 * DAY_MS
STATE_LOOKBACK_MS = 120_000
HISTORY_TTL = 3600
MODELS = ("H100", "GB200")
CLUSTER_NAMES = tuple(target.name for target in K8S_CLUSTERS)


class Priority(StrEnum):
    SYSTEM = "system"
    PRODUCTION = "production"
    INTERACTIVE = "interactive"
    BATCH = "batch"
    UNKNOWN = "unknown_priority"


_PRIORITIES = {f"PRIORITY_BAND_{band.name}": band for band in Priority if band != Priority.UNKNOWN}


class AllocationMetadataSource(Protocol):
    def gpu_allocation_metadata(self, cluster: str, start_ms: int, end_ms: int, *, max_rows: int) -> list[dict]: ...


@dataclass(frozen=True, slots=True)
class Attempt:
    root: str
    task: str
    gpus: int
    model: str
    requested: Priority
    applied: Priority
    current_attempt: int
    attempt: int
    created: int | None
    started: int | None
    finished: int | None
    scope_start: int
    scope_end: int


@dataclass(frozen=True)
class RootLifetimes:
    created: tuple[int, ...]
    finished: tuple[int, ...]
    running: tuple[int, ...]
    stopped: tuple[int, ...]

    def counts(self, at: int) -> tuple[int, int]:
        alive = bisect_right(self.created, at) - bisect_right(self.finished, at)
        running = bisect_right(self.running, at) - bisect_right(self.stopped, at)
        return running, alive - running


def sampling_step(start_ms: int, end_ms: int) -> int:
    """Select whole snapshots at one, five, or fifteen-minute resolution."""
    duration = end_ms - start_ms
    if start_ms < 0 or not 0 < duration <= MAX_WINDOW_MS:
        raise ValueError("GPU allocation history requires a range of at most seven days")
    if duration <= 6 * 3_600_000:
        return 60_000
    if duration <= 2 * DAY_MS:
        return 300_000
    return 900_000


def state_query(start_ms: int, end_ms: int, step_ms: int, clusters: tuple[str, ...]) -> str:
    """Select one whole emission immediately before each sampling instant."""
    cluster_sql = ",".join(f"'{cluster}'" for cluster in clusters)
    # Ceil each emission to a grid point; retain only the last two minutes.
    point = f"date_bin(INTERVAL '{step_ms} milliseconds', ts + INTERVAL '{step_ms - 1} milliseconds')"
    return f"""WITH rollups AS (
 SELECT cluster, {point} AS point, MAX(ts) AS snapshot_ts
 FROM "iris.task_state"
 WHERE root_job_id = '' AND cluster IN ({cluster_sql})
  AND ts >= to_timestamp_millis({max(0, start_ms - STATE_LOOKBACK_MS)})
  AND ts < to_timestamp_millis({end_ms})
  AND ts >= {point} - INTERVAL '2 minutes'
 GROUP BY 1, 2
)
SELECT r.point,s.ts,s.cluster,s.root_job_id,s.assigned,s.building,s.running
FROM "iris.task_state" s JOIN rollups r ON s.cluster=r.cluster AND s.ts=r.snapshot_ts
WHERE s.cluster IN ({cluster_sql})
 AND s.ts >= to_timestamp_millis({max(0, start_ms - STATE_LOOKBACK_MS)})
 AND s.ts < to_timestamp_millis({end_ms})
 AND r.point >= to_timestamp_millis({start_ms}) AND r.point < to_timestamp_millis({end_ms})"""


def _optional_time(row: dict, field: str) -> int | None:
    return int(row[field]) if field in row else None


def metadata_table(rows: list[dict], start_ms: int, end_ms: int) -> pa.Table:
    """Normalize the fixed protobuf JSON response into the shared Arrow cache."""
    schema = pa.schema(
        [
            ("root", pa.string()),
            ("task", pa.string()),
            ("gpus", pa.int32()),
            ("model", pa.string()),
            ("requested", pa.string()),
            ("applied", pa.string()),
            ("current_attempt", pa.int32()),
            ("attempt", pa.int32()),
            ("created", pa.int64()),
            ("started", pa.int64()),
            ("finished", pa.int64()),
            ("scope_start", pa.int64()),
            ("scope_end", pa.int64()),
        ],
        metadata={"scope_end": str(end_ms)},
    )
    return pa.Table.from_pylist(
        [
            {
                "root": row["rootJobId"],
                "task": row["taskId"],
                "gpus": int(row.get("gpuCount", 0)),
                "model": row.get("gpuVariant", "").upper(),
                "requested": _PRIORITIES.get(row.get("requestedPriority", ""), Priority.UNKNOWN),
                "applied": _PRIORITIES.get(row.get("currentAppliedPriority", ""), Priority.UNKNOWN),
                "current_attempt": int(row.get("currentAttemptId", -1)),
                "attempt": int(row.get("attemptId", -1)),
                "created": _optional_time(row, "createdAtMs"),
                "started": _optional_time(row, "startedAtMs"),
                "finished": _optional_time(row, "finishedAtMs"),
                "scope_start": start_ms,
                "scope_end": end_ms,
            }
            for row in rows
        ],
        schema=schema,
    )


def _epoch_ms(value: datetime) -> int:
    return round(value.replace(tzinfo=UTC).timestamp() * 1000)


def _root_lifetimes(attempts: Sequence[Attempt]) -> dict[tuple[int, str], RootLifetimes]:
    roots: dict[tuple[int, str], list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        roots[attempt.scope_start, attempt.root].append(attempt)
    return {
        root: RootLifetimes(
            tuple(sorted(a.created for a in rows if a.created is not None)),
            tuple(sorted(a.finished for a in rows if a.created is not None and a.finished is not None)),
            tuple(sorted(a.started for a in rows if a.started is not None)),
            tuple(sorted(a.finished for a in rows if a.started is not None and a.finished is not None)),
        )
        for root, rows in roots.items()
    }


def _add_interval(differences: list[int], times: list[int], start: int, end: int, gpus: int) -> None:
    if start >= end:
        return
    differences[bisect_left(times, start)] += gpus
    differences[bisect_left(times, end)] -= gpus


def history_rows(
    state: pa.Table,
    metadata: dict[tuple[int, str], pa.Table],
    failures: dict[tuple[int, str], str],
    start_ms: int,
    end_ms: int,
    step_ms: int,
    clusters: tuple[str, ...],
) -> list[dict]:
    """Recover cohorts at common instants, never averaging or unioning a bucket."""
    times = list(range(((start_ms + step_ms - 1) // step_ms) * step_ms, end_ms, step_ms))
    totals = {(model, band): [0] * (len(times) + 1) for model in MODELS for band in Priority}
    setup = {model: [0] * (len(times) + 1) for model in MODELS}
    unresolved = [0] * (len(times) + 1)
    indexes: dict[str, dict[tuple[int, str], RootLifetimes]] = {}
    available_scopes = []
    for (day, cluster), table in metadata.items():
        available_scopes.append((day, int(table.schema.metadata[b"scope_end"])))
        # Each day's evidence is confined to that day, including on partial failure.
        attempts = {
            (row["task"], row["attempt"], row["scope_start"]): Attempt(
                **{**row, "requested": Priority(row["requested"]), "applied": Priority(row["applied"])}
            )
            for row in table.to_pylist()
        }
        indexes.setdefault(cluster, {}).update(_root_lifetimes(tuple(attempts.values())))
        for attempt in attempts.values():
            if not attempt.gpus:
                continue
            if attempt.model not in MODELS:
                if attempt.created is not None:
                    finish = attempt.finished if attempt.finished is not None else attempt.scope_end
                    _add_interval(
                        unresolved,
                        times,
                        max(attempt.created, attempt.scope_start),
                        min(finish, attempt.scope_end),
                        attempt.gpus,
                    )
                continue
            band = attempt.applied if attempt.attempt == attempt.current_attempt else attempt.requested
            if attempt.attempt != attempt.current_attempt and band not in (
                Priority.SYSTEM,
                Priority.PRODUCTION,
                Priority.BATCH,
            ):
                band = Priority.UNKNOWN
            if attempt.started is not None:
                finish = attempt.finished if attempt.finished is not None else attempt.scope_end
                _add_interval(
                    totals[attempt.model, band],
                    times,
                    max(attempt.started, attempt.scope_start),
                    min(finish, attempt.scope_end),
                    attempt.gpus,
                )
            if attempt.created is not None:
                stops = [t for t in (attempt.started, attempt.finished) if t is not None]
                _add_interval(
                    setup[attempt.model],
                    times,
                    max(attempt.created, attempt.scope_start),
                    min([attempt.scope_end, *stops]),
                    attempt.gpus,
                )

    frames: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in state.to_pylist():
        frames[_epoch_ms(row["point"]), row["cluster"]].append(row)
    running = {key: 0 for key in totals}
    waiting = {model: 0 for model in MODELS}
    unknown_model = 0
    result = []
    for index, at in enumerate(times):
        missing = 0
        missing_states = []
        for cluster in clusters:
            frame = frames.get((at, cluster), [])
            rollup = next((r for r in frame if r["root_job_id"] == ""), None)
            if rollup is None:
                missing_states.append(cluster)
                continue
            root_rows = [r for r in frame if r["root_job_id"]]
            count = sum(r["running"] + r["assigned"] + r["building"] for r in root_rows)
            missing += abs(count - rollup["running"] - rollup["assigned"] - rollup["building"])
            for root in root_rows:
                lifetimes = indexes.get(cluster, {}).get(((at // DAY_MS) * DAY_MS, root["root_job_id"]))
                observed_running, observed_setup = lifetimes.counts(_epoch_ms(root["ts"])) if lifetimes else (0, 0)
                missing += abs(root["running"] - observed_running)
                missing += abs(root["assigned"] + root["building"] - observed_setup)
        unknown_model += unresolved[index]
        for key, difference in totals.items():
            running[key] += difference[index]
        for model, difference in setup.items():
            waiting[model] += difference[index]
            status = "Incomplete: historical capacity and pod setup/cleanup lifetimes unavailable"
            unavailable = sorted(
                {cluster for (day, cluster) in failures if day == (at // DAY_MS) * DAY_MS} | set(missing_states)
            )
            if unavailable:
                status += "; missing sources: " + ", ".join(unavailable)
            result.append(
                {
                    "time": at,
                    "model": model,
                    **{
                        band.value: (
                            running[model, band] if any(begin <= at < stop for begin, stop in available_scopes) else None
                        )
                        for band in Priority
                    },
                    "idle": None,
                    "incomplete": 1,
                    "resolution_minutes": step_ms // 60_000,
                    "setup_gpu_requests": waiting[model],
                    "missing_task_metadata": missing,
                    "unknown_model_gpu_requests": unknown_model,
                    "status": status,
                    "missing_clusters": ",".join(unavailable),
                }
            )
    return result


def live_rows(nodes: list[dict], workloads: list[dict], clusters: tuple[str, ...], sampled_at: int) -> list[dict]:
    """Use exactly Cluster Capacity's live numerator and denominator."""
    counts = {model: {band.value: 0 for band in Priority} for model in MODELS}
    capacity = {model: 0 for model in MODELS}
    missing = {r["cluster"] for r in [*nodes, *workloads] if r["cluster"] in clusters and "error_class" in r}
    node_models = {}
    unresolved = 0
    for node in nodes:
        if node["cluster"] not in clusters or "error_class" in node:
            continue
        gpus = node["gpu_allocatable"] or node["gpu_capacity"]
        if not gpus:
            continue
        model = next((m for m in MODELS if node["gpu_model"].upper().startswith(m)), None)
        if model is None:
            unresolved += gpus
            continue
        capacity[model] += gpus
        node_models[node["cluster"], node["node"]] = model
    classes = {f"iris-{band.value}": band for band in Priority if band != Priority.UNKNOWN}
    for pod in workloads:
        if (
            pod["cluster"] not in clusters
            or "error_class" in pod
            or not pod["node"]
            or pod["phase"] in ("Succeeded", "Failed")
        ):
            continue
        if not pod["gpu_request_count"]:
            continue
        model = node_models.get((pod["cluster"], pod["node"]))
        if model is None:
            unresolved += pod["gpu_request_count"]
            continue
        band = classes.get(pod["priority_class"], Priority.UNKNOWN)
        counts[model][band.value] += pod["gpu_request_count"]
    rows = []
    for model in MODELS:
        allocated = sum(counts[model].values())
        complete = not (missing or unresolved or allocated > capacity[model])
        rows.append(
            {
                "time": sampled_at,
                "model": model,
                **counts[model],
                "idle": capacity[model] - allocated if complete else None,
                "capacity": capacity[model] if complete else None,
                "allocated": allocated,
                "incomplete": int(not complete),
                "resolution_minutes": 1,
                "setup_gpu_requests": 0,
                "missing_task_metadata": 0,
                "unknown_model_gpu_requests": unresolved,
                "status": "Live Kubernetes allocation" if complete else "Incomplete live Kubernetes snapshot",
                "missing_clusters": ",".join(sorted(missing)),
            }
        )
    return rows


def allocation_history(
    source: MetricSource,
    registry: AllocationMetadataSource,
    cache: TtlCache[pa.Table],
    start_ms: int,
    end_ms: int,
    *,
    clusters: tuple[str, ...],
    max_rows: int,
    cache_ttl: float,
    now_ms: int,
) -> list[dict]:
    """Return sampled GPU requests and explicit gaps within the existing budget."""
    step = sampling_step(start_ms, end_ms)
    if not clusters or any(c not in CLUSTER_NAMES for c in clusters):
        raise ValueError("GPU allocation history requires configured CoreWeave clusters")
    end_ms = min(end_ms, now_ms)
    if end_ms <= start_ms:
        return []
    states = []
    metadata: dict[tuple[int, str], pa.Table] = {}
    failures: dict[tuple[int, str], str] = {}
    for day in range((start_ms // DAY_MS) * DAY_MS, end_ms, DAY_MS):
        stop = min(day + DAY_MS, now_ms)
        lifetime = HISTORY_TTL if day + DAY_MS <= now_ms else cache_ttl
        sql = state_query(day, stop, step, clusters)
        states.append(
            cache.get_or_compute(
                ("gpu-allocation-state", source.target.name, day, step, clusters),
                partial(source.query, sql, max_rows=max_rows),
                ttl=lifetime,
            )
        )

        with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
            pending = {
                cluster: executor.submit(_metadata_input, cache, registry, cluster, day, stop, max_rows, lifetime)
                for cluster in clusters
            }
            for cluster, future in pending.items():
                try:
                    table = future.result()
                    metadata[day, cluster] = table
                except UpstreamError as error:
                    failures[day, cluster] = str(error)
        tables = [*states, *metadata.values()]
        if sum(table.num_rows for table in tables) > max_rows:
            raise QueryResultTooLargeError("GPU allocation history exceeds the aggregate input row budget")
        if sum(table.nbytes for table in tables) > cache.max_size:
            raise QueryResultTooLargeError("GPU allocation history exceeds the shared input cache budget")
    return history_rows(
        pa.concat_tables(states),
        metadata,
        failures,
        start_ms,
        end_ms,
        step,
        clusters,
    )


def _metadata_input(
    cache: TtlCache[pa.Table],
    registry: AllocationMetadataSource,
    cluster: str,
    start_ms: int,
    end_ms: int,
    max_rows: int,
    ttl: float,
) -> pa.Table:
    def fetch() -> pa.Table:
        rows = registry.gpu_allocation_metadata(cluster, start_ms, end_ms, max_rows=max_rows)
        if len(rows) > max_rows:
            raise QueryResultTooLargeError("GPU allocation metadata exceeds the row budget")
        return metadata_table(rows, start_ms, end_ms)

    return cache.get_or_compute(("gpu-allocation-metadata", cluster, start_ms), fetch, ttl=ttl)
