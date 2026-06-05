# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the Rust vs Python/DuckDB finelog server on the real iris dataset.

In production the Rust ``finelog-server`` adopts the real iris dataset fine but
the iris dashboard's stats ``Query`` RPC is slow (observed "Slow RPC Query:
9457ms"). This script makes that reproducible locally and runs BOTH backends
against the same pulled parquet so a perf gap can be attributed to the engine
(Rust/DataFusion pruning) vs the data layout (re-keying):

  * ``pull``  — copy the per-namespace ``seg_L*.parquet`` segments from the prod
    finelog VM (config "marin") down to a local data dir. Defaults to the
    ``iris.*`` namespaces (the dashboard ``Query`` path). The 25 GB ``log``
    namespace is opt-in: ``--log-gb N`` pulls only the most-recent ~N GB of log
    segments (highest ``min_seq`` first) so FetchLogs can be benched without the
    full egress. ``--with-log`` pulls all of it. Resumable: a file already
    present locally with a matching size is skipped.

  * ``bench`` — for each selected backend (``--backend rust|duckdb|both``), boot
    the server against the pulled data dir, wait for ``/health``, run a warmup,
    then replay the query set N times each and record per-query p50/p95/max
    wall-clock latency measured at the Python client. Backends run one after
    another on the same machine (never concurrent) for a fair compare. Boot /
    adopt time is excluded — only steady-state query latency is timed. Writes a
    side-by-side comparison table to stdout and a machine-readable JSON file.

The query set spans the common finelog query families. Keyed dashboard queries
(see ``DASHBOARD_QUERIES``) are sampled verbatim from the iris dashboard
frontend SQL under ``lib/iris/dashboard/src``. On top of those, ``bench``
discovers REAL key values present in the pulled data and builds:

  * FAMILY A — log tail (FetchLogs RPC, ``log`` namespace): last-N lines for a
    worker, a task attempt, and the controller.
  * FAMILY B — task stats by prefix (``iris.task``): ``task_id LIKE 'Y%'`` and a
    key-range ``task_id >= 'Y' AND task_id < 'Z'`` variant.
  * FAMILY C — profile (``iris.profile``): the ProfileHistory list and the
    single-row blob download for a real ``source``.

The two SQL backends differ on one builtin: DataFusion (Rust) accepts
``length()`` on a binary column; DuckDB requires ``octet_length()``. The
ProfileHistory query is emitted per-backend accordingly (same plan shape,
same blob touch).

Both subcommands take absolute paths and reuse the finelog Python client and
the deploy SSH helper. The VM is touched read-only (sudo find / sudo tar over
SSH); nothing is ever written to or restarted on it.

Example::

    .venv/bin/python lib/finelog/scripts/bench_dashboard_queries.py pull --log-gb 4
    .venv/bin/python lib/finelog/scripts/bench_dashboard_queries.py bench --backend both
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from finelog.client import LogClient
from finelog.deploy._gcp import _ssh_args
from finelog.deploy.config import FinelogConfig, load_finelog_config
from finelog.rpc import logging_pb2
from finelog.schema import schema_from_json
from finelog.store.catalog import CATALOG_DB_FILENAME, Catalog

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

FINELOG_CONFIG_NAME = "marin"
REMOTE_DATA_ROOT = "/var/cache/finelog"
DEFAULT_DATA_DIR = Path("/home/power/finelog-bench/data")
DEFAULT_RESULTS_DIR = Path("/home/power/finelog-bench/results")
DEFAULT_SERVER_BIN = Path("/home/power/code/marin/.worktrees/finelog-rust/rust/target/fast/finelog-server")
DEFAULT_PORT = 19001
DEFAULT_REPEAT = 5

# The Rust server writes this catalog sidecar when it adopts a parquet tree; it
# records each namespace's schema (including key_column). The Python/DuckDB
# server has no adopt-from-parquet path — it rehydrates namespaces purely from
# its own ``_finelog_registry.duckdb`` and then each per-namespace DiskLogStore
# scans its dir for segments. We seed that registry from this Rust catalog so
# both engines serve the identical parquet without a re-ingest.
RUST_CATALOG_SQLITE = "_finelog_catalog.sqlite"


class Backend(StrEnum):
    RUST = "rust"
    DUCKDB = "duckdb"


# Slow-RPC threshold the Rust server logs at; mirrored here for the report flag.
SLOW_MS = 7000.0

# Dashboard query path. The 25 GB ``log`` namespace is FetchLogs, not Query,
# and is deliberately excluded from the default pull (GCP -> open-internet
# egress is cost-sensitive). ``--with-log`` adds it.
IRIS_NAMESPACES = (
    "iris.worker",
    "iris.task",
    "iris.task_status",
    "iris.profile",
    "infra.canary.probes",
)
LOG_NAMESPACE = "log"

SEGMENT_GLOB = "seg_L*.parquet"


# --------------------------------------------------------------------------- #
# Query set — sampled from the iris dashboard frontend (lib/iris/dashboard/src)
# --------------------------------------------------------------------------- #
#
# The dashboard frontend builds these SQL strings and sends them over the
# finelog StatsService ``Query`` RPC (routed via the controller's endpoint
# proxy at /proxy/system.log-server/finelog.stats.StatsService/Query). Real
# requests bind a concrete worker_id / task_id / task-id list; here we use
# representative literals so the query is structurally identical (same scan,
# same predicate shape, same ORDER BY / QUALIFY / LIMIT).


@dataclass(frozen=True)
class BenchQuery:
    """One benchmarked query plus its provenance.

    A query is either a SQL ``Query`` RPC (``sql`` / ``sql_duckdb``) or a
    ``FetchLogs`` RPC (``fetch``). DataFusion and DuckDB disagree on one
    builtin (``length`` vs ``octet_length`` on a binary column), so a query
    may carry a DuckDB-specific SQL variant; when ``sql_duckdb`` is empty the
    same ``sql`` runs on both engines.
    """

    name: str
    # Where in the iris dashboard / which query family this comes from.
    source: str
    sql: str = ""
    sql_duckdb: str = ""
    fetch: logging_pb2.FetchLogsRequest | None = None

    def sql_for(self, backend: Backend) -> str:
        if backend == Backend.DUCKDB and self.sql_duckdb:
            return self.sql_duckdb
        return self.sql


# A literal that is very unlikely to exist in prod data — exercises the
# full-scan / pruning cost without returning a large result. The slowness is
# in the scan, not in row materialization, so a non-matching key is the honest
# "cold lookup" the dashboard pays on every page load.
_PROBE_WORKER = "bench-probe-worker"
_PROBE_TASK = "bench-probe-task"
_PROBE_SOURCE = "/system/worker/bench-probe"


def _build_queries() -> list[BenchQuery]:
    # Representative task-id IN-list for the job/worker rollups (JobDetail
    # batches every task in a job; real lists run to dozens of ids).
    task_id_list = ",".join(f"'{_PROBE_TASK}-{i}'" for i in range(24))

    return [
        # --- WorkerDetail.vue buildStatsSql(): per-worker resource history.
        # Worker detail page; "Live utilization sparklines (iris.worker stats)".
        BenchQuery(
            name="worker_detail_history",
            source="dashboard/src/components/controller/WorkerDetail.vue buildStatsSql",
            sql=f"""
SELECT ts, cpu_pct, mem_bytes, mem_total_bytes,
       disk_used_bytes, disk_total_bytes,
       net_recv_bytes, net_sent_bytes, running_task_count
FROM "iris.worker"
WHERE worker_id = '{_PROBE_WORKER}'
ORDER BY ts DESC
LIMIT 50
""".strip(),
        ),
        # --- TaskDetail.vue buildTaskStatsSql(): per-task per-attempt history.
        # Task detail page resources card; iris.task is the source of truth.
        BenchQuery(
            name="task_detail_history",
            source="dashboard/src/components/controller/TaskDetail.vue buildTaskStatsSql",
            sql=f"""
SELECT ts, cpu_millicores, memory_mb, memory_peak_mb, disk_mb
FROM "iris.task"
WHERE task_id = '{_PROBE_TASK}'
  AND attempt_id = 0
ORDER BY ts DESC
LIMIT 200
""".strip(),
        ),
        # --- JobDetail.vue buildTaskStatsSql(): latest sample per task in a job.
        # Job detail page Memory/Peak/CPU columns. QUALIFY window over a
        # task_id IN-list — the dashboard hot path called out in stats.py.
        BenchQuery(
            name="job_detail_latest_per_task",
            source="dashboard/src/components/controller/JobDetail.vue buildTaskStatsSql",
            sql=f"""
SELECT
  task_id,
  attempt_id,
  cpu_millicores,
  memory_mb * 1024 * 1024 AS memory_bytes,
  memory_peak_mb * 1024 * 1024 AS memory_peak_bytes
FROM "iris.task"
WHERE task_id IN ({task_id_list})
QUALIFY row_number() OVER (PARTITION BY task_id ORDER BY ts DESC) = 1
""".strip(),
        ),
        # --- utils/taskStatus.ts detailSql(): latest status text for one task,
        # within the 10-minute retention window. TaskDetail.vue status card.
        BenchQuery(
            name="task_status_detail",
            source="dashboard/src/utils/taskStatus.ts detailSql",
            sql=f"""
SELECT status_text_detail_md, status_text_summary_md
FROM "iris.task_status"
WHERE task_id = '{_PROBE_TASK}'
  AND ts > now() - INTERVAL '10 minutes'
ORDER BY ts DESC, attempt_id DESC
LIMIT 1
""".strip(),
        ),
        # --- utils/taskStatus.ts batchSummarySql(): one latest summary per task,
        # batched across a job's tasks. JobDetail.vue status column.
        BenchQuery(
            name="task_status_batch_summary",
            source="dashboard/src/utils/taskStatus.ts batchSummarySql",
            sql=f"""
SELECT task_id, status_text_summary_md
FROM "iris.task_status"
WHERE task_id IN ({task_id_list})
  AND ts > now() - INTERVAL '10 minutes'
QUALIFY row_number() OVER (PARTITION BY task_id ORDER BY ts DESC, attempt_id DESC) = 1
""".strip(),
        ),
        # --- ProfileHistory.vue: last 50 profile captures for a source.
        # Profile history panel (worker / controller / task detail pages).
        BenchQuery(
            name="profile_history",
            source="dashboard/src/components/shared/ProfileHistory.vue",
            sql=f"""
SELECT captured_at, type, attempt_id, vm_id, duration_seconds, format, trigger, length(profile_data) AS size_bytes
FROM "iris.profile"
WHERE source = '{_PROBE_SOURCE}'
ORDER BY captured_at DESC
LIMIT 50
""".strip(),
            # DataFusion accepts length() on binary; DuckDB needs octet_length().
            sql_duckdb=f"""
SELECT captured_at, type, attempt_id, vm_id, duration_seconds, format, trigger, octet_length(profile_data) AS size_bytes
FROM "iris.profile"
WHERE source = '{_PROBE_SOURCE}'
ORDER BY captured_at DESC
LIMIT 50
""".strip(),
        ),
        # --- ProfileHistory.vue downloadProfile(): fetch one capture's bytes.
        # Click-to-download a profile row; pulls a large bytes column.
        BenchQuery(
            name="profile_download_one",
            source="dashboard/src/components/shared/ProfileHistory.vue downloadProfile",
            sql=f"""
SELECT profile_data, type, format
FROM "iris.profile"
WHERE source = '{_PROBE_SOURCE}' AND captured_at = '2026-06-01 00:00:00'
LIMIT 1
""".strip(),
        ),
        # --- Representative of the FleetTab worker-status rollup the dashboard
        # shows: latest heartbeat per worker, grouped by status. Not a verbatim
        # frontend string (the live fleet view is served from controller SQLite),
        # but it is the natural iris.worker aggregation a "fleet utilization"
        # panel runs and stresses a GROUP BY over the full worker history.
        BenchQuery(
            name="fleet_status_rollup",
            source="representative of FleetTab fleet rollup over iris.worker",
            sql="""
SELECT status, count(*) AS n, avg(cpu_pct) AS avg_cpu_pct
FROM (
  SELECT worker_id, status, cpu_pct,
         row_number() OVER (PARTITION BY worker_id ORDER BY ts DESC) AS rn
  FROM "iris.worker"
)
WHERE rn = 1
GROUP BY status
ORDER BY n DESC
""".strip(),
        ),
        # --- Representative recent-window scan: rows in the last 15 minutes
        # across the whole worker fleet. Models the "recent activity" the
        # dashboard's auto-refresh polls; a time-window SELECT with no key
        # predicate, so it cannot prune on the worker_id clustering key.
        BenchQuery(
            name="worker_recent_window",
            source="representative recent-time-window scan over iris.worker",
            sql="""
SELECT worker_id, ts, cpu_pct, mem_bytes, running_task_count
FROM "iris.worker"
WHERE ts > now() - INTERVAL '15 minutes'
ORDER BY ts DESC
LIMIT 500
""".strip(),
        ),
        # --- Representative recent-window scan over iris.task: recent attempt
        # resource updates fleet-wide. Same shape against the largest stats
        # namespace (4 GB).
        BenchQuery(
            name="task_recent_window",
            source="representative recent-time-window scan over iris.task",
            sql="""
SELECT task_id, attempt_id, ts, cpu_millicores, memory_mb
FROM "iris.task"
WHERE ts > now() - INTERVAL '15 minutes'
ORDER BY ts DESC
LIMIT 500
""".strip(),
        ),
        # --- infra.canary.probes recent rollup. Small namespace; sanity check
        # that a tiny table stays fast and isolates per-segment open cost.
        BenchQuery(
            name="canary_probes_recent",
            source="representative recent scan over infra.canary.probes",
            sql="""
SELECT *
FROM "infra.canary.probes"
ORDER BY 1 DESC
LIMIT 100
""".strip(),
        ),
    ]


DASHBOARD_QUERIES: list[BenchQuery] = _build_queries()


# --------------------------------------------------------------------------- #
# Real-key discovery — pick concrete key values present in the pulled parquet so
# the broadened families return real rows (the prior probe-literal run was 0-row).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiscoveredKeys:
    """Concrete key values found in the pulled data for the broadened families."""

    task_prefix: str | None = None  # FAMILY B: a real task_id path prefix (LIKE 'Y%')
    task_range_hi: str | None = None  # FAMILY B: exclusive upper bound for the range variant
    profile_source: str | None = None  # FAMILY C: profile source with the most captures
    profile_captured_at: str | None = None  # FAMILY C: one real captured_at for that source
    log_worker_source: str | None = None  # FAMILY A: a /system/worker/<id> log key
    log_task_source: str | None = None  # FAMILY A: a task-attempt log key (e.g. /user/job/0:0)
    log_controller_source: str = "/system/controller"  # FAMILY A: the controller log key


def _segment_paths(namespace_dir: Path) -> list[Path]:
    """Local ``seg_L*_*.parquet`` files for a namespace, newest (highest minseq) last."""

    def minseq(p: Path) -> int:
        m = re.match(r"seg_L\d+_(\d+)\.parquet$", p.name)
        return int(m.group(1)) if m else -1

    return sorted((p for p in namespace_dir.glob(SEGMENT_GLOB) if p.is_file()), key=minseq)


def _top_values(paths: list[Path], column: str, *, limit_segments: int) -> Counter:
    """Aggregate value -> row count for ``column`` across the newest ``limit_segments``."""
    counts: Counter = Counter()
    for path in paths[-limit_segments:]:
        table = pq.ParquetFile(path).read(columns=[column])
        for entry in pc.value_counts(table.column(column)).to_pylist():
            value = entry["values"]
            if value:
                counts[value] += entry["counts"]
    return counts


def _discover_task_prefix(data_dir: Path) -> tuple[str | None, str | None]:
    """Pick the most-common 3-segment ``task_id`` path prefix and its range upper bound.

    iris.task ``task_id`` is a ``/user/job/.../n`` path. A prefix like
    ``/held/iris-run-...`` selects a whole run's attempts. The range upper bound
    is the same prefix with its last byte incremented so ``>= prefix AND < hi``
    is the half-open key range a clustered scan can prune on.
    """
    ns_dir = data_dir / "iris.task"
    if not ns_dir.is_dir():
        return None, None
    paths = _segment_paths(ns_dir)
    if not paths:
        return None, None
    table = pq.ParquetFile(paths[-1]).read(columns=["task_id"])
    prefix_counts: Counter = Counter()
    for tid in pc.unique(table.column("task_id")).to_pylist():
        parts = tid.split("/")
        if len(parts) >= 3:
            prefix_counts["/".join(parts[:3])] += 1
    if not prefix_counts:
        return None, None
    prefix = prefix_counts.most_common(1)[0][0]
    return prefix, _range_upper_bound(prefix)


def _range_upper_bound(prefix: str) -> str:
    """Exclusive upper bound for a half-open key range covering ``prefix*``.

    Increment the final character so ``prefix <= k < hi`` matches exactly the
    keys ``LIKE 'prefix%'`` would (minus the degenerate wrap case, which prod
    keys never hit).
    """
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


def _discover_profile_key(data_dir: Path) -> tuple[str | None, str | None]:
    """Pick the profile ``source`` with the most captures and one real ``captured_at``.

    A high-count source makes the ProfileHistory ``LIMIT 50`` return a full
    page, and gives a concrete ``captured_at`` for the single-row blob download.
    """
    ns_dir = data_dir / "iris.profile"
    if not ns_dir.is_dir():
        return None, None
    paths = _segment_paths(ns_dir)
    if not paths:
        return None, None
    counts: Counter = Counter()
    for path in paths:
        table = pq.ParquetFile(path).read(columns=["source"])
        for entry in pc.value_counts(table.column("source")).to_pylist():
            if entry["values"]:
                counts[entry["values"]] += entry["counts"]
    if not counts:
        return None, None
    source = counts.most_common(1)[0][0]
    captured_at = _first_captured_at(paths, source)
    return source, captured_at


def _first_captured_at(paths: list[Path], source: str) -> str | None:
    """Return one real ``captured_at`` for ``source`` as a microsecond SQL literal."""
    for path in paths:
        table = pq.ParquetFile(path).read(columns=["source", "captured_at"])
        sub = table.filter(pc.equal(table.column("source"), source))
        if sub.num_rows:
            ts = sub.column("captured_at").to_pylist()[0]
            return ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    return None


def _discover_log_sources(data_dir: Path) -> tuple[str | None, str | None]:
    """Pick a real worker log key and a real task-attempt log key from the slice.

    The ``log`` namespace key column is ``key``; worker process logs use the
    ``/system/worker/<id>`` key and task attempts use the wire path
    ``/user/job/.../n:<attempt>``. Pick the highest-volume key of each shape so
    the tail read returns a full page.
    """
    ns_dir = data_dir / "log"
    if not ns_dir.is_dir():
        return None, None
    paths = _segment_paths(ns_dir)
    if not paths:
        return None, None
    counts = _top_values(paths, "key", limit_segments=8)
    worker = _best_key(counts, lambda k: k.startswith("/system/worker/"))
    # A task attempt key carries a ":<attempt>" suffix and is not a /system/ key.
    task = _best_key(counts, lambda k: ":" in k.rsplit("/", 1)[-1] and not k.startswith("/system/"))
    return worker, task


def _best_key(counts: Counter, predicate) -> str | None:
    for key, _count in counts.most_common():
        if predicate(key):
            return key
    return None


def discover_keys(data_dir: Path) -> DiscoveredKeys:
    """Inspect the pulled parquet and return concrete key values for each family."""
    task_prefix, task_hi = _discover_task_prefix(data_dir)
    profile_source, captured_at = _discover_profile_key(data_dir)
    log_worker, log_task = _discover_log_sources(data_dir)
    return DiscoveredKeys(
        task_prefix=task_prefix,
        task_range_hi=task_hi,
        profile_source=profile_source,
        profile_captured_at=captured_at,
        log_worker_source=log_worker,
        log_task_source=log_task,
    )


def _sql_quote(value: str) -> str:
    """Single-quote a SQL string literal, escaping embedded quotes."""
    return "'" + value.replace("'", "''") + "'"


def build_family_queries(keys: DiscoveredKeys) -> list[BenchQuery]:
    """Build FAMILY A/B/C queries bound to the discovered real key values.

    Families whose key could not be discovered (e.g. no log slice pulled) are
    skipped — the report notes the gap rather than running a 0-row query.
    """
    queries: list[BenchQuery] = []

    # FAMILY A — LOG TAIL (FetchLogs RPC, `log` namespace): last 500 lines.
    if keys.log_worker_source:
        queries.append(
            BenchQuery(
                name="log_tail_worker",
                source="FAMILY A FetchLogs tail=500 — worker process log",
                fetch=logging_pb2.FetchLogsRequest(
                    source=keys.log_worker_source,
                    match_scope=logging_pb2.MATCH_SCOPE_EXACT,
                    max_lines=500,
                    tail=True,
                ),
            )
        )
    if keys.log_task_source:
        queries.append(
            BenchQuery(
                name="log_tail_task_attempt",
                source="FAMILY A FetchLogs tail=500 — task attempt log",
                fetch=logging_pb2.FetchLogsRequest(
                    source=keys.log_task_source,
                    match_scope=logging_pb2.MATCH_SCOPE_EXACT,
                    max_lines=500,
                    tail=True,
                ),
            )
        )
    # The controller log only exists if some log slice was pulled at all; gate
    # on having discovered any real log key so we don't bench a 0-row tail.
    if keys.log_worker_source or keys.log_task_source:
        queries.append(
            BenchQuery(
                name="log_tail_controller",
                source="FAMILY A FetchLogs tail=500 — /system/controller",
                fetch=logging_pb2.FetchLogsRequest(
                    source=keys.log_controller_source,
                    match_scope=logging_pb2.MATCH_SCOPE_EXACT,
                    max_lines=500,
                    tail=True,
                ),
            )
        )

    # FAMILY B — TASK STATS BY PREFIX (iris.task, key_column=task_id).
    if keys.task_prefix:
        like_literal = _sql_quote(keys.task_prefix + "%")
        queries.append(
            BenchQuery(
                name="task_stats_prefix_like",
                source="FAMILY B iris.task task_id LIKE 'Y%'",
                sql=f"""
SELECT task_id, attempt_id, ts, cpu_millicores, memory_mb, memory_peak_mb
FROM "iris.task"
WHERE task_id LIKE {like_literal}
ORDER BY ts DESC
LIMIT 500
""".strip(),
            )
        )
        if keys.task_range_hi:
            lo = _sql_quote(keys.task_prefix)
            hi = _sql_quote(keys.task_range_hi)
            queries.append(
                BenchQuery(
                    name="task_stats_prefix_range",
                    source="FAMILY B iris.task task_id >= 'Y' AND task_id < 'Z' (key range)",
                    sql=f"""
SELECT task_id, attempt_id, ts, cpu_millicores, memory_mb, memory_peak_mb
FROM "iris.task"
WHERE task_id >= {lo} AND task_id < {hi}
ORDER BY ts DESC
LIMIT 500
""".strip(),
                )
            )

    # FAMILY C — PROFILE (iris.profile, key_column=source). The ProfileHistory
    # list touches every matched row's blob via length(profile_data); the
    # download fetches one blob. DataFusion accepts length() on binary; DuckDB
    # needs octet_length() — same plan, different builtin name.
    if keys.profile_source:
        src = _sql_quote(keys.profile_source)
        list_cols = (
            "captured_at, type, attempt_id, vm_id, duration_seconds, format, trigger, "
            "{length}(profile_data) AS size_bytes"
        )
        history_sql = f"""
SELECT {list_cols.format(length="length")}
FROM "iris.profile"
WHERE source = {src}
ORDER BY captured_at DESC
LIMIT 50
""".strip()
        history_sql_duckdb = f"""
SELECT {list_cols.format(length="octet_length")}
FROM "iris.profile"
WHERE source = {src}
ORDER BY captured_at DESC
LIMIT 50
""".strip()
        queries.append(
            BenchQuery(
                name="profile_history_real",
                source="FAMILY C ProfileHistory.vue list (real source)",
                sql=history_sql,
                sql_duckdb=history_sql_duckdb,
            )
        )
        if keys.profile_captured_at:
            captured = _sql_quote(keys.profile_captured_at)
            queries.append(
                BenchQuery(
                    name="profile_download_real",
                    source="FAMILY C ProfileHistory.vue downloadProfile (real source+captured_at)",
                    sql=f"""
SELECT profile_data, type, format
FROM "iris.profile"
WHERE source = {src} AND captured_at = {captured}
LIMIT 1
""".strip(),
                )
            )

    return queries


# --------------------------------------------------------------------------- #
# pull
# --------------------------------------------------------------------------- #


@dataclass
class RemoteFile:
    name: str
    size: int


def _ssh_run(cfg: FinelogConfig, command: str, *, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a read-only command on the VM over the deploy SSH helper."""
    return subprocess.run(_ssh_args(cfg, command), capture_output=True, text=True, timeout=timeout)


def _list_remote_segments(cfg: FinelogConfig, namespace: str) -> list[RemoteFile]:
    """Return ``seg_L*.parquet`` files (name, size) for a namespace on the VM.

    Read-only: ``sudo find ... -printf '%f\\t%s\\n'``. The catalog/registry
    sidecar files and ``.parquet.tmp`` staging files are intentionally not
    matched by the glob, so the local server re-adopts namespaces from the
    parquet segments instead of inheriting a prebuilt catalog.
    """
    remote_dir = f"{REMOTE_DATA_ROOT}/{namespace}"
    cmd = (
        f"sudo find {shlex.quote(remote_dir)} -maxdepth 1 -name {shlex.quote(SEGMENT_GLOB)} "
        f"-type f -printf '%f\\t%s\\n' 2>/dev/null"
    )
    result = _ssh_run(cfg, cmd)
    if result.returncode != 0:
        raise RuntimeError(f"listing {namespace} failed (rc={result.returncode}): {result.stderr.strip()[-400:]}")
    files: list[RemoteFile] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, size_str = line.partition("\t")
        files.append(RemoteFile(name=name, size=int(size_str)))
    files.sort(key=lambda f: f.name)
    return files


def _remote_seg_minseq(name: str) -> int:
    m = re.match(r"seg_L\d+_(\d+)\.parquet$", name)
    return int(m.group(1)) if m else -1


def _select_recent_log_slice(remote_files: list[RemoteFile], budget_gb: float) -> list[RemoteFile]:
    """Pick the most-recent log segments (highest ``min_seq`` first) up to ``budget_gb``.

    The 25 GB ``log`` namespace is mostly large L3 segments; selecting by
    descending ``min_seq`` grabs the newest L0/L1/L2 plus the newest-keyed L3
    blocks, which is the recent window FetchLogs tails read. Always keeps at
    least one segment so a tiny budget still yields a benchable slice.
    """
    by_recent = sorted(remote_files, key=lambda f: _remote_seg_minseq(f.name), reverse=True)
    budget_bytes = budget_gb * 1e9
    selected: list[RemoteFile] = []
    acc = 0
    for rf in by_recent:
        if selected and acc + rf.size > budget_bytes:
            break
        selected.append(rf)
        acc += rf.size
    return selected


def _pull_namespace(
    cfg: FinelogConfig,
    namespace: str,
    data_dir: Path,
    *,
    log_gb: float | None = None,
) -> dict:
    """Pull missing ``seg_L*.parquet`` files for one namespace. Returns a summary.

    Resumable: a local file already present with a byte-for-byte matching size
    is skipped. Missing files are streamed in one ``sudo tar`` over SSH (so a
    single sudo elevation moves the whole batch) and extracted locally.

    For the ``log`` namespace, ``log_gb`` bounds the slice to the most-recent
    ~N GB of segments (highest ``min_seq`` first) — pulling all 25 GB is
    avoidable GCP -> open-internet egress.
    """
    remote_files = _list_remote_segments(cfg, namespace)
    remote_total = len(remote_files)
    if namespace == LOG_NAMESPACE and log_gb is not None:
        remote_files = _select_recent_log_slice(remote_files, log_gb)
    local_dir = data_dir / namespace
    local_dir.mkdir(parents=True, exist_ok=True)

    to_pull: list[RemoteFile] = []
    skipped = 0
    for rf in remote_files:
        local_path = local_dir / rf.name
        if local_path.exists() and local_path.stat().st_size == rf.size:
            skipped += 1
            continue
        to_pull.append(rf)

    bytes_transferred = 0
    if to_pull:
        bytes_transferred = _stream_tar(cfg, namespace, local_dir, [rf.name for rf in to_pull])

    selected_note = f" of {remote_total}" if len(remote_files) != remote_total else ""
    print(
        f"  {namespace}: {len(remote_files)}{selected_note} segments "
        f"({skipped} already local, {len(to_pull)} pulled, "
        f"{bytes_transferred / 1e9:.3f} GB transferred)"
    )
    return {
        "namespace": namespace,
        "remote_segments": remote_total,
        "selected_segments": len(remote_files),
        "skipped": skipped,
        "pulled": len(to_pull),
        "bytes_transferred": bytes_transferred,
        "local_dir": str(local_dir),
    }


def _stream_tar(cfg: FinelogConfig, namespace: str, local_dir: Path, names: list[str]) -> int:
    """Stream ``names`` from the VM namespace dir via ``sudo tar`` and extract.

    The VM segments are owned by another user and need ``sudo`` to read, so a
    plain ``gcloud compute scp`` (which runs as the login user) cannot read
    them. Instead we ``sudo tar -cf -`` the requested files on the VM and pipe
    the stream straight into a local ``tar -xf -``. Counts bytes by summing the
    extracted files' sizes against the remote-reported sizes.
    """
    remote_dir = f"{REMOTE_DATA_ROOT}/{namespace}"
    quoted_names = " ".join(shlex.quote(n) for n in names)
    # -C into the namespace dir so the archive holds bare filenames (no path
    # prefix), letting us extract flat into the local namespace dir.
    remote_cmd = f"sudo tar -C {shlex.quote(remote_dir)} -cf - {quoted_names}"
    ssh = _ssh_args(cfg, remote_cmd)

    before = {p.name: p.stat().st_size for p in local_dir.glob(SEGMENT_GLOB) if p.is_file()}
    proc_ssh = subprocess.Popen(ssh, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc_tar = subprocess.Popen(
        ["tar", "-C", str(local_dir), "-xf", "-"],
        stdin=proc_ssh.stdout,
        stderr=subprocess.PIPE,
    )
    if proc_ssh.stdout is not None:
        proc_ssh.stdout.close()  # allow ssh to get SIGPIPE if tar exits
    tar_err = proc_tar.communicate()[1]
    ssh_err = proc_ssh.communicate()[1]
    if proc_ssh.returncode != 0:
        raise RuntimeError(f"sudo tar on VM failed for {namespace}: {ssh_err.decode(errors='replace')[-400:]}")
    if proc_tar.returncode != 0:
        raise RuntimeError(f"local tar extract failed for {namespace}: {tar_err.decode(errors='replace')[-400:]}")

    after = {p.name: p.stat().st_size for p in local_dir.glob(SEGMENT_GLOB) if p.is_file()}
    transferred = 0
    for name in names:
        sz = after.get(name)
        if sz is None:
            raise RuntimeError(f"{namespace}/{name} missing after tar extract")
        transferred += sz
    # Subtract any pre-existing size for names that were overwritten (partial files).
    for name in names:
        if name in before:
            transferred -= before[name]
    return transferred


def cmd_pull(args: argparse.Namespace) -> int:
    cfg = load_finelog_config(args.config)
    namespaces = list(IRIS_NAMESPACES)
    log_gb: float | None = None
    if args.with_log:
        namespaces.append(LOG_NAMESPACE)
    elif args.log_gb is not None:
        namespaces.append(LOG_NAMESPACE)
        log_gb = args.log_gb
    if args.namespaces:
        namespaces = args.namespaces

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pulling namespaces {namespaces} from VM '{cfg.name}' to {data_dir}")
    if LOG_NAMESPACE in namespaces:
        if log_gb is not None:
            print(
                f"  NOTE: pulling only the most-recent ~{log_gb:g} GB of the 'log' namespace "
                f"(GCP -> internet egress; full namespace is ~25 GB)"
            )
        else:
            print("  NOTE: pulling the full 25 GB 'log' namespace (GCP -> internet egress)")

    summaries = []
    total_bytes = 0
    for ns in namespaces:
        summary = _pull_namespace(cfg, ns, data_dir, log_gb=log_gb)
        summaries.append(summary)
        total_bytes += summary["bytes_transferred"]

    print(f"\nTotal transferred this run: {total_bytes / 1e9:.3f} GB (GCP -> open-internet egress)")
    on_disk = sum(p.stat().st_size for p in data_dir.rglob(SEGMENT_GLOB) if p.is_file())
    print(f"Total segment bytes on disk in {data_dir}: {on_disk / 1e9:.3f} GB")
    return 0


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #


@dataclass
class QueryResult:
    name: str
    source: str
    backend: str
    request: str  # the SQL string, or a FetchLogs request summary
    samples_ms: list[float] = field(default_factory=list)
    rows: int = 0
    error: str | None = None

    @property
    def p50_ms(self) -> float | None:
        return _percentile(self.samples_ms, 50)

    @property
    def p95_ms(self) -> float | None:
        return _percentile(self.samples_ms, 95)

    @property
    def max_ms(self) -> float | None:
        return max(self.samples_ms) if self.samples_ms else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    # Nearest-rank percentile (small N; avoids interpolation surprises).
    rank = max(1, round(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _wait_health(port: int, timeout: float = 120.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


def _seed_duckdb_registry(data_dir: Path) -> int:
    """Seed the Python/DuckDB ``_finelog_registry.duckdb`` from the Rust catalog.

    The Python server has no adopt-from-parquet path: it rehydrates namespaces
    from this registry, then each per-namespace DiskLogStore scans its dir and
    picks up the on-disk segments. We copy the per-namespace schema (with its
    ``key_column``) out of the Rust ``.finelog-rust-catalog`` sqlite the Rust
    server wrote when it adopted the same tree, so both engines serve the
    identical parquet without a re-ingest. Idempotent: rebuilt each bench run.

    Returns the number of namespaces seeded.
    """
    rust_catalog = data_dir / RUST_CATALOG_SQLITE
    if not rust_catalog.exists():
        raise RuntimeError(
            f"Rust catalog {rust_catalog} not found — boot the Rust backend at least once "
            f"(it writes the catalog the DuckDB registry is seeded from), or run `bench --backend rust`."
        )
    registry = data_dir / CATALOG_DB_FILENAME
    if registry.exists():
        registry.unlink()

    src = sqlite3.connect(str(rust_catalog))
    src.row_factory = sqlite3.Row
    rows = [(r["namespace"], r["schema_json"]) for r in src.execute("SELECT namespace, schema_json FROM namespaces")]
    src.close()

    catalog = Catalog(data_dir)
    seeded = 0
    try:
        for name, schema_json in rows:
            # ``log`` is materialized by the server's own first-boot fixup; the
            # other namespaces must be present before rehydrate to be adopted.
            if name == LOG_NAMESPACE:
                continue
            catalog.upsert(name, schema_from_json(schema_json))
            seeded += 1
    finally:
        catalog.close()
    return seeded


def _boot_rust(server_bin: Path, data_dir: Path, port: int, log_f) -> subprocess.Popen:
    cmd = [
        str(server_bin),
        "--port",
        str(port),
        "--log-dir",
        str(data_dir),
        "--remote-log-dir",
        "",
        "--log-level",
        "info",
    ]
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=dict(os.environ))


def _boot_duckdb(data_dir: Path, port: int, log_f) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "finelog.server.main",
        "--port",
        str(port),
        "--log-dir",
        str(data_dir),
        "--remote-log-dir",
        "",
        "--log-level",
        "WARNING",
    ]
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=dict(os.environ))


def _boot_backend(backend: Backend, server_bin: Path, data_dir: Path, port: int, log_path: Path):
    """Boot ``backend`` against ``data_dir`` (no remote sync). Returns (Popen, log file).

    For DuckDB the namespace registry is seeded first (see
    ``_seed_duckdb_registry``); boot/adopt time is excluded from the timed runs.
    """
    # Handle outlives this function: returned to the caller, which closes it
    # after the server is torn down, so a context manager doesn't fit here.
    log_f = open(log_path, "w")  # noqa: SIM115
    if backend == Backend.RUST:
        proc = _boot_rust(server_bin, data_dir, port, log_f)
    else:
        seeded = _seed_duckdb_registry(data_dir)
        print(f"  seeded DuckDB registry with {seeded} namespaces from the Rust catalog")
        proc = _boot_duckdb(data_dir, port, log_f)
    return proc, log_f


def _fetch_summary(req: logging_pb2.FetchLogsRequest) -> str:
    scope = logging_pb2.MatchScope.Name(req.match_scope)
    return f"FetchLogs(source={req.source!r}, scope={scope}, max_lines={req.max_lines}, tail={req.tail})"


def _run_one(client: LogClient, query: BenchQuery, backend: Backend) -> tuple[int, float]:
    """Run a query once; return (rows, elapsed_ms)."""
    start = time.perf_counter()
    if query.fetch is not None:
        resp = client.fetch_logs(query.fetch)
        rows = len(resp.entries)
    else:
        tbl = client.query(query.sql_for(backend), max_rows=1_000_000)
        rows = tbl.num_rows
    return rows, (time.perf_counter() - start) * 1000.0


def _run_query_set(
    client: LogClient,
    queries: list[BenchQuery],
    backend: Backend,
    repeat: int,
) -> list[QueryResult]:
    results: list[QueryResult] = []
    for q in queries:
        request = _fetch_summary(q.fetch) if q.fetch is not None else q.sql_for(backend)
        res = QueryResult(name=q.name, source=q.source, backend=str(backend), request=request)
        # 1 warmup (discarded) + ``repeat`` timed runs.
        try:
            res.rows, _ = _run_one(client, q, backend)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            results.append(res)
            print(f"  [{q.name}] ERROR on warmup: {res.error}")
            continue
        for _ in range(repeat):
            try:
                res.rows, elapsed = _run_one(client, q, backend)
            except Exception as exc:
                res.error = f"{type(exc).__name__}: {exc}"
                break
            res.samples_ms.append(elapsed)
        flag = "" if (res.max_ms or 0) < SLOW_MS else "  <-- SLOW (>7s)"
        if res.error:
            print(f"  [{q.name}] ERROR: {res.error}")
        else:
            print(
                f"  [{q.name}] p50={res.p50_ms:.0f}ms p95={res.p95_ms:.0f}ms "
                f"max={res.max_ms:.0f}ms rows={res.rows}{flag}"
            )
        results.append(res)
    return results


def _run_backend(
    backend: Backend,
    queries: list[BenchQuery],
    *,
    server_bin: Path,
    data_dir: Path,
    port: int,
    repeat: int,
    health_timeout: float,
    query_timeout_ms: int,
    server_log: Path,
) -> list[QueryResult]:
    """Boot ``backend``, run every query ``repeat`` times, tear the server down."""
    print(f"\n=== backend={backend} :: booting on port {port} against {data_dir}")
    print(f"  server log -> {server_log}")
    proc, log_f = _boot_backend(backend, server_bin, data_dir, port, server_log)
    results: list[QueryResult] = []
    client: LogClient | None = None
    try:
        if not _wait_health(port, timeout=health_timeout):
            print(f"ERROR: {backend} server did not become healthy; tail of server log:", file=sys.stderr)
            log_f.flush()
            for line in server_log.read_text(errors="replace").splitlines()[-30:]:
                print("   " + line, file=sys.stderr)
            return []
        print(f"  /health ok; replaying {len(queries)} queries x{repeat} (+1 warmup)\n")
        client = LogClient.connect(("127.0.0.1", port), timeout_ms=query_timeout_ms)
        results = _run_query_set(client, queries, backend, repeat)
    finally:
        if client is not None:
            client.close()
        _shutdown_server(proc)
        log_f.flush()
        log_f.close()
    return results


def _slow_rpc_lines(log_path: Path, limit: int = 40) -> list[str]:
    if not log_path.exists():
        return []
    out = [line for line in log_path.read_text(errors="replace").splitlines() if "Slow RPC" in line]
    return out[-limit:]


def _backends_for(choice: str) -> list[Backend]:
    if choice == "both":
        # Rust first: it writes the catalog the DuckDB registry is seeded from.
        return [Backend.RUST, Backend.DUCKDB]
    return [Backend(choice)]


def _ms(value: float | None) -> str:
    return f"{value:.0f}" if value is not None else "-"


def _print_comparison(by_backend: dict[Backend, list[QueryResult]], order: list[str]) -> None:
    """Side-by-side per-query table: Rust p50/p95/max vs DuckDB p50/p95/max + ratio."""
    rust = {r.name: r for r in by_backend.get(Backend.RUST, [])}
    duck = {r.name: r for r in by_backend.get(Backend.DUCKDB, [])}
    have_both = bool(rust) and bool(duck)

    header = (
        f"{'query':<26} {'rows':>6} | {'rust_p50':>9} {'rust_p95':>9} {'rust_max':>9} | "
        f"{'duck_p50':>9} {'duck_p95':>9} {'duck_max':>9} | {'rust/duck':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name in order:
        r = rust.get(name)
        d = duck.get(name)
        rows = (r or d).rows if (r or d) else 0
        ratio = "-"
        if have_both and r and d and r.p50_ms and d.p50_ms and not r.error and not d.error:
            ratio = f"{r.p50_ms / d.p50_ms:.1f}x"
        r_cells = (_ms(r.p50_ms), _ms(r.p95_ms), _ms(r.max_ms)) if r and not r.error else ("ERR", "ERR", "ERR")
        d_cells = (_ms(d.p50_ms), _ms(d.p95_ms), _ms(d.max_ms)) if d and not d.error else ("ERR", "ERR", "ERR")
        print(
            f"{name:<26} {rows:>6} | {r_cells[0]:>9} {r_cells[1]:>9} {r_cells[2]:>9} | "
            f"{d_cells[0]:>9} {d_cells[1]:>9} {d_cells[2]:>9} | {ratio:>9}"
        )


def _result_to_dict(r: QueryResult) -> dict:
    return {
        "name": r.name,
        "source": r.source,
        "backend": r.backend,
        "p50_ms": r.p50_ms,
        "p95_ms": r.p95_ms,
        "max_ms": r.max_ms,
        "rows": r.rows,
        "samples_ms": r.samples_ms,
        "error": r.error,
        "request": r.request,
    }


def cmd_bench(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    server_bin = Path(args.server_bin)
    backends = _backends_for(args.backend)
    if Backend.RUST in backends and not server_bin.exists():
        print(f"ERROR: server binary not found: {server_bin}", file=sys.stderr)
        print("Build it: cd rust && cargo build -p finelog --bin finelog-server --profile fast", file=sys.stderr)
        return 2
    if not data_dir.exists():
        print(f"ERROR: data dir not found: {data_dir}. Run `pull` first.", file=sys.stderr)
        return 2

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results_json = Path(args.out) if args.out else results_dir / f"compare-{stamp}.json"

    # Discover real key values, build the broadened families, and merge with the
    # verbatim keyed dashboard queries (kept as the fast-path control).
    keys = discover_keys(data_dir)
    print("Discovered real keys:")
    print(f"  task_prefix          = {keys.task_prefix}")
    print(f"  task_range_hi        = {keys.task_range_hi}")
    print(f"  profile_source       = {keys.profile_source}")
    print(f"  profile_captured_at  = {keys.profile_captured_at}")
    print(f"  log_worker_source    = {keys.log_worker_source}")
    print(f"  log_task_source      = {keys.log_task_source}")
    print(f"  log_controller       = {keys.log_controller_source}")

    queries = DASHBOARD_QUERIES + build_family_queries(keys)
    if args.only:
        wanted = set(args.only)
        queries = [q for q in queries if q.name in wanted]
        if not queries:
            print(f"ERROR: no queries match --only {args.only}", file=sys.stderr)
            return 2

    by_backend: dict[Backend, list[QueryResult]] = {}
    server_logs: dict[Backend, Path] = {}
    for backend in backends:
        server_log = results_dir / f"server-{backend}-{stamp}.log"
        server_logs[backend] = server_log
        by_backend[backend] = _run_backend(
            backend,
            queries,
            server_bin=server_bin,
            data_dir=data_dir,
            port=args.port,
            repeat=args.repeat,
            health_timeout=args.health_timeout,
            query_timeout_ms=args.query_timeout_ms,
            server_log=server_log,
        )

    order = [q.name for q in queries]
    _print_comparison(by_backend, order)

    slow_lines = {str(b): _slow_rpc_lines(p) for b, p in server_logs.items()}
    for backend, lines in slow_lines.items():
        if lines:
            print(f"\nSlow RPC lines ({backend}):")
            for line in lines:
                print("  " + line)

    summary = {
        "timestamp": stamp,
        "data_dir": str(data_dir),
        "server_bin": str(server_bin),
        "backends": [str(b) for b in backends],
        "port": args.port,
        "repeat": args.repeat,
        "discovered_keys": {
            "task_prefix": keys.task_prefix,
            "task_range_hi": keys.task_range_hi,
            "profile_source": keys.profile_source,
            "profile_captured_at": keys.profile_captured_at,
            "log_worker_source": keys.log_worker_source,
            "log_task_source": keys.log_task_source,
            "log_controller_source": keys.log_controller_source,
        },
        "namespaces_on_disk": _namespaces_on_disk(data_dir),
        "results": {str(b): [_result_to_dict(r) for r in rs] for b, rs in by_backend.items()},
        "server_logs": {str(b): str(p) for b, p in server_logs.items()},
        "slow_rpc_log_lines": slow_lines,
    }
    results_json.write_text(json.dumps(summary, indent=2))
    print(f"\nJSON results -> {results_json}")
    for backend, path in server_logs.items():
        print(f"Server log ({backend}) -> {path}")
    return 0


def _namespaces_on_disk(data_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        segs = [p for p in child.glob(SEGMENT_GLOB) if p.is_file()]
        if not segs:
            continue
        out[child.name] = {
            "segments": len(segs),
            "bytes": sum(p.stat().st_size for p in segs),
        }
    return out


def _shutdown_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help="rsync/tar the iris.* (default) segments from the prod VM")
    p_pull.add_argument("--config", default=FINELOG_CONFIG_NAME, help="finelog config name (default: marin)")
    p_pull.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="local data dir")
    p_pull.add_argument(
        "--log-gb",
        type=float,
        default=None,
        help="pull only the most-recent ~N GB of the 'log' namespace (highest min_seq first)",
    )
    p_pull.add_argument("--with-log", action="store_true", help="pull the FULL 25 GB 'log' namespace (opt-in)")
    p_pull.add_argument(
        "--namespaces",
        nargs="+",
        default=None,
        help="explicit namespace list (overrides default iris.* + --log-gb/--with-log)",
    )
    p_pull.set_defaults(func=cmd_pull)

    p_bench = sub.add_parser("bench", help="boot each backend and compare the query set side by side")
    p_bench.add_argument(
        "--backend",
        choices=["rust", "duckdb", "both"],
        default="both",
        help="which engine(s) to benchmark (default: both, run sequentially)",
    )
    p_bench.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="local data dir to serve")
    p_bench.add_argument("--server-bin", default=str(DEFAULT_SERVER_BIN), help="finelog-server fast binary")
    p_bench.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to bind the server")
    p_bench.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help="timed runs per query (default 5)")
    p_bench.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="dir for JSON + server logs")
    p_bench.add_argument("--out", default=None, help="explicit JSON results path")
    p_bench.add_argument("--only", nargs="+", default=None, help="run only these query names")
    p_bench.add_argument("--health-timeout", type=float, default=180.0, help="seconds to wait for /health")
    p_bench.add_argument(
        "--query-timeout-ms",
        type=int,
        default=120_000,
        help="per-query client timeout (ms); must exceed the slow latency under test",
    )
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
