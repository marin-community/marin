#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage cleanup workflow: delete cold unprotected objects with soft-delete safety net.

Approach:
  1. Resolve the protect-set (wildcard and direct prefixes) into concrete prefix lists.
  2. Load the merged protect set into a DuckDB database.
  3. Scan bucket objects into the DuckDB database for fast querying.
  4. Estimate deletion savings via SQL queries against the object catalog.
  5. Enable soft-delete on each source bucket (3-day retention window).
  6. Delete non-STANDARD objects that fall outside the protect set.
  7. Wait for the soft-delete safety window to elapse.
  8. Disable soft-delete to finalize the purge.

Objects in STANDARD storage class are presumed recently-accessed (Autoclass promotes active
objects) and are never deleted.  The protect-set is always preserved regardless of class.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shlex
import shutil
import duckdb
import subprocess
import sys
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import click
import google.auth
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
STORAGE_DIR = SCRIPT_PATH.parent
REPO_ROOT = STORAGE_DIR.parent.parent
OUTPUT_ROOT = STORAGE_DIR / "purge"
PROTECT_DIR = OUTPUT_ROOT / "protect"
LOG_DIR = OUTPUT_ROOT / "logs"
STORAGE_DB_PATH = OUTPUT_ROOT / "storage.duckdb"
OBJECTS_PARQUET_DIR = OUTPUT_ROOT / "objects_parquet"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARIN_BUCKETS = [
    "marin-eu-west4",
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-east1",
    "marin-us-east5",
    "marin-us-west4",
]

BUCKET_LOCATIONS = {
    "marin-eu-west4": "EUROPE-WEST4",
    "marin-us-central1": "US-CENTRAL1",
    "marin-us-central2": "US-CENTRAL2",
    "marin-us-east1": "US-EAST1",
    "marin-us-east5": "US-EAST5",
    "marin-us-west4": "US-WEST4",
}

NON_STANDARD_STORAGE_CLASSES = frozenset({"NEARLINE", "COLDLINE", "ARCHIVE"})

SOFT_DELETE_RETENTION_SECONDS = 3 * 24 * 3600  # 3 days

GCS_MAX_PAGE_SIZE = 5000

# Partial response fields for list_blobs — only fetch what _blob_to_scanned needs.
_BLOB_FIELDS = "items(name,size,storageClass,timeCreated,updated),prefixes,nextPageToken"
ADAPTIVE_SPLIT_THRESHOLD = 10000  # scan flat if <= this many objects; split otherwise
ADAPTIVE_SCAN_MAX_DEPTH = 2

GCS_DISCOUNT = 0.30

# Deterministic fingerprint for the plan (bucket list).
PLAN_FINGERPRINT = hashlib.sha256(json.dumps(MARIN_BUCKETS, sort_keys=True).encode()).hexdigest()

OBJECT_FLUSH_THRESHOLD = 10_000_000
DELETE_BATCH_SIZE = 1000

# ---------------------------------------------------------------------------
# Step spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSpec:
    action_id: str
    group_name: str
    command_name: str
    description: str
    help_text: str
    mutating: bool
    runner: Callable[[Context, StepSpec], None]
    predecessors: tuple[str, ...] = ()

    scan_workers: bool = False
    settle_hours: bool = False
    optional: bool = False

    def run(self, ctx: Context) -> None:
        self.runner(ctx, self)


@dataclass(frozen=True)
class ScannedObject:
    bucket: str
    name: str
    size_bytes: int
    storage_class_id: int
    created: datetime | None
    updated: datetime | None


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class Context:
    dry_run: bool
    force: bool
    include_optional: bool
    scan_workers: int
    settle_hours: int
    log_path: Path
    timestamp: str
    project: str | None


# ---------------------------------------------------------------------------
# Output directories and cache
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    for path in [OUTPUT_ROOT, PROTECT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    init_db()


# ---------------------------------------------------------------------------
# Module-level DuckDB connection singleton
# ---------------------------------------------------------------------------

_db: duckdb.DuckDBPyConnection | None = None
_db_lock = threading.Lock()


def get_db() -> duckdb.DuckDBPyConnection:
    """Return the module-level DuckDB connection. Raises if init_db() has not been called."""
    if _db is None:
        raise RuntimeError("DuckDB connection not initialized — call init_db() first")
    return _db


def _fetchone_dict(result: duckdb.DuckDBPyConnection) -> dict[str, Any] | None:
    row = result.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in result.description], row, strict=False))


def _fetchall_dicts(result: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row, strict=False)) for row in result.fetchall()]


SCHEMA_VERSION = 9

_OBJECTS_ARROW_SCHEMA = pa.schema(
    [
        ("bucket", pa.string()),
        ("name", pa.string()),
        ("size_bytes", pa.int64()),
        ("storage_class_id", pa.int32()),
        ("created", pa.timestamp("us", tz="UTC")),
        ("updated", pa.timestamp("us", tz="UTC")),
    ]
)

_ARROW_TO_DUCKDB: dict[pa.DataType, str] = {
    pa.string(): "VARCHAR",
    pa.int64(): "BIGINT",
    pa.int32(): "INTEGER",
    pa.timestamp("us", tz="UTC"): "TIMESTAMPTZ",
}


class ObjectBuffer:
    """Buffers scanned objects and flushes them as sorted, ZSTD-compressed parquet segments.

    Owns the parquet segment directory and the DuckDB view that unions all segments.
    Append via `extend`, call `flush(force=True)` when done scanning.
    """

    def __init__(self, parquet_dir: Path, conn: duckdb.DuckDBPyConnection) -> None:
        self._parquet_dir = parquet_dir
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        self._conn = conn
        self._pending: list[ScannedObject] = []

        existing = sorted(self._parquet_dir.glob("objects_*.parquet"))
        self._segment_counter = int(existing[-1].stem.split("_")[1]) if existing else 0
        self._refresh_view()

    def _next_path(self) -> Path:
        self._segment_counter += 1
        return self._parquet_dir / f"objects_{self._segment_counter:06d}.parquet"

    def _refresh_view(self) -> None:
        """Point the ``objects`` view at current parquet segments."""
        has_segments = any(self._parquet_dir.glob("objects_*.parquet"))
        if has_segments:
            glob = str(self._parquet_dir / "objects_*.parquet")
            self._conn.execute(
                f"""
                CREATE OR REPLACE VIEW objects AS
                SELECT * FROM read_parquet('{glob}', union_by_name=true, hive_partitioning=false)
            """
            )
        else:
            cols = ", ".join(f"NULL::{_ARROW_TO_DUCKDB[f.type]} AS {f.name}" for f in _OBJECTS_ARROW_SCHEMA)
            self._conn.execute(f"CREATE OR REPLACE VIEW objects AS SELECT {cols} WHERE false")

    def add(self, objects: list[ScannedObject]) -> None:
        """Append objects to the buffer, flushing to parquet if the threshold is reached."""
        self._pending.extend(objects)
        self.flush()

    def flush(self, *, force: bool = False) -> None:
        if len(self._pending) < OBJECT_FLUSH_THRESHOLD and not force:
            return
        if not self._pending:
            return
        self._pending.sort(key=lambda o: (o.bucket, o.name))
        arrow_table = pa.table(
            {
                "bucket": [o.bucket for o in self._pending],
                "name": [o.name for o in self._pending],
                "size_bytes": [o.size_bytes for o in self._pending],
                "storage_class_id": [o.storage_class_id for o in self._pending],
                "created": [o.created for o in self._pending],
                "updated": [o.updated for o in self._pending],
            },
            schema=_OBJECTS_ARROW_SCHEMA,
        )
        pq.write_table(arrow_table, self._next_path(), compression="zstd")
        self._refresh_view()
        self._pending.clear()

    def reset(self) -> None:
        """Remove all parquet segments and reset counter."""
        shutil.rmtree(self._parquet_dir, ignore_errors=True)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        self._segment_counter = 0
        self._pending.clear()
        self._refresh_view()


def init_db() -> None:
    """Open the module-level DuckDB connection and ensure the schema is current."""
    global _db
    if _db is not None:
        return
    print_summary(f"opening DuckDB catalog: {STORAGE_DB_PATH}")
    conn = duckdb.connect(str(STORAGE_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    row = _fetchone_dict(conn.execute("SELECT value FROM cache_meta WHERE key = 'schema_version'"))
    current_version = int(row["value"]) if row else 0
    if current_version < SCHEMA_VERSION:
        print_summary(f"schema upgrade {current_version} → {SCHEMA_VERSION}, rebuilding tables")
        for tbl in [
            "listing_cache",
            "prefix_estimate_cache",
            "prefix_scan_cache",
            "storage_classes",
            "protect_prefixes",
            "protect_rules",
            "split_cache",
            "scanned_prefixes",
            "step_markers",
        ]:
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        conn.execute("DROP VIEW IF EXISTS objects")
        shutil.rmtree(OBJECTS_PARQUET_DIR, ignore_errors=True)
        conn.execute(
            "INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS storage_classes (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            price_per_gib_month_us REAL NOT NULL,
            price_per_gib_month_eu REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS protect_rules (
            bucket TEXT NOT NULL,
            pattern TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            owners TEXT,
            reasons TEXT,
            sources TEXT,
            PRIMARY KEY (bucket, pattern)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scanned_prefixes (
            bucket TEXT NOT NULL,
            prefix TEXT NOT NULL,
            object_count INTEGER NOT NULL,
            scanned_at TEXT NOT NULL,
            PRIMARY KEY (bucket, prefix)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS split_cache (
            cache_key TEXT PRIMARY KEY,
            entries_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS step_markers (
            action_id TEXT PRIMARY KEY,
            completed_at TEXT NOT NULL,
            dry_run BOOLEAN NOT NULL,
            input_fingerprint TEXT NOT NULL,
            extra_json TEXT
        )
        """
    )
    # Create the objects view over any existing parquet segments
    ObjectBuffer(OBJECTS_PARQUET_DIR, conn)

    # Seed storage classes
    for sc_id, name, us_price, eu_price in [
        (1, "STANDARD", 0.020, 0.023),
        (2, "NEARLINE", 0.010, 0.013),
        (3, "COLDLINE", 0.004, 0.006),
        (4, "ARCHIVE", 0.0012, 0.0025),
    ]:
        conn.execute(
            """
            INSERT INTO storage_classes (id, name, price_per_gib_month_us, price_per_gib_month_eu)
            VALUES (?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (sc_id, name, us_price, eu_price),
        )
    _db = conn


def storage_class_id_map() -> dict[str, int]:
    """Return a mapping from storage class name to its DB id."""
    conn = get_db()
    rows = _fetchall_dicts(conn.execute("SELECT id, name FROM storage_classes"))
    return {row["name"]: row["id"] for row in rows}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def timestamp_string() -> str:
    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def log_line(ctx: Context, message: str) -> None:
    with ctx.log_path.open("a") as f:
        f.write(message.rstrip() + "\n")


def command_env() -> dict[str, str]:
    return dict(os.environ)


@cache
def default_project() -> str:
    for env_var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        value = os.environ.get(env_var)
        if value:
            return value
    completed = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    project = completed.stdout.strip()
    if completed.returncode == 0 and project and project != "(unset)":
        return project
    _, detected_project = google.auth.default()
    if detected_project:
        return detected_project
    raise click.ClickException(
        "Could not determine a GCP project. Pass --project or set GOOGLE_CLOUD_PROJECT / gcloud config project."
    )


def resolved_project(ctx: Context) -> str:
    return ctx.project or default_project()


def storage_client(ctx: Context) -> storage.Client:
    project = resolved_project(ctx)
    credentials, _ = google.auth.default()
    return storage.Client(project=project, credentials=credentials)


def run_subprocess(
    ctx: Context,
    command: list[str],
    *,
    input_text: str | None = None,
    allow_failure: bool = False,
    capture_json: bool = False,
) -> Any:
    rendered = shlex.join(command)
    print_summary(f"$ {rendered}")
    log_line(ctx, f"$ {rendered}")
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        env=command_env(),
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.stdout and completed.stdout.strip():
        print(completed.stdout.rstrip())
        log_line(ctx, completed.stdout)
    if completed.stderr and completed.stderr.strip():
        print(completed.stderr.rstrip(), file=sys.stderr)
        log_line(ctx, completed.stderr)
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed ({completed.returncode}): {rendered}\n{completed.stderr}")
    if capture_json:
        return json.loads(completed.stdout or "{}") if completed.returncode == 0 else None
    return completed


def print_summary(message: str) -> None:
    print(message)


# ---------------------------------------------------------------------------
# CSV / JSON
# ---------------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Markers (resumable step state)
# ---------------------------------------------------------------------------


def marker_matches(action_id: str, input_fingerprint: str) -> bool:
    row = _fetchone_dict(
        get_db().execute("SELECT input_fingerprint FROM step_markers WHERE action_id = ?", (action_id,))
    )
    return row is not None and row["input_fingerprint"] == input_fingerprint


def marker_exists(action_id: str) -> bool:
    row = _fetchone_dict(get_db().execute("SELECT 1 FROM step_markers WHERE action_id = ?", (action_id,)))
    return row is not None


def read_marker_extra(action_id: str) -> dict[str, Any] | None:
    """Read the extra_json blob for a step marker. Returns None if no marker exists."""
    row = _fetchone_dict(get_db().execute("SELECT extra_json FROM step_markers WHERE action_id = ?", (action_id,)))
    if row is None or row["extra_json"] is None:
        return None
    return json.loads(row["extra_json"])


def write_marker(
    action_id: str,
    input_fingerprint: str,
    *,
    dry_run: bool,
    extra: dict[str, Any] | None = None,
) -> None:
    get_db().execute(
        "INSERT OR REPLACE INTO step_markers (action_id, completed_at, dry_run, input_fingerprint, extra_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (action_id, now_utc().isoformat(), dry_run, input_fingerprint, json.dumps(extra) if extra else None),
    )


# ---------------------------------------------------------------------------
# Region / bucket helpers
# ---------------------------------------------------------------------------


def region_from_bucket(bucket: str) -> str:
    return bucket.removeprefix("marin-") if bucket.startswith("marin-") else bucket


def region_bucket(region: str) -> str:
    return f"marin-{region}"


def all_regions() -> list[str]:
    return [region_from_bucket(bucket) for bucket in MARIN_BUCKETS]


def url_bucket(url: str) -> str:
    return url.removeprefix("gs://").split("/", 1)[0]


def url_object_path(url: str) -> str:
    parts = url.removeprefix("gs://").split("/", 1)
    return parts[1] if len(parts) == 2 else ""


def plan_rows() -> list[dict[str, str]]:
    """Return the cleanup plan: one row per bucket with region and location."""
    return [{"region": region_from_bucket(b), "bucket": b, "location": BUCKET_LOCATIONS[b]} for b in MARIN_BUCKETS]


def normalized_prefix_url(url: str) -> str:
    return url if url.endswith("/") else f"{url}/"


def normalize_relative_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


def relative_prefix(prefix_url: str, bucket: str) -> str:
    prefix = url_object_path(prefix_url)
    if prefix.endswith("/"):
        return prefix
    return f"{prefix}/" if prefix else ""


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def continent_for_region(region: str) -> str:
    return "EUROPE" if region.startswith("eu-") else "US"


# ---------------------------------------------------------------------------
# Prefix utilities
# ---------------------------------------------------------------------------


def glob_to_like(gs_glob: str) -> str:
    """Convert a GCS glob pattern to a SQL LIKE pattern relative to the bucket.

    Examples:
        gs://bucket/checkpoints/dclm_1b* → checkpoints/dclm_1b%
        gs://bucket/tokenized/*_marin_tokenizer* → tokenized/%_marin_tokenizer%
        gs://bucket/DL3DV/** → DL3DV/%
    """
    path = url_object_path(gs_glob)
    return path.replace("*", "%").replace("?", "_")


# SQL fragment for checking if an object is protected by any rule.
# Use as: NOT EXISTS (SELECT 1 FROM protect_rules r WHERE {IS_PROTECTED})
# Requires outer query to alias the object table as 'o'.
IS_PROTECTED = """
    SELECT 1 FROM protect_rules r
    WHERE o.bucket = r.bucket
      AND CASE r.pattern_type
          WHEN 'prefix' THEN o.name LIKE r.pattern || '%'
          WHEN 'like' THEN o.name LIKE r.pattern
      END
"""


def gcloud_bucket_describe(ctx: Context, bucket_url: str) -> dict[str, Any]:
    return run_subprocess(
        ctx,
        ["gcloud", "storage", "buckets", "describe", bucket_url, "--format=json"],
        capture_json=True,
    )


def bucket_soft_delete_seconds(metadata: dict[str, Any]) -> int:
    policy = metadata.get("softDeletePolicy") or {}
    value = policy.get("retentionDurationSeconds")
    return int(value) if value is not None else 0


# ---------------------------------------------------------------------------
# Prefix resolution
# ---------------------------------------------------------------------------


# ===========================================================================
# PREP steps
# ===========================================================================


def load_protect_rules(ctx: Context, action: StepSpec) -> None:
    """Load protect globs and direct prefixes into the protect_rules DB table.

    Reads both classified and direct CSVs. Direct prefixes become pattern_type='prefix',
    wildcard globs become pattern_type='like' with * → % conversion. No GCS calls needed.
    """
    classified_path = PROTECT_DIR / "protect_prefixes_classified.csv"
    direct_path = PROTECT_DIR / "protect_prefixes_direct.csv"
    fingerprint = hashlib.sha256((file_digest(classified_path) + file_digest(direct_path)).encode()).hexdigest()
    if not ctx.force and marker_matches(action.action_id, fingerprint):
        print_summary(f"skip {action.action_id}: marker is current")
        return

    classified_rows = read_csv_rows(classified_path)
    direct_rows = read_csv_rows(direct_path)

    # Build DB rows: (bucket, pattern, pattern_type, owners, reasons, sources)
    db_rows: list[tuple[str, str, str, str, str, str]] = []

    # Direct prefixes → pattern_type='prefix'
    for row in direct_rows:
        rel = normalize_relative_prefix(url_object_path(normalized_prefix_url(row["sts_prefix"])))
        db_rows.append((row["bucket"], rel, "prefix", row["owners"], row["reasons"], row["sources"]))

    # Wildcard globs → pattern_type='like'
    for row in classified_rows:
        if row["classification"] != "sts_prefix_via_listing":
            continue
        like_pattern = glob_to_like(row["normalized_glob"])
        db_rows.append((row["bucket"], like_pattern, "like", row["owners"], row["reasons"], row["sources"]))

    # Deduplicate by (bucket, pattern)
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str, str, str, str, str]] = []
    for r in db_rows:
        key = (r[0], r[1])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Write to DB
    conn = get_db()
    conn.execute("DELETE FROM protect_rules")
    if deduped:
        arrow_table = pa.table(
            {
                "bucket": [r[0] for r in deduped],
                "pattern": [r[1] for r in deduped],
                "pattern_type": [r[2] for r in deduped],
                "owners": [r[3] for r in deduped],
                "reasons": [r[4] for r in deduped],
                "sources": [r[5] for r in deduped],
            }
        )
        conn.register("_protect_stage", arrow_table)
        conn.execute(
            """
            INSERT OR REPLACE INTO protect_rules (bucket, pattern, pattern_type, owners, reasons, sources)
            SELECT bucket, pattern, pattern_type, owners, reasons, sources FROM _protect_stage
            """
        )
        conn.unregister("_protect_stage")

    prefix_count = sum(1 for r in deduped if r[2] == "prefix")
    like_count = sum(1 for r in deduped if r[2] == "like")
    print_summary(
        f"{action.action_id}: loaded {len(deduped)} protect rules ({prefix_count} prefix, {like_count} like) into DB"
    )
    write_marker(action.action_id, fingerprint, dry_run=ctx.dry_run)


# ===========================================================================
# PREP: scan objects into DuckDB
# ===========================================================================


@dataclass(frozen=True)
class ScanEvent:
    """Event pushed from a scan worker to the main-thread event loop.

    Kinds:
      progress   — non-terminal status update (object_count is running total)
      leaf_done  — terminal, prefix fully scanned, objects contains all blobs
      split_page — non-terminal, one page of delimiter listing results;
                   objects = root-level blobs, sub_prefixes = discovered children
      split_done — terminal, delimiter listing complete for this prefix
    """

    prefix: str
    depth: int
    worker_id: int  # threading.get_ident()
    kind: str
    objects: list[ScannedObject] = field(default_factory=list)
    object_count: int = 0
    sub_prefixes: list[str] | None = None


def _blob_to_scanned(blob: Any, bucket_name: str, sc_id_map: dict[str, int]) -> ScannedObject:
    sc = blob.storage_class or "STANDARD"
    return ScannedObject(
        bucket=bucket_name,
        name=blob.name,
        size_bytes=int(blob.size or 0),
        storage_class_id=sc_id_map.get(sc, sc_id_map["STANDARD"]),
        created=blob.time_created,
        updated=blob.updated,
    )


_SENTINEL = (None, -1)  # poison pill for worker shutdown


def _scan_one_prefix(
    client: storage.Client,
    bucket_name: str,
    prefix: str,
    depth: int,
    sc_id_map: dict[str, int],
    wid: int,
    eq: queue.Queue[ScanEvent],
) -> None:
    """Scan a single prefix, pushing ScanEvents to the queue."""

    def _put(kind: str, **kwargs: Any) -> None:
        eq.put(ScanEvent(prefix=prefix, depth=depth, worker_id=wid, kind=kind, **kwargs))

    _put("progress", object_count=0)

    if prefix == "":
        objects: list[ScannedObject] = []
        for blob in client.list_blobs(bucket_name, delimiter="/", page_size=GCS_MAX_PAGE_SIZE, fields=_BLOB_FIELDS):
            objects.append(_blob_to_scanned(blob, bucket_name, sc_id_map))
        _put("leaf_done", objects=objects, object_count=len(objects))
        return

    # Optimistic flat scan — read pages up to ADAPTIVE_SPLIT_THRESHOLD.
    # If the prefix has fewer objects than the threshold, scan it flat
    # even if it spans multiple pages (avoids needless splitting on
    # prefixes like SpatialVID/videos/group_0030 with ~5k objects).
    iterator = client.list_blobs(bucket_name, prefix=prefix, page_size=GCS_MAX_PAGE_SIZE, fields=_BLOB_FIELDS)
    pages_iter = iterator.pages
    probe_objects: list[ScannedObject] = []
    is_small = False
    for page in pages_iter:
        page_items = [_blob_to_scanned(blob, bucket_name, sc_id_map) for blob in page]
        probe_objects.extend(page_items)
        if len(page_items) < GCS_MAX_PAGE_SIZE:
            # Partial page means we've exhausted the prefix.
            is_small = True
            break
        if len(probe_objects) >= ADAPTIVE_SPLIT_THRESHOLD:
            break

    if is_small:
        _put("leaf_done", objects=probe_objects, object_count=len(probe_objects))
        return

    if depth >= ADAPTIVE_SCAN_MAX_DEPTH:
        # At max depth — continue the flat scan from where the probe left off.
        all_objects = probe_objects
        for page in pages_iter:
            all_objects.extend(_blob_to_scanned(blob, bucket_name, sc_id_map) for blob in page)
            _put("progress", object_count=len(all_objects))
        _put("leaf_done", objects=all_objects, object_count=len(all_objects))
        return

    # Large prefix, can still split. Discard probe_objects and do a
    # delimiter listing, yielding sub-prefixes page-by-page.
    sub_iterator = client.list_blobs(
        bucket_name, prefix=prefix, delimiter="/", page_size=GCS_MAX_PAGE_SIZE, fields=_BLOB_FIELDS
    )
    all_root_objects: list[ScannedObject] = []
    any_sub_prefixes = False
    for page in sub_iterator.pages:
        page_roots = [_blob_to_scanned(blob, bucket_name, sc_id_map) for blob in page]
        page_subs = list(page.prefixes)
        all_root_objects.extend(page_roots)
        if page_subs:
            any_sub_prefixes = True
            _put("split_page", objects=page_roots, object_count=len(all_root_objects), sub_prefixes=page_subs)
        else:
            _put("progress", object_count=len(all_root_objects))

    if not any_sub_prefixes:
        _put("leaf_done", objects=all_root_objects, object_count=len(all_root_objects))
    else:
        _put("split_done", object_count=len(all_root_objects))


def _scan_worker_loop(
    ctx: Context,
    bucket_name: str,
    sc_id_map: dict[str, int],
    work_queue: queue.Queue[tuple[str | None, int]],
    event_queue: queue.Queue[ScanEvent],
) -> None:
    """Long-lived worker: pull (prefix, depth) from work_queue, scan, repeat.

    Exits when it receives _SENTINEL.
    """
    wid = threading.get_ident()
    client = storage_client(ctx)

    while True:
        item = work_queue.get()
        if item == _SENTINEL:
            return
        prefix, depth = item
        try:
            _scan_one_prefix(client, bucket_name, prefix, depth, sc_id_map, wid, event_queue)
        except Exception:
            log.exception("worker error scanning %s/%s", bucket_name, prefix)
            event_queue.put(ScanEvent(prefix=prefix, depth=depth, worker_id=wid, kind="error"))


METADATA_FLUSH_THRESHOLD = 500


@dataclass
class ScanBuffer:
    objects: ObjectBuffer
    prefixes: list[tuple[str, str, int, str]] = field(default_factory=list)
    split_cache: list[tuple[str, str, str]] = field(default_factory=list)


def _buffer_prefix_scanned(buf: ScanBuffer, bucket_name: str, prefix: str, object_count: int) -> None:
    buf.prefixes.append((bucket_name, prefix, object_count, now_utc().isoformat()))


def _buffer_split_cache(
    buf: ScanBuffer,
    cache: dict[str, list[str]],
    bucket_name: str,
    prefix: str,
    children: list[str],
) -> None:
    key = _split_cache_key(bucket_name, prefix)
    cache[key] = children
    buf.split_cache.append((key, json.dumps(children), now_utc().isoformat()))


def _flush_metadata(conn: duckdb.DuckDBPyConnection, buf: ScanBuffer, *, force: bool = False) -> None:
    """Batch-flush buffered prefix and split-cache writes via PyArrow staging."""
    total = len(buf.prefixes) + len(buf.split_cache)
    if total < METADATA_FLUSH_THRESHOLD and not force:
        return
    if buf.prefixes:
        arrow_table = pa.table(
            {
                "bucket": [r[0] for r in buf.prefixes],
                "prefix": [r[1] for r in buf.prefixes],
                "object_count": [r[2] for r in buf.prefixes],
                "scanned_at": [r[3] for r in buf.prefixes],
            }
        )
        conn.register("_pfx_stage", arrow_table)
        conn.execute(
            """
            INSERT OR REPLACE INTO scanned_prefixes (bucket, prefix, object_count, scanned_at)
            SELECT bucket, prefix, object_count, scanned_at FROM _pfx_stage
            """
        )
        conn.unregister("_pfx_stage")
        buf.prefixes.clear()
    if buf.split_cache:
        arrow_table = pa.table(
            {
                "cache_key": [r[0] for r in buf.split_cache],
                "entries_json": [r[1] for r in buf.split_cache],
                "updated_at": [r[2] for r in buf.split_cache],
            }
        )
        conn.register("_sc_stage", arrow_table)
        conn.execute(
            """
            INSERT OR REPLACE INTO split_cache (cache_key, entries_json, updated_at)
            SELECT cache_key, entries_json, updated_at FROM _sc_stage
            """
        )
        conn.unregister("_sc_stage")
        buf.split_cache.clear()


@dataclass
class WorkerSlot:
    """State of a single worker thread as seen by the display."""

    slot_id: int
    prefix: str = ""
    start_time: float = 0.0
    object_count: int = 0
    sub_prefix_count: int = 0  # sub-prefixes found during split phase


@dataclass
class ScanProgress:
    """Rich Live display for the adaptive scan loop with per-worker tracking."""

    bucket_name: str
    num_workers: int
    total_objects: int = 0
    prefixes_completed: int = 0
    prefixes_expanded: int = 0
    prefixes_queued: int = 0
    _start_time: float = field(default_factory=time.monotonic)
    # Maps thread_ident -> stable display slot
    _thread_slots: dict[int, WorkerSlot] = field(default_factory=dict)
    _next_slot: int = 0
    _progress: Progress = field(init=False)
    _task_id: Any = field(init=False)
    _live: Live = field(init=False)
    _console: Console = field(init=False)

    def __post_init__(self) -> None:
        self._console = Console()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=120),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._task_id = self._progress.add_task(f"scan {self.bucket_name}", total=0)
        self._live = Live(self._render(), console=self._console, refresh_per_second=4)

    def _slot(self, worker_id: int) -> WorkerSlot:
        if worker_id not in self._thread_slots:
            self._thread_slots[worker_id] = WorkerSlot(slot_id=self._next_slot)
            self._next_slot += 1
        return self._thread_slots[worker_id]

    def start(self) -> None:
        self._live.start()

    def stop(self) -> None:
        self._live.stop()

    def set_total(self, total: int) -> None:
        self.prefixes_queued = total
        self._progress.update(self._task_id, total=total)

    def add_to_total(self, n: int) -> None:
        self.prefixes_queued += n
        self._progress.update(self._task_id, total=self.prefixes_queued)

    def handle_event(self, event: ScanEvent) -> None:
        slot = self._slot(event.worker_id)
        # Use delta-based counting so total_objects updates incrementally
        # from every event type, not just terminal ones.
        delta = event.object_count - slot.object_count
        if event.kind == "progress":
            self.total_objects += delta
            slot.object_count = event.object_count
        elif event.kind == "split_page":
            self.total_objects += delta
            slot.object_count = event.object_count
            slot.sub_prefix_count += len(event.sub_prefixes or [])
        elif event.kind == "leaf_done":
            self.prefixes_completed += 1
            self.total_objects += delta
            self._clear_slot(slot)
            self._progress.advance(self._task_id)
        elif event.kind == "split_done":
            self.prefixes_expanded += 1
            self.total_objects += delta
            self._clear_slot(slot)
            self._progress.advance(self._task_id)
        elif event.kind == "error":
            self.prefixes_completed += 1  # count as done so we don't hang
            self._clear_slot(slot)
            self._progress.advance(self._task_id)

    def _clear_slot(self, slot: WorkerSlot) -> None:
        slot.prefix = ""
        slot.object_count = 0
        slot.sub_prefix_count = 0

    def mark_worker_start(self, worker_id: int, prefix: str) -> None:
        slot = self._slot(worker_id)
        slot.prefix = prefix
        slot.start_time = time.monotonic()
        slot.object_count = 0
        slot.sub_prefix_count = 0

    def mark_queued_split(self, children_count: int, already_done: int = 0) -> None:
        """Account for a cached split expansion (no worker involved)."""
        self.prefixes_expanded += 1
        self.prefixes_completed += already_done
        self.add_to_total(children_count - 1)
        # Advance for the parent + already-done children
        self._progress.advance(self._task_id, advance=1 + already_done)

    def _refresh(self) -> None:
        self._live.update(self._render())

    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        if seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m{s:02d}s"
        h, remainder = divmod(int(seconds), 3600)
        m, _ = divmod(remainder, 60)
        return f"{h}h{m:02d}m"

    def _render(self) -> Group:
        done = self.prefixes_completed + self.prefixes_expanded
        pending = self.prefixes_queued - done
        elapsed = time.monotonic() - self._start_time

        # ETA based on completion rate
        eta_str = ""
        if done > 0 and pending > 0 and elapsed > 5:
            rate = done / elapsed
            eta_seconds = pending / rate
            eta_str = self._format_duration(eta_seconds)

        stats = Text()
        stats.append("  objects: ", style="dim")
        stats.append(f"{self.total_objects:,}", style="bold green")
        stats.append("    splits: ", style="dim")
        stats.append(f"{self.prefixes_expanded}", style="yellow")
        stats.append("    queued: ", style="dim")
        stats.append(f"{pending:,}")

        active_slots = [s for s in self._thread_slots.values() if s.prefix]
        stats.append("    workers: ", style="dim")
        stats.append(f"{len(active_slots)}/{self.num_workers}")

        if eta_str:
            stats.append("    eta: ", style="dim")
            stats.append(eta_str, style="bold cyan")

        table = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
        table.add_column("w", style="dim", width=3)
        table.add_column("prefix", style="cyan", no_wrap=True, max_width=50)
        table.add_column("objects", style="green", justify="right", width=9)
        table.add_column("subs", style="magenta", justify="right", width=6)
        table.add_column("elapsed", style="yellow", justify="right", width=8)

        now = time.monotonic()
        # Show all slots sorted by slot_id; active first, then idle
        all_slots = sorted(self._thread_slots.values(), key=lambda s: (not s.prefix, s.slot_id))
        for slot in all_slots[: self.num_workers]:
            slot_label = f"w{slot.slot_id}"
            if not slot.prefix:
                table.add_row(slot_label, "(idle)", "", "", "", style="dim")
                continue
            elapsed = now - slot.start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
            display_prefix = slot.prefix.rstrip("/") if slot.prefix else "(root)"
            obj_str = f"{slot.object_count:,}" if slot.object_count else ""
            sub_str = f"{slot.sub_prefix_count:,}" if slot.sub_prefix_count else ""
            table.add_row(slot_label, display_prefix, obj_str, sub_str, time_str)

        return Group(self._progress, stats, table)


def _split_cache_key(bucket_name: str, prefix: str) -> str:
    return f"scan://{bucket_name}/{prefix}"


def _load_split_cache(conn: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """Load the split_cache table into memory for fast lookups."""
    rows = conn.execute("SELECT cache_key, entries_json FROM split_cache").fetchall()
    cache: dict[str, list[str]] = {}
    for cache_key, entries_json in rows:
        entries = json.loads(entries_json)
        if isinstance(entries, list):
            cache[cache_key] = entries
    return cache


def _read_split_cache(cache: dict[str, list[str]], bucket_name: str, prefix: str) -> list[str] | None:
    return cache.get(_split_cache_key(bucket_name, prefix))


def _run_adaptive_scan(
    ctx: Context,
    bucket_name: str,
    sc_id_map: dict[str, int],
    pending: list[tuple[str, int]],
    num_workers: int,
    already_scanned: set[str],
    db_conn: duckdb.DuckDBPyConnection,
    progress: ScanProgress,
    buf: ScanBuffer,
) -> None:
    """Two-queue adaptive scan: workers are self-scheduling.

    work_queue:  (prefix, depth) items — workers pull from this autonomously
    event_queue: ScanEvent items — workers push status here

    Workers are long-lived loops that pull work, scan, push events, repeat.
    The main thread drains events, does DB writes, and puts new work
    (from splits or cache) onto the work queue.  Workers never wait on
    the main thread for dispatch.
    """
    work_queue: queue.Queue[tuple[str | None, int]] = queue.Queue()
    event_queue: queue.Queue[ScanEvent] = queue.Queue()
    in_flight = 0
    split_children: dict[str, list[str]] = {}
    split_cache = _load_split_cache(db_conn)

    def _enqueue(prefix: str, depth: int) -> None:
        nonlocal in_flight
        in_flight += 1
        work_queue.put((prefix, depth))

    def _expand_cached_or_enqueue(prefix: str, depth: int) -> None:
        """Use cached split children if available, otherwise enqueue for workers."""
        cached_children = _read_split_cache(split_cache, bucket_name, prefix)
        if cached_children is not None:
            new_children = [sp for sp in cached_children if sp not in already_scanned]
            skipped = len(cached_children) - len(new_children)
            # Only count un-scanned children in the total.
            # Already-scanned children count as both queued and completed.
            progress.mark_queued_split(len(cached_children), already_done=skipped)
            for sp in new_children:
                _expand_cached_or_enqueue(sp, depth + 1)
        else:
            _enqueue(prefix, depth)

    def _process_event(event: ScanEvent) -> None:
        nonlocal in_flight

        slot = progress._slot(event.worker_id)
        if not slot.prefix:
            progress.mark_worker_start(event.worker_id, event.prefix)

        if event.kind == "progress":
            progress.handle_event(event)
            return

        if event.kind == "split_page":
            if event.objects:
                buf.objects.add(event.objects)
            page_subs = event.sub_prefixes or []
            split_children.setdefault(event.prefix, []).extend(page_subs)
            new_children = [sp for sp in page_subs if sp not in already_scanned]
            progress.add_to_total(len(new_children))
            progress.handle_event(event)
            for sp in new_children:
                _expand_cached_or_enqueue(sp, event.depth + 1)
            return

        if event.kind == "split_done":
            all_children = split_children.pop(event.prefix, [])
            _buffer_split_cache(buf, split_cache, bucket_name, event.prefix, sorted(all_children))
            _flush_metadata(db_conn, buf)
            progress.handle_event(event)
            in_flight -= 1
            return

        if event.kind == "leaf_done":
            if event.objects:
                buf.objects.add(event.objects)
            _buffer_prefix_scanned(buf, bucket_name, event.prefix, event.object_count)
            _flush_metadata(db_conn, buf)
            already_scanned.add(event.prefix)
            progress.handle_event(event)
            in_flight -= 1
            return

        if event.kind == "error":
            log.warning("scan failed for %s/%s — will retry on next run", bucket_name, event.prefix)
            # Don't mark as scanned — it'll be retried on resume.
            # Clean up any partial split state.
            split_children.pop(event.prefix, None)
            progress.handle_event(event)
            in_flight -= 1

    # Seed the work queue
    for p, d in pending:
        _expand_cached_or_enqueue(p, d)

    # Start worker threads
    worker_threads = []
    for _ in range(num_workers):
        t = threading.Thread(
            target=_scan_worker_loop,
            args=(ctx, bucket_name, sc_id_map, work_queue, event_queue),
            daemon=True,
        )
        t.start()
        worker_threads.append(t)

    try:
        while in_flight > 0:
            try:
                event = event_queue.get(timeout=0.25)
            except queue.Empty:
                progress._refresh()
                continue

            # Drain all available events
            batch = [event]
            while True:
                try:
                    batch.append(event_queue.get_nowait())
                except queue.Empty:
                    break

            for ev in batch:
                _process_event(ev)

            progress._refresh()
    finally:
        # Flush any remaining buffered objects and metadata
        buf.objects.flush(force=True)
        _flush_metadata(db_conn, buf, force=True)
        # Shut down workers
        for _ in worker_threads:
            work_queue.put(_SENTINEL)
        for t in worker_threads:
            t.join(timeout=5)


def scan_objects(ctx: Context, action: StepSpec) -> None:
    """Scan bucket objects into the DuckDB database for subsequent querying.

    Uses adaptive prefix splitting: each prefix is optimistically scanned with
    a single page. If the page is full (prefix is large), the prefix is split
    into sub-prefixes via a delimiter listing and those are enqueued for
    parallel scanning.  This avoids single-thread bottlenecks on giant flat
    extractions like SpatialVID/.
    """
    fingerprint = PLAN_FINGERPRINT
    if not ctx.force and marker_matches(action.action_id, fingerprint):
        print_summary(f"skip {action.action_id}: marker is current")
        return

    sc_id_map = storage_class_id_map()
    remote_summary: dict[str, Any] = {"project": resolved_project(ctx), "regions": {}}

    db_conn = get_db()
    for plan_row in plan_rows():
        region = plan_row["region"]
        bucket_name = plan_row["bucket"]

        # Discover top-level prefixes
        print_summary(f"{action.action_id}: discovering top-level prefixes in {bucket_name}")
        client = storage_client(ctx)
        iterator = client.list_blobs(bucket_name, delimiter="/", fields="items(name),prefixes,nextPageToken")
        top_level_prefixes: list[str] = []
        has_root_objects = False
        for page in iterator.pages:
            top_level_prefixes.extend(page.prefixes)
            if not has_root_objects:
                for _ in page:
                    has_root_objects = True
                    break

        initial_prefixes: list[tuple[str, int]] = [(p, 0) for p in sorted(set(top_level_prefixes))]
        if has_root_objects:
            initial_prefixes = [("", 0), *initial_prefixes]

        # Load already-scanned prefixes for resume support
        already_scanned: set[str] = set()
        if not ctx.force:
            already_scanned = {
                r["prefix"]
                for r in _fetchall_dicts(
                    db_conn.execute("SELECT prefix FROM scanned_prefixes WHERE bucket = ?", (bucket_name,))
                )
            }

        pending: list[tuple[str, int]] = [(p, d) for p, d in initial_prefixes if p not in already_scanned]
        if len(pending) < len(initial_prefixes):
            skipped = len(initial_prefixes) - len(pending)
            print_summary(f"{action.action_id}: {bucket_name}: {skipped} prefixes already scanned, skipping")

        if not pending:
            total_row = _fetchone_dict(
                db_conn.execute(
                    "SELECT COUNT(*) as total FROM objects WHERE bucket = ?",
                    (bucket_name,),
                )
            )
            total_objects = int(total_row["total"])
            remote_summary["regions"][region] = {
                "bucket": bucket_name,
                "prefixes_scanned": len(already_scanned),
                "total_objects": total_objects,
            }
            print_summary(f"{region}: {total_objects} objects across {len(already_scanned)} prefixes (all cached)")
            continue

        print_summary(
            f"{action.action_id}: scanning {bucket_name}: {len(pending)} initial prefixes "
            f"with {ctx.scan_workers} workers (adaptive splitting, max depth {ADAPTIVE_SCAN_MAX_DEPTH})"
        )

        workers = max(1, ctx.scan_workers)
        progress = ScanProgress(bucket_name=bucket_name, num_workers=workers)
        progress.set_total(len(pending))
        progress.start()
        buf = ScanBuffer(objects=ObjectBuffer(OBJECTS_PARQUET_DIR, db_conn))

        try:
            _run_adaptive_scan(ctx, bucket_name, sc_id_map, pending, workers, already_scanned, db_conn, progress, buf)
        finally:
            progress.stop()

        if progress.prefixes_expanded:
            n = progress.prefixes_expanded
            print_summary(f"{action.action_id}: {bucket_name}: {n} prefixes expanded via adaptive splitting")

        total_row = _fetchone_dict(
            db_conn.execute(
                "SELECT COUNT(*) as total FROM objects WHERE bucket = ?",
                (bucket_name,),
            )
        )
        grand_total = int(total_row["total"])

        prefix_count = _fetchone_dict(
            db_conn.execute(
                "SELECT COUNT(*) as cnt FROM scanned_prefixes WHERE bucket = ?",
                (bucket_name,),
            )
        )

        remote_summary["regions"][region] = {
            "bucket": bucket_name,
            "prefixes_scanned": int(prefix_count["cnt"]),
            "total_objects": grand_total,
        }
        print_summary(
            f"{region}: scanned {progress.total_objects} new objects, {grand_total} total "
            f"across {int(prefix_count['cnt'])} prefixes"
        )

    write_marker(action.action_id, fingerprint, dry_run=ctx.dry_run, extra={"remote_summary": remote_summary})


# ===========================================================================
# PREP: estimate deletion savings (SQL-based)
# ===========================================================================


def estimate_savings(ctx: Context, action: StepSpec) -> None:
    """Estimate deletion savings using SQL queries against the scanned object catalog."""
    fingerprint = hashlib.sha256((PLAN_FINGERPRINT + "estimate").encode()).hexdigest()
    if not ctx.force and marker_matches(action.action_id, fingerprint):
        print_summary(f"skip {action.action_id}: marker is current")
        return

    summary_rows: list[dict[str, str]] = []
    remote_summary: dict[str, Any] = {"project": resolved_project(ctx), "regions": {}}
    conn = get_db()

    for plan_row in plan_rows():
        region = plan_row["region"]
        bucket_name = plan_row["bucket"]
        continent = continent_for_region(region)
        price_column = "price_per_gib_month_us" if continent == "US" else "price_per_gib_month_eu"

        totals = _fetchone_dict(
            conn.execute(
                "SELECT COUNT(*) as cnt FROM objects WHERE bucket = ?",
                (bucket_name,),
            )
        )
        total_objects = int(totals["cnt"])

        delete_rows = _fetchall_dicts(
            conn.execute(
                f"""
            SELECT sc.name as storage_class,
                   COUNT(*) as cnt,
                   COALESCE(SUM(o.size_bytes), 0) as total_bytes,
                   COALESCE(SUM(o.size_bytes), 0) / (1024.0*1024.0*1024.0) * sc.{price_column} * ? as monthly_cost
            FROM objects o
            JOIN storage_classes sc ON o.storage_class_id = sc.id
            WHERE o.bucket = ?
              AND sc.name != 'STANDARD'
              AND NOT EXISTS ({IS_PROTECTED})
            GROUP BY sc.name
            """,
                (1.0 - GCS_DISCOUNT, bucket_name),
            )
        )

        protect_rows = _fetchall_dicts(
            conn.execute(
                f"""
            SELECT sc.name as storage_class,
                   COUNT(*) as cnt,
                   COALESCE(SUM(o.size_bytes), 0) as total_bytes
            FROM objects o
            JOIN storage_classes sc ON o.storage_class_id = sc.id
            WHERE o.bucket = ?
              AND (sc.name = 'STANDARD'
                   OR EXISTS ({IS_PROTECTED}))
            GROUP BY sc.name
            """,
                (bucket_name,),
            )
        )

        delete_bytes_by_class: dict[str, int] = {}
        delete_count_by_class: dict[str, int] = {}
        monthly_savings = 0.0
        for row in delete_rows:
            sc = row["storage_class"]
            delete_bytes_by_class[sc] = int(row["total_bytes"])
            delete_count_by_class[sc] = int(row["cnt"])
            monthly_savings += float(row["monthly_cost"])

        protect_bytes_by_class: dict[str, int] = {}
        protect_count_by_class: dict[str, int] = {}
        for row in protect_rows:
            sc = row["storage_class"]
            protect_bytes_by_class[sc] = int(row["total_bytes"])
            protect_count_by_class[sc] = int(row["cnt"])

        delete_total_bytes = sum(delete_bytes_by_class.values())
        delete_total_count = sum(delete_count_by_class.values())

        summary_rows.append(
            {
                "region": region,
                "bucket": bucket_name,
                "total_objects": str(total_objects),
                "delete_objects": str(delete_total_count),
                "delete_bytes": str(delete_total_bytes),
                "delete_human_bytes": human_bytes(delete_total_bytes),
                "estimated_monthly_savings_usd": f"{monthly_savings:.2f}",
            }
        )
        remote_summary["regions"][region] = {
            "bucket": bucket_name,
            "total_objects": total_objects,
            "delete_objects": delete_total_count,
            "delete_bytes": delete_total_bytes,
            "delete_human_bytes": human_bytes(delete_total_bytes),
            "delete_bytes_by_class": delete_bytes_by_class,
            "estimated_monthly_savings_usd": round(monthly_savings, 2),
        }
        print_summary(
            f"{region}: will delete {human_bytes(delete_total_bytes)} ({delete_total_count} objects) "
            f"~${monthly_savings:,.2f}/mo savings"
        )
        for sc in sorted(delete_bytes_by_class):
            print_summary(
                f"  {sc:>12}: {human_bytes(delete_bytes_by_class[sc]):>12}  " f"{delete_count_by_class[sc]:>8} objects"
            )

    total_monthly = sum(float(row["estimated_monthly_savings_usd"]) for row in summary_rows)
    total_annual = total_monthly * 12
    total_delete_bytes = sum(int(row["delete_bytes"]) for row in summary_rows)
    print_summary(f"{action.action_id}: estimated deletion savings for {len(summary_rows)} regions")
    print_summary(
        f"  total deletable: {human_bytes(total_delete_bytes)} — "
        f"~${total_monthly:,.2f}/mo, ~${total_annual:,.2f}/yr savings (after 50% discount)"
    )
    write_marker(action.action_id, fingerprint, dry_run=ctx.dry_run, extra={"remote_summary": remote_summary})


# ===========================================================================
# CLEANUP steps
# ===========================================================================


def enable_soft_delete(ctx: Context, action: StepSpec) -> None:
    fingerprint = PLAN_FINGERPRINT
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows():
        region = row["region"]
        bucket_url = f"gs://{row['bucket']}"
        metadata = gcloud_bucket_describe(ctx, bucket_url)
        current_seconds = bucket_soft_delete_seconds(metadata)
        if current_seconds >= SOFT_DELETE_RETENTION_SECONDS and not ctx.force:
            print_summary(
                f"{action.action_id}: {bucket_url} already has soft-delete >= {SOFT_DELETE_RETENTION_SECONDS}s"
            )
            remote_summary["regions"][region] = {"soft_delete_seconds": current_seconds, "action": "already_enabled"}
            continue
        print_summary(
            f"{action.action_id}: {'would enable' if ctx.dry_run else 'enabling'} "
            f"soft-delete ({SOFT_DELETE_RETENTION_SECONDS}s) on {bucket_url}"
        )
        if not ctx.dry_run:
            run_subprocess(
                ctx,
                [
                    "gcloud",
                    "storage",
                    "buckets",
                    "update",
                    bucket_url,
                    f"--soft-delete-duration={SOFT_DELETE_RETENTION_SECONDS}s",
                ],
            )
            after = gcloud_bucket_describe(ctx, bucket_url)
            after_seconds = bucket_soft_delete_seconds(after)
            if after_seconds < SOFT_DELETE_RETENTION_SECONDS:
                raise RuntimeError(
                    f"Soft-delete on {bucket_url} is {after_seconds}s, expected >= {SOFT_DELETE_RETENTION_SECONDS}s"
                )
            remote_summary["regions"][region] = {"soft_delete_seconds": after_seconds, "action": "enabled"}
        else:
            remote_summary["regions"][region] = {"action": "dry_run"}
    if not ctx.dry_run:
        write_marker(action.action_id, fingerprint, dry_run=ctx.dry_run, extra={"remote_summary": remote_summary})


def _delete_prefix_objects(
    ctx: Context,
    bucket_name: str,
    prefix: str,
) -> tuple[int, int, dict[str, int]]:
    """Delete cold unprotected objects under a single prefix. Returns (count, bytes, by_class)."""
    with _db_lock:
        rows = _fetchall_dicts(
            get_db().execute(
                f"""
            SELECT o.name, o.size_bytes, sc.name as storage_class
            FROM objects o
            JOIN storage_classes sc ON o.storage_class_id = sc.id
            WHERE o.bucket = ?
              AND o.name LIKE ? || '%'
              AND sc.name != 'STANDARD'
              AND NOT EXISTS ({IS_PROTECTED})
            ORDER BY o.name
            """,
                (bucket_name, prefix),
            )
        )

    if not rows:
        return 0, 0, {}

    client = storage_client(ctx)
    bucket_obj = client.bucket(bucket_name)
    deleted_count = 0
    deleted_bytes = 0
    deleted_by_class: dict[str, int] = defaultdict(int)
    batch: list[storage.Blob] = []

    for row in rows:
        deleted_count += 1
        deleted_bytes += int(row["size_bytes"])
        deleted_by_class[row["storage_class"]] += 1
        batch.append(bucket_obj.blob(row["name"]))

        if len(batch) >= DELETE_BATCH_SIZE:
            if not ctx.dry_run:
                with client.batch():
                    for b in batch:
                        b.delete()
            batch.clear()

    if batch and not ctx.dry_run:
        with client.batch():
            for b in batch:
                b.delete()

    return deleted_count, deleted_bytes, dict(deleted_by_class)


def delete_cold_unprotected(ctx: Context, action: StepSpec) -> None:
    """Delete non-STANDARD objects outside the protect set, driven by the objects DB."""
    fingerprint = PLAN_FINGERPRINT
    if not ctx.force and marker_matches(action.action_id, fingerprint):
        print_summary(f"skip {action.action_id}: already completed")
        return

    remote_summary: dict[str, Any] = {"regions": {}}
    conn = get_db()

    for row in plan_rows():
        region = row["region"]
        bucket_name = row["bucket"]

        prefix_rows = _fetchall_dicts(
            conn.execute(
                "SELECT prefix FROM scanned_prefixes WHERE bucket = ? ORDER BY prefix",
                (bucket_name,),
            )
        )
        prefixes = [r["prefix"] for r in prefix_rows]

        print_summary(
            f"{action.action_id}: deleting cold unprotected objects from {bucket_name} "
            f"({len(prefixes)} prefixes, {ctx.scan_workers} workers)"
        )

        total_deleted_count = 0
        total_deleted_bytes = 0
        total_deleted_by_class: dict[str, int] = defaultdict(int)

        standard_row = _fetchone_dict(
            conn.execute(
                """
            SELECT COUNT(*) as cnt FROM objects o
            JOIN storage_classes sc ON o.storage_class_id = sc.id
            WHERE o.bucket = ? AND sc.name = 'STANDARD'
            """,
                (bucket_name,),
            )
        )
        total_skipped_standard = int(standard_row["cnt"])

        protected_row = _fetchone_dict(
            conn.execute(
                f"""
            SELECT COUNT(*) as cnt FROM objects o
            JOIN storage_classes sc ON o.storage_class_id = sc.id
            WHERE o.bucket = ?
              AND sc.name != 'STANDARD'
              AND EXISTS ({IS_PROTECTED})
            """,
                (bucket_name,),
            )
        )
        total_skipped_protected = int(protected_row["cnt"])

        progress = tqdm(total=len(prefixes), desc=f"delete {bucket_name}", unit="prefix", leave=True)
        workers = max(1, min(ctx.scan_workers, len(prefixes) or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_prefix = {executor.submit(_delete_prefix_objects, ctx, bucket_name, p): p for p in prefixes}
            for future in as_completed(future_to_prefix):
                p = future_to_prefix[future]
                count, nbytes, by_class = future.result()
                total_deleted_count += count
                total_deleted_bytes += nbytes
                for sc, c in by_class.items():
                    total_deleted_by_class[sc] += c
                progress.set_postfix_str(f"{p[:50]} ({count} del)")
                progress.update(1)
        progress.close()

        remote_summary["regions"][region] = {
            "deleted_count": total_deleted_count,
            "deleted_bytes": total_deleted_bytes,
            "skipped_protected": total_skipped_protected,
            "skipped_standard": total_skipped_standard,
        }
        verb = "would delete" if ctx.dry_run else "deleted"
        print_summary(
            f"{region}: {verb} {total_deleted_count} objects ({human_bytes(total_deleted_bytes)}), "
            f"skipped {total_skipped_protected} protected + {total_skipped_standard} STANDARD"
        )
    if not ctx.dry_run:
        write_marker(action.action_id, fingerprint, dry_run=ctx.dry_run, extra={"remote_summary": remote_summary})


def wait_for_soft_delete_window(ctx: Context, action: StepSpec) -> None:
    if not marker_exists("cleanup.delete_cold_objects"):
        raise RuntimeError("cleanup.wait_for_safety_window requires cleanup.delete_cold_objects to be complete first")
    fingerprint = PLAN_FINGERPRINT
    settle_deadline = now_utc() + timedelta(hours=ctx.settle_hours)

    existing_extra = read_marker_extra(action.action_id)
    if existing_extra is not None and not ctx.force:
        existing_deadline = datetime.fromisoformat(existing_extra["settle_deadline"])
        if now_utc() < existing_deadline:
            remaining = existing_deadline - now_utc()
            raise RuntimeError(
                f"Soft-delete safety window still open until {existing_deadline.isoformat()} "
                f"({remaining} remaining). Rerun after the deadline or use --force to override."
            )
        print_summary(f"{action.action_id}: soft-delete safety window already elapsed")
        return

    print_summary(
        f"{action.action_id}: recording safety window for {ctx.settle_hours} hours "
        f"(until {settle_deadline.isoformat()})"
    )
    write_marker(
        action.action_id,
        fingerprint,
        dry_run=ctx.dry_run,
        extra={"settle_deadline": settle_deadline.isoformat()},
    )


def disable_soft_delete(ctx: Context, action: StepSpec) -> None:
    fingerprint = PLAN_FINGERPRINT
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows():
        region = row["region"]
        bucket_url = f"gs://{row['bucket']}"
        metadata = gcloud_bucket_describe(ctx, bucket_url)
        current_seconds = bucket_soft_delete_seconds(metadata)
        if current_seconds == 0 and not ctx.force:
            print_summary(f"{action.action_id}: {bucket_url} already has soft-delete disabled")
            remote_summary["regions"][region] = {"soft_delete_seconds": 0, "action": "already_disabled"}
            continue
        print_summary(
            f"{action.action_id}: {'would disable' if ctx.dry_run else 'disabling'} soft-delete on {bucket_url}"
        )
        if not ctx.dry_run:
            run_subprocess(ctx, ["gcloud", "storage", "buckets", "update", bucket_url, "--clear-soft-delete"])
            after = gcloud_bucket_describe(ctx, bucket_url)
            after_seconds = bucket_soft_delete_seconds(after)
            if after_seconds != 0:
                raise RuntimeError(f"Soft-delete on {bucket_url} is still {after_seconds}s after clear")
            remote_summary["regions"][region] = {"soft_delete_seconds": 0, "action": "disabled"}
        else:
            remote_summary["regions"][region] = {"action": "dry_run"}
    if not ctx.dry_run:
        write_marker(action.action_id, fingerprint, dry_run=ctx.dry_run, extra={"remote_summary": remote_summary})


# ===========================================================================
# Step registry
# ===========================================================================

STEPS: list[StepSpec] = [
    StepSpec(
        action_id="prep.load_protect_rules",
        group_name="prep",
        command_name="load-protect-rules",
        description="Load protect globs and direct prefixes into the protect_rules DB table.",
        help_text=(
            "Reads protect_prefixes_classified.csv and protect_prefixes_direct.csv, converts wildcard "
            "globs to SQL LIKE patterns, and inserts all rules into the protect_rules table. "
            "No GCS listing calls needed — protection is resolved against scanned objects via SQL."
        ),
        mutating=False,
        runner=load_protect_rules,
    ),
    StepSpec(
        action_id="prep.scan_objects",
        group_name="prep",
        command_name="scan-objects",
        description="Scan bucket objects into the DuckDB catalog.",
        help_text=(
            "List every object in each bucket and insert into the local DuckDB database. "
            "Fans out over top-level prefixes with `--scan-workers` concurrent threads. "
            "Skips already-scanned prefixes unless `--force` is given."
        ),
        mutating=False,
        runner=scan_objects,
        predecessors=("prep.load_protect_rules",),
        scan_workers=True,
    ),
    StepSpec(
        action_id="prep.estimate_savings",
        group_name="prep",
        command_name="estimate-savings",
        description="Estimate deletion savings via SQL queries against the object catalog.",
        help_text=(
            "Query the DuckDB object catalog to classify every object as protected/deletable by storage class, "
            "and compute the monthly cost savings from deletion. Prints per-region summaries."
        ),
        mutating=False,
        runner=estimate_savings,
        predecessors=("prep.scan_objects",),
    ),
    StepSpec(
        action_id="cleanup.enable_soft_delete",
        group_name="cleanup",
        command_name="enable-soft-delete",
        description="Enable 3-day soft-delete retention on source buckets.",
        help_text=(
            "Enable soft-delete with a 3-day retention window on each source bucket. This ensures "
            "that any objects deleted in the next step can be restored if something goes wrong."
        ),
        mutating=True,
        runner=enable_soft_delete,
        predecessors=("prep.estimate_savings",),
    ),
    StepSpec(
        action_id="cleanup.delete_cold_objects",
        group_name="cleanup",
        command_name="delete-cold-objects",
        description="Delete non-STANDARD objects outside the protect set.",
        help_text=(
            "Query the DuckDB catalog for cold unprotected objects and batch-delete them from GCS. "
            "Fans out over top-level prefixes with workers for throughput. "
            "Soft-delete must be enabled first so deletions are recoverable during the safety window."
        ),
        mutating=True,
        runner=delete_cold_unprotected,
        predecessors=("cleanup.enable_soft_delete",),
        scan_workers=True,
    ),
    StepSpec(
        action_id="cleanup.wait_for_safety_window",
        group_name="cleanup",
        command_name="wait-for-safety-window",
        description="Record and honor the soft-delete safety window.",
        help_text=(
            "Record the soft-delete safety window and refuse to disable soft-delete until that window "
            "has elapsed. This is a checkpoint, not a sleep. Use `--settle-hours` to adjust the window."
        ),
        mutating=False,
        runner=wait_for_soft_delete_window,
        predecessors=("cleanup.delete_cold_objects",),
        settle_hours=True,
    ),
    StepSpec(
        action_id="cleanup.disable_soft_delete",
        group_name="cleanup",
        command_name="disable-soft-delete",
        description="Disable soft-delete after the safety window has elapsed.",
        help_text=(
            "Disable soft-delete on each source bucket, permanently removing the soft-deleted objects. "
            "Only run this after the safety window has elapsed and you have confirmed no important "
            "data was accidentally deleted."
        ),
        mutating=True,
        runner=disable_soft_delete,
        predecessors=("cleanup.wait_for_safety_window",),
        optional=True,
    ),
]

STEP_INDEX = {step.action_id: step for step in STEPS}

# ===========================================================================
# CLI
# ===========================================================================


def selected_steps(
    *,
    from_action: str | None,
    to_action: str | None,
    only_action: str | None,
    include_optional: bool,
) -> list[StepSpec]:
    if only_action is not None:
        return [STEP_INDEX[only_action]]
    start_index = 0 if from_action is None else next(i for i, s in enumerate(STEPS) if s.action_id == from_action)
    end_index = len(STEPS) - 1 if to_action is None else next(i for i, s in enumerate(STEPS) if s.action_id == to_action)
    steps = STEPS[start_index : end_index + 1]
    if include_optional:
        return steps
    if to_action is not None and STEP_INDEX[to_action].optional:
        return steps
    return [s for s in steps if not s.optional]


def assert_step_predecessors(ctx: Context, step: StepSpec) -> None:
    if ctx.force:
        return
    missing = [p for p in step.predecessors if not marker_exists(p)]
    if missing:
        raise RuntimeError(
            f"{step.action_id} requires these predecessor steps to be complete first: {', '.join(missing)}"
        )


def build_context(
    *,
    dry_run: bool,
    force: bool,
    include_optional: bool,
    scan_workers: int,
    settle_hours: int,
    log_prefix: str,
    project: str | None,
) -> Context:
    ensure_output_dirs()
    return Context(
        dry_run=dry_run,
        force=force,
        include_optional=include_optional,
        scan_workers=scan_workers,
        settle_hours=settle_hours,
        log_path=LOG_DIR / f"{log_prefix}_{timestamp_string()}.log",
        timestamp=timestamp_string(),
        project=project,
    )


def runtime_options(
    *,
    scan_workers: bool = False,
    settle_hours: bool = False,
    include_optional: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        options: list[Callable[[Callable[..., Any]], Callable[..., Any]]] = [
            click.option("--force", is_flag=True, help="Ignore cached markers and recompute."),
            click.option("--dry-run", is_flag=True, help="Read-only mode: inspect but never mutate remote state."),
            click.option("--project", help="Override the GCP project for Cloud Storage API calls."),
        ]
        if include_optional:
            options.append(click.option("--include-optional", is_flag=True, help="Include optional cleanup steps."))
        if scan_workers:
            options.append(
                click.option(
                    "--scan-workers", default=64, show_default=True, type=int, help="Concurrent scan/delete workers."
                )
            )
        if settle_hours:
            options.append(
                click.option(
                    "--settle-hours", default=72, show_default=True, type=int, help="Soft-delete safety window (hours)."
                )
            )
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Storage cleanup workflow: delete cold unprotected objects from Marin buckets "
        "with a soft-delete safety net. Run `plan` to see step status, or use the "
        "prep/cleanup subcommands to execute individual steps."
    ),
)
def cli() -> None:
    pass


@cli.command("plan", help="Show ordered step list and completion status.")
def plan_cli() -> None:
    ensure_output_dirs()
    print("Ordered storage cleanup steps:")
    for step in STEPS:
        suffix = " [optional]" if step.optional else ""
        status = "done" if marker_exists(step.action_id) else "pending"
        print(f"  {status:7}  {step.action_id}{suffix}")


@cli.command(
    "run",
    help="Execute a contiguous slice of the ordered workflow.",
)
@runtime_options(scan_workers=True, settle_hours=True, include_optional=True)
@click.option("--from", "from_action", type=click.Choice(sorted(STEP_INDEX)), help="Start from this step.")
@click.option("--to", "to_action", type=click.Choice(sorted(STEP_INDEX)), help="Stop after this step.")
@click.option("--only", "only_action", type=click.Choice(sorted(STEP_INDEX)), help="Run exactly one step.")
def run_cli(
    dry_run: bool,
    force: bool,
    include_optional: bool,
    scan_workers: int,
    settle_hours: int,
    project: str | None,
    from_action: str | None,
    to_action: str | None,
    only_action: str | None,
) -> None:
    ctx = build_context(
        dry_run=dry_run,
        force=force,
        include_optional=include_optional,
        scan_workers=scan_workers,
        settle_hours=settle_hours,
        log_prefix="run",
        project=project,
    )
    steps = selected_steps(
        from_action=from_action,
        to_action=to_action,
        only_action=only_action,
        include_optional=include_optional,
    )
    print_summary(f"running {len(steps)} steps; log: {ctx.log_path.relative_to(REPO_ROOT)}")
    for step in steps:
        print_summary(f"==> {step.action_id}: {step.description}")
        assert_step_predecessors(ctx, step)
        step.run(ctx)
    print_summary("completed selected steps")


@cli.group(help="Preparation commands: resolve protect set, scan objects, estimate savings. Read-only against buckets.")
def prep() -> None:
    pass


@cli.group(help="Cleanup commands: enable soft-delete, delete cold objects, finalize.")
def cleanup() -> None:
    pass


GROUPS: dict[str, click.Group] = {"prep": prep, "cleanup": cleanup}


def register_step_command(group: click.Group, step: StepSpec) -> None:
    @runtime_options(
        scan_workers=step.scan_workers,
        settle_hours=step.settle_hours,
    )
    def command(
        dry_run: bool,
        force: bool,
        project: str | None,
        scan_workers: int = 64,
        settle_hours: int = 72,
    ) -> None:
        ctx = build_context(
            dry_run=dry_run,
            force=force,
            include_optional=False,
            scan_workers=scan_workers,
            settle_hours=settle_hours,
            log_prefix=step.action_id.replace(".", "__"),
            project=project,
        )
        print_summary(f"running 1 step; log: {ctx.log_path.relative_to(REPO_ROOT)}")
        print_summary(f"==> {step.action_id}: {step.description}")
        assert_step_predecessors(ctx, step)
        step.run(ctx)
        print_summary("completed selected steps")

    command.__name__ = step.action_id.replace(".", "_").replace("-", "_")
    group.command(name=step.command_name, help=step.help_text)(command)


for _step in STEPS:
    register_step_command(GROUPS[_step.group_name], _step)


if __name__ == "__main__":
    cli()
