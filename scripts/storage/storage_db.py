# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared DuckDB storage catalog: schemas, init, queries, and helpers.

Used by both the purge CLI and the delete-o-tron dashboard server.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

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

METADATA_FLUSH_THRESHOLD = 500


# ---------------------------------------------------------------------------
# Data classes
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
# Module-level DuckDB connection singleton
# ---------------------------------------------------------------------------

_db: duckdb.DuckDBPyConnection | None = None
_db_lock = threading.Lock()

SCHEMA_VERSION = 11  # protect_rules.id uses GENERATED ALWAYS AS IDENTITY


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
    Append via `add`, call `flush(force=True)` when done scanning.
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
        print_summary(f"schema upgrade {current_version} → {SCHEMA_VERSION}")
        if current_version < 11:
            # v11: protect_rules.id was INTEGER PRIMARY KEY (no auto-increment in DuckDB).
            # Add a sequence and set it as the default so INSERTs without explicit id work.
            conn.execute("CREATE SEQUENCE IF NOT EXISTS protect_rules_id_seq START 1")
            try:
                conn.execute("ALTER TABLE protect_rules ALTER COLUMN id SET DEFAULT nextval('protect_rules_id_seq')")
                # Advance the sequence past any existing rows
                row = _fetchone_dict(conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM protect_rules"))
                if row and row["max_id"] > 0:
                    conn.execute(f"SELECT setval('protect_rules_id_seq', {row['max_id'] + 1})")
            except duckdb.CatalogException:
                pass  # table doesn't exist yet, CREATE TABLE below will handle it
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
    conn.execute("CREATE SEQUENCE IF NOT EXISTS protect_rules_id_seq START 1")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS protect_rules (
            id INTEGER PRIMARY KEY DEFAULT nextval('protect_rules_id_seq'),
            bucket TEXT NOT NULL,
            pattern TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            owners TEXT,
            reasons TEXT,
            sources TEXT,
            UNIQUE (bucket, pattern)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rule_costs (
            rule_id INTEGER NOT NULL REFERENCES protect_rules(id),
            storage_class_id INTEGER NOT NULL REFERENCES storage_classes(id),
            object_count INTEGER NOT NULL,
            total_bytes BIGINT NOT NULL,
            monthly_cost_usd REAL NOT NULL,
            PRIMARY KEY (rule_id, storage_class_id)
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
# SQL fragments
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def timestamp_string() -> str:
    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def print_summary(message: str) -> None:
    print(message)


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


def region_from_bucket(bucket: str) -> str:
    return bucket.removeprefix("marin-") if bucket.startswith("marin-") else bucket


def region_bucket(region: str) -> str:
    return f"marin-{region}"


def all_regions() -> list[str]:
    return [region_from_bucket(bucket) for bucket in MARIN_BUCKETS]


def plan_rows() -> list[dict[str, str]]:
    """Return the cleanup plan: one row per bucket with region and location."""
    return [{"region": region_from_bucket(b), "bucket": b, "location": BUCKET_LOCATIONS[b]} for b in MARIN_BUCKETS]


def glob_to_like(gs_glob: str) -> str:
    """Convert a GCS glob pattern to a SQL LIKE pattern (e.g. 'foo/bar*' → 'foo/bar%')."""
    return gs_glob.replace("*", "%").replace("?", "_")


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
# Prefix utilities
# ---------------------------------------------------------------------------


def url_bucket(url: str) -> str:
    return url.removeprefix("gs://").split("/", 1)[0]


def url_object_path(url: str) -> str:
    parts = url.removeprefix("gs://").split("/", 1)
    return parts[1] if len(parts) == 2 else ""


def normalized_prefix_url(url: str) -> str:
    return url if url.endswith("/") else f"{url}/"


def normalize_relative_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


def relative_prefix(prefix_url: str, bucket: str) -> str:
    prefix = url_object_path(prefix_url)
    if prefix.endswith("/"):
        return prefix
    return f"{prefix}/" if prefix else ""


# ---------------------------------------------------------------------------
# Scan buffer helpers
# ---------------------------------------------------------------------------


@dataclass
class ScanBuffer:
    objects: ObjectBuffer
    prefixes: list[tuple[str, str, int, str]] = field(default_factory=list)
    split_cache: list[tuple[str, str, str]] = field(default_factory=list)


def buffer_prefix_scanned(buf: ScanBuffer, bucket_name: str, prefix: str, object_count: int) -> None:
    buf.prefixes.append((bucket_name, prefix, object_count, now_utc().isoformat()))


def split_cache_key(bucket_name: str, prefix: str) -> str:
    return f"scan://{bucket_name}/{prefix}"


def buffer_split_cache(
    buf: ScanBuffer,
    cache: dict[str, list[str]],
    bucket_name: str,
    prefix: str,
    children: list[str],
) -> None:
    key = split_cache_key(bucket_name, prefix)
    cache[key] = children
    buf.split_cache.append((key, json.dumps(children), now_utc().isoformat()))


def flush_metadata(conn: duckdb.DuckDBPyConnection, buf: ScanBuffer, *, force: bool = False) -> None:
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


def load_split_cache(conn: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """Load the split_cache table into memory for fast lookups."""
    rows = conn.execute("SELECT cache_key, entries_json FROM split_cache").fetchall()
    cache: dict[str, list[str]] = {}
    for cache_key, entries_json in rows:
        entries = json.loads(entries_json)
        if isinstance(entries, list):
            cache[cache_key] = entries
    return cache


def read_split_cache(cache: dict[str, list[str]], bucket_name: str, prefix: str) -> list[str] | None:
    return cache.get(split_cache_key(bucket_name, prefix))


# ---------------------------------------------------------------------------
# Rule cost materialization
# ---------------------------------------------------------------------------


def materialize_rule_costs(conn: duckdb.DuckDBPyConnection) -> int:
    """Compute and store per-rule costs in the rule_costs table. Returns number of rows inserted."""
    conn.execute("DELETE FROM rule_costs")
    total_inserted = 0

    for plan_row in plan_rows():
        bucket_name = plan_row["bucket"]
        region = plan_row["region"]
        continent = continent_for_region(region)
        price_column = "price_per_gib_month_us" if continent == "US" else "price_per_gib_month_eu"

        conn.execute(
            f"""
            INSERT INTO rule_costs (rule_id, storage_class_id, object_count, total_bytes, monthly_cost_usd)
            SELECT r.id,
                   sc.id,
                   COUNT(*) as object_count,
                   COALESCE(SUM(o.size_bytes), 0) as total_bytes,
                   COALESCE(SUM(o.size_bytes), 0) / (1024.0*1024.0*1024.0)
                       * sc.{price_column} * ? as monthly_cost_usd
            FROM protect_rules r
            JOIN objects o ON o.bucket = r.bucket
                AND CASE r.pattern_type
                    WHEN 'prefix' THEN o.name LIKE r.pattern || '%'
                    WHEN 'like' THEN o.name LIKE r.pattern
                END
            JOIN storage_classes sc ON o.storage_class_id = sc.id
            WHERE r.bucket = ?
            GROUP BY r.id, sc.id
            """,
            (1.0 - GCS_DISCOUNT, bucket_name),
        )
        row = _fetchone_dict(conn.execute("SELECT changes() as cnt"))
        total_inserted += int(row["cnt"]) if row else 0

    return total_inserted


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    for path in [OUTPUT_ROOT, PROTECT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    init_db()
