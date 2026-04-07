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
from functools import cached_property
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


@dataclass(frozen=True)
class StorageCatalog:
    """Owns all workspace paths derived from a single root directory."""

    root: Path

    @cached_property
    def db_path(self) -> Path:
        return self.root / "storage.duckdb"

    @cached_property
    def objects_parquet_dir(self) -> Path:
        return self.root / "objects_parquet"

    @cached_property
    def dir_summary_parquet_dir(self) -> Path:
        return self.root / "dir_summary_parquet"

    @cached_property
    def protect_rules_json(self) -> Path:
        return self.root / "protect_rules.json"

    @cached_property
    def delete_rules_json(self) -> Path:
        return self.root / "delete_rules.json"

    @cached_property
    def deletion_manifest_csv(self) -> Path:
        return self.root / "deletion_manifest.csv"

    @cached_property
    def protect_dir(self) -> Path:
        return self.root / "protect"

    @cached_property
    def log_dir(self) -> Path:
        return self.root / "logs"

    def open_db(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def ensure_dirs(self) -> None:
        for d in [self.root, self.objects_parquet_dir, self.dir_summary_parquet_dir, self.protect_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)


DEFAULT_CATALOG = StorageCatalog(STORAGE_DIR / "purge")

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

# dir_summary collapse thresholds: dirs deeper than this with less than
# DIR_SUMMARY_MIN_BYTES are rolled up into their depth-2 ancestor.
DIR_SUMMARY_MIN_DEPTH = 2
DIR_SUMMARY_MIN_BYTES = 10_000_000_000  # 10 GB

GCS_MAX_PAGE_SIZE = 5000

# Partial response fields for list_blobs — only fetch what _blob_to_scanned needs.
_BLOB_FIELDS = "items(name,size,storageClass,timeCreated,updated),prefixes,nextPageToken"
ADAPTIVE_SPLIT_THRESHOLD = 10000  # scan flat if <= this many objects; split otherwise
ADAPTIVE_SCAN_MAX_DEPTH = 2

GCS_DISCOUNT = 0.30

OBJECT_FLUSH_THRESHOLD = 10_000_000
DELETE_BATCH_SIZE = 1000

METADATA_FLUSH_THRESHOLD = 500


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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
    conn: duckdb.DuckDBPyConnection
    db_lock: threading.Lock
    dry_run: bool
    force: bool
    include_optional: bool
    scan_workers: int
    settle_hours: int
    log_path: Path
    timestamp: str
    project: str | None


SCHEMA_VERSION = 17

_ARROW_TO_DUCKDB: dict[pa.DataType, str] = {
    pa.string(): "VARCHAR",
    pa.int64(): "BIGINT",
    pa.int32(): "INTEGER",
    pa.timestamp("us", tz="UTC"): "TIMESTAMPTZ",
}


def _fetchone_dict(result: duckdb.DuckDBPyConnection) -> dict[str, Any] | None:
    row = result.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in result.description], row, strict=False))


def _fetchall_dicts(result: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row, strict=False)) for row in result.fetchall()]


# ---------------------------------------------------------------------------
# Arrow schemas
# ---------------------------------------------------------------------------

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

_DIR_SUMMARY_ARROW_SCHEMA = pa.schema(
    [
        ("bucket", pa.string()),
        ("dir_prefix", pa.string()),
        ("standard_count", pa.int32()),
        ("standard_bytes", pa.int64()),
        ("nearline_count", pa.int32()),
        ("nearline_bytes", pa.int64()),
        ("coldline_count", pa.int32()),
        ("coldline_bytes", pa.int64()),
        ("archive_count", pa.int32()),
        ("archive_bytes", pa.int64()),
    ]
)

_DIR_SUMMARY_ARROW_TO_DUCKDB: dict[pa.DataType, str] = {
    pa.string(): "VARCHAR",
    pa.int64(): "BIGINT",
    pa.int32(): "INTEGER",
}


# ---------------------------------------------------------------------------
# ObjectBuffer
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DirSummaryBuffer
# ---------------------------------------------------------------------------


class DirSummaryBuffer:
    """Manages parquet segments for dir_summary and maintains a DuckDB view over them."""

    def __init__(self, parquet_dir: Path, conn: duckdb.DuckDBPyConnection) -> None:
        self._parquet_dir = parquet_dir
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        self._conn = conn

        existing = sorted(self._parquet_dir.glob("dir_summary_*.parquet"))
        self._segment_counter = int(existing[-1].stem.split("_")[2]) if existing else 0
        self._refresh_view()

    def _next_path(self) -> Path:
        self._segment_counter += 1
        return self._parquet_dir / f"dir_summary_{self._segment_counter:06d}.parquet"

    def _refresh_view(self) -> None:
        """Point the ``dir_summary`` view at current parquet segments."""
        has_segments = any(self._parquet_dir.glob("dir_summary_*.parquet"))
        if has_segments:
            glob = str(self._parquet_dir / "dir_summary_*.parquet")
            self._conn.execute(
                f"""
                CREATE OR REPLACE VIEW dir_summary AS
                SELECT * FROM read_parquet('{glob}', union_by_name=true, hive_partitioning=false)
            """
            )
        else:
            cols = ", ".join(
                f"NULL::{_DIR_SUMMARY_ARROW_TO_DUCKDB[f.type]} AS {f.name}" for f in _DIR_SUMMARY_ARROW_SCHEMA
            )
            self._conn.execute(f"CREATE OR REPLACE VIEW dir_summary AS SELECT {cols} WHERE false")

    def write_arrow_table(self, arrow_table: pa.Table) -> None:
        pq.write_table(arrow_table, self._next_path(), compression="zstd")
        self._refresh_view()

    def write_from_query(self, conn: duckdb.DuckDBPyConnection, bucket: str) -> None:
        """Run the dir_summary aggregation query for a single bucket and write results to parquet."""
        result = conn.execute(
            """
            SELECT o.bucket,
                   left(o.name, length(o.name) - position('/' in reverse(o.name)) + 1) as dir_prefix,
                   COUNT(*) FILTER (WHERE sc.name = 'STANDARD') as standard_count,
                   COALESCE(SUM(o.size_bytes) FILTER (WHERE sc.name = 'STANDARD'), 0) as standard_bytes,
                   COUNT(*) FILTER (WHERE sc.name = 'NEARLINE') as nearline_count,
                   COALESCE(SUM(o.size_bytes) FILTER (WHERE sc.name = 'NEARLINE'), 0) as nearline_bytes,
                   COUNT(*) FILTER (WHERE sc.name = 'COLDLINE') as coldline_count,
                   COALESCE(SUM(o.size_bytes) FILTER (WHERE sc.name = 'COLDLINE'), 0) as coldline_bytes,
                   COUNT(*) FILTER (WHERE sc.name = 'ARCHIVE') as archive_count,
                   COALESCE(SUM(o.size_bytes) FILTER (WHERE sc.name = 'ARCHIVE'), 0) as archive_bytes
            FROM objects o
            JOIN storage_classes sc ON o.storage_class_id = sc.id
            WHERE o.bucket = ?
            GROUP BY o.bucket, dir_prefix
            """,
            (bucket,),
        )
        arrow_table = result.fetch_arrow_table()
        if len(arrow_table) > 0:
            self.write_arrow_table(arrow_table)

    def reset(self) -> None:
        """Remove all parquet segments and reset counter."""
        shutil.rmtree(self._parquet_dir, ignore_errors=True)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        self._segment_counter = 0
        self._refresh_view()


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------


def init_db(catalog: StorageCatalog = DEFAULT_CATALOG) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection and ensure the schema is current."""
    print_summary(f"opening DuckDB catalog: {catalog.db_path}")
    conn = catalog.open_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    _run_migrations(conn, catalog)
    return conn


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------

# Each migration is an idempotent function that brings the DB from version N-1
# to version N.  They run in order inside a single transaction per migration.


def _migrate_to_11(conn: duckdb.DuckDBPyConnection) -> None:
    """Add auto-increment sequence to protect_rules.id."""
    conn.execute("CREATE SEQUENCE IF NOT EXISTS protect_rules_id_seq START 1")
    try:
        conn.execute("ALTER TABLE protect_rules ALTER COLUMN id SET DEFAULT nextval('protect_rules_id_seq')")
        row = _fetchone_dict(conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM protect_rules"))
        if row and row["max_id"] > 0:
            # DuckDB has no setval(); advance by calling nextval() until past max_id
            current = 0
            while current <= row["max_id"]:
                current = conn.execute("SELECT nextval('protect_rules_id_seq')").fetchone()[0]
    except duckdb.CatalogException:
        pass  # table doesn't exist yet; created by _ensure_current_schema


def _migrate_to_12(conn: duckdb.DuckDBPyConnection) -> None:
    """Recreate rule_costs with a bucket column for wildcard rule expansion."""
    try:
        conn.execute("DROP TABLE rule_costs")
    except duckdb.CatalogException:
        pass


def _migrate_to_13(conn: duckdb.DuckDBPyConnection) -> None:
    """Add delete_rules and delete_rule_costs tables."""
    conn.execute("CREATE SEQUENCE IF NOT EXISTS delete_rules_id_seq START 1")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS delete_rules (
            id INTEGER PRIMARY KEY DEFAULT nextval('delete_rules_id_seq'),
            pattern TEXT NOT NULL,
            storage_class TEXT,
            description TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS delete_rule_costs (
            rule_id INTEGER NOT NULL,
            bucket TEXT NOT NULL,
            storage_class_id INTEGER NOT NULL,
            object_count INTEGER NOT NULL,
            total_bytes BIGINT NOT NULL,
            monthly_cost_usd REAL NOT NULL,
            PRIMARY KEY (rule_id, storage_class_id, bucket)
        )
        """
    )


def _migrate_to_14(conn: duckdb.DuckDBPyConnection, catalog: StorageCatalog) -> None:
    """Move dir_summary from a DuckDB table to a parquet-backed view."""
    try:
        row_count = conn.execute("SELECT COUNT(*) FROM dir_summary").fetchone()[0]
        if row_count > 0:
            catalog.dir_summary_parquet_dir.mkdir(parents=True, exist_ok=True)
            segment_path = catalog.dir_summary_parquet_dir / "dir_summary_000001.parquet"
            conn.execute(f"COPY dir_summary TO '{segment_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            print_summary(f"  migrated {row_count} dir_summary rows to {segment_path}")
        conn.execute("DROP TABLE dir_summary")
    except duckdb.CatalogException:
        pass  # table doesn't exist (fresh DB)


# Migrations 11-13 take (conn); migration 14 also needs the catalog.
def _migrate_to_16(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop rules tables; replaced by views over JSON files."""
    for table in ("delete_rule_costs", "rule_costs", "delete_rules", "protect_rules"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    for seq in ("protect_rules_id_seq", "delete_rules_id_seq"):
        conn.execute(f"DROP SEQUENCE IF EXISTS {seq}")


def _migrate_to_17(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop materialized cost tables; costs are now computed on the fly."""
    for table in ("rule_costs", "delete_rule_costs"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")


_MIGRATIONS_SIMPLE: list[tuple[int, Callable[[duckdb.DuckDBPyConnection], None]]] = [
    (11, _migrate_to_11),
    (12, _migrate_to_12),
    (13, _migrate_to_13),
    (16, _migrate_to_16),
    (17, _migrate_to_17),
]


def _run_migrations(conn: duckdb.DuckDBPyConnection, catalog: StorageCatalog) -> None:
    """Run pending migrations, then ensure the current schema exists."""
    row = _fetchone_dict(conn.execute("SELECT value FROM cache_meta WHERE key = 'schema_version'"))
    current_version = int(row["value"]) if row else 0

    for target_version, migrate_fn in _MIGRATIONS_SIMPLE:
        if current_version >= target_version:
            continue
        print_summary(f"  migration → v{target_version}: {migrate_fn.__doc__}")
        conn.execute("BEGIN TRANSACTION")
        migrate_fn(conn)
        conn.execute(
            "INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('schema_version', ?)",
            (str(target_version),),
        )
        conn.execute("COMMIT")

    if current_version < 14:
        print_summary(f"  migration → v14: {_migrate_to_14.__doc__}")
        conn.execute("BEGIN TRANSACTION")
        _migrate_to_14(conn, catalog)
        conn.execute(
            "INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('schema_version', '14')",
        )
        conn.execute("COMMIT")
        current_version = target_version

    _ensure_current_schema(conn, catalog)
    conn.execute("VACUUM")


def _ensure_current_schema(conn: duckdb.DuckDBPyConnection, catalog: StorageCatalog) -> None:
    """Idempotent: create all tables/views/sequences for the current schema version."""
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
    # Rules are views over JSON files — the JSON is the source of truth.
    _create_rules_views(conn, catalog)
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
    ObjectBuffer(catalog.objects_parquet_dir, conn)
    # Drop leftover dir_summary table from a partially-completed v14 migration
    try:
        conn.execute("DROP TABLE dir_summary")
    except duckdb.CatalogException:
        pass
    DirSummaryBuffer(catalog.dir_summary_parquet_dir, conn)

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


def storage_class_id_map(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Return a mapping from storage class name to its DB id."""
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
    WHERE (o.bucket = r.bucket OR r.bucket = '*')
      AND o.name LIKE r.pattern
"""

IS_DELETE_TARGET = """
    SELECT 1 FROM delete_rules dr
    WHERE o.name LIKE dr.pattern
      AND (dr.storage_class IS NULL OR sc.name = dr.storage_class)
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
    """Convert a GCS glob pattern to a SQL LIKE pattern (e.g. 'foo/bar*' -> 'foo/bar%')."""
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


def materialize_dir_summary(conn: duckdb.DuckDBPyConnection, catalog: StorageCatalog = DEFAULT_CATALOG) -> int:
    """Aggregate objects into per-directory rows with per-storage-class columns.

    Directories deeper than DIR_SUMMARY_MIN_DEPTH with fewer than
    DIR_SUMMARY_MIN_BYTES total bytes are rolled up into their
    depth-MIN_DEPTH ancestor to keep the row count manageable.

    Returns total rows written.
    """
    buf = DirSummaryBuffer(catalog.dir_summary_parquet_dir, conn)
    buf.reset()

    min_depth = DIR_SUMMARY_MIN_DEPTH
    min_bytes = DIR_SUMMARY_MIN_BYTES

    for plan_row in plan_rows():
        bucket_name = plan_row["bucket"]

        print_summary(f"  materializing dir_summary for {bucket_name}...")

        # Aggregate per-directory, then collapse small deep dirs into their
        # depth-{min_depth} ancestor in a single query.
        result = conn.execute(
            f"""
            WITH raw AS (
                SELECT o.bucket,
                       left(o.name, length(o.name) - position('/' in reverse(o.name)) + 1) as dir_prefix,
                       COUNT(*) FILTER (WHERE sc.name = 'STANDARD') as standard_count,
                       COALESCE(SUM(o.size_bytes) FILTER (WHERE sc.name = 'STANDARD'), 0) as standard_bytes,
                       COUNT(*) FILTER (WHERE sc.name = 'NEARLINE') as nearline_count,
                       COALESCE(SUM(o.size_bytes) FILTER (WHERE sc.name = 'NEARLINE'), 0) as nearline_bytes,
                       COUNT(*) FILTER (WHERE sc.name = 'COLDLINE') as coldline_count,
                       COALESCE(SUM(o.size_bytes) FILTER (WHERE sc.name = 'COLDLINE'), 0) as coldline_bytes,
                       COUNT(*) FILTER (WHERE sc.name = 'ARCHIVE') as archive_count,
                       COALESCE(SUM(o.size_bytes) FILTER (WHERE sc.name = 'ARCHIVE'), 0) as archive_bytes
                FROM objects o
                JOIN storage_classes sc ON o.storage_class_id = sc.id
                WHERE o.bucket = ?
                GROUP BY o.bucket, dir_prefix
            ),
            tagged AS (
                SELECT *,
                    CASE WHEN len(string_split(dir_prefix, '/')) - 1 > {min_depth}
                              AND (standard_bytes + nearline_bytes + coldline_bytes + archive_bytes) < {min_bytes}
                         THEN array_to_string(string_split(dir_prefix, '/')[1:{min_depth + 1}], '/') || '/'
                         ELSE dir_prefix
                    END AS effective_prefix
                FROM raw
            )
            SELECT bucket, effective_prefix AS dir_prefix,
                   SUM(standard_count) as standard_count, SUM(standard_bytes) as standard_bytes,
                   SUM(nearline_count) as nearline_count, SUM(nearline_bytes) as nearline_bytes,
                   SUM(coldline_count) as coldline_count, SUM(coldline_bytes) as coldline_bytes,
                   SUM(archive_count) as archive_count, SUM(archive_bytes) as archive_bytes
            FROM tagged
            GROUP BY bucket, effective_prefix
            """,
            (bucket_name,),
        )

        arrow_table = result.fetch_arrow_table()
        if len(arrow_table) > 0:
            buf.write_arrow_table(arrow_table)

    total = _fetchone_dict(conn.execute("SELECT COUNT(*) as cnt FROM dir_summary"))
    return int(total["cnt"])


# ---------------------------------------------------------------------------
# JSON-backed rules persistence
# ---------------------------------------------------------------------------


_EMPTY_PROTECT_RULES_VIEW = (
    "CREATE OR REPLACE VIEW protect_rules AS "
    "SELECT NULL::BIGINT as id, NULL::VARCHAR as bucket, NULL::VARCHAR as pattern, "
    "NULL::VARCHAR as pattern_type, NULL::VARCHAR as owners, "
    "NULL::VARCHAR as reasons, NULL::VARCHAR as sources WHERE false"
)

_EMPTY_DELETE_RULES_VIEW = (
    "CREATE OR REPLACE VIEW delete_rules AS "
    "SELECT NULL::BIGINT as id, NULL::VARCHAR as pattern, NULL::VARCHAR as storage_class, "
    "NULL::VARCHAR as description, NULL::VARCHAR as created_at WHERE false"
)


def _json_file_has_rows(path: Path) -> bool:
    """Return True if path exists and contains a non-empty JSON array."""
    if not path.exists():
        return False
    text = path.read_text().strip()
    return text not in ("", "[]")


def _create_rules_views(conn: duckdb.DuckDBPyConnection, catalog: StorageCatalog) -> None:
    """Create (or replace) views over the JSON rule files.

    If a JSON file doesn't exist or is empty, creates an empty view with the
    right column schema.  DuckDB's read_json_auto cannot infer columns from an
    empty array, so we must handle that case explicitly.
    """
    if _json_file_has_rows(catalog.protect_rules_json):
        conn.execute(
            f"CREATE OR REPLACE VIEW protect_rules AS " f"SELECT * FROM read_json_auto('{catalog.protect_rules_json}')"
        )
    else:
        conn.execute(_EMPTY_PROTECT_RULES_VIEW)

    if _json_file_has_rows(catalog.delete_rules_json):
        conn.execute(
            f"CREATE OR REPLACE VIEW delete_rules AS " f"SELECT * FROM read_json_auto('{catalog.delete_rules_json}')"
        )
    else:
        conn.execute(_EMPTY_DELETE_RULES_VIEW)


def _next_rule_id(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    """Return the next available id for a rules JSON file."""
    row = conn.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table}").fetchone()
    return int(row[0])


def flush_protect_rules(conn: duckdb.DuckDBPyConnection, catalog: StorageCatalog) -> None:
    """Refresh the protect_rules view after the JSON file has been updated."""
    _create_rules_views(conn, catalog)


def flush_delete_rules(conn: duckdb.DuckDBPyConnection, catalog: StorageCatalog) -> None:
    """Refresh the delete_rules view after the JSON file has been updated."""
    _create_rules_views(conn, catalog)


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------


def ensure_output_dirs(catalog: StorageCatalog = DEFAULT_CATALOG) -> duckdb.DuckDBPyConnection:
    """Create workspace directories and return an initialized DB connection."""
    catalog.ensure_dirs()
    return init_db(catalog)
