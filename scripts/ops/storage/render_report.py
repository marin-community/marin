#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage usage report generator.

Builds a bounded directory-level rollup (`dir_summary`) from raw parquet
object listings in a single streaming pass, then computes every report
section from the rollup. Produces a markdown report with size, cost, and
trend breakdowns.

The rollup groups objects by `(bucket, storage_class_id, dir_prefix)` for
the dir summary and `(bucket, storage_class_id, created_month, age_bucket)`
for the time summary. `dir_prefix` is the first DIR_DEPTH path components.
This keeps the working set in the low millions of rows regardless of how
many billions of objects the scan produced.

Usage (standalone):
    uv run scripts/ops/storage/render_report.py PARQUET_DIR

PARQUET_DIR (a local dir or gs:// path) is required; it points at the parquet
segments written by the scan stage.
"""

import hashlib
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import duckdb
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from tqdm import tqdm

from scripts.ops.storage.constants import (
    BUCKET_LOCATIONS,
    DISCOUNT_FACTOR,
    STORAGE_CLASS_PRICING,
    human_bytes,
)


def _download_gcs_parquet(gcs_dir: str, local_dir: Path) -> Path:
    """Mirror all *.parquet files from gcs_dir to local_dir via fsspec.

    Deletes local parquets that no longer exist at the source — without
    this, segments from prior scans accumulate locally and the report ends
    up reading a stale union of every historical scan.

    Returns the local directory.
    """

    local_dir.mkdir(parents=True, exist_ok=True)
    fs, _ = fsspec.core.url_to_fs(gcs_dir)
    pattern = f"{gcs_dir.rstrip('/')}/*.parquet"
    remote_paths = fs.glob(pattern)
    remote_basenames = {Path(p).name for p in remote_paths}

    pruned = 0
    for existing in local_dir.glob("*.parquet"):
        if existing.name not in remote_basenames:
            existing.unlink()
            pruned += 1
    if pruned:
        print(f"  pruned {pruned} stale local parquets")

    print(f"Mirroring {len(remote_paths)} parquets {gcs_dir} -> {local_dir} ...")
    pbar = tqdm(remote_paths, unit="file", dynamic_ncols=True)
    for remote in pbar:
        name = Path(remote).name
        local = local_dir / name
        if local.exists() and local.stat().st_size == fs.info(remote)["size"]:
            continue
        with fs.open(remote, "rb") as src, open(local, "wb") as dst:
            while chunk := src.read(8 * 1024 * 1024):
                dst.write(chunk)
    print(f"  {len(remote_paths)} parquet files local")
    return local_dir


def _init_storage_classes(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE storage_classes (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price_per_gib_month_us REAL NOT NULL,
            price_per_gib_month_eu REAL NOT NULL
        )
    """
    )
    for sc_id, name, us_price, eu_price in STORAGE_CLASS_PRICING:
        conn.execute(
            "INSERT INTO storage_classes VALUES (?, ?, ?, ?)",
            (sc_id, name, us_price, eu_price),
        )


# Depth at which we roll up directory paths. First DIR_DEPTH components
# of each object name form the `dir_prefix` key; deeper structure is discarded.
DIR_DEPTH = 3


def _dir_agg_sql(source: str) -> str:
    return f"""
    SELECT
        bucket,
        storage_class_id,
        array_to_string(list_slice(string_split(name, '/'), 1, {DIR_DEPTH}), '/') AS dir_prefix,
        COUNT(*) AS object_count,
        SUM(size_bytes) AS total_bytes
    FROM {source}
    GROUP BY 1, 2, 3
    """


def _time_agg_sql(source: str) -> str:
    return f"""
    SELECT
        bucket,
        storage_class_id,
        date_trunc('month', created) AS created_month,
        CASE
            WHEN created >= CURRENT_TIMESTAMP - INTERVAL  '7 days'  THEN '<7d'
            WHEN created >= CURRENT_TIMESTAMP - INTERVAL '30 days'  THEN '7-30d'
            WHEN created >= CURRENT_TIMESTAMP - INTERVAL '90 days'  THEN '30-90d'
            WHEN created >= CURRENT_TIMESTAMP - INTERVAL '365 days' THEN '90-365d'
            WHEN created IS NULL                                    THEN NULL
            ELSE '>365d'
        END AS age_bucket,
        COUNT(*) AS object_count,
        SUM(size_bytes) AS total_bytes
    FROM {source}
    GROUP BY 1, 2, 3, 4
    """


def _batch_key(batch: list[str], depth: int) -> str:
    """Stable content-addressed key for a batch — invariant to run ordering.

    Changing DIR_DEPTH or the files in a batch invalidates the cache."""
    h = hashlib.sha1()
    for p in sorted(batch):
        h.update(p.encode())
        h.update(b"\0")
    h.update(f"d{depth}".encode())
    return h.hexdigest()[:16]


def _build_summaries(
    conn: duckdb.DuckDBPyConnection,
    parquet_files: list[str],
    summary_cache: Path,
) -> None:
    """Map step: per-batch aggregates written to cached parquets.

    Per-batch parquets are written to `summary_cache` with content-addressed
    names, so reruns skip already-computed batches. `dir_summary` and
    `time_summary` are then registered as views over the leaf parquets —
    the full-cardinality reduce is skipped because every report query
    except `top_dir3` has tiny group cardinality, and `top_dir3` prunes
    via hierarchical top-K.
    """
    batch_size = 32
    batches = [parquet_files[i : i + batch_size] for i in range(0, len(parquet_files), batch_size)]

    leaf_root = summary_cache / f"d{DIR_DEPTH}"
    dir_leaves = leaf_root / "dir"
    time_leaves = leaf_root / "time"
    for d in (dir_leaves, time_leaves):
        d.mkdir(parents=True, exist_ok=True)

    # Prune leaves from prior runs whose batch keys are not in the current
    # batch set. Otherwise stale per-batch aggregates (from objects parquets
    # that have since been removed) silently fold into the dir_summary view
    # below and the report shows ghost objects.
    expected_keys = {_batch_key(batch, DIR_DEPTH) for batch in batches}
    pruned = 0
    for leaves_dir in (dir_leaves, time_leaves):
        for existing in leaves_dir.glob("*.parquet"):
            if existing.stem not in expected_keys:
                existing.unlink()
                pruned += 1
    if pruned:
        print(f"pruned {pruned} stale summary leaves from {leaf_root}", file=sys.stderr)

    total_objects = 0
    total_bytes = 0
    pbar = tqdm(batches, desc="map (per-batch agg)", unit="batch", dynamic_ncols=True)
    cache_hits = 0
    for batch in pbar:
        key = _batch_key(batch, DIR_DEPTH)
        dir_out = dir_leaves / f"{key}.parquet"
        time_out = time_leaves / f"{key}.parquet"

        if dir_out.exists() and time_out.exists():
            cache_hits += 1
        else:
            path_list = ", ".join(f"'{p}'" for p in batch)
            source = f"read_parquet([{path_list}], union_by_name=true)"
            conn.execute(f"CREATE TEMP VIEW _batch AS SELECT * FROM {source}")
            # Write to a .tmp then rename so partial files don't poison the cache.
            dir_tmp = dir_out.with_suffix(".parquet.tmp")
            time_tmp = time_out.with_suffix(".parquet.tmp")
            conn.execute(f"COPY ({_dir_agg_sql('_batch')}) TO '{dir_tmp}' (FORMAT parquet)")
            conn.execute(f"COPY ({_time_agg_sql('_batch')}) TO '{time_tmp}' (FORMAT parquet)")
            conn.execute("DROP VIEW _batch")
            dir_tmp.rename(dir_out)
            time_tmp.rename(time_out)

        oc, tb = conn.execute(f"SELECT SUM(object_count), SUM(total_bytes) FROM read_parquet('{dir_out}')").fetchone()
        total_objects += oc or 0
        total_bytes += tb or 0
        pbar.set_postfix(
            cached=cache_hits,
            objects=f"{total_objects / 1e6:.1f}M",
            size=f"{total_bytes / 1e12:.2f}TB",
            refresh=False,
        )

    print(f"map done: {cache_hits}/{len(batches)} batches from cache", file=sys.stderr)

    # Skip the reduce step: report queries with small group cardinality
    # (by_bucket, by_class, top_dir1, top_dir2, age, monthly) run happily
    # against the per-batch leaf parquets directly. The only query that
    # would need a full (bucket, dir_prefix) reduce is top_dir3, which is
    # handled via hierarchical top-K in `_query_top_dir_prefix`.
    dir_files = sorted(dir_leaves.glob("*.parquet"))
    time_files = sorted(time_leaves.glob("*.parquet"))
    dir_list = ", ".join(f"'{p}'" for p in dir_files)
    time_list = ", ".join(f"'{p}'" for p in time_files)
    conn.execute(f"CREATE VIEW dir_summary AS SELECT * FROM read_parquet([{dir_list}])")
    conn.execute(f"CREATE VIEW time_summary AS SELECT * FROM read_parquet([{time_list}])")
    print(
        f"views registered: dir_summary over {len(dir_files)} leaves, time_summary over {len(time_files)} leaves",
        file=sys.stderr,
    )


def _tune_duckdb(conn: duckdb.DuckDBPyConnection, scratch: Path) -> None:
    """Cap working set and give DuckDB a big temp dir on the main disk.

    /tmp on macOS is tiny; DuckDB otherwise auto-caps its spill based on
    /tmp's free space and bails at ~2GiB mid-reduce. Pointing it at the
    same scratch we use for summary cache avoids that.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    spill = scratch / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    conn.execute("SET threads=2")
    conn.execute("SET memory_limit='8GB'")
    conn.execute("SET preserve_insertion_order=false")
    conn.execute(f"SET temp_directory='{spill}'")
    conn.execute("SET max_temp_directory_size='100GiB'")


def load_parquet_db(parquet_dir: Path | str, local_cache: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Build an in-memory DuckDB with a pre-aggregated `dir_summary` table.

    For gs:// paths, downloads files to local_cache (or a temp dir) first so
    DuckDB reads from local disk — avoids the GCS auth maze.
    """
    dir_str = str(parquet_dir)
    if dir_str.startswith("gs://"):
        if local_cache is None:
            cache_root = Path("/tmp/storage-scan-cache")
            subpath = dir_str.removeprefix("gs://").replace("/", "_")
            local_cache = cache_root / subpath
        local_dir = _download_gcs_parquet(dir_str, local_cache)
    else:
        local_dir = Path(dir_str)

    conn = duckdb.connect(":memory:")
    files = sorted(str(p) for p in local_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet files found under {local_dir}")
    summary_cache = local_dir.parent / (local_dir.name + "_summaries")
    _tune_duckdb(conn, summary_cache)
    _build_summaries(conn, files, summary_cache)
    _init_storage_classes(conn)
    return conn


def load_parquet_db_from_paths(paths: list[str], summary_cache: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Build an in-memory DuckDB from explicit parquet paths (local or GCS)."""
    conn = duckdb.connect(":memory:")
    if summary_cache is None:
        h = hashlib.sha1()
        for p in sorted(paths):
            h.update(p.encode() + b"\0")
        summary_cache = Path("/tmp/storage-scan-cache/_from_paths_summaries") / h.hexdigest()[:16]
    _tune_duckdb(conn, summary_cache)
    _build_summaries(conn, list(paths), summary_cache)
    _init_storage_classes(conn)
    return conn


# ---------------------------------------------------------------------------
# Cost SQL fragment (reused across queries)
# ---------------------------------------------------------------------------

# Per-summary-row cost expression (use inside SUM(...) or as a column).
# Works on any summary table aliased `s` joined to `storage_classes sc`,
# as long as `s` exposes `bucket` and `total_bytes`.
_ROW_COST = f"""
    s.total_bytes / (1024.0 * 1024.0 * 1024.0)
        * CASE WHEN s.bucket LIKE '%eu%' THEN sc.price_per_gib_month_eu
               ELSE sc.price_per_gib_month_us END
        * {DISCOUNT_FACTOR}
"""


# ---------------------------------------------------------------------------
# Query functions (all read from dir_summary)
# ---------------------------------------------------------------------------


def _query_overview(conn: duckdb.DuckDBPyConnection) -> dict:
    row = conn.execute(
        f"""
        SELECT
            SUM(s.object_count),
            SUM(s.total_bytes),
            SUM({_ROW_COST})
        FROM dir_summary s
        JOIN storage_classes sc ON s.storage_class_id = sc.id
        """
    ).fetchone()
    return {"total_objects": row[0], "total_bytes": row[1], "monthly_cost": row[2]}


def _query_by_bucket(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            s.bucket,
            SUM(s.object_count) AS object_count,
            SUM(s.total_bytes) AS total_bytes,
            SUM({_ROW_COST}) AS monthly_cost
        FROM dir_summary s
        JOIN storage_classes sc ON s.storage_class_id = sc.id
        GROUP BY s.bucket
        ORDER BY monthly_cost DESC
        """
    ).fetchall()
    return [
        {
            "bucket": r[0],
            "region": BUCKET_LOCATIONS.get(r[0], "?"),
            "object_count": r[1],
            "total_bytes": r[2],
            "monthly_cost": r[3],
        }
        for r in rows
    ]


def _query_by_storage_class(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            sc.name,
            SUM(s.object_count) AS object_count,
            SUM(s.total_bytes) AS total_bytes,
            SUM({_ROW_COST}) AS monthly_cost
        FROM dir_summary s
        JOIN storage_classes sc ON s.storage_class_id = sc.id
        GROUP BY sc.name
        ORDER BY monthly_cost DESC
        """
    ).fetchall()
    grand_total = sum(r[2] for r in rows) or 1
    return [
        {
            "name": r[0],
            "object_count": r[1],
            "total_bytes": r[2],
            "monthly_cost": r[3],
            "pct": r[2] / grand_total * 100,
        }
        for r in rows
    ]


def _query_top_dir1(conn: duckdb.DuckDBPyConnection, limit: int = 30) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            s.bucket,
            split_part(s.dir_prefix, '/', 1) AS dir1,
            SUM(s.object_count) AS object_count,
            SUM(s.total_bytes) AS total_bytes,
            SUM({_ROW_COST}) AS monthly_cost
        FROM dir_summary s
        JOIN storage_classes sc ON s.storage_class_id = sc.id
        WHERE s.dir_prefix LIKE '%/%'
        GROUP BY s.bucket, dir1
        ORDER BY monthly_cost DESC
        LIMIT {limit}
        """
    ).fetchall()
    return [
        {
            "bucket": r[0],
            "prefix": r[1] + "/",
            "object_count": r[2],
            "total_bytes": r[3],
            "monthly_cost": r[4],
        }
        for r in rows
    ]


def _query_top_dir2(conn: duckdb.DuckDBPyConnection, limit: int = 30) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            s.bucket,
            split_part(s.dir_prefix, '/', 1) || '/' || split_part(s.dir_prefix, '/', 2) AS prefix2,
            SUM(s.object_count) AS object_count,
            SUM(s.total_bytes) AS total_bytes,
            SUM({_ROW_COST}) AS monthly_cost
        FROM dir_summary s
        JOIN storage_classes sc ON s.storage_class_id = sc.id
        WHERE s.dir_prefix LIKE '%/%/%'
        GROUP BY s.bucket, prefix2
        ORDER BY monthly_cost DESC
        LIMIT {limit}
        """
    ).fetchall()
    return [
        {
            "bucket": r[0],
            "prefix": r[1],
            "object_count": r[2],
            "total_bytes": r[3],
            "monthly_cost": r[4],
        }
        for r in rows
    ]


# Safety margin for hierarchical top-K: true top-`limit` dir3 prefixes are
# guaranteed to live under dir2 parents whose total cost ≥ the cost of the
# top-`limit`-th dir3. We don't know that cost up front, so we include the
# top-N dir2s with N >> limit. N=200 x ~few-thousand dir3 children per dir2
# keeps the second-pass hash table in the hundreds of thousands of keys.
_TOP_DIR3_CANDIDATE_DIR2S = 200


def _query_top_dir_prefix(conn: duckdb.DuckDBPyConnection, limit: int = 30) -> list[dict]:
    # Two-pass: restrict the (bucket, dir_prefix) GROUP BY to dir3 rows whose
    # (bucket, dir1/dir2) parent is in the top-N dir2s by cost. This bounds
    # the hash table instead of materializing ~20M global dir_prefix groups.
    rows = conn.execute(
        f"""
        WITH top_dir2 AS (
            SELECT
                s.bucket,
                split_part(s.dir_prefix, '/', 1) || '/' || split_part(s.dir_prefix, '/', 2) AS dir2,
                SUM({_ROW_COST}) AS c
            FROM dir_summary s
            JOIN storage_classes sc ON s.storage_class_id = sc.id
            WHERE s.dir_prefix LIKE '%/%/%'
            GROUP BY s.bucket, dir2
            ORDER BY c DESC
            LIMIT {_TOP_DIR3_CANDIDATE_DIR2S}
        )
        SELECT
            s.bucket,
            s.dir_prefix AS prefix,
            SUM(s.object_count) AS object_count,
            SUM(s.total_bytes) AS total_bytes,
            SUM({_ROW_COST}) AS monthly_cost
        FROM dir_summary s
        JOIN storage_classes sc ON s.storage_class_id = sc.id
        JOIN top_dir2 t
          ON t.bucket = s.bucket
         AND t.dir2 = split_part(s.dir_prefix, '/', 1) || '/' || split_part(s.dir_prefix, '/', 2)
        GROUP BY s.bucket, s.dir_prefix
        ORDER BY monthly_cost DESC
        LIMIT {limit}
        """
    ).fetchall()
    return [
        {
            "bucket": r[0],
            "prefix": r[1],
            "object_count": r[2],
            "total_bytes": r[3],
            "monthly_cost": r[4],
        }
        for r in rows
    ]


def _query_age_distribution(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            s.age_bucket,
            SUM(s.object_count) AS object_count,
            SUM(s.total_bytes) AS total_bytes,
            SUM({_ROW_COST}) AS monthly_cost
        FROM time_summary s
        JOIN storage_classes sc ON s.storage_class_id = sc.id
        WHERE s.age_bucket IS NOT NULL
        GROUP BY s.age_bucket
        ORDER BY CASE s.age_bucket
            WHEN '<7d' THEN 1
            WHEN '7-30d' THEN 2
            WHEN '30-90d' THEN 3
            WHEN '90-365d' THEN 4
            ELSE 5
        END
        """
    ).fetchall()
    return [
        {
            "age_bucket": r[0],
            "object_count": r[1],
            "total_bytes": r[2],
            "monthly_cost": r[3],
        }
        for r in rows
    ]


def _query_monthly_growth(conn: duckdb.DuckDBPyConnection, months: int = 12) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            strftime(created_month, '%Y-%m') AS month,
            SUM(object_count) AS object_count,
            SUM(total_bytes) AS total_bytes
        FROM time_summary
        WHERE created_month IS NOT NULL
          AND created_month >= date_trunc('month', CURRENT_TIMESTAMP) - INTERVAL '{months} months'
        GROUP BY month
        ORDER BY month DESC
        """
    ).fetchall()
    return [{"month": r[0], "object_count": r[1], "total_bytes": r[2]} for r in rows]


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:,.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:,.1f}K"
    return f"{n:,}"


def _fmt_tb(b: int) -> str:
    return f"{b / 1e12:,.2f}"


def _fmt_cost(c: float) -> str:
    return f"${c:,.0f}"


def _md_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    """Render a markdown table. align entries: 'l', 'r', or 'c'."""
    if not rows:
        return "_No data._\n"
    if align is None:
        align = ["l"] * len(headers)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    sep_map = {"l": ":---", "r": "---:", "c": ":---:"}
    lines.append("| " + " | ".join(sep_map.get(a, "---") for a in align) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Week-over-week snapshots + diff
#
# Each run archives a compact per-(bucket, dir_prefix) snapshot to a stable
# history directory. The next run loads the most recent prior snapshot and
# flags prefixes whose size moved by more than a threshold. The snapshot only
# keeps prefixes above SNAPSHOT_MIN_BYTES so it stays a few thousand rows (KB),
# cheap to archive weekly and diff — small churn below the floor is noise.
# ---------------------------------------------------------------------------

# Prefixes smaller than this are not archived (keeps snapshots tiny). A prefix
# that grows from below the floor to above the change threshold still shows up
# as "new" because its prior bytes are treated as zero.
SNAPSHOT_MIN_BYTES = 1 * 1024**3  # 1 GiB

# A prefix whose size moves by at least this much between runs is flagged.
DEFAULT_CHANGE_THRESHOLD_BYTES = 100 * 1024**3  # 100 GiB

_SNAPSHOT_SCHEMA = pa.schema(
    [
        ("bucket", pa.string()),
        ("dir_prefix", pa.string()),
        ("object_count", pa.int64()),
        ("total_bytes", pa.int64()),
        ("monthly_cost", pa.float64()),
    ]
)

_SNAPSHOT_NAME_RE = re.compile(r"dir_summary-(\d{4}-\d{2}-\d{2})\.parquet$")


def snapshot_dir_summary(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per-(bucket, dir_prefix) rollup (>= SNAPSHOT_MIN_BYTES) for diffing."""
    rows = conn.execute(
        f"""
        SELECT
            s.bucket,
            s.dir_prefix,
            SUM(s.object_count) AS object_count,
            SUM(s.total_bytes) AS total_bytes,
            SUM({_ROW_COST}) AS monthly_cost
        FROM dir_summary s
        JOIN storage_classes sc ON s.storage_class_id = sc.id
        GROUP BY s.bucket, s.dir_prefix
        HAVING SUM(s.total_bytes) >= {SNAPSHOT_MIN_BYTES}
        """
    ).fetchall()
    return [
        {
            "bucket": r[0],
            "dir_prefix": r[1],
            "object_count": int(r[2]),
            "total_bytes": int(r[3]),
            "monthly_cost": float(r[4]),
        }
        for r in rows
    ]


def snapshot_path(history_dir: str, date_str: str) -> str:
    return f"{history_dir.rstrip('/')}/dir_summary-{date_str}.parquet"


def write_snapshot(rows: list[dict], path: str) -> None:
    table = pa.Table.from_pylist(rows, schema=_SNAPSHOT_SCHEMA)
    with StoragePath(path).open("wb") as f:
        pq.write_table(table, f)


def read_snapshot(path: str) -> list[dict]:
    with StoragePath(path).open("rb") as f:
        return pq.read_table(f).to_pylist()


def find_latest_snapshot(history_dir: str, *, before_date: str | None = None) -> tuple[str, str] | None:
    """Return (full_path, date) of the most recent snapshot, or None.

    ``before_date`` (YYYY-MM-DD) excludes snapshots on or after that date so a
    run never diffs against itself when re-run on the same day.
    """
    try:
        candidates = StoragePath(f"{history_dir.rstrip('/')}/dir_summary-*.parquet").glob()
    except FileNotFoundError:
        return None
    dated: list[tuple[str, str]] = []
    for candidate in candidates:
        path = str(candidate)
        match = _SNAPSHOT_NAME_RE.search(path)
        if not match:
            continue
        date = match.group(1)
        if before_date is not None and date >= before_date:
            continue
        dated.append((date, path))
    if not dated:
        return None
    date, path = max(dated)
    return path, date


def compute_changes(current: list[dict], previous: list[dict], *, threshold_bytes: int) -> list[dict]:
    """Flag (bucket, dir_prefix) prefixes whose byte size moved >= threshold."""
    cur_by_key = {(r["bucket"], r["dir_prefix"]): r for r in current}
    prev_by_key = {(r["bucket"], r["dir_prefix"]): r for r in previous}

    changes: list[dict] = []
    for key in set(cur_by_key) | set(prev_by_key):
        cur = cur_by_key.get(key)
        prev = prev_by_key.get(key)
        now_bytes = cur["total_bytes"] if cur else 0
        was_bytes = prev["total_bytes"] if prev else 0
        delta = now_bytes - was_bytes
        if abs(delta) < threshold_bytes:
            continue
        if prev is None:
            status = "new"
        elif cur is None:
            status = "gone"
        elif delta > 0:
            status = "grew"
        else:
            status = "shrank"
        changes.append(
            {
                "bucket": key[0],
                "dir_prefix": key[1],
                "delta_bytes": delta,
                "now_bytes": now_bytes,
                "was_bytes": was_bytes,
                "status": status,
            }
        )
    changes.sort(key=lambda c: abs(c["delta_bytes"]), reverse=True)
    return changes


def _fmt_delta(num_bytes: int) -> str:
    sign = "+" if num_bytes >= 0 else "-"
    return f"{sign}{human_bytes(abs(num_bytes))}"


def _changes_table(changes: list[dict], *, limit: int) -> str:
    shown = changes[:limit]
    rows = [
        [
            c["bucket"],
            c["dir_prefix"],
            _fmt_delta(c["delta_bytes"]),
            _fmt_tb(c["now_bytes"]),
            _fmt_tb(c["was_bytes"]),
            c["status"],
        ]
        for c in shown
    ]
    table = _md_table(
        ["Bucket", "Prefix", "Δ Size", "Now (TB)", "Was (TB)", "Status"],
        rows,
        align=["l", "l", "r", "r", "r", "l"],
    )
    if len(changes) > limit:
        table += f"\n_(+{len(changes) - limit} more above threshold)_\n"
    return table


def render_changes_section(
    changes: list[dict],
    *,
    previous_date: str | None,
    threshold_bytes: int,
    limit: int = 30,
) -> str:
    """Markdown for the week-over-week section. ``previous_date=None`` = baseline.

    Splits into Increases (growth — more alarming, shown first) and Decreases
    (cleanup), each sorted by magnitude.
    """
    if previous_date is None:
        return "## Week-over-Week Changes\n\n_No prior snapshot — baseline established for next run._\n"

    threshold_human = human_bytes(threshold_bytes)
    header = f"## Week-over-Week Changes\n\n_Prefixes whose size moved by ≥ {threshold_human} since {previous_date}._\n"
    if not changes:
        return header + "\n_No changes above threshold._\n"

    increases = sorted((c for c in changes if c["delta_bytes"] > 0), key=lambda c: c["delta_bytes"], reverse=True)
    decreases = sorted((c for c in changes if c["delta_bytes"] < 0), key=lambda c: c["delta_bytes"])

    parts = [header]
    parts.append("\n### Increases\n")
    parts.append("\n" + (_changes_table(increases, limit=limit) if increases else "_None above threshold._\n"))
    parts.append("\n### Decreases\n")
    parts.append("\n" + (_changes_table(decreases, limit=limit) if decreases else "_None above threshold._\n"))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(conn: duckdb.DuckDBPyConnection, *, changes_section: str | None = None) -> str:
    """Generate a full storage usage report as markdown.

    ``changes_section`` is the pre-rendered week-over-week block (see
    ``render_changes_section``); when provided it is inserted after Overview.
    """
    overview = _query_overview(conn)
    by_bucket = _query_by_bucket(conn)
    by_class = _query_by_storage_class(conn)
    top_dir1 = _query_top_dir1(conn)
    top_dir2 = _query_top_dir2(conn)
    top_prefix = _query_top_dir_prefix(conn)
    age_dist = _query_age_distribution(conn)
    monthly = _query_monthly_growth(conn)

    parts: list[str] = []

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts.append(f"# GCS Storage Report\n\nGenerated: {ts}\n")

    parts.append("## Overview\n")
    parts.append(
        _md_table(
            ["Metric", "Value"],
            [
                ["Total Objects", _fmt_count(overview["total_objects"])],
                ["Total Size", f"{_fmt_tb(overview['total_bytes'])} TB"],
                ["Est. Monthly Cost", _fmt_cost(overview["monthly_cost"])],
                ["Annual Estimate", _fmt_cost(overview["monthly_cost"] * 12)],
            ],
            align=["l", "r"],
        )
    )

    if changes_section is not None:
        parts.append(changes_section)

    parts.append("## By Bucket\n")
    parts.append(
        _md_table(
            ["Bucket", "Region", "Objects", "Size (TB)", "Monthly Cost"],
            [
                [
                    r["bucket"],
                    r["region"],
                    _fmt_count(r["object_count"]),
                    _fmt_tb(r["total_bytes"]),
                    _fmt_cost(r["monthly_cost"]),
                ]
                for r in by_bucket
            ],
            align=["l", "l", "r", "r", "r"],
        )
    )

    parts.append("## By Storage Class\n")
    parts.append(
        _md_table(
            ["Class", "Objects", "Size (TB)", "Monthly Cost", "% of Total"],
            [
                [
                    r["name"],
                    _fmt_count(r["object_count"]),
                    _fmt_tb(r["total_bytes"]),
                    _fmt_cost(r["monthly_cost"]),
                    f"{r['pct']:.1f}%",
                ]
                for r in by_class
            ],
            align=["l", "r", "r", "r", "r"],
        )
    )

    parts.append("## Top First-Level Directories\n")
    parts.append(
        _md_table(
            ["Bucket", "Directory", "Objects", "Size (TB)", "Monthly Cost"],
            [
                [
                    r["bucket"],
                    r["prefix"],
                    _fmt_count(r["object_count"]),
                    _fmt_tb(r["total_bytes"]),
                    _fmt_cost(r["monthly_cost"]),
                ]
                for r in top_dir1
            ],
            align=["l", "l", "r", "r", "r"],
        )
    )

    parts.append("## Top Two-Level Prefixes\n")
    parts.append(
        _md_table(
            ["Bucket", "Prefix", "Objects", "Size (TB)", "Monthly Cost"],
            [
                [
                    r["bucket"],
                    r["prefix"],
                    _fmt_count(r["object_count"]),
                    _fmt_tb(r["total_bytes"]),
                    _fmt_cost(r["monthly_cost"]),
                ]
                for r in top_dir2
            ],
            align=["l", "l", "r", "r", "r"],
        )
    )

    parts.append(f"## Top {DIR_DEPTH}-Level Prefixes\n")
    parts.append(
        _md_table(
            ["Bucket", "Prefix", "Objects", "Size (TB)", "Monthly Cost"],
            [
                [
                    r["bucket"],
                    r["prefix"],
                    _fmt_count(r["object_count"]),
                    _fmt_tb(r["total_bytes"]),
                    _fmt_cost(r["monthly_cost"]),
                ]
                for r in top_prefix
            ],
            align=["l", "l", "r", "r", "r"],
        )
    )

    parts.append("## Age Distribution\n")
    parts.append(
        _md_table(
            ["Age", "Objects", "Size (TB)", "Monthly Cost"],
            [
                [
                    r["age_bucket"],
                    _fmt_count(r["object_count"]),
                    _fmt_tb(r["total_bytes"]),
                    _fmt_cost(r["monthly_cost"]),
                ]
                for r in age_dist
            ],
            align=["l", "r", "r", "r"],
        )
    )

    parts.append("## Monthly Creation Trend\n")
    parts.append(
        _md_table(
            ["Month", "Objects Created", "Size Created (TB)"],
            [[r["month"], _fmt_count(r["object_count"]), _fmt_tb(r["total_bytes"])] for r in monthly],
            align=["l", "r", "r"],
        )
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument("parquet_dir")
@click.option(
    "--output",
    "-o",
    help="Write markdown report to file (local path or gs:// URL; default: stdout).",
)
@click.option(
    "--history-dir",
    help="Directory (local or gs://) of dated snapshots. When set, the report "
    "archives this run's snapshot and adds a week-over-week changes section "
    "diffing against the most recent prior snapshot.",
)
@click.option(
    "--change-threshold-gib",
    default=DEFAULT_CHANGE_THRESHOLD_BYTES / 1024**3,
    show_default=True,
    type=float,
    help="Flag prefixes whose size moved by at least this many GiB.",
)
def main(parquet_dir: str, output: str | None, history_dir: str | None, change_threshold_gib: float) -> None:
    """Generate a storage usage report from parquet output of a scan.

    PARQUET_DIR may be a local directory or a gs:// path (auto-downloaded
    to /tmp/storage-scan-cache via fsspec).

    Examples:
        uv run scripts/ops/storage/render_report.py gs://marin-us-central2/tmp/storage-scan-v7
        uv run scripts/ops/storage/render_report.py ./local_parquet -o report.md
    """
    conn = load_parquet_db(parquet_dir)
    changes_section = None
    current_snapshot = None
    if history_dir:
        threshold_bytes = int(change_threshold_gib * 1024**3)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        current_snapshot = snapshot_dir_summary(conn)
        prior = find_latest_snapshot(history_dir, before_date=today)
        if prior is None:
            changes_section = render_changes_section([], previous_date=None, threshold_bytes=threshold_bytes)
        else:
            prev_path, prev_date = prior
            changes = compute_changes(current_snapshot, read_snapshot(prev_path), threshold_bytes=threshold_bytes)
            changes_section = render_changes_section(changes, previous_date=prev_date, threshold_bytes=threshold_bytes)
    report = generate_report(conn, changes_section=changes_section)
    if history_dir and current_snapshot is not None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        write_snapshot(current_snapshot, snapshot_path(history_dir, today))
        print(f"Snapshot archived: {snapshot_path(history_dir, today)}", file=sys.stderr)
    if output:
        if output.startswith("gs://"):
            with fsspec.open(output, "w") as f:
                f.write(report)
        else:
            Path(output).write_text(report)
        print(f"Report written to {output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
