#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage usage report generator.

Loads object-listing parquet files into an in-memory DuckDB, runs analysis
queries, and produces a markdown report with size, cost, and trend breakdowns.

Usage (standalone):
    uv run scripts/storage/report.py [PARQUET_DIR]

The default parquet directory is scripts/storage/purge/objects_parquet/.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import duckdb

from scripts.storage.constants import (
    BUCKET_LOCATIONS,
    DISCOUNT_FACTOR,
    STORAGE_CLASS_PRICING,
)


def _download_gcs_parquet(gcs_dir: str, local_dir: Path) -> Path:
    """Download all *.parquet files from gcs_dir to local_dir using gcloud.

    Skips files already present locally. Returns the local directory.
    """
    import subprocess

    local_dir.mkdir(parents=True, exist_ok=True)
    src = gcs_dir.rstrip("/") + "/*.parquet"
    print(f"Downloading {src} -> {local_dir} ...")
    # rsync-style: only copies new files
    result = subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive", gcs_dir.rstrip("/"), str(local_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gcloud rsync failed: {result.stderr}")
    file_count = len(list(local_dir.glob("*.parquet")))
    print(f"  {file_count} parquet files local")
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


def load_parquet_db(parquet_dir: Path | str, local_cache: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB with objects view over a parquet directory.

    For gs:// paths, downloads files to local_cache (or a temp dir) first so
    DuckDB reads from local disk — avoids the GCS auth maze.
    """
    dir_str = str(parquet_dir)
    if dir_str.startswith("gs://"):
        if local_cache is None:
            # Default cache: /tmp/storage-scan-cache/<bucket-path>
            cache_root = Path("/tmp/storage-scan-cache")
            subpath = dir_str.removeprefix("gs://").replace("/", "_")
            local_cache = cache_root / subpath
        local_dir = _download_gcs_parquet(dir_str, local_cache)
    else:
        local_dir = Path(dir_str)

    conn = duckdb.connect(":memory:")
    conn.execute("SET threads=8")
    glob_pattern = str(local_dir / "*.parquet")
    conn.execute(
        f"""
        CREATE VIEW objects AS
        SELECT * FROM read_parquet('{glob_pattern}', union_by_name=true)
    """
    )
    _init_storage_classes(conn)
    return conn


def load_parquet_db_from_paths(paths: list[str]) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB with objects view over explicit parquet paths (local or GCS)."""
    conn = duckdb.connect(":memory:")
    conn.execute("SET threads=8")

    path_list = ", ".join(f"'{p}'" for p in paths)
    conn.execute(
        f"""
        CREATE VIEW objects AS
        SELECT * FROM read_parquet([{path_list}], union_by_name=true)
    """
    )
    _init_storage_classes(conn)
    return conn


# ---------------------------------------------------------------------------
# Cost SQL fragment (reused across queries)
# ---------------------------------------------------------------------------

# Per-row cost expression (use inside SUM(...) or as a column)
_ROW_COST = f"""
    o.size_bytes / (1024.0 * 1024.0 * 1024.0)
        * CASE WHEN o.bucket LIKE '%eu%' THEN sc.price_per_gib_month_eu
               ELSE sc.price_per_gib_month_us END
        * {DISCOUNT_FACTOR}
"""


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def _query_overview(conn: duckdb.DuckDBPyConnection) -> dict:
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) as total_objects,
            SUM(o.size_bytes) as total_bytes,
            SUM({_ROW_COST}) as monthly_cost
        FROM objects o
        JOIN storage_classes sc ON o.storage_class_id = sc.id
    """
    ).fetchone()
    return {"total_objects": row[0], "total_bytes": row[1], "monthly_cost": row[2]}


def _query_by_bucket(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            o.bucket,
            COUNT(*) as object_count,
            SUM(o.size_bytes) as total_bytes,
            SUM({_ROW_COST}) as monthly_cost
        FROM objects o
        JOIN storage_classes sc ON o.storage_class_id = sc.id
        GROUP BY o.bucket
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
            COUNT(*) as object_count,
            SUM(o.size_bytes) as total_bytes,
            SUM({_ROW_COST}) as monthly_cost
        FROM objects o
        JOIN storage_classes sc ON o.storage_class_id = sc.id
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
            o.bucket,
            split_part(o.name, '/', 1) as dir1,
            COUNT(*) as object_count,
            SUM(o.size_bytes) as total_bytes,
            SUM({_ROW_COST}) as monthly_cost
        FROM objects o
        JOIN storage_classes sc ON o.storage_class_id = sc.id
        WHERE o.name LIKE '%/%'
        GROUP BY o.bucket, dir1
        ORDER BY monthly_cost DESC
        LIMIT {limit}
    """
    ).fetchall()
    return [
        {"bucket": r[0], "prefix": r[1] + "/", "object_count": r[2], "total_bytes": r[3], "monthly_cost": r[4]}
        for r in rows
    ]


def _query_top_dir2(conn: duckdb.DuckDBPyConnection, limit: int = 30) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            o.bucket,
            split_part(o.name, '/', 1) || '/' || split_part(o.name, '/', 2) as prefix2,
            COUNT(*) as object_count,
            SUM(o.size_bytes) as total_bytes,
            SUM({_ROW_COST}) as monthly_cost
        FROM objects o
        JOIN storage_classes sc ON o.storage_class_id = sc.id
        WHERE o.name LIKE '%/%/%'
        GROUP BY o.bucket, prefix2
        ORDER BY monthly_cost DESC
        LIMIT {limit}
    """
    ).fetchall()
    return [
        {"bucket": r[0], "prefix": r[1], "object_count": r[2], "total_bytes": r[3], "monthly_cost": r[4]} for r in rows
    ]


def _query_top_prefix50(conn: duckdb.DuckDBPyConnection, limit: int = 30) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            o.bucket,
            left(o.name, 50) as prefix50,
            COUNT(*) as object_count,
            SUM(o.size_bytes) as total_bytes,
            SUM({_ROW_COST}) as monthly_cost
        FROM objects o
        JOIN storage_classes sc ON o.storage_class_id = sc.id
        GROUP BY o.bucket, prefix50
        ORDER BY monthly_cost DESC
        LIMIT {limit}
    """
    ).fetchall()
    return [
        {"bucket": r[0], "prefix": r[1], "object_count": r[2], "total_bytes": r[3], "monthly_cost": r[4]} for r in rows
    ]


def _query_age_distribution(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            CASE
                WHEN created >= CURRENT_TIMESTAMP - INTERVAL '7 days' THEN '<7d'
                WHEN created >= CURRENT_TIMESTAMP - INTERVAL '30 days' THEN '7-30d'
                WHEN created >= CURRENT_TIMESTAMP - INTERVAL '90 days' THEN '30-90d'
                WHEN created >= CURRENT_TIMESTAMP - INTERVAL '365 days' THEN '90-365d'
                ELSE '>365d'
            END as age_bucket,
            COUNT(*) as object_count,
            SUM(o.size_bytes) as total_bytes,
            SUM({_ROW_COST}) as monthly_cost
        FROM objects o
        JOIN storage_classes sc ON o.storage_class_id = sc.id
        WHERE created IS NOT NULL
        GROUP BY age_bucket
        ORDER BY CASE age_bucket
            WHEN '<7d' THEN 1
            WHEN '7-30d' THEN 2
            WHEN '30-90d' THEN 3
            WHEN '90-365d' THEN 4
            ELSE 5
        END
    """
    ).fetchall()
    return [{"age_bucket": r[0], "object_count": r[1], "total_bytes": r[2], "monthly_cost": r[3]} for r in rows]


def _query_monthly_growth(conn: duckdb.DuckDBPyConnection, months: int = 12) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
            strftime(created, '%Y-%m') as month,
            COUNT(*) as object_count,
            SUM(size_bytes) as total_bytes
        FROM objects
        WHERE created IS NOT NULL
          AND created >= CURRENT_TIMESTAMP - INTERVAL '{months} months'
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
# Report generation
# ---------------------------------------------------------------------------


def generate_report(conn: duckdb.DuckDBPyConnection) -> str:
    """Generate a full storage usage report as markdown."""
    overview = _query_overview(conn)
    by_bucket = _query_by_bucket(conn)
    by_class = _query_by_storage_class(conn)
    top_dir1 = _query_top_dir1(conn)
    top_dir2 = _query_top_dir2(conn)
    top_p50 = _query_top_prefix50(conn)
    age_dist = _query_age_distribution(conn)
    monthly = _query_monthly_growth(conn)

    parts: list[str] = []

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts.append(f"# GCS Storage Report\n\nGenerated: {ts}\n")

    # Overview
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

    # By Bucket
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

    # By Storage Class
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

    # Top First-Level Directories
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

    # Top Two-Level Prefixes
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

    # Top 50-char Prefixes
    parts.append("## Top Prefixes (50-char grouping)\n")
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
                for r in top_p50
            ],
            align=["l", "l", "r", "r", "r"],
        )
    )

    # Age Distribution
    parts.append("## Age Distribution\n")
    parts.append(
        _md_table(
            ["Age", "Objects", "Size (TB)", "Monthly Cost"],
            [
                [r["age_bucket"], _fmt_count(r["object_count"]), _fmt_tb(r["total_bytes"]), _fmt_cost(r["monthly_cost"])]
                for r in age_dist
            ],
            align=["l", "r", "r", "r"],
        )
    )

    # Monthly Creation Trend
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
@click.option("--output", "-o", type=click.Path(), help="Write markdown report to file (default: stdout).")
def main(parquet_dir: str, output: str | None) -> None:
    """Generate a storage usage report from parquet output of a scan.

    PARQUET_DIR may be a local directory or a gs:// path (auto-downloaded
    to /tmp/storage-scan-cache via gcloud rsync).

    Examples:
        uv run scripts/storage/report.py gs://marin-us-central2/tmp/storage-scan-v7
        uv run scripts/storage/report.py ./local_parquet -o report.md
    """
    conn = load_parquet_db(parquet_dir)
    report = generate_report(conn)
    if output:
        Path(output).write_text(report)
        print(f"Report written to {output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
