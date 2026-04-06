#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""delete-o-tron: storage cost dashboard server.

Serves a Vue frontend and JSON API endpoints from the DuckDB object catalog
populated by the purge workflow.

Usage:
    uv run scripts/storage/delete_o_tron.py [--port 8000] [--dev]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import duckdb
import uvicorn
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from scripts.storage.storage_db import (
    GCS_DISCOUNT,
    IS_PROTECTED,
    OBJECTS_PARQUET_DIR,
    STORAGE_DB_PATH,
    continent_for_region,
    human_bytes,
    plan_rows,
    region_from_bucket,
)

log = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent / "dashboard"


def open_readonly_db() -> duckdb.DuckDBPyConnection:
    """Open the storage DuckDB in read-only mode for serving queries."""
    conn = duckdb.connect(str(STORAGE_DB_PATH), read_only=True)
    # Set up the objects view over existing parquet segments
    has_segments = any(OBJECTS_PARQUET_DIR.glob("objects_*.parquet"))
    if has_segments:
        glob = str(OBJECTS_PARQUET_DIR / "objects_*.parquet")
        conn.execute(
            f"""
            CREATE OR REPLACE VIEW objects AS
            SELECT * FROM read_parquet('{glob}', union_by_name=true, hive_partitioning=false)
        """
        )
    return conn


def _fetchall(conn: duckdb.DuckDBPyConnection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    result = conn.execute(sql, params)
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row, strict=False)) for row in result.fetchall()]


def _fetchone(conn: duckdb.DuckDBPyConnection, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    result = conn.execute(sql, params)
    row = result.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in result.description], row, strict=False))


def create_app() -> FastAPI:
    app = FastAPI(title="delete-o-tron", docs_url="/api/docs")
    db = open_readonly_db()

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        regions = []
        grand_total_objects = 0
        grand_total_bytes = 0
        grand_total_cost = 0.0

        for plan_row in plan_rows():
            region = plan_row["region"]
            bucket_name = plan_row["bucket"]
            continent = continent_for_region(region)
            price_col = "price_per_gib_month_us" if continent == "US" else "price_per_gib_month_eu"

            by_class = _fetchall(
                db,
                f"""
                SELECT sc.name as class,
                       COUNT(*) as objects,
                       COALESCE(SUM(o.size_bytes), 0) as bytes,
                       COALESCE(SUM(o.size_bytes), 0) / (1024.0*1024.0*1024.0)
                           * sc.{price_col} * ? as monthly_cost_usd
                FROM objects o
                JOIN storage_classes sc ON o.storage_class_id = sc.id
                WHERE o.bucket = ?
                GROUP BY sc.name
                ORDER BY sc.id
                """,
                (1.0 - GCS_DISCOUNT, bucket_name),
            )

            total_objects = sum(int(r["objects"]) for r in by_class)
            total_bytes = sum(int(r["bytes"]) for r in by_class)
            total_cost = sum(float(r["monthly_cost_usd"]) for r in by_class)

            regions.append(
                {
                    "region": region,
                    "bucket": bucket_name,
                    "continent": continent,
                    "total_objects": total_objects,
                    "total_bytes": total_bytes,
                    "total_bytes_human": human_bytes(total_bytes),
                    "total_monthly_cost_usd": round(total_cost, 2),
                    "by_storage_class": [
                        {
                            "class": r["class"],
                            "objects": int(r["objects"]),
                            "bytes": int(r["bytes"]),
                            "bytes_human": human_bytes(int(r["bytes"])),
                            "monthly_cost_usd": round(float(r["monthly_cost_usd"]), 2),
                        }
                        for r in by_class
                    ],
                }
            )

            grand_total_objects += total_objects
            grand_total_bytes += total_bytes
            grand_total_cost += total_cost

        return {
            "regions": regions,
            "totals": {
                "total_objects": grand_total_objects,
                "total_bytes": grand_total_bytes,
                "total_bytes_human": human_bytes(grand_total_bytes),
                "total_monthly_cost_usd": round(grand_total_cost, 2),
            },
            "discount": GCS_DISCOUNT,
        }

    @app.get("/api/savings")
    def savings() -> dict[str, Any]:
        regions = []
        grand_deletable_objects = 0
        grand_deletable_bytes = 0
        grand_savings = 0.0

        for plan_row in plan_rows():
            region = plan_row["region"]
            bucket_name = plan_row["bucket"]
            continent = continent_for_region(region)
            price_col = "price_per_gib_month_us" if continent == "US" else "price_per_gib_month_eu"

            by_class = _fetchall(
                db,
                f"""
                SELECT sc.name as class,
                       COUNT(*) as objects,
                       COALESCE(SUM(o.size_bytes), 0) as bytes,
                       COALESCE(SUM(o.size_bytes), 0) / (1024.0*1024.0*1024.0)
                           * sc.{price_col} * ? as monthly_cost_usd
                FROM objects o
                JOIN storage_classes sc ON o.storage_class_id = sc.id
                WHERE o.bucket = ?
                  AND sc.name != 'STANDARD'
                  AND NOT EXISTS ({IS_PROTECTED})
                GROUP BY sc.name
                ORDER BY sc.id
                """,
                (1.0 - GCS_DISCOUNT, bucket_name),
            )

            deletable_objects = sum(int(r["objects"]) for r in by_class)
            deletable_bytes = sum(int(r["bytes"]) for r in by_class)
            monthly_savings = sum(float(r["monthly_cost_usd"]) for r in by_class)

            regions.append(
                {
                    "region": region,
                    "deletable_objects": deletable_objects,
                    "deletable_bytes": deletable_bytes,
                    "deletable_bytes_human": human_bytes(deletable_bytes),
                    "monthly_savings_usd": round(monthly_savings, 2),
                    "by_storage_class": [
                        {
                            "class": r["class"],
                            "objects": int(r["objects"]),
                            "bytes": int(r["bytes"]),
                            "monthly_cost_usd": round(float(r["monthly_cost_usd"]), 2),
                        }
                        for r in by_class
                    ],
                }
            )

            grand_deletable_objects += deletable_objects
            grand_deletable_bytes += deletable_bytes
            grand_savings += monthly_savings

        return {
            "regions": regions,
            "totals": {
                "deletable_objects": grand_deletable_objects,
                "deletable_bytes": grand_deletable_bytes,
                "deletable_bytes_human": human_bytes(grand_deletable_bytes),
                "monthly_savings_usd": round(grand_savings, 2),
            },
        }

    @app.get("/api/rules")
    def rules() -> dict[str, Any]:
        raw_rules = _fetchall(
            db,
            """
            SELECT r.id, r.bucket, r.pattern, r.pattern_type, r.owners, r.reasons,
                   COALESCE(SUM(rc.object_count), 0) as total_objects,
                   COALESCE(SUM(rc.total_bytes), 0) as total_bytes,
                   COALESCE(SUM(rc.monthly_cost_usd), 0) as monthly_cost_usd
            FROM protect_rules r
            LEFT JOIN rule_costs rc ON rc.rule_id = r.id
            GROUP BY r.id, r.bucket, r.pattern, r.pattern_type, r.owners, r.reasons
            ORDER BY monthly_cost_usd DESC
            """,
        )

        result_rules = []
        for r in raw_rules:
            by_class = _fetchall(
                db,
                """
                SELECT sc.name as class,
                       rc.object_count as objects,
                       rc.total_bytes as bytes,
                       rc.monthly_cost_usd
                FROM rule_costs rc
                JOIN storage_classes sc ON rc.storage_class_id = sc.id
                WHERE rc.rule_id = ?
                ORDER BY sc.id
                """,
                (r["id"],),
            )
            result_rules.append(
                {
                    "id": r["id"],
                    "bucket": r["bucket"],
                    "region": region_from_bucket(r["bucket"]),
                    "pattern": r["pattern"],
                    "pattern_type": r["pattern_type"],
                    "owners": r["owners"] or "",
                    "reasons": r["reasons"] or "",
                    "total_objects": int(r["total_objects"]),
                    "total_bytes": int(r["total_bytes"]),
                    "total_bytes_human": human_bytes(int(r["total_bytes"])),
                    "monthly_cost_usd": round(float(r["monthly_cost_usd"]), 4),
                    "by_storage_class": [
                        {
                            "class": c["class"],
                            "objects": int(c["objects"]),
                            "bytes": int(c["bytes"]),
                            "monthly_cost_usd": round(float(c["monthly_cost_usd"]), 4),
                        }
                        for c in by_class
                    ],
                }
            )

        return {"rules": result_rules}

    @app.get("/api/rules/simulate")
    def simulate(exclude: list[int] = Query(default=[])) -> dict[str, Any]:
        """Compute savings if the given rule IDs were removed from the protect set.

        Objects currently protected only by excluded rules become deletable
        (unless they are STANDARD class).
        """
        if not exclude:
            return savings()

        # Build a temporary view of the remaining protect rules
        placeholders = ", ".join("?" * len(exclude))
        remaining_protected = f"""
            SELECT 1 FROM protect_rules r
            WHERE r.id NOT IN ({placeholders})
              AND o.bucket = r.bucket
              AND CASE r.pattern_type
                  WHEN 'prefix' THEN o.name LIKE r.pattern || '%'
                  WHEN 'like' THEN o.name LIKE r.pattern
              END
        """

        regions = []
        grand_deletable_objects = 0
        grand_deletable_bytes = 0
        grand_savings = 0.0

        for plan_row in plan_rows():
            region = plan_row["region"]
            bucket_name = plan_row["bucket"]
            continent = continent_for_region(region)
            price_col = "price_per_gib_month_us" if continent == "US" else "price_per_gib_month_eu"

            by_class = _fetchall(
                db,
                f"""
                SELECT sc.name as class,
                       COUNT(*) as objects,
                       COALESCE(SUM(o.size_bytes), 0) as bytes,
                       COALESCE(SUM(o.size_bytes), 0) / (1024.0*1024.0*1024.0)
                           * sc.{price_col} * ? as monthly_cost_usd
                FROM objects o
                JOIN storage_classes sc ON o.storage_class_id = sc.id
                WHERE o.bucket = ?
                  AND sc.name != 'STANDARD'
                  AND NOT EXISTS ({remaining_protected})
                GROUP BY sc.name
                ORDER BY sc.id
                """,
                (1.0 - GCS_DISCOUNT, bucket_name, *exclude),
            )

            deletable_objects = sum(int(r["objects"]) for r in by_class)
            deletable_bytes = sum(int(r["bytes"]) for r in by_class)
            monthly_savings = sum(float(r["monthly_cost_usd"]) for r in by_class)

            regions.append(
                {
                    "region": region,
                    "deletable_objects": deletable_objects,
                    "deletable_bytes": deletable_bytes,
                    "deletable_bytes_human": human_bytes(deletable_bytes),
                    "monthly_savings_usd": round(monthly_savings, 2),
                    "by_storage_class": [
                        {
                            "class": r["class"],
                            "objects": int(r["objects"]),
                            "bytes": int(r["bytes"]),
                            "monthly_cost_usd": round(float(r["monthly_cost_usd"]), 2),
                        }
                        for r in by_class
                    ],
                }
            )

            grand_deletable_objects += deletable_objects
            grand_deletable_bytes += deletable_bytes
            grand_savings += monthly_savings

        return {
            "regions": regions,
            "totals": {
                "deletable_objects": grand_deletable_objects,
                "deletable_bytes": grand_deletable_bytes,
                "deletable_bytes_human": human_bytes(grand_deletable_bytes),
                "monthly_savings_usd": round(grand_savings, 2),
            },
        }

    # Serve dashboard static files (no build step — plain JS + HTML)
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

    @app.get("/{path:path}")
    def spa_fallback(path: str) -> FileResponse:
        """Serve index.html for all non-API routes (SPA routing)."""
        return FileResponse(DASHBOARD_DIR / "index.html")

    return app


@click.command()
@click.option("--port", default=8000, show_default=True, type=int, help="Port to listen on.")
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind to.")
@click.option("--dev", is_flag=True, help="Enable auto-reload for development.")
def main(port: int, host: str, dev: bool) -> None:
    """Start the delete-o-tron dashboard server."""
    log_level = "info" if dev else "warning"
    if dev:
        uvicorn.run(
            "scripts.storage.delete_o_tron:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=["scripts/storage"],
            log_level=log_level,
        )
    else:
        app = create_app()
        uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
