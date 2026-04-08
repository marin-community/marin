#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage dashboard server.

Serves a Vue frontend and JSON API endpoints from the DuckDB object catalog
populated by the purge workflow.

Usage:
    uv run scripts/storage/server.py [--port 8000] [--dev]
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import logging
import os
import secrets
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import duckdb
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse, JSONResponse


from scripts.storage.db import (
    DEFAULT_CATALOG,
    GCS_DISCOUNT,
    StorageCatalog,
    _next_rule_id,
    continent_for_region,
    flush_delete_rules,
    flush_protect_rules,
    human_bytes,
    init_db,
    plan_rows,
    region_from_bucket,
)

log = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent / "dashboard"

SYNC_INTERVAL = 600  # 10 minutes

_db_lock = threading.RLock()


class _LoginRequest(BaseModel):
    password: str


class DeletePatternsRequest(BaseModel):
    patterns: list[str]


class DeleteRuleCreate(BaseModel):
    pattern: str
    storage_class: str | None = None
    description: str | None = None


class ProtectRuleCreate(BaseModel):
    bucket: str
    pattern: str
    owners: str | None = None
    reasons: str | None = None


@dataclass
class SyncState:
    last_sync: str | None = None
    syncing: bool = False
    error: str | None = None


_sync_state = SyncState()


def open_db(catalog: StorageCatalog) -> duckdb.DuckDBPyConnection:
    return init_db(catalog)


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


STATUS_ORDER = {"delete": 0, "mixed": 1, "keep": 2}


def _run_sync(gcs_prefix: str, catalog: StorageCatalog) -> None:
    """Upload rules JSON files to GCS and write a timestamped archive copy."""
    global _sync_state
    try:
        _sync_state.syncing = True
        _sync_state.error = None
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        for name in ("protect_rules.json", "delete_rules.json"):
            local = catalog.root / name
            if local.exists():
                subprocess.run(
                    ["gcloud", "storage", "cp", str(local), f"{gcs_prefix}/{name}"],
                    check=True,
                    capture_output=True,
                )
                stem = name.replace(".json", "")
                subprocess.run(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        str(local),
                        f"{gcs_prefix}/archive/{stem}_{ts}.json",
                    ],
                    check=True,
                    capture_output=True,
                )
        _sync_state.last_sync = datetime.now(UTC).isoformat()
    except Exception as e:
        log.exception("sync failed")
        _sync_state.error = str(e)
    finally:
        _sync_state.syncing = False


async def _periodic_sync(gcs_prefix: str, catalog: StorageCatalog) -> None:
    """Background task: sync rules JSON to GCS every 10 minutes with a timestamped archive copy."""
    while True:
        await asyncio.sleep(SYNC_INTERVAL)
        _run_sync(gcs_prefix, catalog)


ALL_SC_COLS: list[tuple[str, str, str]] = [
    ("STANDARD", "standard_count", "standard_bytes"),
    ("NEARLINE", "nearline_count", "nearline_bytes"),
    ("COLDLINE", "coldline_count", "coldline_bytes"),
    ("ARCHIVE", "archive_count", "archive_bytes"),
]


def _query_explore_segments(
    conn: duckdb.DuckDBPyConnection,
    prefix_len: int,
    prefix: str,
    bucket: str | None,
    storage_class: str | None,
) -> list[dict[str, Any]]:
    """Query dir_summary segments with inline keep/delete status.

    Computes status at query time by joining against the small protect_rules
    and delete_rules tables. No materialized status table needed.

    Status logic:
      - protected dir_prefixes (matched by any protect_rule) → keep
      - unprotected dir_prefixes matched by a delete_rule → delete
      - everything else → keep (unmatched is implicitly keep)
    Segment-level: all delete → delete, any delete → mixed, else → keep.
    """
    sc_cols = [c for c in ALL_SC_COLS if storage_class is None or c[0] == storage_class]
    sum_exprs = ", ".join(f"SUM(d.{cc}) as {cc}, SUM(d.{bc}) as {bc}" for _, cc, bc in sc_cols)

    # Build WHERE clause with optional bucket filter
    where = "d.dir_prefix LIKE ? || '%'"
    params: list[Any] = [prefix]
    if bucket:
        where += " AND d.bucket = ?"
        params.append(bucket)

    # Storage class filter for delete rules.
    # Without a class filter, only match rules that delete ALL classes
    # (storage_class IS NULL). Class-specific rules are only relevant
    # when the user filters by that class.
    if storage_class:
        sc_filter = "AND (dr.storage_class IS NULL OR dr.storage_class = ?)"
        sc_params: list[Any] = [storage_class]
    else:
        sc_filter = "AND dr.storage_class IS NULL"
        sc_params = []

    # Bucket filter repeated for CTEs
    cte_bucket = "AND d.bucket = ?" if bucket else ""
    cte_bucket_params = [bucket] if bucket else []

    query = f"""
        WITH protected AS (
            SELECT DISTINCT d.bucket, d.dir_prefix
            FROM protect_rules p
            JOIN dir_summary d
                ON d.dir_prefix LIKE p.pattern
                AND (d.bucket = p.bucket OR p.bucket = '*')
            WHERE d.dir_prefix LIKE ? || '%' {cte_bucket}
        ),
        deleted AS (
            SELECT DISTINCT d.bucket, d.dir_prefix
            FROM delete_rules dr
            JOIN dir_summary d ON d.dir_prefix LIKE dr.pattern
            LEFT JOIN protected p USING (bucket, dir_prefix)
            WHERE d.dir_prefix LIKE ? || '%' {cte_bucket}
              AND p.dir_prefix IS NULL
              {sc_filter}
        )
        SELECT
            split_part(substr(d.dir_prefix, ?), '/', 1) AS segment,
            {sum_exprs},
            CASE
                WHEN COUNT(del.dir_prefix) = COUNT(*) THEN 'delete'
                WHEN COUNT(del.dir_prefix) > 0 THEN 'mixed'
                ELSE 'keep'
            END AS status,
            SUM(CASE WHEN del.dir_prefix IS NULL
                THEN d.standard_bytes + d.nearline_bytes + d.coldline_bytes + d.archive_bytes
                ELSE 0 END) AS kept_bytes,
            SUM(d.standard_bytes + d.nearline_bytes + d.coldline_bytes + d.archive_bytes) AS total_bytes
        FROM dir_summary d
        LEFT JOIN deleted del USING (bucket, dir_prefix)
        WHERE {where}
        GROUP BY segment
        ORDER BY segment
    """

    all_params = [
        prefix,
        *cte_bucket_params,  # protected CTE
        prefix,
        *cte_bucket_params,  # deleted CTE
        *sc_params,  # storage class filter
        prefix_len,  # split_part offset
        *params,  # main WHERE
    ]
    return _fetchall(conn, query, tuple(all_params))


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

# Tokens are HMAC-SHA256(secret_key, token_nonce). The secret_key is derived
# from DASHBOARD_PASSWORD so tokens survive server restarts.
_secret_key: bytes = hashlib.sha256(os.environ.get("DASHBOARD_PASSWORD", "").encode()).digest()


def _make_token() -> str:
    nonce = secrets.token_hex(32)
    sig = hmac.new(_secret_key, nonce.encode(), hashlib.sha256).hexdigest()
    return f"{nonce}.{sig}"


def _verify_token(token: str) -> bool:
    if not token:
        return False
    try:
        nonce, sig = token.rsplit(".", 1)
    except ValueError:
        return False
    expected = hmac.new(_secret_key, nonce.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def create_app(catalog: StorageCatalog = DEFAULT_CATALOG) -> FastAPI:
    app = FastAPI(title="storage-dashboard", docs_url="/api/docs")
    _db: duckdb.DuckDBPyConnection | None = None
    _background_tasks: set[asyncio.Task] = set()

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path != "/api/login":
            auth = request.headers.get("Authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if not _verify_token(token):
                log.warning("401 Unauthorized: %s %s (token=%r)", request.method, path, token[:8] if token else "")
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    @app.post("/api/login")
    def login(body: _LoginRequest) -> dict[str, str]:
        password = os.environ.get("DASHBOARD_PASSWORD", "")
        if not hmac.compare_digest(body.password, password):
            raise HTTPException(status_code=401, detail="Incorrect password")
        return {"token": _make_token()}

    def db() -> duckdb.DuckDBPyConnection:
        if _db is None:
            raise RuntimeError("DB not initialized yet")
        return _db

    def serialized(fn):
        """Serialize DB access through a single lock.

        DuckDB cursors from the same connection share internal state and are not
        safe under concurrent access from FastAPI's threadpool.
        """

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _db_lock:
                return fn(*args, **kwargs)

        return wrapper

    @app.on_event("startup")
    async def _download_data() -> None:
        """Sync all data from GCS at startup."""
        gcs_prefix = os.environ.get("GCS_DATA_PREFIX")
        if not gcs_prefix:
            return
        for name in ("protect_rules.json", "delete_rules.json"):
            local = catalog.root / name
            log.info("downloading %s from %s", name, gcs_prefix)
            try:
                subprocess.run(
                    ["gcloud", "storage", "cp", f"{gcs_prefix}/{name}", str(local)],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                log.warning("could not download %s — starting with empty rules", name)
        for name, local_dir in [
            ("objects_parquet", catalog.objects_parquet_dir),
            ("dir_summary_parquet", catalog.dir_summary_parquet_dir),
        ]:
            local_dir.mkdir(parents=True, exist_ok=True)
            log.info("syncing %s from %s", name, gcs_prefix)
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    "--recursive",
                    f"{gcs_prefix}/{name}/",
                    str(local_dir) + "/",
                ],
                check=True,
                capture_output=True,
            )

    def _cache_dir_summary(conn: duckdb.DuckDBPyConnection) -> None:
        """Load the parquet-backed dir_summary view into a temp table for fast queries.

        Restores the parquet view first in case a previous server run left it
        pointing at a temp table that no longer exists.
        """
        if catalog.dir_summary_parquet_dir.exists() and any(
            catalog.dir_summary_parquet_dir.glob("dir_summary_*.parquet")
        ):
            parquet_glob = str(catalog.dir_summary_parquet_dir / "dir_summary_*.parquet")
            conn.execute(
                f"CREATE OR REPLACE VIEW dir_summary AS "
                f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true, hive_partitioning=false)"
            )
        try:
            row_count = conn.execute("SELECT COUNT(*) FROM dir_summary").fetchone()[0]
        except duckdb.CatalogException:
            return
        if row_count == 0:
            return
        conn.execute("CREATE TEMPORARY TABLE _dir_summary_mem AS SELECT * FROM dir_summary")
        conn.execute("CREATE OR REPLACE VIEW dir_summary AS SELECT * FROM _dir_summary_mem")
        log.info("cached %d dir_summary rows into memory", row_count)

    @app.on_event("startup")
    async def _init_db() -> None:
        nonlocal _db
        _db = open_db(catalog)
        _cache_dir_summary(_db)

    @app.on_event("startup")
    async def _start_sync() -> None:
        gcs_prefix = os.environ.get("GCS_DATA_PREFIX")
        if gcs_prefix:
            assert _db is not None
            task = asyncio.create_task(_periodic_sync(gcs_prefix, catalog))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

    @app.get("/api/overview")
    @serialized
    def overview() -> dict[str, Any]:
        regions = []
        grand_total_objects = 0
        grand_total_bytes = 0
        grand_total_cost = 0.0
        grand_protected_objects = 0
        grand_protected_bytes = 0
        grand_protected_cost = 0.0

        # Fetch prices once
        prices_us = {
            r["name"]: float(r["price"])
            for r in _fetchall(db(), "SELECT name, price_per_gib_month_us as price FROM storage_classes")
        }
        prices_eu = {
            r["name"]: float(r["price"])
            for r in _fetchall(db(), "SELECT name, price_per_gib_month_eu as price FROM storage_classes")
        }

        for plan_row in plan_rows():
            region = plan_row["region"]
            bucket_name = plan_row["bucket"]
            continent = continent_for_region(region)
            prices = prices_us if continent == "US" else prices_eu
            discount_factor = 1.0 - GCS_DISCOUNT

            totals = _fetchone(
                db(),
                """
                SELECT SUM(standard_count) as standard_count, SUM(standard_bytes) as standard_bytes,
                       SUM(nearline_count) as nearline_count, SUM(nearline_bytes) as nearline_bytes,
                       SUM(coldline_count) as coldline_count, SUM(coldline_bytes) as coldline_bytes,
                       SUM(archive_count) as archive_count, SUM(archive_bytes) as archive_bytes
                FROM dir_summary WHERE bucket = ?
                """,
                (bucket_name,),
            )

            if not totals or totals["standard_count"] is None:
                continue

            protected_row = _fetchone(
                db(),
                """
                SELECT COALESCE(SUM(d.standard_count), 0) as p_standard_count,
                       COALESCE(SUM(d.standard_bytes), 0) as p_standard_bytes,
                       COALESCE(SUM(d.nearline_count), 0) as p_nearline_count,
                       COALESCE(SUM(d.nearline_bytes), 0) as p_nearline_bytes,
                       COALESCE(SUM(d.coldline_count), 0) as p_coldline_count,
                       COALESCE(SUM(d.coldline_bytes), 0) as p_coldline_bytes,
                       COALESCE(SUM(d.archive_count), 0) as p_archive_count,
                       COALESCE(SUM(d.archive_bytes), 0) as p_archive_bytes
                FROM protect_rules r
                JOIN dir_summary d ON d.dir_prefix LIKE r.pattern
                    AND (d.bucket = r.bucket OR r.bucket = '*')
                WHERE d.bucket = ?
                """,
                (bucket_name,),
            )

            by_class = []
            total_objects = 0
            total_bytes = 0
            total_cost = 0.0
            protected_objects = 0
            protected_bytes = 0
            protected_cost = 0.0
            for sc_name, count_col, bytes_col in ALL_SC_COLS:
                cnt = int(totals[count_col] or 0)
                bts = int(totals[bytes_col] or 0)
                cost = bts / (1024.0 * 1024.0 * 1024.0) * prices[sc_name] * discount_factor
                if cnt > 0:
                    by_class.append(
                        {
                            "class": sc_name,
                            "objects": cnt,
                            "bytes": bts,
                            "bytes_human": human_bytes(bts),
                            "monthly_cost_usd": round(cost, 2),
                        }
                    )
                total_objects += cnt
                total_bytes += bts
                total_cost += cost

                p_cnt = int(protected_row[f"p_{count_col}"] or 0) if protected_row else 0
                p_bts = int(protected_row[f"p_{bytes_col}"] or 0) if protected_row else 0
                protected_objects += p_cnt
                protected_bytes += p_bts
                protected_cost += p_bts / (1024.0 * 1024.0 * 1024.0) * prices[sc_name] * discount_factor

            deletable_objects = total_objects - protected_objects
            deletable_bytes = total_bytes - protected_bytes
            deletable_cost = total_cost - protected_cost

            regions.append(
                {
                    "region": region,
                    "bucket": bucket_name,
                    "continent": continent,
                    "total_objects": total_objects,
                    "total_bytes": total_bytes,
                    "total_bytes_human": human_bytes(total_bytes),
                    "total_monthly_cost_usd": round(total_cost, 2),
                    "protected_objects": protected_objects,
                    "protected_bytes": protected_bytes,
                    "protected_bytes_human": human_bytes(protected_bytes),
                    "protected_monthly_cost_usd": round(protected_cost, 2),
                    "deletable_objects": deletable_objects,
                    "deletable_bytes": deletable_bytes,
                    "deletable_bytes_human": human_bytes(deletable_bytes),
                    "deletable_monthly_cost_usd": round(deletable_cost, 2),
                    "by_storage_class": by_class,
                }
            )

            grand_total_objects += total_objects
            grand_total_bytes += total_bytes
            grand_total_cost += total_cost
            grand_protected_objects += protected_objects
            grand_protected_bytes += protected_bytes
            grand_protected_cost += protected_cost

        grand_deletable_objects = grand_total_objects - grand_protected_objects
        grand_deletable_bytes = grand_total_bytes - grand_protected_bytes
        grand_deletable_cost = grand_total_cost - grand_protected_cost

        return {
            "regions": regions,
            "totals": {
                "total_objects": grand_total_objects,
                "total_bytes": grand_total_bytes,
                "total_bytes_human": human_bytes(grand_total_bytes),
                "total_monthly_cost_usd": round(grand_total_cost, 2),
                "protected_objects": grand_protected_objects,
                "protected_bytes": grand_protected_bytes,
                "protected_bytes_human": human_bytes(grand_protected_bytes),
                "protected_monthly_cost_usd": round(grand_protected_cost, 2),
                "deletable_objects": grand_deletable_objects,
                "deletable_bytes": grand_deletable_bytes,
                "deletable_bytes_human": human_bytes(grand_deletable_bytes),
                "deletable_monthly_cost_usd": round(grand_deletable_cost, 2),
            },
            "discount": GCS_DISCOUNT,
        }

    @app.get("/api/savings")
    @serialized
    def savings() -> dict[str, Any]:
        """Derive deletable = total - protected from dir_summary and protect_rules."""
        regions = []
        grand_deletable_objects = 0
        grand_deletable_bytes = 0
        grand_savings = 0.0

        prices_us = {
            r["name"]: float(r["price"])
            for r in _fetchall(db(), "SELECT name, price_per_gib_month_us as price FROM storage_classes")
        }
        prices_eu = {
            r["name"]: float(r["price"])
            for r in _fetchall(db(), "SELECT name, price_per_gib_month_eu as price FROM storage_classes")
        }

        for plan_row in plan_rows():
            region = plan_row["region"]
            bucket_name = plan_row["bucket"]
            continent = continent_for_region(region)
            prices = prices_us if continent == "US" else prices_eu
            discount_factor = 1.0 - GCS_DISCOUNT

            totals = _fetchone(
                db(),
                """
                SELECT SUM(standard_count) as standard_count, SUM(standard_bytes) as standard_bytes,
                       SUM(nearline_count) as nearline_count, SUM(nearline_bytes) as nearline_bytes,
                       SUM(coldline_count) as coldline_count, SUM(coldline_bytes) as coldline_bytes,
                       SUM(archive_count) as archive_count, SUM(archive_bytes) as archive_bytes
                FROM dir_summary WHERE bucket = ?
                """,
                (bucket_name,),
            )

            protected_row = _fetchone(
                db(),
                """
                SELECT COALESCE(SUM(d.standard_count), 0) as p_standard_count,
                       COALESCE(SUM(d.standard_bytes), 0) as p_standard_bytes,
                       COALESCE(SUM(d.nearline_count), 0) as p_nearline_count,
                       COALESCE(SUM(d.nearline_bytes), 0) as p_nearline_bytes,
                       COALESCE(SUM(d.coldline_count), 0) as p_coldline_count,
                       COALESCE(SUM(d.coldline_bytes), 0) as p_coldline_bytes,
                       COALESCE(SUM(d.archive_count), 0) as p_archive_count,
                       COALESCE(SUM(d.archive_bytes), 0) as p_archive_bytes
                FROM protect_rules r
                JOIN dir_summary d ON d.dir_prefix LIKE r.pattern
                    AND (d.bucket = r.bucket OR r.bucket = '*')
                WHERE d.bucket = ?
                """,
                (bucket_name,),
            )

            if not totals:
                continue

            by_class = []
            deletable_objects = 0
            deletable_bytes = 0
            monthly_savings = 0.0
            for sc_name, count_col, bytes_col in [
                ("STANDARD", "standard_count", "standard_bytes"),
                ("NEARLINE", "nearline_count", "nearline_bytes"),
                ("COLDLINE", "coldline_count", "coldline_bytes"),
                ("ARCHIVE", "archive_count", "archive_bytes"),
            ]:
                total_cnt = int(totals[count_col] or 0)
                total_bts = int(totals[bytes_col] or 0)
                p_cnt = int(protected_row[f"p_{count_col}"] or 0) if protected_row else 0
                p_bts = int(protected_row[f"p_{bytes_col}"] or 0) if protected_row else 0
                d_cnt = total_cnt - p_cnt
                d_bts = total_bts - p_bts
                if d_cnt <= 0:
                    continue
                cost = d_bts / (1024.0 * 1024.0 * 1024.0) * prices[sc_name] * discount_factor
                by_class.append(
                    {
                        "class": sc_name,
                        "objects": d_cnt,
                        "bytes": d_bts,
                        "monthly_cost_usd": round(cost, 2),
                    }
                )
                deletable_objects += d_cnt
                deletable_bytes += d_bts
                monthly_savings += cost

            regions.append(
                {
                    "region": region,
                    "deletable_objects": deletable_objects,
                    "deletable_bytes": deletable_bytes,
                    "deletable_bytes_human": human_bytes(deletable_bytes),
                    "monthly_savings_usd": round(monthly_savings, 2),
                    "by_storage_class": by_class,
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

    @app.get("/api/buckets")
    def list_buckets() -> dict[str, Any]:
        return {"buckets": plan_rows()}

    @app.get("/api/rules")
    @serialized
    def list_protect_rules() -> dict[str, Any]:
        rows = _fetchall(
            db(),
            "SELECT id, bucket, pattern, pattern_type, owners, reasons FROM protect_rules ORDER BY pattern ASC",
        )
        return {"rules": rows}

    @app.get("/api/rules/{rule_id}/cost")
    @serialized
    def get_protect_rule_cost(rule_id: int) -> dict[str, Any]:
        conn = db()
        rule = _fetchone(conn, "SELECT id FROM protect_rules WHERE id = ?", (rule_id,))
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")

        prices_us = {
            r["name"]: float(r["price"])
            for r in _fetchall(conn, "SELECT name, price_per_gib_month_us as price FROM storage_classes")
        }
        prices_eu = {
            r["name"]: float(r["price"])
            for r in _fetchall(conn, "SELECT name, price_per_gib_month_eu as price FROM storage_classes")
        }
        discount_factor = 1.0 - GCS_DISCOUNT

        raw = _fetchall(
            conn,
            """
            SELECT d.bucket as data_bucket,
                   SUM(d.standard_count) as standard_count, SUM(d.standard_bytes) as standard_bytes,
                   SUM(d.nearline_count) as nearline_count, SUM(d.nearline_bytes) as nearline_bytes,
                   SUM(d.coldline_count) as coldline_count, SUM(d.coldline_bytes) as coldline_bytes,
                   SUM(d.archive_count) as archive_count, SUM(d.archive_bytes) as archive_bytes
            FROM protect_rules r
            LEFT JOIN dir_summary d ON d.dir_prefix LIKE r.pattern
                AND (d.bucket = r.bucket OR r.bucket = '*')
            WHERE r.id = ?
            GROUP BY d.bucket
            """,
            (rule_id,),
        )

        total_objects = 0
        total_bytes = 0
        total_cost = 0.0
        by_storage_class: dict[str, dict[str, int | float]] = {}

        for row in raw:
            data_bucket = row["data_bucket"]
            if data_bucket is None:
                continue
            continent = continent_for_region(region_from_bucket(data_bucket))
            prices = prices_us if continent == "US" else prices_eu

            for sc_name, count_col, bytes_col in ALL_SC_COLS:
                cnt = int(row[count_col] or 0)
                bts = int(row[bytes_col] or 0)
                if cnt == 0:
                    continue
                cost = bts / (1024.0 * 1024.0 * 1024.0) * prices[sc_name] * discount_factor
                total_objects += cnt
                total_bytes += bts
                total_cost += cost

                if sc_name not in by_storage_class:
                    by_storage_class[sc_name] = {"class": sc_name, "objects": 0, "bytes": 0, "monthly_cost_usd": 0.0}
                by_storage_class[sc_name]["objects"] += cnt
                by_storage_class[sc_name]["bytes"] += bts
                by_storage_class[sc_name]["monthly_cost_usd"] += cost

        return {
            "total_objects": total_objects,
            "total_bytes": total_bytes,
            "total_bytes_human": human_bytes(total_bytes),
            "monthly_cost_usd": round(total_cost, 4),
            "by_storage_class": [
                {**v, "monthly_cost_usd": round(v["monthly_cost_usd"], 4)} for v in by_storage_class.values()
            ],
        }

    @app.post("/api/rules")
    @serialized
    def create_protect_rule(req: ProtectRuleCreate) -> dict[str, Any]:
        conn = db()
        new_id = _next_rule_id(conn, "protect_rules")
        import json as _json

        path = catalog.protect_rules_json
        rules = _json.loads(path.read_text()) if path.exists() else []
        new_rule = {
            "id": new_id,
            "bucket": req.bucket,
            "pattern": req.pattern,
            "pattern_type": "like",
            "owners": req.owners,
            "reasons": req.reasons,
            "sources": None,
        }
        rules.append(new_rule)
        path.write_text(_json.dumps(rules, indent=2) + "\n")
        flush_protect_rules(conn, catalog)
        return new_rule

    @app.delete("/api/rules/{rule_id}")
    @serialized
    def remove_protect_rule(rule_id: int) -> dict[str, Any]:
        import json as _json

        conn = db()
        path = catalog.protect_rules_json
        rules = _json.loads(path.read_text()) if path.exists() else []
        rules = [r for r in rules if r["id"] != rule_id]
        path.write_text(_json.dumps(rules, indent=2) + "\n")
        flush_protect_rules(conn, catalog)
        return {"deleted": rule_id}

    @app.get("/api/rules/simulate")
    @serialized
    def simulate(exclude: list[int] = Query(default=[])) -> dict[str, Any]:
        """Estimate savings if the given rule IDs were removed from the protect set.

        Uses an approximation: adds the excluded rules' costs to the baseline
        savings. This double-counts objects protected by multiple rules but
        gives a fast upper-bound estimate without scanning the full objects table.
        """
        if not exclude:
            return savings()

        baseline = savings()
        placeholders = ", ".join("?" * len(exclude))

        prices_us = {
            r["name"]: float(r["price"])
            for r in _fetchall(db(), "SELECT name, price_per_gib_month_us as price FROM storage_classes")
        }
        prices_eu = {
            r["name"]: float(r["price"])
            for r in _fetchall(db(), "SELECT name, price_per_gib_month_eu as price FROM storage_classes")
        }
        discount_factor = 1.0 - GCS_DISCOUNT

        excluded_costs = _fetchall(
            db(),
            f"""
            SELECT r.bucket as rule_bucket, d.bucket,
                   SUM(d.standard_count) as standard_count, SUM(d.standard_bytes) as standard_bytes,
                   SUM(d.nearline_count) as nearline_count, SUM(d.nearline_bytes) as nearline_bytes,
                   SUM(d.coldline_count) as coldline_count, SUM(d.coldline_bytes) as coldline_bytes,
                   SUM(d.archive_count) as archive_count, SUM(d.archive_bytes) as archive_bytes
            FROM protect_rules r
            JOIN dir_summary d ON d.dir_prefix LIKE r.pattern
                AND (d.bucket = r.bucket OR r.bucket = '*')
            WHERE r.id IN ({placeholders})
            GROUP BY r.bucket, d.bucket
            """,
            tuple(exclude),
        )

        extra_by_bucket: dict[str, list[dict[str, Any]]] = {}
        for row in excluded_costs:
            bucket = row["bucket"]
            continent = continent_for_region(region_from_bucket(bucket))
            prices = prices_us if continent == "US" else prices_eu
            for sc_name, count_col, bytes_col in ALL_SC_COLS:
                cnt = int(row[count_col] or 0)
                bts = int(row[bytes_col] or 0)
                if cnt == 0:
                    continue
                cost = bts / (1024.0 * 1024.0 * 1024.0) * prices[sc_name] * discount_factor
                extra_by_bucket.setdefault(bucket, []).append(
                    {
                        "class": sc_name,
                        "objects": cnt,
                        "bytes": bts,
                        "monthly_cost_usd": round(cost, 4),
                    }
                )

        # Add excluded rule costs to baseline savings
        regions = []
        grand_deletable_objects = 0
        grand_deletable_bytes = 0
        grand_savings = 0.0

        for rs in baseline["regions"]:
            extras = extra_by_bucket.get(next((p["bucket"] for p in plan_rows() if p["region"] == rs["region"]), ""), [])
            extra_objects = sum(int(e["objects"]) for e in extras)
            extra_bytes = sum(int(e["bytes"]) for e in extras)
            extra_cost = sum(float(e["monthly_cost_usd"]) for e in extras)

            deletable_objects = rs["deletable_objects"] + extra_objects
            deletable_bytes = rs["deletable_bytes"] + extra_bytes
            monthly_savings = rs["monthly_savings_usd"] + extra_cost

            by_class = list(rs["by_storage_class"])
            existing_classes = {c["class"] for c in by_class}
            for e in extras:
                if e["class"] in existing_classes:
                    for c in by_class:
                        if c["class"] == e["class"]:
                            c["objects"] += int(e["objects"])
                            c["bytes"] += int(e["bytes"])
                            c["monthly_cost_usd"] = round(c["monthly_cost_usd"] + float(e["monthly_cost_usd"]), 2)
                else:
                    by_class.append(
                        {
                            "class": e["class"],
                            "objects": int(e["objects"]),
                            "bytes": int(e["bytes"]),
                            "monthly_cost_usd": round(float(e["monthly_cost_usd"]), 2),
                        }
                    )

            regions.append(
                {
                    "region": rs["region"],
                    "deletable_objects": deletable_objects,
                    "deletable_bytes": deletable_bytes,
                    "deletable_bytes_human": human_bytes(deletable_bytes),
                    "monthly_savings_usd": round(monthly_savings, 2),
                    "by_storage_class": by_class,
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

    @app.get("/api/delete-rules")
    @serialized
    def list_delete_rules() -> dict[str, Any]:
        rows = _fetchall(
            db(),
            "SELECT id, pattern, storage_class, description, created_at FROM delete_rules ORDER BY pattern ASC",
        )
        return {"rules": rows}

    @app.get("/api/delete-rules/{rule_id}/cost")
    @serialized
    def get_delete_rule_cost(rule_id: int) -> dict[str, Any]:
        conn = db()
        rule = _fetchone(conn, "SELECT id, storage_class FROM delete_rules WHERE id = ?", (rule_id,))
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")

        rule_sc = rule["storage_class"]

        prices_us = {
            r["name"]: float(r["price"])
            for r in _fetchall(conn, "SELECT name, price_per_gib_month_us as price FROM storage_classes")
        }
        prices_eu = {
            r["name"]: float(r["price"])
            for r in _fetchall(conn, "SELECT name, price_per_gib_month_eu as price FROM storage_classes")
        }
        discount_factor = 1.0 - GCS_DISCOUNT

        raw = _fetchall(
            conn,
            """
            SELECT d.bucket as data_bucket,
                   SUM(d.standard_count) as standard_count, SUM(d.standard_bytes) as standard_bytes,
                   SUM(d.nearline_count) as nearline_count, SUM(d.nearline_bytes) as nearline_bytes,
                   SUM(d.coldline_count) as coldline_count, SUM(d.coldline_bytes) as coldline_bytes,
                   SUM(d.archive_count) as archive_count, SUM(d.archive_bytes) as archive_bytes
            FROM delete_rules dr
            LEFT JOIN dir_summary d ON d.dir_prefix LIKE dr.pattern
            LEFT JOIN protect_rules p ON d.dir_prefix LIKE p.pattern
                AND (d.bucket = p.bucket OR p.bucket = '*')
            WHERE dr.id = ? AND p.id IS NULL
            GROUP BY d.bucket
            """,
            (rule_id,),
        )

        total_objects = 0
        total_bytes = 0
        total_cost = 0.0
        by_storage_class: dict[str, dict[str, int | float]] = {}

        for row in raw:
            data_bucket = row["data_bucket"]
            if data_bucket is None:
                continue
            continent = continent_for_region(region_from_bucket(data_bucket))
            prices = prices_us if continent == "US" else prices_eu

            for sc_name, count_col, bytes_col in ALL_SC_COLS:
                if rule_sc and rule_sc != sc_name:
                    continue
                cnt = int(row[count_col] or 0)
                bts = int(row[bytes_col] or 0)
                if cnt == 0:
                    continue
                cost = bts / (1024.0 * 1024.0 * 1024.0) * prices[sc_name] * discount_factor
                total_objects += cnt
                total_bytes += bts
                total_cost += cost

                if sc_name not in by_storage_class:
                    by_storage_class[sc_name] = {"class": sc_name, "objects": 0, "bytes": 0, "monthly_cost_usd": 0.0}
                by_storage_class[sc_name]["objects"] += cnt
                by_storage_class[sc_name]["bytes"] += bts
                by_storage_class[sc_name]["monthly_cost_usd"] += cost

        return {
            "total_objects": total_objects,
            "total_bytes": total_bytes,
            "total_bytes_human": human_bytes(total_bytes),
            "monthly_cost_usd": round(total_cost, 4),
            "by_storage_class": [
                {**v, "monthly_cost_usd": round(v["monthly_cost_usd"], 4)} for v in by_storage_class.values()
            ],
        }

    @app.post("/api/delete-rules")
    @serialized
    def create_delete_rule(req: DeleteRuleCreate) -> dict[str, Any]:
        import json as _json

        conn = db()
        new_id = _next_rule_id(conn, "delete_rules")
        now = datetime.now(UTC).isoformat()
        path = catalog.delete_rules_json
        rules = _json.loads(path.read_text()) if path.exists() else []
        new_rule = {
            "id": new_id,
            "pattern": req.pattern,
            "storage_class": req.storage_class,
            "description": req.description,
            "created_at": now,
        }
        rules.append(new_rule)
        path.write_text(_json.dumps(rules, indent=2) + "\n")
        flush_delete_rules(conn, catalog)
        return new_rule

    @app.delete("/api/delete-rules/{rule_id}")
    @serialized
    def remove_delete_rule(rule_id: int) -> dict[str, Any]:
        import json as _json

        conn = db()
        path = catalog.delete_rules_json
        rules = _json.loads(path.read_text()) if path.exists() else []
        rules = [r for r in rules if r["id"] != rule_id]
        path.write_text(_json.dumps(rules, indent=2) + "\n")
        flush_delete_rules(conn, catalog)
        return {"deleted": rule_id}

    @app.post("/api/delete-patterns/estimate")
    @serialized
    def estimate_delete_patterns(req: DeletePatternsRequest) -> dict[str, Any]:
        """Estimate cost of objects matching the given LIKE patterns across all buckets.

        Patterns are matched against dir_prefix in dir_summary. Each pattern
        should use SQL LIKE syntax (% wildcard). Patterns without a wildcard
        get % appended automatically.
        """
        if not req.patterns:
            return {"patterns": [], "totals": {"objects": 0, "bytes": 0, "monthly_cost_usd": 0}}

        # Normalize patterns
        like_patterns = [p if "%" in p else p + "%" for p in req.patterns]

        prices_us = {
            r["name"]: float(r["price"])
            for r in _fetchall(db(), "SELECT name, price_per_gib_month_us as price FROM storage_classes")
        }
        prices_eu = {
            r["name"]: float(r["price"])
            for r in _fetchall(db(), "SELECT name, price_per_gib_month_eu as price FROM storage_classes")
        }

        pattern_results = []
        for pattern in like_patterns:
            pattern_total_objects = 0
            pattern_total_bytes = 0
            pattern_total_cost = 0.0
            pattern_by_region: list[dict[str, Any]] = []

            for plan_row in plan_rows():
                region = plan_row["region"]
                bucket_name = plan_row["bucket"]
                continent = continent_for_region(region)
                prices = prices_us if continent == "US" else prices_eu
                discount_factor = 1.0 - GCS_DISCOUNT

                row = _fetchone(
                    db(),
                    """
                    SELECT SUM(standard_count) as standard_count, SUM(standard_bytes) as standard_bytes,
                           SUM(nearline_count) as nearline_count, SUM(nearline_bytes) as nearline_bytes,
                           SUM(coldline_count) as coldline_count, SUM(coldline_bytes) as coldline_bytes,
                           SUM(archive_count) as archive_count, SUM(archive_bytes) as archive_bytes
                    FROM dir_summary
                    WHERE bucket = ? AND dir_prefix LIKE ?
                    """,
                    (bucket_name, pattern),
                )

                if not row or row["standard_count"] is None:
                    continue

                region_objects = 0
                region_bytes = 0
                region_cost = 0.0
                by_class = []
                for sc_name, count_col, bytes_col in [
                    ("STANDARD", "standard_count", "standard_bytes"),
                    ("NEARLINE", "nearline_count", "nearline_bytes"),
                    ("COLDLINE", "coldline_count", "coldline_bytes"),
                    ("ARCHIVE", "archive_count", "archive_bytes"),
                ]:
                    cnt = int(row[count_col] or 0)
                    bts = int(row[bytes_col] or 0)
                    if cnt == 0:
                        continue
                    cost = bts / (1024.0 * 1024.0 * 1024.0) * prices[sc_name] * discount_factor
                    by_class.append({"class": sc_name, "objects": cnt, "bytes": bts, "monthly_cost_usd": round(cost, 2)})
                    region_objects += cnt
                    region_bytes += bts
                    region_cost += cost

                if region_objects > 0:
                    pattern_by_region.append(
                        {
                            "region": region,
                            "bucket": bucket_name,
                            "objects": region_objects,
                            "bytes": region_bytes,
                            "bytes_human": human_bytes(region_bytes),
                            "monthly_cost_usd": round(region_cost, 2),
                            "by_storage_class": by_class,
                        }
                    )
                    pattern_total_objects += region_objects
                    pattern_total_bytes += region_bytes
                    pattern_total_cost += region_cost

            pattern_results.append(
                {
                    "pattern": pattern,
                    "objects": pattern_total_objects,
                    "bytes": pattern_total_bytes,
                    "bytes_human": human_bytes(pattern_total_bytes),
                    "monthly_cost_usd": round(pattern_total_cost, 2),
                    "regions": pattern_by_region,
                }
            )

        grand_objects = sum(p["objects"] for p in pattern_results)
        grand_bytes = sum(p["bytes"] for p in pattern_results)
        grand_cost = sum(p["monthly_cost_usd"] for p in pattern_results)

        return {
            "patterns": pattern_results,
            "totals": {
                "objects": grand_objects,
                "bytes": grand_bytes,
                "bytes_human": human_bytes(grand_bytes),
                "monthly_cost_usd": round(grand_cost, 2),
            },
        }

    EXPLORE_BUCKET_SIZE = 5000

    @app.get("/api/explore")
    @serialized
    def explore(bucket: str, prefix: str = "", max_children: int = EXPLORE_BUCKET_SIZE) -> dict[str, Any]:
        """Browse dir_summary hierarchically.

        Returns direct children of `prefix` with aggregated stats. If there
        are more than `max_children` unique segments, they are bucketed into
        lexicographic ranges.
        """
        conn = db()
        region = region_from_bucket(bucket)
        continent = continent_for_region(region)
        prices_row = _fetchall(conn, "SELECT name, price_per_gib_month_us, price_per_gib_month_eu FROM storage_classes")
        prices = {
            r["name"]: float(r["price_per_gib_month_us"] if continent == "US" else r["price_per_gib_month_eu"])
            for r in prices_row
        }
        discount_factor = 1.0 - GCS_DISCOUNT

        # Query children grouped by next path segment
        prefix_len = len(prefix) + 1  # +1 for the SQL 1-based indexing
        rows = _fetchall(
            conn,
            """
            SELECT split_part(substr(dir_prefix, ?), '/', 1) as segment,
                   SUM(standard_count) as standard_count, SUM(standard_bytes) as standard_bytes,
                   SUM(nearline_count) as nearline_count, SUM(nearline_bytes) as nearline_bytes,
                   SUM(coldline_count) as coldline_count, SUM(coldline_bytes) as coldline_bytes,
                   SUM(archive_count) as archive_count, SUM(archive_bytes) as archive_bytes
            FROM dir_summary
            WHERE bucket = ? AND dir_prefix LIKE ? || '%'
            GROUP BY segment
            ORDER BY segment
            """,
            (prefix_len, bucket, prefix),
        )

        def entry_stats(row: dict[str, Any]) -> dict[str, Any]:
            total_objects = 0
            total_bytes = 0
            total_cost = 0.0
            by_class = []
            for sc_name, count_col, bytes_col in [
                ("STANDARD", "standard_count", "standard_bytes"),
                ("NEARLINE", "nearline_count", "nearline_bytes"),
                ("COLDLINE", "coldline_count", "coldline_bytes"),
                ("ARCHIVE", "archive_count", "archive_bytes"),
            ]:
                cnt = int(row[count_col] or 0)
                bts = int(row[bytes_col] or 0)
                cost = bts / (1024.0 * 1024.0 * 1024.0) * prices[sc_name] * discount_factor
                total_objects += cnt
                total_bytes += bts
                total_cost += cost
                if cnt > 0:
                    by_class.append({"class": sc_name, "objects": cnt, "bytes": bts, "monthly_cost_usd": round(cost, 2)})
            return {
                "objects": total_objects,
                "bytes": total_bytes,
                "bytes_human": human_bytes(total_bytes),
                "monthly_cost_usd": round(total_cost, 2),
                "by_storage_class": by_class,
            }

        if len(rows) <= max_children:
            entries = []
            for row in rows:
                stats = entry_stats(row)
                stats["name"] = row["segment"] + "/"
                stats["prefix"] = prefix + row["segment"] + "/"
                entries.append(stats)
            return {"type": "children", "bucket": bucket, "prefix": prefix, "entries": entries}

        # Too many children — bucket into lexicographic ranges
        entries = []
        for i in range(0, len(rows), max_children):
            chunk = rows[i : i + max_children]
            # Merge stats across the chunk
            merged: dict[str, int] = {}
            for col in [
                "standard_count",
                "standard_bytes",
                "nearline_count",
                "nearline_bytes",
                "coldline_count",
                "coldline_bytes",
                "archive_count",
                "archive_bytes",
            ]:
                merged[col] = sum(int(r[col] or 0) for r in chunk)
            stats = entry_stats(merged)
            first_seg = chunk[0]["segment"]
            last_seg = chunk[-1]["segment"]
            stats["name"] = f"{first_seg} ... {last_seg}" if first_seg != last_seg else first_seg + "/"
            stats["first"] = first_seg
            stats["last"] = last_seg
            stats["child_count"] = len(chunk)
            entries.append(stats)
        return {"type": "buckets", "bucket": bucket, "prefix": prefix, "entries": entries}

    @app.get("/api/explore/unified")
    @serialized
    def explore_unified(
        prefix: str = "",
        bucket: str | None = None,
        storage_class: str | None = None,
        max_children: int = EXPLORE_BUCKET_SIZE,
    ) -> dict[str, Any]:
        """Hierarchical explorer with keep/delete status.

        Status is computed at query time by joining against protect/delete
        rules — no materialized status table. Unmatched dirs are implicitly
        "keep". Segment status: all-delete → delete, any-delete → mixed,
        else → keep.
        """
        conn = db()
        discount_factor = 1.0 - GCS_DISCOUNT

        # Resolve pricing
        prices_row = _fetchall(conn, "SELECT name, price_per_gib_month_us, price_per_gib_month_eu FROM storage_classes")
        if bucket:
            region = region_from_bucket(bucket)
            continent = continent_for_region(region)
            prices = {
                r["name"]: float(r["price_per_gib_month_us"] if continent == "US" else r["price_per_gib_month_eu"])
                for r in prices_row
            }
        else:
            prices = {
                r["name"]: (float(r["price_per_gib_month_us"]) + float(r["price_per_gib_month_eu"])) / 2.0
                for r in prices_row
            }

        sc_cols = [c for c in ALL_SC_COLS if storage_class is None or c[0] == storage_class]
        prefix_len = len(prefix) + 1  # +1 for SQL 1-based indexing

        rows = _query_explore_segments(conn, prefix_len, prefix, bucket, storage_class)

        def entry_stats(row: dict[str, Any]) -> dict[str, Any]:
            total_objects = 0
            total_bytes = 0
            total_cost = 0.0
            by_class = []
            for sc_name, count_col, bytes_col in sc_cols:
                cnt = int(row.get(count_col) or 0)
                bts = int(row.get(bytes_col) or 0)
                cost = bts / (1024.0 * 1024.0 * 1024.0) * prices[sc_name] * discount_factor
                total_objects += cnt
                total_bytes += bts
                total_cost += cost
                if cnt > 0:
                    by_class.append({"class": sc_name, "objects": cnt, "bytes": bts, "monthly_cost_usd": round(cost, 2)})
            return {
                "objects": total_objects,
                "bytes": total_bytes,
                "bytes_human": human_bytes(total_bytes),
                "monthly_cost_usd": round(total_cost, 2),
                "by_storage_class": by_class,
            }

        def _annotate(stats: dict[str, Any], status: str) -> dict[str, Any]:
            stats["status"] = status
            stats["status_order"] = STATUS_ORDER.get(status, 9)
            return stats

        if len(rows) <= max_children:
            entries = []
            for row in rows:
                seg = row["segment"]
                stats = entry_stats(row)
                stats["name"] = seg + "/"
                stats["prefix"] = prefix + seg + "/"
                tb = float(row["total_bytes"] or 0)
                kb = float(row["kept_bytes"] or 0)
                frac = kb / tb if tb > 0 else 1.0
                stats["kept_cost"] = round(stats["monthly_cost_usd"] * frac, 2)
                _annotate(stats, row["status"])
                entries.append(stats)
            return {"type": "children", "prefix": prefix, "entries": entries}

        # Too many children — collapse mixed → keep, sort by status then lex,
        # bucket within each status group so no range straddles two statuses.
        SUB_ORDER = {"delete": 0, "keep": 1}
        for row in rows:
            row["_sub_status"] = "keep" if row["status"] == "mixed" else row["status"]
        rows.sort(key=lambda r: (SUB_ORDER.get(r["_sub_status"], 1), r["segment"]))

        col_names = [cc for _, cc, _ in sc_cols] + [bc for _, _, bc in sc_cols]
        entries = []
        i = 0
        while i < len(rows):
            cur_status = rows[i]["_sub_status"]
            j = i
            while j < len(rows) and rows[j]["_sub_status"] == cur_status:
                j += 1
            status_run = rows[i:j]

            for k in range(0, len(status_run), max_children):
                chunk = status_run[k : k + max_children]
                merged: dict[str, int] = {}
                for col in col_names:
                    merged[col] = sum(int(r.get(col) or 0) for r in chunk)
                stats = entry_stats(merged)
                first_seg = chunk[0]["segment"]
                last_seg = chunk[-1]["segment"]
                if first_seg == last_seg:
                    stats["name"] = first_seg + "/"
                    stats["prefix"] = prefix + first_seg + "/"
                else:
                    stats["name"] = f"{first_seg} ... {last_seg}"
                stats["first"] = first_seg
                stats["last"] = last_seg
                stats["child_count"] = len(chunk)
                _annotate(stats, cur_status)
                entries.append(stats)

            i = j

        return {"type": "buckets", "prefix": prefix, "entries": entries}

    @app.post("/api/sync")
    @serialized
    def trigger_sync() -> dict[str, Any]:
        gcs_prefix = os.environ.get("GCS_DATA_PREFIX")
        if not gcs_prefix:
            return {"status": "no_gcs_prefix", "state": _sync_state.__dict__}
        if _sync_state.syncing:
            return {"status": "already_syncing", "state": _sync_state.__dict__}
        _run_sync(gcs_prefix, catalog)
        return {"status": "ok", "state": _sync_state.__dict__}

    @app.get("/api/sync/status")
    def sync_status() -> dict[str, Any]:
        return _sync_state.__dict__

    @app.middleware("http")
    async def no_cache_static(request: Request, call_next):
        """Add Cache-Control: no-cache to all non-API responses.

        ETags are already emitted by Starlette, so browsers will still do
        conditional GETs (304 Not Modified) for unchanged files — this just
        prevents stale JS from being served after a deploy.
        """
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

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
    """Start the storage dashboard server."""
    if not os.environ.get("DASHBOARD_PASSWORD"):
        raise SystemExit("DASHBOARD_PASSWORD env var must be set")
    if dev:
        uvicorn.run(
            "scripts.storage.server:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=["scripts/storage"],
            log_level="debug",
            access_log=True,
        )
    else:
        app = create_app()
        uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
