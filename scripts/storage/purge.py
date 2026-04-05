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
import fnmatch
import hashlib
import json
import logging
import os
import shlex
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
RESOLVE_DIR = OUTPUT_ROOT / "resolve"
BACKUP_DIR = OUTPUT_ROOT / "backup"
STATE_DIR = OUTPUT_ROOT / "state"
LOG_DIR = OUTPUT_ROOT / "logs"
CACHE_DB_PATH = OUTPUT_ROOT / "cache.duckdb"

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

OBJECT_FLUSH_THRESHOLD = 500_000
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
    outputs: tuple[Path, ...]
    mutating: bool
    runner: Callable[[Context, StepSpec], None]
    predecessors: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    listing_workers: bool = False
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
    listing_workers: int
    scan_workers: int
    settle_hours: int
    selected_regions: set[str] | None
    log_path: Path
    timestamp: str
    project: str | None

    @property
    def region_key(self) -> str:
        if not self.selected_regions:
            return "all"
        return "_".join(sorted(self.selected_regions))

    def state_path(self, action_id: str) -> Path:
        sanitized = action_id.replace(".", "__")
        return STATE_DIR / f"{sanitized}__{self.region_key}.json"


# ---------------------------------------------------------------------------
# Output directories and cache
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    for path in [OUTPUT_ROOT, PROTECT_DIR, RESOLVE_DIR, BACKUP_DIR, STATE_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    init_cache_db()


def cache_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(CACHE_DB_PATH))


def _fetchone_dict(result: duckdb.DuckDBPyConnection) -> dict[str, Any] | None:
    row = result.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in result.description], row, strict=False))


def _fetchall_dicts(result: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row, strict=False)) for row in result.fetchall()]


CACHE_SCHEMA_VERSION = 6


def init_cache_db() -> None:
    print_summary(f"opening DuckDB catalog: {CACHE_DB_PATH}")
    conn = cache_connection()
    try:
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
        if current_version < CACHE_SCHEMA_VERSION:
            print_summary(f"schema upgrade {current_version} → {CACHE_SCHEMA_VERSION}, rebuilding tables")
            conn.execute("DROP TABLE IF EXISTS listing_cache")
            conn.execute("DROP TABLE IF EXISTS prefix_estimate_cache")
            conn.execute("DROP TABLE IF EXISTS prefix_scan_cache")
            conn.execute("DROP TABLE IF EXISTS storage_classes")
            conn.execute("DROP TABLE IF EXISTS protect_prefixes")
            conn.execute("DROP TABLE IF EXISTS scanned_prefixes")
            conn.execute("DROP TABLE IF EXISTS objects")
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('schema_version', ?)",
                (str(CACHE_SCHEMA_VERSION),),
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS listing_cache (
                listing_prefix TEXT PRIMARY KEY,
                entries_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
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
            CREATE TABLE IF NOT EXISTS protect_prefixes (
                bucket TEXT NOT NULL,
                prefix TEXT NOT NULL,
                owners TEXT,
                reasons TEXT,
                sources TEXT,
                PRIMARY KEY (bucket, prefix)
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
            CREATE TABLE IF NOT EXISTS objects (
                bucket TEXT NOT NULL,
                name TEXT NOT NULL,
                size_bytes BIGINT NOT NULL,
                storage_class_id INTEGER NOT NULL REFERENCES storage_classes(id),
                created TIMESTAMPTZ,
                updated TIMESTAMPTZ,
                PRIMARY KEY (bucket, name)
            )
            """
        )

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
    finally:
        conn.close()


def storage_class_id_map() -> dict[str, int]:
    """Return a mapping from storage class name to its DB id."""
    conn = cache_connection()
    try:
        rows = _fetchall_dicts(conn.execute("SELECT id, name FROM storage_classes"))
    finally:
        conn.close()
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


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Markers (resumable step state)
# ---------------------------------------------------------------------------


def marker_matches(ctx: Context, step: StepSpec, input_fingerprint: str) -> bool:
    marker_path = ctx.state_path(step.action_id)
    if not marker_path.exists():
        return False
    marker = read_json(marker_path)
    if marker.get("input_fingerprint") != input_fingerprint:
        return False
    return all(path.exists() for path in step.outputs)


def write_marker(
    ctx: Context,
    step: StepSpec,
    input_fingerprint: str,
    *,
    outputs: list[Path],
    remote_summary: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "action_id": step.action_id,
        "completed_at": now_utc().isoformat(),
        "dry_run": ctx.dry_run,
        "input_fingerprint": input_fingerprint,
        "outputs": [str(path.relative_to(REPO_ROOT)) for path in outputs],
        "remote_summary": remote_summary or {},
    }
    if extra:
        payload.update(extra)
    write_json(ctx.state_path(step.action_id), payload)


# ---------------------------------------------------------------------------
# Region / bucket helpers
# ---------------------------------------------------------------------------


def region_from_bucket(bucket: str) -> str:
    return bucket.removeprefix("marin-") if bucket.startswith("marin-") else bucket


def region_bucket(region: str) -> str:
    return f"marin-{region}"


def selected_regions(ctx: Context) -> list[str]:
    if ctx.selected_regions is None:
        return [region_from_bucket(bucket) for bucket in MARIN_BUCKETS]
    return sorted(ctx.selected_regions)


def bucket_selected(ctx: Context, bucket: str) -> bool:
    return ctx.selected_regions is None or region_from_bucket(bucket) in ctx.selected_regions


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


def filter_rows_by_region(rows: list[dict[str, str]], ctx: Context, bucket_key: str = "bucket") -> list[dict[str, str]]:
    return [row for row in rows if bucket_selected(ctx, row[bucket_key])]


# ---------------------------------------------------------------------------
# Listing cache
# ---------------------------------------------------------------------------


def listing_cache_lookup(listing_prefix: str) -> list[str] | None:
    with cache_connection() as conn:
        row = _fetchone_dict(
            conn.execute(
                "SELECT entries_json FROM listing_cache WHERE listing_prefix = ?",
                (listing_prefix,),
            )
        )
    if row is None:
        return None
    entries = json.loads(row["entries_json"])
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        return None
    return entries


def write_listing_cache(listing_prefix: str, entries: list[str]) -> None:
    with cache_connection() as conn:
        conn.execute(
            """
            INSERT INTO listing_cache (listing_prefix, entries_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(listing_prefix) DO UPDATE SET
                entries_json = excluded.entries_json,
                updated_at = excluded.updated_at
            """,
            (listing_prefix, json.dumps(entries), now_utc().isoformat()),
        )


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------


def cached_child_prefixes(ctx: Context, listing_prefix: str) -> list[str]:
    if not ctx.force:
        cached_entries = listing_cache_lookup(listing_prefix)
        if cached_entries is not None:
            print_summary(f"using cached child-prefix listing: {listing_prefix}")
            return cached_entries

    bucket = url_bucket(listing_prefix)
    prefix = url_object_path(normalized_prefix_url(listing_prefix))
    print_summary(f"$ storage.list bucket={bucket} prefix={prefix!r} delimiter='/' project={resolved_project(ctx)!r}")
    client = storage_client(ctx)
    iterator = client.list_blobs(bucket, prefix=prefix, delimiter="/", fields="prefixes,nextPageToken")
    child_prefixes: set[str] = set()
    for page in iterator.pages:
        child_prefixes.update(page.prefixes)
    entries = [f"gs://{bucket}/{cp}" for cp in sorted(child_prefixes)]
    write_listing_cache(listing_prefix, entries)
    return entries


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


def resolve_prefix_from_match(normalized_glob: str, listing_prefix: str, candidate_url: str) -> str | None:
    bucket = url_bucket(normalized_glob)
    glob_parts = [part for part in url_object_path(normalized_glob).split("/") if part]
    wildcard_index = next((i for i, part in enumerate(glob_parts) if any(ch in part for ch in "*?[]")), None)
    if wildcard_index is None:
        return None
    prefix_parts = glob_parts[:wildcard_index]
    wildcard_segment = glob_parts[wildcard_index]
    suffix_parts = glob_parts[wildcard_index + 1 :]

    candidate_suffix = candidate_url.removeprefix(listing_prefix).strip("/")
    if not candidate_suffix:
        return None
    candidate_segment = candidate_suffix.split("/", 1)[0]
    if not fnmatch.fnmatch(candidate_segment, wildcard_segment):
        return None
    resolved_parts = [*prefix_parts, candidate_segment, *suffix_parts]
    resolved_path = "/".join(resolved_parts).rstrip("*").rstrip("/")
    return f"gs://{bucket}/{resolved_path}/"


# ===========================================================================
# PREP steps
# ===========================================================================


def resolve_listing_prefixes(ctx: Context, action: StepSpec) -> None:
    classified_rows = filter_rows_by_region(read_csv_rows(PROTECT_DIR / "protect_prefixes_classified.csv"), ctx)
    listing_rows = [row for row in classified_rows if row["classification"] == "sts_prefix_via_listing"]
    fingerprint = hashlib.sha256(
        (file_digest(PROTECT_DIR / "protect_prefixes_classified.csv") + ctx.region_key).encode()
    ).hexdigest()
    expected_outputs = [RESOLVE_DIR / f"listing_prefixes_{region}.csv" for region in selected_regions(ctx)] + [
        RESOLVE_DIR / f"resolved_prefixes_{region}.csv" for region in selected_regions(ctx)
    ]
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: outputs and marker are current")
        return

    listing_by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in listing_rows:
        listing_by_region[region_from_bucket(row["bucket"])].append(row)

    remote_summary: dict[str, Any] = {"regions": {}}
    written_outputs: list[Path] = []
    for region in selected_regions(ctx):
        region_rows = sorted(listing_by_region.get(region, []), key=lambda row: row["normalized_glob"])
        listing_input_path = RESOLVE_DIR / f"listing_prefixes_{region}.csv"
        resolved_output_path = RESOLVE_DIR / f"resolved_prefixes_{region}.csv"
        print_summary(f"{action.action_id}: region {region} has {len(region_rows)} listing-based families to resolve")
        listing_output_rows = [
            {
                "listing_prefix": row["listing_prefix"],
                "concrete_prefix_hint": row["concrete_prefix_hint"],
                "bucket": row["bucket"],
                "owners": row["owners"],
                "reasons": row["reasons"],
                "sources": row["sources"],
                "artifact_kinds": row["artifact_kinds"],
                "priority_max": row["priority_max"],
                "normalized_glob": row["normalized_glob"],
            }
            for row in region_rows
        ]
        write_csv_rows(
            listing_input_path,
            listing_output_rows,
            fieldnames=[
                "listing_prefix",
                "concrete_prefix_hint",
                "bucket",
                "owners",
                "reasons",
                "sources",
                "artifact_kinds",
                "priority_max",
                "normalized_glob",
            ],
        )
        written_outputs.append(listing_input_path)

        resolved_rows: list[dict[str, str]] = []
        rows_by_listing_prefix: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in region_rows:
            rows_by_listing_prefix[row["listing_prefix"]].append(row)
        listing_prefixes = sorted(rows_by_listing_prefix)
        listing_calls = 0
        candidate_prefixes_by_listing_prefix: dict[str, list[str]] = {}
        workers = max(1, min(ctx.listing_workers, len(listing_prefixes) or 1))
        print_summary(
            f"{action.action_id}: region {region} resolving {len(listing_prefixes)} unique listing prefixes "
            f"with {workers} workers"
        )
        progress = tqdm(total=len(listing_prefixes), desc=f"resolve {region}", unit="prefix", leave=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_lp = {executor.submit(cached_child_prefixes, ctx, lp): lp for lp in listing_prefixes}
            for future in as_completed(future_to_lp):
                lp = future_to_lp[future]
                listing_calls += 1
                progress.set_postfix_str(lp.removeprefix(f"gs://{region_bucket(region)}/")[:60])
                candidate_prefixes_by_listing_prefix[lp] = future.result()
                progress.update(1)
        progress.close()
        for lp in listing_prefixes:
            candidate_prefixes = candidate_prefixes_by_listing_prefix[lp]
            for row in rows_by_listing_prefix[lp]:
                for candidate in candidate_prefixes:
                    resolved = resolve_prefix_from_match(row["normalized_glob"], row["listing_prefix"], candidate)
                    if resolved is not None:
                        resolved_rows.append(
                            {
                                "prefix_url": resolved,
                                "bucket": row["bucket"],
                                "owners": row["owners"],
                                "reasons": row["reasons"],
                                "sources": row["sources"],
                                "artifact_kinds": row["artifact_kinds"],
                                "priority_max": row["priority_max"],
                                "normalized_glob": row["normalized_glob"],
                                "listing_prefix": row["listing_prefix"],
                            }
                        )
        resolved_rows = sorted(
            {
                (
                    row["prefix_url"],
                    row["bucket"],
                    row["owners"],
                    row["reasons"],
                    row["sources"],
                    row["artifact_kinds"],
                    row["priority_max"],
                    row["normalized_glob"],
                    row["listing_prefix"],
                ): row
                for row in resolved_rows
            }.values(),
            key=lambda row: row["prefix_url"],
        )
        write_csv_rows(
            resolved_output_path,
            resolved_rows,
            fieldnames=[
                "prefix_url",
                "bucket",
                "owners",
                "reasons",
                "sources",
                "artifact_kinds",
                "priority_max",
                "normalized_glob",
                "listing_prefix",
            ],
        )
        written_outputs.append(resolved_output_path)
        remote_summary["regions"][region] = {
            "listing_rows": len(region_rows),
            "resolved_prefix_rows": len(resolved_rows),
            "listing_calls": listing_calls,
        }
    print_summary(f"{action.action_id}: resolved listing-based prefixes for {len(selected_regions(ctx))} regions")
    write_marker(ctx, action, fingerprint, outputs=written_outputs or expected_outputs, remote_summary=remote_summary)


def load_protect_set(ctx: Context, action: StepSpec) -> None:
    """Merge direct + resolved prefixes into the protect_prefixes DB table and write cleanup_plan.csv."""
    direct_rows = filter_rows_by_region(read_csv_rows(PROTECT_DIR / "protect_prefixes_direct.csv"), ctx)
    fingerprint = hashlib.sha256(
        (
            file_digest(PROTECT_DIR / "protect_prefixes_direct.csv")
            + "".join(
                (
                    str((RESOLVE_DIR / f"resolved_prefixes_{region}.csv").stat().st_mtime_ns)
                    if (RESOLVE_DIR / f"resolved_prefixes_{region}.csv").exists()
                    else "missing"
                )
                for region in selected_regions(ctx)
            )
            + ctx.region_key
        ).encode()
    ).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: outputs and marker are current")
        return

    direct_by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in direct_rows:
        direct_by_region[region_from_bucket(row["bucket"])].append(
            {
                "prefix_url": normalized_prefix_url(row["sts_prefix"]),
                "bucket": row["bucket"],
                "owners": row["owners"],
                "reasons": row["reasons"],
                "sources": row["sources"],
                "artifact_kinds": row["artifact_kinds"],
                "priority_max": row["priority_max"],
                "normalized_glob": row["normalized_glob"],
                "origin": "direct",
            }
        )

    written_outputs: list[Path] = []
    plan_rows: list[dict[str, str]] = []
    all_db_rows: list[tuple[str, str, str, str, str]] = []

    for region in selected_regions(ctx):
        resolved_path = RESOLVE_DIR / f"resolved_prefixes_{region}.csv"
        resolved_rows = read_csv_rows(resolved_path) if resolved_path.exists() else []
        combined_rows = [
            *direct_by_region.get(region, []),
            *[
                {
                    "prefix_url": normalized_prefix_url(row["prefix_url"]),
                    "bucket": row["bucket"],
                    "owners": row["owners"],
                    "reasons": row["reasons"],
                    "sources": row["sources"],
                    "artifact_kinds": row["artifact_kinds"],
                    "priority_max": row["priority_max"],
                    "normalized_glob": row["normalized_glob"],
                    "origin": "resolved",
                }
                for row in resolved_rows
            ],
        ]
        deduped = list({(row["prefix_url"], row["bucket"]): row for row in combined_rows}.values())
        deduped.sort(key=lambda row: row["prefix_url"])

        # Write per-region CSV (same as before)
        output_path = BACKUP_DIR / f"protect_prefixes_{region}.csv"
        write_csv_rows(
            output_path,
            deduped,
            fieldnames=[
                "prefix_url",
                "bucket",
                "owners",
                "reasons",
                "sources",
                "artifact_kinds",
                "priority_max",
                "normalized_glob",
                "origin",
            ],
        )
        written_outputs.append(output_path)

        # Collect rows for DB insert
        for row in deduped:
            rel = normalize_relative_prefix(url_object_path(row["prefix_url"]))
            all_db_rows.append((row["bucket"], rel, row["owners"], row["reasons"], row["sources"]))

        bucket = region_bucket(region)
        plan_rows.append(
            {
                "region": region,
                "bucket": bucket,
                "location": BUCKET_LOCATIONS[bucket],
                "protect_prefix_csv": str(output_path.relative_to(REPO_ROOT)),
                "prefix_count": str(len(deduped)),
            }
        )

    plan_path = BACKUP_DIR / "cleanup_plan.csv"
    write_csv_rows(
        plan_path,
        plan_rows,
        fieldnames=["region", "bucket", "location", "protect_prefix_csv", "prefix_count"],
    )
    written_outputs.append(plan_path)

    # Populate protect_prefixes table (clear and repopulate)
    with cache_connection() as conn:
        conn.execute("DELETE FROM protect_prefixes")
        if all_db_rows:
            arrow_table = pa.table(
                {
                    "bucket": [r[0] for r in all_db_rows],
                    "prefix": [r[1] for r in all_db_rows],
                    "owners": [r[2] for r in all_db_rows],
                    "reasons": [r[3] for r in all_db_rows],
                    "sources": [r[4] for r in all_db_rows],
                }
            )
            conn.register("_protect_stage", arrow_table)
            conn.execute(
                """
                INSERT OR REPLACE INTO protect_prefixes (bucket, prefix, owners, reasons, sources)
                SELECT bucket, prefix, owners, reasons, sources FROM _protect_stage
                """
            )
            conn.unregister("_protect_stage")

    print_summary(
        f"{action.action_id}: wrote merged protect inputs for {len(plan_rows)} regions, "
        f"{len(all_db_rows)} prefixes loaded into DB"
    )
    write_marker(ctx, action, fingerprint, outputs=written_outputs, extra={"regions": plan_rows})


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
        probe_objects.extend(_blob_to_scanned(blob, bucket_name, sc_id_map) for blob in page)
        if len(probe_objects) < GCS_MAX_PAGE_SIZE:
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
    objects: list[ScannedObject] = field(default_factory=list)
    prefixes: list[tuple[str, str, int, str]] = field(default_factory=list)
    split_cache: list[tuple[str, str, str]] = field(default_factory=list)


def _flush_objects(
    conn: duckdb.DuckDBPyConnection, buf: ScanBuffer, objects: list[ScannedObject], *, force: bool = False
) -> None:
    """Buffer scanned objects and bulk-insert when the buffer is large enough.

    Uses PyArrow tables registered as virtual views for fast columnar inserts.
    Plain INSERT (no conflict resolution) is safe because scanned_prefixes
    tracking prevents re-scanning the same prefix within a single run.
    """
    buf.objects.extend(objects)
    if len(buf.objects) < OBJECT_FLUSH_THRESHOLD and not force:
        return
    if not buf.objects:
        return
    buf.objects.sort(key=lambda o: (o.bucket, o.name))
    arrow_table = pa.table(
        {
            "bucket": [o.bucket for o in buf.objects],
            "name": [o.name for o in buf.objects],
            "size_bytes": [o.size_bytes for o in buf.objects],
            "storage_class_id": [o.storage_class_id for o in buf.objects],
            "created": [o.created for o in buf.objects],
            "updated": [o.updated for o in buf.objects],
        },
        schema=pa.schema(
            [
                ("bucket", pa.string()),
                ("name", pa.string()),
                ("size_bytes", pa.int64()),
                ("storage_class_id", pa.int32()),
                ("created", pa.timestamp("us", tz="UTC")),
                ("updated", pa.timestamp("us", tz="UTC")),
            ]
        ),
    )
    conn.register("_obj_stage", arrow_table)
    conn.execute("INSERT INTO objects SELECT * FROM _obj_stage")
    conn.unregister("_obj_stage")
    buf.objects.clear()


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
                "listing_prefix": [r[0] for r in buf.split_cache],
                "entries_json": [r[1] for r in buf.split_cache],
                "updated_at": [r[2] for r in buf.split_cache],
            }
        )
        conn.register("_sc_stage", arrow_table)
        conn.execute(
            """
            INSERT OR REPLACE INTO listing_cache (listing_prefix, entries_json, updated_at)
            SELECT listing_prefix, entries_json, updated_at FROM _sc_stage
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
            BarColumn(bar_width=40),
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
    """Load the entire listing_cache table into memory for fast lookups."""
    rows = conn.execute("SELECT listing_prefix, entries_json FROM listing_cache").fetchall()
    cache: dict[str, list[str]] = {}
    for listing_prefix, entries_json in rows:
        entries = json.loads(entries_json)
        if isinstance(entries, list):
            cache[listing_prefix] = entries
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
                _flush_objects(db_conn, buf, event.objects)
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
                _flush_objects(db_conn, buf, event.objects)
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
        _flush_objects(db_conn, buf, [], force=True)
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
    plan_path = BACKUP_DIR / "cleanup_plan.csv"
    if not plan_path.exists():
        raise RuntimeError("Missing cleanup_plan.csv. Run `prep load-protect-set` first.")
    plan_rows = read_csv_rows(plan_path)
    fingerprint = hashlib.sha256((file_digest(plan_path) + ctx.region_key).encode()).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: outputs and marker are current")
        return

    sc_id_map = storage_class_id_map()
    remote_summary: dict[str, Any] = {"project": resolved_project(ctx), "regions": {}}

    db_conn = cache_connection()
    try:
        for plan_row in plan_rows:
            region = plan_row["region"]
            if ctx.selected_regions and region not in ctx.selected_regions:
                continue
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
                        "SELECT COALESCE(SUM(object_count), 0) as total FROM scanned_prefixes WHERE bucket = ?",
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
            buf = ScanBuffer()

            try:
                _run_adaptive_scan(
                    ctx, bucket_name, sc_id_map, pending, workers, already_scanned, db_conn, progress, buf
                )
            finally:
                progress.stop()

            if progress.prefixes_expanded:
                n = progress.prefixes_expanded
                print_summary(f"{action.action_id}: {bucket_name}: {n} prefixes expanded via adaptive splitting")

            total_row = _fetchone_dict(
                db_conn.execute(
                    "SELECT COALESCE(SUM(object_count), 0) as total FROM scanned_prefixes WHERE bucket = ?",
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
    finally:
        db_conn.close()

    write_marker(ctx, action, fingerprint, outputs=[], remote_summary=remote_summary)


# ===========================================================================
# PREP: estimate deletion savings (SQL-based)
# ===========================================================================


def size_estimate_path(region: str) -> Path:
    return BACKUP_DIR / f"deletion_estimate_{region}.json"


def estimate_savings(ctx: Context, action: StepSpec) -> None:
    """Estimate deletion savings using SQL queries against the scanned object catalog."""
    plan_path = BACKUP_DIR / "cleanup_plan.csv"
    if not plan_path.exists():
        raise RuntimeError("Missing cleanup_plan.csv. Run `prep load-protect-set` first.")
    plan_rows = read_csv_rows(plan_path)
    fingerprint = hashlib.sha256((file_digest(plan_path) + ctx.region_key + "scan").encode()).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: outputs and marker are current")
        return

    summary_rows: list[dict[str, str]] = []
    outputs: list[Path] = []
    remote_summary: dict[str, Any] = {"project": resolved_project(ctx), "regions": {}}
    conn = cache_connection()

    try:
        for plan_row in plan_rows:
            region = plan_row["region"]
            if ctx.selected_regions and region not in ctx.selected_regions:
                continue
            bucket_name = plan_row["bucket"]
            continent = continent_for_region(region)
            price_column = "price_per_gib_month_us" if continent == "US" else "price_per_gib_month_eu"

            totals = _fetchone_dict(
                conn.execute(
                    "SELECT COUNT(*) as cnt, COALESCE(SUM(size_bytes), 0) as total_bytes FROM objects WHERE bucket = ?",
                    (bucket_name,),
                )
            )
            total_objects = int(totals["cnt"])
            total_bytes = int(totals["total_bytes"])

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
                  AND NOT EXISTS (
                    SELECT 1 FROM protect_prefixes p
                    WHERE o.bucket = p.bucket AND o.name LIKE p.prefix || '%'
                  )
                GROUP BY sc.name
                """,
                    (1.0 - GCS_DISCOUNT, bucket_name),
                )
            )

            protect_rows = _fetchall_dicts(
                conn.execute(
                    """
                SELECT sc.name as storage_class,
                       COUNT(*) as cnt,
                       COALESCE(SUM(o.size_bytes), 0) as total_bytes
                FROM objects o
                JOIN storage_classes sc ON o.storage_class_id = sc.id
                WHERE o.bucket = ?
                  AND (sc.name = 'STANDARD'
                       OR EXISTS (
                         SELECT 1 FROM protect_prefixes p
                         WHERE o.bucket = p.bucket AND o.name LIKE p.prefix || '%'
                       ))
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

            region_output = size_estimate_path(region)
            write_json(
                region_output,
                {
                    "region": region,
                    "project": resolved_project(ctx),
                    "bucket": bucket_name,
                    "total_objects_scanned": total_objects,
                    "total_bytes_scanned": total_bytes,
                    "total_human_bytes_scanned": human_bytes(total_bytes),
                    "delete_object_count": delete_total_count,
                    "delete_total_bytes": delete_total_bytes,
                    "delete_human_bytes": human_bytes(delete_total_bytes),
                    "delete_bytes_by_class": delete_bytes_by_class,
                    "delete_count_by_class": delete_count_by_class,
                    "protect_object_count": sum(protect_count_by_class.values()),
                    "protect_total_bytes": sum(protect_bytes_by_class.values()),
                    "protect_bytes_by_class": protect_bytes_by_class,
                    "estimated_monthly_savings_usd": round(monthly_savings, 2),
                },
            )
            outputs.append(region_output)
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
                    f"  {sc:>12}: {human_bytes(delete_bytes_by_class[sc]):>12}  "
                    f"{delete_count_by_class[sc]:>8} objects"
                )
    finally:
        conn.close()

    summary_path = BACKUP_DIR / "deletion_estimate_summary.csv"
    write_csv_rows(
        summary_path,
        summary_rows,
        fieldnames=[
            "region",
            "bucket",
            "total_objects",
            "delete_objects",
            "delete_bytes",
            "delete_human_bytes",
            "estimated_monthly_savings_usd",
        ],
    )
    outputs.append(summary_path)
    total_monthly = sum(float(row["estimated_monthly_savings_usd"]) for row in summary_rows)
    total_annual = total_monthly * 12
    total_delete_bytes = sum(int(row["delete_bytes"]) for row in summary_rows)
    print_summary(f"{action.action_id}: wrote deletion estimates for {len(summary_rows)} regions")
    print_summary(
        f"  total deletable: {human_bytes(total_delete_bytes)} — "
        f"~${total_monthly:,.2f}/mo, ~${total_annual:,.2f}/yr savings (after 50% discount)"
    )
    write_marker(ctx, action, fingerprint, outputs=outputs, remote_summary=remote_summary)


# ===========================================================================
# CLEANUP steps
# ===========================================================================


def enable_soft_delete(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "cleanup_plan.csv")
    fingerprint = hashlib.sha256((file_digest(BACKUP_DIR / "cleanup_plan.csv") + ctx.region_key).encode()).hexdigest()
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
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
        write_marker(ctx, action, fingerprint, outputs=[], remote_summary=remote_summary)


def _delete_prefix_objects(
    ctx: Context,
    bucket_name: str,
    prefix: str,
) -> tuple[int, int, dict[str, int]]:
    """Delete cold unprotected objects under a single prefix. Returns (count, bytes, by_class)."""
    with cache_connection() as conn:
        rows = _fetchall_dicts(
            conn.execute(
                """
            SELECT o.name, o.size_bytes, sc.name as storage_class
            FROM objects o
            JOIN storage_classes sc ON o.storage_class_id = sc.id
            WHERE o.bucket = ?
              AND o.name LIKE ? || '%'
              AND sc.name != 'STANDARD'
              AND NOT EXISTS (
                SELECT 1 FROM protect_prefixes p
                WHERE o.bucket = p.bucket AND o.name LIKE p.prefix || '%'
              )
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
    plan_rows = read_csv_rows(BACKUP_DIR / "cleanup_plan.csv")
    fingerprint = hashlib.sha256((file_digest(BACKUP_DIR / "cleanup_plan.csv") + ctx.region_key).encode()).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: already completed")
        return

    remote_summary: dict[str, Any] = {"regions": {}}
    outputs: list[Path] = []
    conn = cache_connection()

    try:
        for row in plan_rows:
            region = row["region"]
            if ctx.selected_regions and region not in ctx.selected_regions:
                continue
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
                    """
                SELECT COUNT(*) as cnt FROM objects o
                JOIN storage_classes sc ON o.storage_class_id = sc.id
                WHERE o.bucket = ?
                  AND sc.name != 'STANDARD'
                  AND EXISTS (
                    SELECT 1 FROM protect_prefixes p
                    WHERE o.bucket = p.bucket AND o.name LIKE p.prefix || '%'
                  )
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

            deletion_log_path = BACKUP_DIR / f"deletion_log_{region}.json"
            write_json(
                deletion_log_path,
                {
                    "region": region,
                    "bucket": bucket_name,
                    "deleted_count": total_deleted_count,
                    "deleted_bytes": total_deleted_bytes,
                    "deleted_human_bytes": human_bytes(total_deleted_bytes),
                    "deleted_by_class": dict(total_deleted_by_class),
                    "skipped_protected": total_skipped_protected,
                    "skipped_standard": total_skipped_standard,
                    "dry_run": ctx.dry_run,
                },
            )
            outputs.append(deletion_log_path)
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
    finally:
        conn.close()
    if not ctx.dry_run:
        write_marker(ctx, action, fingerprint, outputs=outputs, remote_summary=remote_summary)


def wait_for_soft_delete_window(ctx: Context, action: StepSpec) -> None:
    prerequisite_marker = ctx.state_path("cleanup.delete_cold_objects")
    if not prerequisite_marker.exists():
        raise RuntimeError("cleanup.wait_for_safety_window requires cleanup.delete_cold_objects to be complete first")
    fingerprint = file_digest(prerequisite_marker)
    marker_path = ctx.state_path(action.action_id)
    settle_deadline = now_utc() + timedelta(hours=ctx.settle_hours)
    if marker_path.exists() and not ctx.force:
        marker = read_json(marker_path)
        existing_deadline = datetime.fromisoformat(marker["settle_deadline"])
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
        ctx,
        action,
        fingerprint,
        outputs=[],
        extra={"settle_deadline": settle_deadline.isoformat()},
    )


def disable_soft_delete(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "cleanup_plan.csv")
    fingerprint = hashlib.sha256((file_digest(BACKUP_DIR / "cleanup_plan.csv") + ctx.region_key).encode()).hexdigest()
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
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
        write_marker(ctx, action, fingerprint, outputs=[], remote_summary=remote_summary)


# ===========================================================================
# Step registry
# ===========================================================================

ALL_REGIONS = [region_from_bucket(b) for b in MARIN_BUCKETS]

STEPS = [
    StepSpec(
        action_id="prep.resolve_listing_prefixes",
        group_name="prep",
        command_name="resolve-listing-prefixes",
        description="Resolve listing-based protect families into concrete prefixes.",
        help_text=(
            "Resolve wildcard protect families into concrete prefixes by listing one level below each parent prefix. "
            "Use `--listing-workers` to control parallel listing."
        ),
        outputs=tuple(
            sorted(
                [
                    *(RESOLVE_DIR / f"listing_prefixes_{r}.csv" for r in ALL_REGIONS),
                    *(RESOLVE_DIR / f"resolved_prefixes_{r}.csv" for r in ALL_REGIONS),
                ]
            )
        ),
        mutating=False,
        runner=resolve_listing_prefixes,
        listing_workers=True,
        requirements=(str(PROTECT_DIR / "protect_prefixes_classified.csv"),),
    ),
    StepSpec(
        action_id="prep.load_protect_set",
        group_name="prep",
        command_name="load-protect-set",
        description="Merge direct and resolved prefixes into the protect set (DB + CSV).",
        help_text=(
            "Merge direct keep prefixes with resolved wildcard prefixes into one protect-set per region. "
            "Populates the protect_prefixes DB table and writes per-region CSVs and cleanup_plan.csv."
        ),
        outputs=tuple(
            [
                *(BACKUP_DIR / f"protect_prefixes_{r}.csv" for r in ALL_REGIONS),
                BACKUP_DIR / "cleanup_plan.csv",
            ]
        ),
        mutating=False,
        runner=load_protect_set,
        predecessors=("prep.resolve_listing_prefixes",),
        requirements=(str(PROTECT_DIR / "protect_prefixes_direct.csv"),),
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
        outputs=(),
        mutating=False,
        runner=scan_objects,
        predecessors=("prep.load_protect_set",),
        requirements=(str(BACKUP_DIR / "cleanup_plan.csv"),),
        scan_workers=True,
    ),
    StepSpec(
        action_id="prep.estimate_savings",
        group_name="prep",
        command_name="estimate-savings",
        description="Estimate deletion savings via SQL queries against the object catalog.",
        help_text=(
            "Query the DuckDB object catalog to classify every object as protected/deletable by storage class, "
            "and compute the monthly cost savings from deletion. Produces per-region JSON estimates "
            "and a summary CSV."
        ),
        outputs=tuple(
            [
                *(size_estimate_path(r) for r in ALL_REGIONS),
                BACKUP_DIR / "deletion_estimate_summary.csv",
            ]
        ),
        mutating=False,
        runner=estimate_savings,
        predecessors=("prep.scan_objects",),
        requirements=(str(BACKUP_DIR / "cleanup_plan.csv"),),
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
        outputs=(),
        mutating=True,
        runner=enable_soft_delete,
        predecessors=("prep.estimate_savings",),
        requirements=(str(BACKUP_DIR / "cleanup_plan.csv"),),
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
        outputs=tuple(BACKUP_DIR / f"deletion_log_{r}.json" for r in ALL_REGIONS),
        mutating=True,
        runner=delete_cold_unprotected,
        predecessors=("cleanup.enable_soft_delete",),
        requirements=(str(BACKUP_DIR / "cleanup_plan.csv"),),
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
        outputs=(),
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
        outputs=(),
        mutating=True,
        runner=disable_soft_delete,
        predecessors=("cleanup.wait_for_safety_window",),
        requirements=(str(BACKUP_DIR / "cleanup_plan.csv"),),
        optional=True,
    ),
]

STEP_INDEX = {step.action_id: step for step in STEPS}

# ===========================================================================
# CLI
# ===========================================================================


def parse_regions(raw_regions: list[str] | None) -> set[str] | None:
    if not raw_regions:
        return None
    allowed = {region_from_bucket(bucket) for bucket in MARIN_BUCKETS}
    selected = set(raw_regions)
    unknown = selected - allowed
    if unknown:
        raise ValueError(f"Unknown regions: {', '.join(sorted(unknown))}")
    return selected


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
    missing = [p for p in step.predecessors if not ctx.state_path(p).exists()]
    if missing:
        raise RuntimeError(
            f"{step.action_id} requires these predecessor steps to be complete first: {', '.join(missing)}"
        )


def build_context(
    *,
    dry_run: bool,
    force: bool,
    include_optional: bool,
    listing_workers: int,
    scan_workers: int,
    settle_hours: int,
    regions: tuple[str, ...],
    log_prefix: str,
    project: str | None,
) -> Context:
    ensure_output_dirs()
    return Context(
        dry_run=dry_run,
        force=force,
        include_optional=include_optional,
        listing_workers=listing_workers,
        scan_workers=scan_workers,
        settle_hours=settle_hours,
        selected_regions=parse_regions(list(regions) or None),
        log_path=LOG_DIR / f"{log_prefix}_{timestamp_string()}.log",
        timestamp=timestamp_string(),
        project=project,
    )


def runtime_options(
    *,
    listing_workers: bool = False,
    scan_workers: bool = False,
    settle_hours: bool = False,
    include_optional: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        options: list[Callable[[Callable[..., Any]], Callable[..., Any]]] = [
            click.option("--region", "regions", multiple=True, help="Limit to specific Marin storage regions."),
            click.option("--force", is_flag=True, help="Ignore cached markers and recompute."),
            click.option("--dry-run", is_flag=True, help="Read-only mode: inspect but never mutate remote state."),
            click.option("--project", help="Override the GCP project for Cloud Storage API calls."),
        ]
        if include_optional:
            options.append(click.option("--include-optional", is_flag=True, help="Include optional cleanup steps."))
        if listing_workers:
            options.append(
                click.option(
                    "--listing-workers", default=32, show_default=True, type=int, help="Concurrent listing fetches."
                )
            )
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
@click.option("--region", "regions", multiple=True, help="Limit to specific regions.")
def plan_cli(regions: tuple[str, ...]) -> None:
    ensure_output_dirs()
    parsed = parse_regions(list(regions) or None)
    print("Ordered storage cleanup steps:")
    for step in STEPS:
        suffix = " [optional]" if step.optional else ""
        dummy_ctx = Context(
            dry_run=False,
            force=False,
            include_optional=False,
            listing_workers=32,
            scan_workers=64,
            settle_hours=72,
            selected_regions=parsed,
            log_path=LOG_DIR / "plan.log",
            timestamp=timestamp_string(),
            project=None,
        )
        status = "done" if dummy_ctx.state_path(step.action_id).exists() else "pending"
        print(f"  {status:7}  {step.action_id}{suffix}")


@cli.command(
    "run",
    help="Execute a contiguous slice of the ordered workflow.",
)
@runtime_options(listing_workers=True, scan_workers=True, settle_hours=True, include_optional=True)
@click.option("--from", "from_action", type=click.Choice(sorted(STEP_INDEX)), help="Start from this step.")
@click.option("--to", "to_action", type=click.Choice(sorted(STEP_INDEX)), help="Stop after this step.")
@click.option("--only", "only_action", type=click.Choice(sorted(STEP_INDEX)), help="Run exactly one step.")
def run_cli(
    regions: tuple[str, ...],
    dry_run: bool,
    force: bool,
    include_optional: bool,
    listing_workers: int,
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
        listing_workers=listing_workers,
        scan_workers=scan_workers,
        settle_hours=settle_hours,
        regions=regions,
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
        listing_workers=step.listing_workers,
        scan_workers=step.scan_workers,
        settle_hours=step.settle_hours,
    )
    def command(
        regions: tuple[str, ...],
        dry_run: bool,
        force: bool,
        project: str | None,
        listing_workers: int = 32,
        scan_workers: int = 64,
        settle_hours: int = 72,
    ) -> None:
        ctx = build_context(
            dry_run=dry_run,
            force=force,
            include_optional=False,
            listing_workers=listing_workers,
            scan_workers=scan_workers,
            settle_hours=settle_hours,
            regions=regions,
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
