#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage cleanup workflow: delete cold unprotected objects with soft-delete safety net.

Approach:
  1. Resolve the protect-set (wildcard and direct prefixes) into concrete prefix lists.
  2. Estimate deletion savings by scanning objects and grouping by storage class.
  3. Enable soft-delete on each source bucket (3-day retention window).
  4. Delete non-STANDARD objects that fall outside the protect set.
  5. Wait for the soft-delete safety window to elapse.
  6. Disable soft-delete to finalize the purge.

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
import sqlite3
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import click
import google.auth
from google.cloud import storage
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
CACHE_DB_PATH = OUTPUT_ROOT / "cache.sqlite3"

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

# GCS storage pricing (USD / GiB / month).  Source: cloud.google.com/storage/pricing
# We apply a 50 % CUD/negotiated discount.
GCS_PRICE_PER_GIB_MONTH: dict[str, dict[str, float]] = {
    "US": {
        "STANDARD": 0.020,
        "NEARLINE": 0.010,
        "COLDLINE": 0.004,
        "ARCHIVE": 0.0012,
    },
    "EUROPE": {
        "STANDARD": 0.023,
        "NEARLINE": 0.013,
        "COLDLINE": 0.006,
        "ARCHIVE": 0.0025,
    },
}
GCS_DISCOUNT = 0.50

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
    estimate_workers: bool = False
    delete_workers: bool = False
    settle_hours: bool = False
    optional: bool = False

    def run(self, ctx: Context) -> None:
        self.runner(ctx, self)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixEstimate:
    prefix: str
    object_count: int
    total_bytes: int
    bytes_by_class: dict[str, int]
    count_by_class: dict[str, int]


@dataclass
class Context:
    dry_run: bool
    force: bool
    include_optional: bool
    listing_workers: int
    estimate_workers: int
    delete_workers: int
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


def cache_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(CACHE_DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


CACHE_SCHEMA_VERSION = 2  # bump to invalidate stale caches


def init_cache_db() -> None:
    with cache_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        row = conn.execute("SELECT value FROM cache_meta WHERE key = 'schema_version'").fetchone()
        current_version = int(row["value"]) if row else 0
        if current_version < CACHE_SCHEMA_VERSION:
            conn.execute("DROP TABLE IF EXISTS listing_cache")
            conn.execute("DROP TABLE IF EXISTS prefix_estimate_cache")
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
            CREATE TABLE IF NOT EXISTS prefix_estimate_cache (
                bucket_name TEXT NOT NULL,
                prefix TEXT NOT NULL,
                object_count INTEGER NOT NULL,
                total_bytes INTEGER NOT NULL,
                bytes_by_class_json TEXT NOT NULL DEFAULT '{}',
                count_by_class_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (bucket_name, prefix)
            )
            """
        )


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


def estimate_monthly_cost_by_class(bytes_by_class: dict[str, int], region: str) -> float:
    continent = continent_for_region(region)
    prices = GCS_PRICE_PER_GIB_MONTH.get(continent, GCS_PRICE_PER_GIB_MONTH["US"])
    total = 0.0
    for storage_class, num_bytes in bytes_by_class.items():
        price = prices.get(storage_class, prices["STANDARD"])
        total += (num_bytes / (1024**3)) * price * (1.0 - GCS_DISCOUNT)
    return total


# ---------------------------------------------------------------------------
# Prefix utilities
# ---------------------------------------------------------------------------


def collapse_overlapping_prefixes(prefixes: list[str]) -> tuple[list[str], list[str]]:
    normalized = sorted(
        {normalize_relative_prefix(p) for p in prefixes},
        key=lambda p: (len(p), p),
    )
    kept: list[str] = []
    skipped: list[str] = []
    for prefix in normalized:
        if any(prefix.startswith(existing) for existing in kept):
            skipped.append(prefix)
            continue
        kept.append(prefix)
    return kept, skipped


def relative_object_paths_for_bucket(prefix_rows: list[dict[str, str]], bucket: str) -> list[str]:
    values = [relative_prefix(row["prefix_url"], bucket) for row in prefix_rows if row["bucket"] == bucket]
    return sorted(dict.fromkeys(values))


def filter_rows_by_region(rows: list[dict[str, str]], ctx: Context, bucket_key: str = "bucket") -> list[dict[str, str]]:
    return [row for row in rows if bucket_selected(ctx, row[bucket_key])]


# ---------------------------------------------------------------------------
# Listing cache
# ---------------------------------------------------------------------------


def listing_cache_lookup(listing_prefix: str) -> list[str] | None:
    with cache_connection() as conn:
        row = conn.execute(
            "SELECT entries_json FROM listing_cache WHERE listing_prefix = ?",
            (listing_prefix,),
        ).fetchone()
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
# Estimate cache (v2: per-class breakdowns)
# ---------------------------------------------------------------------------


def estimate_cache_lookup(bucket_name: str, prefix: str) -> PrefixEstimate | None:
    with cache_connection() as conn:
        row = conn.execute(
            """
            SELECT object_count, total_bytes, bytes_by_class_json, count_by_class_json
            FROM prefix_estimate_cache
            WHERE bucket_name = ? AND prefix = ?
            """,
            (bucket_name, prefix),
        ).fetchone()
    if row is None:
        return None
    return PrefixEstimate(
        prefix=prefix,
        object_count=int(row["object_count"]),
        total_bytes=int(row["total_bytes"]),
        bytes_by_class=json.loads(row["bytes_by_class_json"]),
        count_by_class=json.loads(row["count_by_class_json"]),
    )


def write_estimate_cache(estimate: PrefixEstimate, bucket_name: str) -> None:
    with cache_connection() as conn:
        conn.execute(
            """
            INSERT INTO prefix_estimate_cache
                (bucket_name, prefix, object_count, total_bytes, bytes_by_class_json, count_by_class_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bucket_name, prefix) DO UPDATE SET
                object_count = excluded.object_count,
                total_bytes = excluded.total_bytes,
                bytes_by_class_json = excluded.bytes_by_class_json,
                count_by_class_json = excluded.count_by_class_json,
                updated_at = excluded.updated_at
            """,
            (
                bucket_name,
                estimate.prefix,
                estimate.object_count,
                estimate.total_bytes,
                json.dumps(estimate.bytes_by_class),
                json.dumps(estimate.count_by_class),
                now_utc().isoformat(),
            ),
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
    iterator = client.list_blobs(bucket, prefix=prefix, delimiter="/")
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
# Prefix resolution (shared with old script)
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


def build_protect_inputs(ctx: Context, action: StepSpec) -> None:
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
    print_summary(f"{action.action_id}: wrote merged protect inputs for {len(plan_rows)} regions")
    write_marker(ctx, action, fingerprint, outputs=written_outputs, extra={"regions": plan_rows})


def build_protect_prefix_set(ctx: Context) -> dict[str, set[str]]:
    """Load the merged protect prefix set per bucket, for matching during deletion."""
    plan_path = BACKUP_DIR / "cleanup_plan.csv"
    plan_rows = read_csv_rows(plan_path)
    result: dict[str, set[str]] = {}
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        prefix_rows = read_csv_rows(REPO_ROOT / row["protect_prefix_csv"])
        bucket = row["bucket"]
        prefixes = {normalize_relative_prefix(url_object_path(r["prefix_url"])) for r in prefix_rows}
        result[bucket] = prefixes
    return result


def object_is_protected(object_name: str, protect_prefixes: set[str]) -> bool:
    for prefix in protect_prefixes:
        if object_name.startswith(prefix):
            return True
    return False


# ===========================================================================
# PREP: estimate deletion savings (per-class pricing)
# ===========================================================================


def size_estimate_path(region: str) -> Path:
    return BACKUP_DIR / f"deletion_estimate_{region}.json"


def estimate_prefix_with_classes(ctx: Context, bucket_name: str, prefix: str) -> PrefixEstimate:
    client = storage_client(ctx)
    object_count = 0
    total_bytes = 0
    bytes_by_class: dict[str, int] = defaultdict(int)
    count_by_class: dict[str, int] = defaultdict(int)
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        size = int(blob.size or 0)
        sc = blob.storage_class or "STANDARD"
        object_count += 1
        total_bytes += size
        bytes_by_class[sc] += size
        count_by_class[sc] += 1
    return PrefixEstimate(
        prefix=prefix,
        object_count=object_count,
        total_bytes=total_bytes,
        bytes_by_class=dict(bytes_by_class),
        count_by_class=dict(count_by_class),
    )


def estimate_deletion_savings(ctx: Context, action: StepSpec) -> None:
    """Estimate how much data will be deleted and the monthly savings by storage class."""
    plan_path = BACKUP_DIR / "cleanup_plan.csv"
    if not plan_path.exists():
        raise RuntimeError("Missing cleanup_plan.csv. Run `prep build-protect-inputs` first.")
    plan_rows = read_csv_rows(plan_path)
    fingerprint = hashlib.sha256((file_digest(plan_path) + ctx.region_key).encode()).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: outputs and marker are current")
        return

    protect_set = build_protect_prefix_set(ctx)
    summary_rows: list[dict[str, str]] = []
    outputs: list[Path] = []
    remote_summary: dict[str, Any] = {"project": resolved_project(ctx), "regions": {}}

    for plan_row in plan_rows:
        region = plan_row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        bucket_name = plan_row["bucket"]
        bucket_protect = protect_set.get(bucket_name, set())

        # Scan the entire bucket, tallying protected vs deletable by class
        print_summary(f"{action.action_id}: scanning {bucket_name} (this may take a while)")
        client = storage_client(ctx)
        delete_bytes_by_class: dict[str, int] = defaultdict(int)
        delete_count_by_class: dict[str, int] = defaultdict(int)
        protect_bytes_by_class: dict[str, int] = defaultdict(int)
        protect_count_by_class: dict[str, int] = defaultdict(int)
        total_objects = 0
        total_bytes = 0

        progress = tqdm(desc=f"scan {bucket_name}", unit=" obj", leave=True)
        for blob in client.list_blobs(bucket_name):
            size = int(blob.size or 0)
            sc = blob.storage_class or "STANDARD"
            total_objects += 1
            total_bytes += size
            progress.update(1)

            protected = object_is_protected(blob.name, bucket_protect)
            is_cold = sc in NON_STANDARD_STORAGE_CLASSES

            if protected or not is_cold:
                protect_bytes_by_class[sc] += size
                protect_count_by_class[sc] += 1
            else:
                delete_bytes_by_class[sc] += size
                delete_count_by_class[sc] += 1
        progress.close()

        delete_total_bytes = sum(delete_bytes_by_class.values())
        delete_total_count = sum(delete_count_by_class.values())
        monthly_savings = estimate_monthly_cost_by_class(delete_bytes_by_class, region)

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
                "delete_bytes_by_class": dict(delete_bytes_by_class),
                "delete_count_by_class": dict(delete_count_by_class),
                "protect_object_count": sum(protect_count_by_class.values()),
                "protect_total_bytes": sum(protect_bytes_by_class.values()),
                "protect_bytes_by_class": dict(protect_bytes_by_class),
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
            "delete_bytes_by_class": dict(delete_bytes_by_class),
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


def delete_cold_unprotected(ctx: Context, action: StepSpec) -> None:
    """Delete non-STANDARD objects outside the protect set."""
    plan_rows = read_csv_rows(BACKUP_DIR / "cleanup_plan.csv")
    fingerprint = hashlib.sha256((file_digest(BACKUP_DIR / "cleanup_plan.csv") + ctx.region_key).encode()).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: already completed")
        return

    protect_set = build_protect_prefix_set(ctx)
    remote_summary: dict[str, Any] = {"regions": {}}
    outputs: list[Path] = []

    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        bucket_name = row["bucket"]
        bucket_protect = protect_set.get(bucket_name, set())
        client = storage_client(ctx)
        bucket_obj = client.bucket(bucket_name)

        deleted_count = 0
        deleted_bytes = 0
        skipped_protected = 0
        skipped_standard = 0
        deleted_by_class: dict[str, int] = defaultdict(int)

        print_summary(f"{action.action_id}: scanning {bucket_name} for cold unprotected objects")
        progress = tqdm(desc=f"delete {bucket_name}", unit=" obj", leave=True)

        delete_batch: list[storage.Blob] = []
        BATCH_SIZE = 1000

        for blob in client.list_blobs(bucket_name):
            sc = blob.storage_class or "STANDARD"
            size = int(blob.size or 0)
            progress.update(1)

            if sc not in NON_STANDARD_STORAGE_CLASSES:
                skipped_standard += 1
                continue
            if object_is_protected(blob.name, bucket_protect):
                skipped_protected += 1
                continue

            deleted_count += 1
            deleted_bytes += size
            deleted_by_class[sc] += 1
            delete_batch.append(bucket_obj.blob(blob.name))

            if len(delete_batch) >= BATCH_SIZE:
                if not ctx.dry_run:
                    with client.batch():
                        for b in delete_batch:
                            b.delete()
                delete_batch.clear()

        if delete_batch and not ctx.dry_run:
            with client.batch():
                for b in delete_batch:
                    b.delete()
            delete_batch.clear()
        progress.close()

        deletion_log_path = BACKUP_DIR / f"deletion_log_{region}.json"
        write_json(
            deletion_log_path,
            {
                "region": region,
                "bucket": bucket_name,
                "deleted_count": deleted_count,
                "deleted_bytes": deleted_bytes,
                "deleted_human_bytes": human_bytes(deleted_bytes),
                "deleted_by_class": dict(deleted_by_class),
                "skipped_protected": skipped_protected,
                "skipped_standard": skipped_standard,
                "dry_run": ctx.dry_run,
            },
        )
        outputs.append(deletion_log_path)
        remote_summary["regions"][region] = {
            "deleted_count": deleted_count,
            "deleted_bytes": deleted_bytes,
            "skipped_protected": skipped_protected,
            "skipped_standard": skipped_standard,
        }
        verb = "would delete" if ctx.dry_run else "deleted"
        print_summary(
            f"{region}: {verb} {deleted_count} objects ({human_bytes(deleted_bytes)}), "
            f"skipped {skipped_protected} protected + {skipped_standard} STANDARD"
        )
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
        action_id="prep.build_protect_inputs",
        group_name="prep",
        command_name="build-protect-inputs",
        description="Merge direct and resolved prefixes into the protect set.",
        help_text=(
            "Merge direct keep prefixes with resolved wildcard prefixes into one protect-set per region. "
            "This step is local-only and produces the CSVs that later cleanup steps consume."
        ),
        outputs=tuple(
            [
                *(BACKUP_DIR / f"protect_prefixes_{r}.csv" for r in ALL_REGIONS),
                BACKUP_DIR / "cleanup_plan.csv",
            ]
        ),
        mutating=False,
        runner=build_protect_inputs,
        predecessors=("prep.resolve_listing_prefixes",),
        requirements=(str(PROTECT_DIR / "protect_prefixes_direct.csv"),),
    ),
    StepSpec(
        action_id="prep.estimate_deletion_savings",
        group_name="prep",
        command_name="estimate-deletion-savings",
        description="Estimate deletion savings by scanning objects and their storage classes.",
        help_text=(
            "Scan each source bucket, classify every object as protected/deletable by storage class, "
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
        runner=estimate_deletion_savings,
        predecessors=("prep.build_protect_inputs",),
        requirements=(str(BACKUP_DIR / "cleanup_plan.csv"),),
        estimate_workers=True,
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
        predecessors=("prep.estimate_deletion_savings",),
        requirements=(str(BACKUP_DIR / "cleanup_plan.csv"),),
    ),
    StepSpec(
        action_id="cleanup.delete_cold_objects",
        group_name="cleanup",
        command_name="delete-cold-objects",
        description="Delete non-STANDARD objects outside the protect set.",
        help_text=(
            "Scan each source bucket and delete objects whose storage class is NEARLINE, COLDLINE, or "
            "ARCHIVE and that do not fall under any protected prefix. Uses batch deletes for throughput. "
            "Soft-delete must be enabled first so deletions are recoverable during the safety window."
        ),
        outputs=tuple(BACKUP_DIR / f"deletion_log_{r}.json" for r in ALL_REGIONS),
        mutating=True,
        runner=delete_cold_unprotected,
        predecessors=("cleanup.enable_soft_delete",),
        requirements=(str(BACKUP_DIR / "cleanup_plan.csv"),),
        delete_workers=True,
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
    estimate_workers: int,
    delete_workers: int,
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
        estimate_workers=estimate_workers,
        delete_workers=delete_workers,
        settle_hours=settle_hours,
        selected_regions=parse_regions(list(regions) or None),
        log_path=LOG_DIR / f"{log_prefix}_{timestamp_string()}.log",
        timestamp=timestamp_string(),
        project=project,
    )


def runtime_options(
    *,
    listing_workers: bool = False,
    estimate_workers: bool = False,
    delete_workers: bool = False,
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
        if estimate_workers:
            options.append(
                click.option(
                    "--estimate-workers", default=32, show_default=True, type=int, help="Concurrent prefix scans."
                )
            )
        if delete_workers:
            options.append(
                click.option(
                    "--delete-workers", default=16, show_default=True, type=int, help="Concurrent delete workers."
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
            estimate_workers=32,
            delete_workers=16,
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
@runtime_options(
    listing_workers=True, estimate_workers=True, delete_workers=True, settle_hours=True, include_optional=True
)
@click.option("--from", "from_action", type=click.Choice(sorted(STEP_INDEX)), help="Start from this step.")
@click.option("--to", "to_action", type=click.Choice(sorted(STEP_INDEX)), help="Stop after this step.")
@click.option("--only", "only_action", type=click.Choice(sorted(STEP_INDEX)), help="Run exactly one step.")
def run_cli(
    regions: tuple[str, ...],
    dry_run: bool,
    force: bool,
    include_optional: bool,
    listing_workers: int,
    estimate_workers: int,
    delete_workers: int,
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
        estimate_workers=estimate_workers,
        delete_workers=delete_workers,
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


@cli.group(help="Preparation commands: resolve protect set, estimate savings. Read-only against buckets.")
def prep() -> None:
    pass


@cli.group(help="Cleanup commands: enable soft-delete, delete cold objects, finalize.")
def cleanup() -> None:
    pass


GROUPS: dict[str, click.Group] = {"prep": prep, "cleanup": cleanup}


def register_step_command(group: click.Group, step: StepSpec) -> None:
    @runtime_options(
        listing_workers=step.listing_workers,
        estimate_workers=step.estimate_workers,
        delete_workers=step.delete_workers,
        settle_hours=step.settle_hours,
    )
    def command(
        regions: tuple[str, ...],
        dry_run: bool,
        force: bool,
        project: str | None,
        listing_workers: int = 32,
        estimate_workers: int = 32,
        delete_workers: int = 16,
        settle_hours: int = 72,
    ) -> None:
        ctx = build_context(
            dry_run=dry_run,
            force=force,
            include_optional=False,
            listing_workers=listing_workers,
            estimate_workers=estimate_workers,
            delete_workers=delete_workers,
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
