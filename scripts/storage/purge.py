#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the one-off storage purge workflow as a resumable ordered driver."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fnmatch
import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any
from collections.abc import Callable

import click
import google.auth
from google.cloud import storage
from tqdm.auto import tqdm

SCRIPT_PATH = Path(__file__).resolve()
STORAGE_DIR = SCRIPT_PATH.parent
REPO_ROOT = STORAGE_DIR.parent.parent
OUTPUT_ROOT = STORAGE_DIR / "purge"
PROTECT_DIR = OUTPUT_ROOT / "protect"
RESOLVE_DIR = OUTPUT_ROOT / "resolve"
BACKUP_DIR = OUTPUT_ROOT / "backup"
PURGE_DIR = OUTPUT_ROOT / "purge"
STATE_DIR = OUTPUT_ROOT / "state"
LOG_DIR = OUTPUT_ROOT / "logs"
CACHE_DB_PATH = OUTPUT_ROOT / "cache.sqlite3"

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

NON_STANDARD_STORAGE_CLASSES = ["NEARLINE", "COLDLINE", "ARCHIVE"]

# GCS Standard Storage list prices (USD per GiB per month).
# Source: https://cloud.google.com/storage/pricing
# We apply a 50% CUD/negotiated discount.
GCS_STANDARD_PRICE_PER_GIB_MONTH: dict[str, float] = {
    "US": 0.020,
    "EUROPE": 0.023,
}
GCS_DISCOUNT = 0.50


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
    sample_size: bool = False
    settle_hours: bool = False
    optional: bool = False

    def run(self, ctx: Context) -> None:
        self.runner(ctx, self)


@dataclass(frozen=True)
class PrefixEstimate:
    prefix: str
    object_count: int
    total_bytes: int


@dataclass
class Context:
    dry_run: bool
    force: bool
    include_optional: bool
    listing_workers: int
    estimate_workers: int
    sample_size: int
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


def ensure_output_dirs() -> None:
    for path in [OUTPUT_ROOT, PROTECT_DIR, RESOLVE_DIR, BACKUP_DIR, PURGE_DIR, STATE_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    init_cache_db()


def cache_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(CACHE_DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def init_cache_db() -> None:
    with cache_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS listing_cache (
                listing_prefix TEXT PRIMARY KEY,
                entries_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prefix_estimate_cache (
                bucket_name TEXT NOT NULL,
                prefix TEXT NOT NULL,
                object_count INTEGER NOT NULL,
                total_bytes INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (bucket_name, prefix)
            )
            """
        )


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


def run_command(
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
    if completed.stdout:
        if completed.stdout.strip():
            print(completed.stdout.rstrip())
        log_line(ctx, completed.stdout)
    if completed.stderr:
        if completed.stderr.strip():
            print(completed.stderr.rstrip(), file=sys.stderr)
        log_line(ctx, completed.stderr)
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed ({completed.returncode}): {rendered}\n{completed.stderr}")
    if capture_json:
        if completed.returncode != 0:
            return None
        return json.loads(completed.stdout or "{}")
    return completed


def print_summary(message: str) -> None:
    print(message)


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


def stat_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_dir():
            continue
        stat = path.stat()
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


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
    payload = {
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


def region_from_bucket(bucket: str) -> str:
    prefix = "marin-"
    return bucket.removeprefix(prefix) if bucket.startswith(prefix) else bucket


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


def join_gs_url(bucket: str, object_path: str) -> str:
    cleaned = object_path.lstrip("/")
    return f"gs://{bucket}/{cleaned}" if cleaned else f"gs://{bucket}"


def normalized_prefix_url(url: str) -> str:
    return url if url.endswith("/") else f"{url}/"


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def estimate_monthly_cost(total_bytes: int, region: str) -> float:
    """Estimate monthly GCS Standard storage cost in USD, after discount."""
    continent = "EUROPE" if region.startswith("eu-") else "US"
    list_price = GCS_STANDARD_PRICE_PER_GIB_MONTH.get(continent, GCS_STANDARD_PRICE_PER_GIB_MONTH["US"])
    gib = total_bytes / (1024**3)
    return gib * list_price * (1.0 - GCS_DISCOUNT)


def normalize_relative_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


def collapse_overlapping_prefixes(prefixes: list[str]) -> tuple[list[str], list[str]]:
    normalized = sorted(
        {normalize_relative_prefix(prefix) for prefix in prefixes},
        key=lambda prefix: (len(prefix), prefix),
    )
    kept: list[str] = []
    skipped: list[str] = []
    for prefix in normalized:
        if any(prefix.startswith(existing) for existing in kept):
            skipped.append(prefix)
            continue
        kept.append(prefix)
    return kept, skipped


def recursive_listing_url(prefix_url: str) -> str:
    return normalized_prefix_url(prefix_url) + "**"


def relative_prefix(prefix_url: str, bucket: str) -> str:
    prefix = url_object_path(prefix_url)
    if prefix.endswith("/"):
        return prefix
    return f"{prefix}/" if prefix else ""


def suggest_backup_bucket(bucket: str) -> str:
    return f"marin-tmp-backup-{region_from_bucket(bucket)}-purge-tmp-20260326"


def stable_job_name(region: str, include_prefixes: list[str]) -> str:
    digest = hashlib.sha256("\n".join(include_prefixes).encode()).hexdigest()[:10]
    return f"storage-purge-backup-{region}-{digest}"


def size_estimate_path(region: str) -> Path:
    return BACKUP_DIR / f"backup_size_estimate_{region}.json"


def gcloud_bucket_describe(ctx: Context, bucket_url: str) -> dict[str, Any]:
    return run_command(
        ctx,
        ["gcloud", "storage", "buckets", "describe", bucket_url, "--format=json"],
        capture_json=True,
    )


def gcloud_object_describe(ctx: Context, object_url: str) -> dict[str, Any] | None:
    return run_command(
        ctx,
        ["gcloud", "storage", "objects", "describe", object_url, "--format=json"],
        allow_failure=True,
        capture_json=True,
    )


def gcloud_transfer_job_describe(ctx: Context, job_name: str) -> dict[str, Any] | None:
    return run_command(
        ctx,
        ["gcloud", "transfer", "jobs", "describe", job_name, "--format=json"],
        allow_failure=True,
        capture_json=True,
    )


def gcloud_ls(ctx: Context, path: str, *, recursive: bool = False) -> list[str]:
    command = ["gcloud", "storage", "ls"]
    if recursive:
        command.append("--recursive")
    command.append(path)
    completed = run_command(ctx, command)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def listing_cache_lookup(listing_prefix: str) -> list[str] | None:
    with cache_connection() as connection:
        row = connection.execute(
            "SELECT entries_json FROM listing_cache WHERE listing_prefix = ?",
            (listing_prefix,),
        ).fetchone()
    if row is None:
        return None
    entries = json.loads(row["entries_json"])
    if not isinstance(entries, list) or not all(isinstance(entry, str) for entry in entries):
        return None
    return entries


def write_listing_cache(listing_prefix: str, entries: list[str]) -> None:
    with cache_connection() as connection:
        connection.execute(
            """
            INSERT INTO listing_cache (listing_prefix, entries_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(listing_prefix) DO UPDATE SET
                entries_json = excluded.entries_json,
                updated_at = excluded.updated_at
            """,
            (listing_prefix, json.dumps(entries), now_utc().isoformat()),
        )


def estimate_cache_lookup(bucket_name: str, prefix: str) -> PrefixEstimate | None:
    with cache_connection() as connection:
        row = connection.execute(
            """
            SELECT object_count, total_bytes
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
    )


def write_estimate_cache(estimate: PrefixEstimate, bucket_name: str) -> None:
    with cache_connection() as connection:
        connection.execute(
            """
            INSERT INTO prefix_estimate_cache (bucket_name, prefix, object_count, total_bytes, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(bucket_name, prefix) DO UPDATE SET
                object_count = excluded.object_count,
                total_bytes = excluded.total_bytes,
                updated_at = excluded.updated_at
            """,
            (
                bucket_name,
                estimate.prefix,
                estimate.object_count,
                estimate.total_bytes,
                now_utc().isoformat(),
            ),
        )


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
    entries = [f"gs://{bucket}/{child_prefix}" for child_prefix in sorted(child_prefixes)]
    write_listing_cache(listing_prefix, entries)
    return entries


def bucket_autoclass_enabled(metadata: dict[str, Any]) -> bool:
    autoclass = metadata.get("autoclass") or {}
    return bool(autoclass.get("enabled", False))


def bucket_soft_delete_seconds(metadata: dict[str, Any]) -> int:
    policy = metadata.get("softDeletePolicy") or {}
    value = policy.get("retentionDurationSeconds")
    if value is None:
        return 0
    return int(value)


def lifecycle_rules_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    lifecycle = metadata.get("lifecycle") or {}
    rule = lifecycle.get("rule")
    if isinstance(rule, list):
        return rule
    return []


def is_delete_nonstandard_rule(rule: dict[str, Any]) -> bool:
    action = rule.get("action") or {}
    condition = rule.get("condition") or {}
    matches = condition.get("matchesStorageClass")
    return action.get("type") == "Delete" and matches == NON_STANDARD_STORAGE_CLASSES


def relative_object_paths_for_bucket(prefix_rows: list[dict[str, str]], bucket: str) -> list[str]:
    values = [relative_prefix(row["prefix_url"], bucket) for row in prefix_rows if row["bucket"] == bucket]
    return sorted(dict.fromkeys(values))


def preview_rows(rows: list[dict[str, Any]], *, limit: int = 3) -> str:
    if not rows:
        return "0 rows"
    sample = ", ".join(str(row) for row in rows[:limit])
    suffix = "" if len(rows) <= limit else f", ... ({len(rows)} rows total)"
    return sample + suffix


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


def filter_rows_by_region(rows: list[dict[str, str]], ctx: Context, bucket_key: str = "bucket") -> list[dict[str, str]]:
    return [row for row in rows if bucket_selected(ctx, row[bucket_key])]


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
            future_to_listing_prefix = {
                executor.submit(cached_child_prefixes, ctx, listing_prefix): listing_prefix
                for listing_prefix in listing_prefixes
            }
            for future in as_completed(future_to_listing_prefix):
                listing_prefix = future_to_listing_prefix[future]
                listing_calls += 1
                progress.set_postfix_str(listing_prefix.removeprefix(f"gs://{region_bucket(region)}/")[:60])
                candidate_prefixes_by_listing_prefix[listing_prefix] = future.result()
                progress.update(1)
        progress.close()
        for listing_prefix in listing_prefixes:
            candidate_prefixes = candidate_prefixes_by_listing_prefix[listing_prefix]
            for row in rows_by_listing_prefix[listing_prefix]:
                for candidate in candidate_prefixes:
                    resolved_prefix = resolve_prefix_from_match(row["normalized_glob"], row["listing_prefix"], candidate)
                    if resolved_prefix is None:
                        continue
                    resolved_rows.append(
                        {
                            "prefix_url": resolved_prefix,
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


def build_backup_inputs(ctx: Context, action: StepSpec) -> None:
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
        output_path = BACKUP_DIR / f"backup_prefixes_{region}.csv"
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
                "source_bucket": bucket,
                "location": BUCKET_LOCATIONS[bucket],
                "backup_bucket": suggest_backup_bucket(bucket),
                "backup_prefix_csv": str(output_path.relative_to(REPO_ROOT)),
                "prefix_count": str(len(deduped)),
            }
        )
    plan_path = BACKUP_DIR / "backup_bucket_plan.csv"
    write_csv_rows(
        plan_path,
        plan_rows,
        fieldnames=["region", "source_bucket", "location", "backup_bucket", "backup_prefix_csv", "prefix_count"],
    )
    written_outputs.append(plan_path)
    print_summary(f"{action.action_id}: wrote merged backup inputs for {len(plan_rows)} regions")
    write_marker(ctx, action, fingerprint, outputs=written_outputs, extra={"regions": plan_rows})


def create_backup_buckets(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "backup_bucket_plan.csv")
    fingerprint = hashlib.sha256(
        (file_digest(BACKUP_DIR / "backup_bucket_plan.csv") + ctx.region_key).encode()
    ).hexdigest()
    verification_rows: list[dict[str, str]] = []
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        bucket_url = f"gs://{row['backup_bucket']}"
        metadata = run_command(
            ctx,
            ["gcloud", "storage", "buckets", "describe", bucket_url, "--format=json"],
            allow_failure=True,
            capture_json=True,
        )
        status = "exists"
        if metadata is None:
            status = "missing"
            print_summary(f"{action.action_id}: {'would create' if ctx.dry_run else 'creating'} {bucket_url}")
            if not ctx.dry_run:
                run_command(
                    ctx,
                    [
                        "gcloud",
                        "storage",
                        "buckets",
                        "create",
                        bucket_url,
                        f"--location={row['location']}",
                        "--soft-delete-duration=0",
                        "--no-enable-autoclass",
                    ],
                )
            metadata = gcloud_bucket_describe(ctx, bucket_url)
            status = "created"
        else:
            print_summary(f"{action.action_id}: {'would reconcile' if ctx.dry_run else 'reconciling'} {bucket_url}")
            if not ctx.dry_run:
                run_command(
                    ctx,
                    [
                        "gcloud",
                        "storage",
                        "buckets",
                        "update",
                        bucket_url,
                        "--clear-soft-delete",
                        "--no-enable-autoclass",
                    ],
                )
            metadata = gcloud_bucket_describe(ctx, bucket_url)
            status = "updated" if not ctx.dry_run else "planned"
        autoclass_enabled = bucket_autoclass_enabled(metadata)
        soft_delete_seconds = bucket_soft_delete_seconds(metadata)
        verification_rows.append(
            {
                "region": region,
                "backup_bucket": row["backup_bucket"],
                "status": status,
                "location": metadata.get("location", ""),
                "autoclass_enabled": str(autoclass_enabled).lower(),
                "soft_delete_seconds": str(soft_delete_seconds),
            }
        )
        if metadata.get("location") != row["location"]:
            raise RuntimeError(f"Backup bucket {bucket_url} is in {metadata.get('location')} not {row['location']}")
        if autoclass_enabled:
            raise RuntimeError(f"Backup bucket {bucket_url} still has Autoclass enabled")
        if soft_delete_seconds != 0:
            raise RuntimeError(f"Backup bucket {bucket_url} still has soft delete configured")
        remote_summary["regions"][region] = verification_rows[-1]
    output_path = BACKUP_DIR / "backup_bucket_state.csv"
    write_csv_rows(
        output_path,
        verification_rows,
        fieldnames=["region", "backup_bucket", "status", "location", "autoclass_enabled", "soft_delete_seconds"],
    )
    print_summary(f"{action.action_id}: reconciled {len(verification_rows)} backup buckets")
    if not ctx.dry_run:
        write_marker(ctx, action, fingerprint, outputs=[output_path], remote_summary=remote_summary)


def prepare_backup_jobs(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "backup_bucket_plan.csv")
    fingerprint = hashlib.sha256(
        (file_digest(BACKUP_DIR / "backup_bucket_plan.csv") + ctx.region_key).encode()
    ).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: outputs and marker are current")
        return

    outputs: list[Path] = []
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        prefix_rows = read_csv_rows(REPO_ROOT / row["backup_prefix_csv"])
        include_prefixes = relative_object_paths_for_bucket(prefix_rows, row["source_bucket"])
        job_name = stable_job_name(region, include_prefixes)
        spec_path = BACKUP_DIR / f"sts_job_{region}.json"
        prefix_list_path = BACKUP_DIR / f"sts_job_include_prefixes_{region}.txt"
        spec = {
            "name": job_name,
            "description": f"Protect-set backup for {region}",
            "source": f"gs://{row['source_bucket']}",
            "destination": f"gs://{row['backup_bucket']}",
            "include_prefixes": include_prefixes,
        }
        write_json(spec_path, spec)
        prefix_list_path.write_text("\n".join(include_prefixes) + ("\n" if include_prefixes else ""))
        outputs.extend([spec_path, prefix_list_path])
    print_summary(f"{action.action_id}: wrote backup job specs for {len(outputs) // 2} regions")
    write_marker(ctx, action, fingerprint, outputs=outputs)


def estimate_backup_prefix(ctx: Context, bucket_name: str, prefix: str) -> PrefixEstimate:
    client = storage_client(ctx)
    object_count = 0
    total_bytes = 0
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        object_count += 1
        total_bytes += int(blob.size or 0)
    return PrefixEstimate(prefix=prefix, object_count=object_count, total_bytes=total_bytes)


def estimate_backup_size(ctx: Context, action: StepSpec) -> None:
    regions = selected_regions(ctx)
    spec_paths = [BACKUP_DIR / f"sts_job_{region}.json" for region in regions]
    missing_specs = [path for path in spec_paths if not path.exists()]
    if missing_specs:
        raise RuntimeError(
            "Missing STS job specs. Run `prep prepare-backup-jobs` first:\n"
            + "\n".join(str(path.relative_to(REPO_ROOT)) for path in missing_specs)
        )
    fingerprint = hashlib.sha256(
        ("".join(file_digest(path) for path in spec_paths) + ctx.region_key).encode()
    ).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: outputs and marker are current")
        return

    summary_rows: list[dict[str, str]] = []
    outputs: list[Path] = []
    remote_summary: dict[str, Any] = {"project": resolved_project(ctx), "regions": {}}
    for region in regions:
        spec_path = BACKUP_DIR / f"sts_job_{region}.json"
        spec = read_json(spec_path)
        source = spec.get("source")
        include_prefixes = spec.get("include_prefixes")
        if not isinstance(source, str) or not source.startswith("gs://"):
            raise RuntimeError(f"Invalid or missing `source` in {spec_path.relative_to(REPO_ROOT)}")
        if not isinstance(include_prefixes, list) or not all(isinstance(prefix, str) for prefix in include_prefixes):
            raise RuntimeError(f"Invalid or missing `include_prefixes` in {spec_path.relative_to(REPO_ROOT)}")
        bucket_name = url_bucket(source)
        kept_prefixes, skipped_prefixes = collapse_overlapping_prefixes(include_prefixes)
        cached_estimates: list[PrefixEstimate] = []
        pending_prefixes: list[str] = []
        for prefix in kept_prefixes:
            cached_estimate = estimate_cache_lookup(bucket_name, prefix)
            if cached_estimate is None:
                pending_prefixes.append(prefix)
                continue
            cached_estimates.append(cached_estimate)
        workers = max(1, min(ctx.estimate_workers, len(pending_prefixes) or 1))
        print_summary(
            f"{action.action_id}: region {region} estimating {len(kept_prefixes)} deduped prefixes "
            f"from {len(include_prefixes)} inputs"
        )
        print_summary(
            f"{action.action_id}: region {region} reused {len(cached_estimates)} cached prefix estimates, "
            f"scanning {len(pending_prefixes)} prefixes"
        )
        estimates = list(cached_estimates)
        if pending_prefixes:
            if len(pending_prefixes) <= 5:
                print_summary(f"{action.action_id}: region {region} live prefixes: {', '.join(pending_prefixes)}")
            else:
                print_summary(f"{action.action_id}: region {region} scanning with {workers} workers")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_prefix = {
                    executor.submit(estimate_backup_prefix, ctx, bucket_name, prefix): prefix
                    for prefix in pending_prefixes
                }
                progress = tqdm(total=len(pending_prefixes), desc=f"estimate {region}", unit="prefix", leave=True)
                if len(pending_prefixes) == 1:
                    progress.set_postfix_str(pending_prefixes[0][:60])
                for future in as_completed(future_to_prefix):
                    prefix = future_to_prefix[future]
                    estimate = future.result()
                    write_estimate_cache(estimate, bucket_name)
                    estimates.append(estimate)
                    progress.set_postfix_str(prefix[:60])
                    progress.update(1)
                progress.close()
        else:
            print_summary(f"{action.action_id}: region {region} has no uncached prefixes to scan")
        estimates.sort(key=lambda item: item.total_bytes, reverse=True)
        total_objects = sum(item.object_count for item in estimates)
        total_bytes = sum(item.total_bytes for item in estimates)
        top_prefixes = [
            {
                "prefix": item.prefix,
                "object_count": item.object_count,
                "total_bytes": item.total_bytes,
                "total_human_bytes": human_bytes(item.total_bytes),
            }
            for item in estimates[:50]
        ]
        monthly_cost = estimate_monthly_cost(total_bytes, region)
        region_output = size_estimate_path(region)
        write_json(
            region_output,
            {
                "region": region,
                "project": resolved_project(ctx),
                "source_bucket": bucket_name,
                "spec_path": str(spec_path.relative_to(REPO_ROOT)),
                "input_prefix_count": len(include_prefixes),
                "deduped_prefix_count": len(kept_prefixes),
                "skipped_nested_prefix_count": len(skipped_prefixes),
                "skipped_nested_prefixes": skipped_prefixes,
                "total_object_count": total_objects,
                "total_bytes": total_bytes,
                "total_human_bytes": human_bytes(total_bytes),
                "estimated_monthly_cost_usd": round(monthly_cost, 2),
                "top_prefixes": top_prefixes,
            },
        )
        outputs.append(region_output)
        summary_rows.append(
            {
                "region": region,
                "source_bucket": bucket_name,
                "input_prefix_count": str(len(include_prefixes)),
                "deduped_prefix_count": str(len(kept_prefixes)),
                "total_object_count": str(total_objects),
                "total_bytes": str(total_bytes),
                "total_human_bytes": human_bytes(total_bytes),
                "estimated_monthly_cost_usd": f"{monthly_cost:.2f}",
            }
        )
        remote_summary["regions"][region] = {
            "source_bucket": bucket_name,
            "total_object_count": total_objects,
            "total_bytes": total_bytes,
            "total_human_bytes": human_bytes(total_bytes),
            "estimated_monthly_cost_usd": round(monthly_cost, 2),
            "top_prefixes": top_prefixes,
        }
        print_summary(
            f"{region}: {human_bytes(total_bytes)} ({total_objects} objects) "
            f"~${monthly_cost:,.2f}/mo (after 50% discount)"
        )
        if top_prefixes:
            print_summary("top 5 prefixes:")
            for prefix_summary in top_prefixes:
                print_summary(
                    f"  {prefix_summary['total_human_bytes']:>12}  "
                    f"{prefix_summary['object_count']:>8}  {prefix_summary['prefix']}"
                )

    summary_path = BACKUP_DIR / "backup_size_summary.csv"
    write_csv_rows(
        summary_path,
        summary_rows,
        fieldnames=[
            "region",
            "source_bucket",
            "input_prefix_count",
            "deduped_prefix_count",
            "total_object_count",
            "total_bytes",
            "total_human_bytes",
            "estimated_monthly_cost_usd",
        ],
    )
    outputs.append(summary_path)
    total_monthly = sum(float(row["estimated_monthly_cost_usd"]) for row in summary_rows)
    total_annual = total_monthly * 12
    total_all_bytes = sum(int(row["total_bytes"]) for row in summary_rows)
    print_summary(f"{action.action_id}: wrote size estimates for {len(summary_rows)} regions")
    print_summary(
        f"  total: {human_bytes(total_all_bytes)} — "
        f"~${total_monthly:,.2f}/mo, ~${total_annual:,.2f}/yr (GCS Standard, 50% discount)"
    )
    write_marker(ctx, action, fingerprint, outputs=outputs, remote_summary=remote_summary)


def create_backup_jobs(ctx: Context, action: StepSpec) -> None:
    spec_paths = [BACKUP_DIR / f"sts_job_{region}.json" for region in selected_regions(ctx)]
    fingerprint = hashlib.sha256(
        ("".join(file_digest(path) for path in spec_paths if path.exists()) + ctx.region_key).encode()
    ).hexdigest()
    marker_path = ctx.state_path(action.action_id)
    if marker_path.exists() and not ctx.force:
        marker = read_json(marker_path)
        if marker.get("input_fingerprint") == fingerprint:
            job_names = marker.get("job_names", [])
            if job_names and all(gcloud_transfer_job_describe(ctx, job_name) is not None for job_name in job_names):
                print_summary(f"skip {action.action_id}: STS jobs already exist and were verified")
                return

    created_jobs: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for region in selected_regions(ctx):
        spec_path = BACKUP_DIR / f"sts_job_{region}.json"
        if not spec_path.exists():
            raise RuntimeError(f"Missing STS spec for {region}: {spec_path}")
        spec = read_json(spec_path)
        prefix_list_path = BACKUP_DIR / f"sts_job_include_prefixes_{region}.txt"
        outputs.extend([spec_path, prefix_list_path])
        job_name = spec["name"]
        existing = gcloud_transfer_job_describe(ctx, job_name)
        if existing is not None and not ctx.force:
            created_jobs.append(existing)
            continue
        print_summary(f"{action.action_id}: {'would create' if ctx.dry_run else 'creating'} STS job {job_name}")
        if ctx.dry_run:
            continue
        command = [
            "gcloud",
            "transfer",
            "jobs",
            "create",
            spec["source"],
            spec["destination"],
            f"--name={job_name}",
            f"--description={spec['description']}",
            "--do-not-run",
            f"--include-prefixes={','.join(spec['include_prefixes'])}",
            "--format=json",
        ]
        created = run_command(ctx, command, capture_json=True)
        verified = gcloud_transfer_job_describe(ctx, job_name)
        if verified is None:
            raise RuntimeError(f"Failed to verify created STS job {job_name}")
        created_jobs.append(created or verified)
    if not ctx.dry_run:
        write_marker(
            ctx,
            action,
            fingerprint,
            outputs=outputs,
            remote_summary={"created_jobs": created_jobs},
            extra={"job_names": [job["name"] for job in created_jobs if "name" in job]},
        )


def validate_backup_sample(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "backup_bucket_plan.csv")
    fingerprint = hashlib.sha256(
        (file_digest(BACKUP_DIR / "backup_bucket_plan.csv") + ctx.region_key).encode()
    ).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: outputs and marker are current")
        return

    verification_rows: list[dict[str, str]] = []
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        prefix_rows = read_csv_rows(REPO_ROOT / row["backup_prefix_csv"])
        sampled = prefix_rows[: ctx.sample_size]
        found = 0
        for prefix_row in sampled:
            source_prefix = prefix_row["prefix_url"]
            backup_prefix = source_prefix.replace(
                f"gs://{row['source_bucket']}/",
                f"gs://{row['backup_bucket']}/",
                1,
            )
            objects = gcloud_ls(ctx, recursive_listing_url(backup_prefix), recursive=False)
            if objects:
                found += 1
        verification_rows.append(
            {
                "region": region,
                "sampled_prefixes": str(len(sampled)),
                "prefixes_with_objects": str(found),
                "backup_bucket": row["backup_bucket"],
            }
        )
        remote_summary["regions"][region] = verification_rows[-1]
        if sampled and found == 0:
            raise RuntimeError(f"No sampled backup prefixes were readable in {row['backup_bucket']}")
    output_path = BACKUP_DIR / "backup_validation.csv"
    write_csv_rows(
        output_path,
        verification_rows,
        fieldnames=["region", "sampled_prefixes", "prefixes_with_objects", "backup_bucket"],
    )
    print_summary(f"{action.action_id}: validated sampled backup prefixes")
    write_marker(ctx, action, fingerprint, outputs=[output_path], remote_summary=remote_summary)


def materialize_hold_manifest(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "backup_bucket_plan.csv")
    fingerprint = hashlib.sha256(
        (file_digest(BACKUP_DIR / "backup_bucket_plan.csv") + ctx.region_key).encode()
    ).hexdigest()
    if not ctx.force and marker_matches(ctx, action, fingerprint):
        print_summary(f"skip {action.action_id}: outputs and marker are current")
        return

    outputs: list[Path] = []
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        prefix_rows = read_csv_rows(REPO_ROOT / row["backup_prefix_csv"])
        hold_rows: list[dict[str, str]] = []
        for prefix_row in prefix_rows:
            for object_url in gcloud_ls(ctx, recursive_listing_url(prefix_row["prefix_url"])):
                hold_rows.append(
                    {
                        "object_url": object_url,
                        "bucket": row["source_bucket"],
                        "prefix_url": prefix_row["prefix_url"],
                    }
                )
        output_path = PURGE_DIR / f"hold_objects_{region}.csv"
        write_csv_rows(output_path, hold_rows, fieldnames=["object_url", "bucket", "prefix_url"])
        outputs.append(output_path)
        remote_summary["regions"][region] = {"objects": len(hold_rows)}
    print_summary(f"{action.action_id}: materialized exact hold objects")
    write_marker(ctx, action, fingerprint, outputs=outputs, remote_summary=remote_summary)


def verify_object_holds(ctx: Context, object_urls: list[str], *, expected: bool) -> bool:
    for object_url in object_urls:
        metadata = gcloud_object_describe(ctx, object_url)
        if metadata is None:
            return False
        actual = bool(metadata.get("temporaryHold", False))
        if actual != expected:
            return False
    return True


def update_temporary_holds(ctx: Context, object_urls: list[str], *, enable: bool) -> None:
    if not object_urls:
        return
    flag = "--temporary-hold" if enable else "--no-temporary-hold"
    run_command(
        ctx,
        ["gcloud", "storage", "objects", "update", "--read-paths-from-stdin", flag],
        input_text="\n".join(object_urls) + "\n",
    )


def apply_temporary_holds(ctx: Context, action: StepSpec) -> None:
    manifest_paths = [PURGE_DIR / f"hold_objects_{region}.csv" for region in selected_regions(ctx)]
    fingerprint = hashlib.sha256(
        ("".join(file_digest(path) for path in manifest_paths if path.exists()) + ctx.region_key).encode()
    ).hexdigest()
    object_urls: list[str] = []
    for path in manifest_paths:
        if path.exists():
            object_urls.extend(row["object_url"] for row in read_csv_rows(path))
    if (
        not ctx.force
        and marker_matches(ctx, action, fingerprint)
        and verify_object_holds(ctx, object_urls, expected=True)
    ):
        print_summary(f"skip {action.action_id}: temporary holds already verified")
        return

    print_summary(f"{action.action_id}: applying temporary holds to {len(object_urls)} objects")
    if ctx.dry_run:
        print_summary(f"{action.action_id}: dry-run only, would apply temporary holds")
        return
    update_temporary_holds(ctx, object_urls, enable=True)
    if not verify_object_holds(ctx, object_urls, expected=True):
        raise RuntimeError("Temporary hold verification failed after update")
    write_marker(
        ctx,
        action,
        fingerprint,
        outputs=manifest_paths,
        remote_summary={"object_count": len(object_urls), "temporary_hold": True},
    )


def disable_autoclass(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "backup_bucket_plan.csv")
    fingerprint = hashlib.sha256(
        (file_digest(BACKUP_DIR / "backup_bucket_plan.csv") + ctx.region_key).encode()
    ).hexdigest()
    snapshot_paths: list[Path] = []
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        bucket_url = f"gs://{row['source_bucket']}"
        metadata = gcloud_bucket_describe(ctx, bucket_url)
        before_path = PURGE_DIR / f"bucket_before_autoclass_disable_{region}.json"
        write_json(before_path, metadata)
        snapshot_paths.append(before_path)
        if not ctx.force and marker_matches(ctx, action, fingerprint) and not bucket_autoclass_enabled(metadata):
            remote_summary["regions"][region] = {"autoclass_enabled": False}
            continue
        print_summary(f"{action.action_id}: {'would disable' if ctx.dry_run else 'disabling'} Autoclass on {bucket_url}")
        if not ctx.dry_run:
            run_command(ctx, ["gcloud", "storage", "buckets", "update", bucket_url, "--no-enable-autoclass"])
        after = gcloud_bucket_describe(ctx, bucket_url)
        if not ctx.dry_run and bucket_autoclass_enabled(after):
            raise RuntimeError(f"Autoclass is still enabled for {bucket_url}")
        remote_summary["regions"][region] = {"autoclass_enabled": bucket_autoclass_enabled(after)}
    if not ctx.dry_run:
        write_marker(ctx, action, fingerprint, outputs=snapshot_paths, remote_summary=remote_summary)


def lifecycle_rule_payload(existing_rules: list[dict[str, Any]]) -> dict[str, Any]:
    merged_rules = [rule for rule in existing_rules if not is_delete_nonstandard_rule(rule)]
    merged_rules.append(
        {
            "action": {"type": "Delete"},
            "condition": {"matchesStorageClass": NON_STANDARD_STORAGE_CLASSES},
        }
    )
    return {"rule": merged_rules}


def apply_delete_lifecycle(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "backup_bucket_plan.csv")
    fingerprint = hashlib.sha256(
        (file_digest(BACKUP_DIR / "backup_bucket_plan.csv") + ctx.region_key).encode()
    ).hexdigest()
    outputs: list[Path] = []
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        bucket_url = f"gs://{row['source_bucket']}"
        before = gcloud_bucket_describe(ctx, bucket_url)
        before_path = PURGE_DIR / f"bucket_lifecycle_before_{region}.json"
        desired_path = PURGE_DIR / f"lifecycle_delete_nonstandard_{region}.json"
        write_json(before_path, before)
        payload = lifecycle_rule_payload(lifecycle_rules_from_metadata(before))
        write_json(desired_path, payload)
        outputs.extend([before_path, desired_path])
        if (
            not ctx.force
            and marker_matches(ctx, action, fingerprint)
            and any(is_delete_nonstandard_rule(rule) for rule in lifecycle_rules_from_metadata(before))
        ):
            remote_summary["regions"][region] = {"delete_rule_present": True}
            continue
        print_summary(
            f"{action.action_id}: {'would apply' if ctx.dry_run else 'applying'} lifecycle delete to {bucket_url}"
        )
        if not ctx.dry_run:
            run_command(
                ctx,
                ["gcloud", "storage", "buckets", "update", bucket_url, f"--lifecycle-file={desired_path}"],
            )
        after = gcloud_bucket_describe(ctx, bucket_url)
        if not ctx.dry_run and not any(
            is_delete_nonstandard_rule(rule) for rule in lifecycle_rules_from_metadata(after)
        ):
            raise RuntimeError(f"Lifecycle delete rule is missing on {bucket_url}")
        remote_summary["regions"][region] = {"delete_rule_present": True}
    if not ctx.dry_run:
        write_marker(ctx, action, fingerprint, outputs=outputs, remote_summary=remote_summary)


def wait_for_lifecycle(ctx: Context, action: StepSpec) -> None:
    prerequisite_marker = ctx.state_path("purge.apply_delete_lifecycle")
    if not prerequisite_marker.exists():
        raise RuntimeError("purge.wait_for_lifecycle requires purge.apply_delete_lifecycle to be complete first")
    fingerprint = file_digest(prerequisite_marker)
    marker_path = ctx.state_path(action.action_id)
    settle_deadline = now_utc() + timedelta(hours=ctx.settle_hours)
    if marker_path.exists() and not ctx.force:
        marker = read_json(marker_path)
        existing_deadline = datetime.fromisoformat(marker["settle_deadline"])
        if now_utc() < existing_deadline:
            raise RuntimeError(
                f"Lifecycle settle window still open until {existing_deadline.isoformat()}. "
                "Rerun after the deadline or use --force to override."
            )
        print_summary(f"{action.action_id}: lifecycle settle window already elapsed")
        return
    print_summary(
        f"{action.action_id}: recording lifecycle settle window for {ctx.settle_hours} hours "
        f"(until {settle_deadline.isoformat()})"
    )
    write_marker(
        ctx,
        action,
        fingerprint,
        outputs=[],
        extra={"settle_deadline": settle_deadline.isoformat()},
    )


def remove_delete_lifecycle(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "backup_bucket_plan.csv")
    fingerprint = hashlib.sha256(
        (file_digest(BACKUP_DIR / "backup_bucket_plan.csv") + ctx.region_key).encode()
    ).hexdigest()
    outputs: list[Path] = []
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        bucket_url = f"gs://{row['source_bucket']}"
        before_path = PURGE_DIR / f"bucket_lifecycle_before_{region}.json"
        if not before_path.exists():
            raise RuntimeError(f"Missing previous lifecycle snapshot for {region}: {before_path}")
        previous = read_json(before_path)
        previous_rules = lifecycle_rules_from_metadata(previous)
        previous_path = PURGE_DIR / f"lifecycle_restore_{region}.json"
        if previous_rules:
            write_json(previous_path, {"rule": previous_rules})
            outputs.append(previous_path)
        metadata = gcloud_bucket_describe(ctx, bucket_url)
        if (
            not ctx.force
            and marker_matches(ctx, action, fingerprint)
            and not any(is_delete_nonstandard_rule(rule) for rule in lifecycle_rules_from_metadata(metadata))
        ):
            remote_summary["regions"][region] = {"delete_rule_present": False}
            continue
        print_summary(f"{action.action_id}: {'would restore' if ctx.dry_run else 'restoring'} lifecycle on {bucket_url}")
        if not ctx.dry_run:
            if previous_rules:
                run_command(
                    ctx,
                    ["gcloud", "storage", "buckets", "update", bucket_url, f"--lifecycle-file={previous_path}"],
                )
            else:
                run_command(ctx, ["gcloud", "storage", "buckets", "update", bucket_url, "--clear-lifecycle"])
        after = gcloud_bucket_describe(ctx, bucket_url)
        if not ctx.dry_run and any(is_delete_nonstandard_rule(rule) for rule in lifecycle_rules_from_metadata(after)):
            raise RuntimeError(f"Temporary lifecycle delete rule still present on {bucket_url}")
        remote_summary["regions"][region] = {"delete_rule_present": False}
    if not ctx.dry_run:
        write_marker(ctx, action, fingerprint, outputs=outputs, remote_summary=remote_summary)


def reenable_autoclass(ctx: Context, action: StepSpec) -> None:
    plan_rows = read_csv_rows(BACKUP_DIR / "backup_bucket_plan.csv")
    fingerprint = hashlib.sha256(
        (file_digest(BACKUP_DIR / "backup_bucket_plan.csv") + ctx.region_key).encode()
    ).hexdigest()
    remote_summary: dict[str, Any] = {"regions": {}}
    outputs: list[Path] = []
    for row in plan_rows:
        region = row["region"]
        if ctx.selected_regions and region not in ctx.selected_regions:
            continue
        bucket_url = f"gs://{row['source_bucket']}"
        snapshot_path = PURGE_DIR / f"bucket_after_autoclass_reenable_{region}.json"
        metadata = gcloud_bucket_describe(ctx, bucket_url)
        if not ctx.force and marker_matches(ctx, action, fingerprint) and bucket_autoclass_enabled(metadata):
            write_json(snapshot_path, metadata)
            outputs.append(snapshot_path)
            remote_summary["regions"][region] = {"autoclass_enabled": True}
            continue
        print_summary(f"{action.action_id}: {'would enable' if ctx.dry_run else 'enabling'} Autoclass on {bucket_url}")
        if not ctx.dry_run:
            run_command(ctx, ["gcloud", "storage", "buckets", "update", bucket_url, "--enable-autoclass"])
        after = gcloud_bucket_describe(ctx, bucket_url)
        write_json(snapshot_path, after)
        outputs.append(snapshot_path)
        if not ctx.dry_run and not bucket_autoclass_enabled(after):
            raise RuntimeError(f"Autoclass is still disabled for {bucket_url}")
        remote_summary["regions"][region] = {"autoclass_enabled": bucket_autoclass_enabled(after)}
    if not ctx.dry_run:
        write_marker(ctx, action, fingerprint, outputs=outputs, remote_summary=remote_summary)


def clear_temporary_holds(ctx: Context, action: StepSpec) -> None:
    manifest_paths = [PURGE_DIR / f"hold_objects_{region}.csv" for region in selected_regions(ctx)]
    fingerprint = hashlib.sha256(
        ("".join(file_digest(path) for path in manifest_paths if path.exists()) + ctx.region_key).encode()
    ).hexdigest()
    object_urls: list[str] = []
    for path in manifest_paths:
        if path.exists():
            object_urls.extend(row["object_url"] for row in read_csv_rows(path))
    if (
        not ctx.force
        and marker_matches(ctx, action, fingerprint)
        and verify_object_holds(ctx, object_urls, expected=False)
    ):
        print_summary(f"skip {action.action_id}: temporary holds already cleared")
        return
    print_summary(f"{action.action_id}: {'would clear' if ctx.dry_run else 'clearing'} temporary holds")
    if ctx.dry_run:
        return
    update_temporary_holds(ctx, object_urls, enable=False)
    if not verify_object_holds(ctx, object_urls, expected=False):
        raise RuntimeError("Temporary hold verification failed after clear")
    write_marker(
        ctx,
        action,
        fingerprint,
        outputs=manifest_paths,
        remote_summary={"object_count": len(object_urls), "temporary_hold": False},
    )


STEPS = [
    StepSpec(
        action_id="prep.resolve_listing_prefixes",
        group_name="prep",
        command_name="resolve-listing-prefixes",
        description="Resolve listing-based protect families into concrete prefixes.",
        help_text=(
            "Resolve wildcard protect families into concrete prefixes by listing one level below each parent prefix. "
            "This step explains exactly what will be copied later, caches the child-prefix listings, and avoids "
            "recursive subtree walks. Use `--listing-workers` to control parallel listing."
        ),
        outputs=tuple(
            sorted(
                [
                    *(
                        RESOLVE_DIR / f"listing_prefixes_{region}.csv"
                        for region in [region_from_bucket(bucket) for bucket in MARIN_BUCKETS]
                    ),
                    *(
                        RESOLVE_DIR / f"resolved_prefixes_{region}.csv"
                        for region in [region_from_bucket(bucket) for bucket in MARIN_BUCKETS]
                    ),
                ]
            )
        ),
        mutating=False,
        runner=resolve_listing_prefixes,
        listing_workers=True,
        requirements=(str(PROTECT_DIR / "protect_prefixes_classified.csv"),),
    ),
    StepSpec(
        action_id="prep.build_backup_inputs",
        group_name="prep",
        command_name="build-backup-inputs",
        description="Merge direct and resolved prefixes into backup inputs and a bucket plan.",
        help_text=(
            "Merge direct keep prefixes with the resolved wildcard prefixes into one backup input set per region. "
            "This step is local-only and produces the CSVs that the copy stage and later purge steps consume. "
            "Run it again whenever the resolved prefix outputs change."
        ),
        outputs=tuple(
            [
                *(BACKUP_DIR / f"backup_prefixes_{region_from_bucket(bucket)}.csv" for bucket in MARIN_BUCKETS),
                BACKUP_DIR / "backup_bucket_plan.csv",
            ]
        ),
        mutating=False,
        runner=build_backup_inputs,
        predecessors=("prep.resolve_listing_prefixes",),
        requirements=(str(PROTECT_DIR / "protect_prefixes_direct.csv"),),
    ),
    StepSpec(
        action_id="prep.prepare_backup_jobs",
        group_name="prep",
        command_name="prepare-backup-jobs",
        description="Write STS job specs for backup copies.",
        help_text=(
            "Write STS job specs and include-prefix lists for each region without creating any transfer jobs yet. "
            "This step turns the backup input CSVs into stable, reviewable artifacts that the copy stage can execute "
            "directly. Run it before size estimation or backup-bucket creation so later commands have one canonical "
            "spec per region."
        ),
        outputs=tuple(
            [
                *(BACKUP_DIR / f"sts_job_{region_from_bucket(bucket)}.json" for bucket in MARIN_BUCKETS),
                *(BACKUP_DIR / f"sts_job_include_prefixes_{region_from_bucket(bucket)}.txt" for bucket in MARIN_BUCKETS),
            ]
        ),
        mutating=False,
        runner=prepare_backup_jobs,
        predecessors=("prep.build_backup_inputs",),
        requirements=(str(BACKUP_DIR / "backup_bucket_plan.csv"),),
    ),
    StepSpec(
        action_id="prep.estimate_backup_size",
        group_name="prep",
        command_name="estimate-backup-size",
        description="Estimate backup size from the prepared STS job specs.",
        help_text=(
            "Estimate how much data each prepared STS backup job will copy by scanning the source prefixes in Cloud "
            "Storage. This step deduplicates nested prefixes first so totals are not double-counted, then writes one "
            "JSON estimate per region plus a human-readable summary CSV. Use `--estimate-workers` to control parallel "
            "prefix scans and `--project` if the storage client should use a non-default GCP project."
        ),
        outputs=tuple(
            [
                *(size_estimate_path(region_from_bucket(bucket)) for bucket in MARIN_BUCKETS),
                BACKUP_DIR / "backup_size_summary.csv",
            ]
        ),
        mutating=False,
        runner=estimate_backup_size,
        predecessors=("prep.prepare_backup_jobs",),
        requirements=tuple(str(BACKUP_DIR / f"sts_job_{region_from_bucket(bucket)}.json") for bucket in MARIN_BUCKETS),
        estimate_workers=True,
    ),
    StepSpec(
        action_id="copy.create_backup_buckets",
        group_name="copy",
        command_name="create-backup-buckets",
        description="Create or reconcile temporary same-region backup buckets.",
        help_text=(
            "Create each temporary same-region backup bucket if it is missing, and reconcile its settings if it "
            "already exists. This step ensures the backup buckets, not the source buckets, have Autoclass disabled "
            "and soft delete cleared before any transfer jobs are created. Use `--dry-run` to inspect which buckets "
            "would be created or updated without mutating Cloud Storage."
        ),
        outputs=(BACKUP_DIR / "backup_bucket_state.csv",),
        mutating=True,
        runner=create_backup_buckets,
        predecessors=("prep.prepare_backup_jobs",),
        requirements=(str(BACKUP_DIR / "backup_bucket_plan.csv"),),
    ),
    StepSpec(
        action_id="copy.create_backup_jobs",
        group_name="copy",
        command_name="create-backup-jobs",
        description="Create STS jobs for same-region backup copies.",
        help_text=(
            "Create same-region Storage Transfer Service jobs from the prepared specs. This is the first "
            "remote-mutation step in the workflow, and it verifies that the expected STS jobs exist before marking "
            "itself complete. Use `--dry-run` to inspect the generated specs without creating jobs."
        ),
        outputs=tuple(
            [
                *(BACKUP_DIR / f"sts_job_{region_from_bucket(bucket)}.json" for bucket in MARIN_BUCKETS),
                *(BACKUP_DIR / f"sts_job_include_prefixes_{region_from_bucket(bucket)}.txt" for bucket in MARIN_BUCKETS),
            ]
        ),
        mutating=True,
        runner=create_backup_jobs,
        predecessors=("copy.create_backup_buckets",),
        requirements=tuple(str(BACKUP_DIR / f"sts_job_{region_from_bucket(bucket)}.json") for bucket in MARIN_BUCKETS),
    ),
    StepSpec(
        action_id="copy.validate_backup_sample",
        group_name="copy",
        command_name="validate-backup-sample",
        description="Inspect sampled prefixes in backup buckets.",
        help_text=(
            "Inspect a small sample of copied prefixes in each backup bucket to confirm that the transferred "
            "content is present and readable. This step gives a lightweight safety check before any source-bucket "
            "protection or deletion logic begins. Use `--sample-size` to adjust how many prefixes are sampled."
        ),
        outputs=(BACKUP_DIR / "backup_validation.csv",),
        mutating=False,
        runner=validate_backup_sample,
        predecessors=("copy.create_backup_jobs",),
        requirements=(str(BACKUP_DIR / "backup_bucket_plan.csv"),),
        sample_size=True,
    ),
    StepSpec(
        action_id="purge.materialize_hold_manifest",
        group_name="purge",
        command_name="materialize-hold-manifest",
        description="Expand the keep set to exact object URLs for temporary holds.",
        help_text=(
            "Expand the keep prefixes into exact object URLs that will receive temporary holds in the source "
            "buckets. This step is read-only against storage and produces the concrete hold manifest used by later "
            "purge commands. It makes the protected object set explicit before any bucket settings are changed."
        ),
        outputs=tuple(PURGE_DIR / f"hold_objects_{region_from_bucket(bucket)}.csv" for bucket in MARIN_BUCKETS),
        mutating=False,
        runner=materialize_hold_manifest,
        predecessors=("copy.validate_backup_sample",),
        requirements=(str(BACKUP_DIR / "backup_bucket_plan.csv"),),
    ),
    StepSpec(
        action_id="purge.apply_temporary_holds",
        group_name="purge",
        command_name="apply-temporary-holds",
        description="Apply temporary holds to the protected in-place objects.",
        help_text=(
            "Apply temporary holds to every object in the materialized hold manifest so lifecycle deletion cannot "
            "remove them. This is the main guardrail that lets the later lifecycle rule stay broad without touching "
            "protected artifacts. Use `--dry-run` to inspect counts and inputs without changing object metadata."
        ),
        outputs=tuple(PURGE_DIR / f"hold_objects_{region_from_bucket(bucket)}.csv" for bucket in MARIN_BUCKETS),
        mutating=True,
        runner=apply_temporary_holds,
        predecessors=("purge.materialize_hold_manifest",),
    ),
    StepSpec(
        action_id="purge.disable_autoclass",
        group_name="purge",
        command_name="disable-autoclass",
        description="Disable Autoclass on the source buckets.",
        help_text=(
            "Disable Autoclass on the source buckets so the current storage classes stop changing during the purge "
            "window. This step snapshots bucket metadata first and verifies that Autoclass is truly off afterward. "
            "It should only run after backup validation and temporary holds are in place."
        ),
        outputs=tuple(
            PURGE_DIR / f"bucket_before_autoclass_disable_{region_from_bucket(bucket)}.json" for bucket in MARIN_BUCKETS
        ),
        mutating=True,
        runner=disable_autoclass,
        predecessors=("purge.apply_temporary_holds",),
        requirements=(str(BACKUP_DIR / "backup_bucket_plan.csv"),),
    ),
    StepSpec(
        action_id="purge.apply_delete_lifecycle",
        group_name="purge",
        command_name="apply-delete-lifecycle",
        description="Apply the temporary non-STANDARD delete lifecycle rule.",
        help_text=(
            "Apply the temporary lifecycle rule that deletes non-`STANDARD` objects from the source buckets. This "
            "is the broad deletion mechanism for cold, unprotected data, so it writes the exact lifecycle JSON "
            "alongside the before-state snapshots. Use `--dry-run` to review the planned lifecycle payloads "
            "without updating buckets."
        ),
        outputs=tuple(
            [
                *(PURGE_DIR / f"bucket_lifecycle_before_{region_from_bucket(bucket)}.json" for bucket in MARIN_BUCKETS),
                *(
                    PURGE_DIR / f"lifecycle_delete_nonstandard_{region_from_bucket(bucket)}.json"
                    for bucket in MARIN_BUCKETS
                ),
            ]
        ),
        mutating=True,
        runner=apply_delete_lifecycle,
        predecessors=("purge.disable_autoclass",),
        requirements=(str(BACKUP_DIR / "backup_bucket_plan.csv"),),
    ),
    StepSpec(
        action_id="purge.wait_for_lifecycle",
        group_name="purge",
        command_name="wait-for-lifecycle",
        description="Record and honor the lifecycle settle window.",
        help_text=(
            "Record the lifecycle settle window and refuse cleanup until that window has elapsed. This step is "
            "intentionally a checkpoint rather than a long sleep, so reruns remain explicit and auditable. "
            "Use `--settle-hours` to adjust the recorded waiting period."
        ),
        outputs=tuple(),
        mutating=False,
        runner=wait_for_lifecycle,
        predecessors=("purge.apply_delete_lifecycle",),
        settle_hours=True,
    ),
    StepSpec(
        action_id="purge.remove_delete_lifecycle",
        group_name="purge",
        command_name="remove-delete-lifecycle",
        description="Restore each source bucket's pre-purge lifecycle configuration.",
        help_text=(
            "Remove the temporary delete rule and restore the previous lifecycle configuration on each source "
            "bucket. This step verifies that the non-`STANDARD` delete rule is gone before it marks itself "
            "complete. It should only run after the recorded settle window has elapsed and the purge result "
            "has been reviewed."
        ),
        outputs=tuple(PURGE_DIR / f"lifecycle_restore_{region_from_bucket(bucket)}.json" for bucket in MARIN_BUCKETS),
        mutating=True,
        runner=remove_delete_lifecycle,
        predecessors=("purge.wait_for_lifecycle",),
        requirements=tuple(
            str(PURGE_DIR / f"bucket_lifecycle_before_{region_from_bucket(bucket)}.json") for bucket in MARIN_BUCKETS
        ),
    ),
    StepSpec(
        action_id="purge.reenable_autoclass",
        group_name="purge",
        command_name="reenable-autoclass",
        description="Re-enable Autoclass on the source buckets.",
        help_text=(
            "Re-enable Autoclass on the source buckets so they return to normal steady-state storage behavior. "
            "This step snapshots the post-change bucket metadata and confirms that Autoclass is enabled again. "
            "It should run after lifecycle cleanup, not before."
        ),
        outputs=tuple(
            PURGE_DIR / f"bucket_after_autoclass_reenable_{region_from_bucket(bucket)}.json" for bucket in MARIN_BUCKETS
        ),
        mutating=True,
        runner=reenable_autoclass,
        predecessors=("purge.remove_delete_lifecycle",),
        requirements=(str(BACKUP_DIR / "backup_bucket_plan.csv"),),
    ),
    StepSpec(
        action_id="purge.clear_temporary_holds",
        group_name="purge",
        command_name="clear-temporary-holds",
        description="Optionally clear temporary holds after purge confidence is high.",
        help_text=(
            "Clear temporary holds from the protected objects once you are satisfied with the backup and purge "
            "result. This is optional cleanup, not part of the default end-to-end run. Use `--dry-run` if you "
            "want to inspect the manifest size without modifying object metadata."
        ),
        outputs=tuple(PURGE_DIR / f"hold_objects_{region_from_bucket(bucket)}.csv" for bucket in MARIN_BUCKETS),
        mutating=True,
        runner=clear_temporary_holds,
        predecessors=("purge.reenable_autoclass",),
        optional=True,
    ),
]

STEP_INDEX = {step.action_id: step for step in STEPS}


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
    start_index = 0 if from_action is None else next(i for i, step in enumerate(STEPS) if step.action_id == from_action)
    end_index = (
        len(STEPS) - 1 if to_action is None else next(i for i, step in enumerate(STEPS) if step.action_id == to_action)
    )
    steps = STEPS[start_index : end_index + 1]
    if include_optional:
        return steps
    if to_action is not None and STEP_INDEX[to_action].optional:
        return steps
    return [step for step in steps if not step.optional]


def plan_command(regions: set[str] | None) -> None:
    ensure_output_dirs()
    print("Ordered storage purge steps:")
    for step in STEPS:
        if step.optional:
            suffix = " [optional]"
        else:
            suffix = ""
        marker_path = Context(
            dry_run=False,
            force=False,
            include_optional=False,
            listing_workers=32,
            estimate_workers=32,
            sample_size=0,
            settle_hours=48,
            selected_regions=regions,
            log_path=LOG_DIR / f"plan_{timestamp_string()}.log",
            timestamp=timestamp_string(),
            project=None,
        ).state_path(step.action_id)
        status = "done" if marker_path.exists() else "pending"
        print(f"  {status:7}  {step.action_id}{suffix}")


def run_command_mode(
    *,
    from_action: str | None,
    to_action: str | None,
    only_action: str | None,
    dry_run: bool,
    force: bool,
    include_optional: bool,
    listing_workers: int,
    estimate_workers: int,
    sample_size: int,
    settle_hours: int,
    regions: set[str] | None,
    project: str | None,
) -> int:
    ensure_output_dirs()
    ctx = Context(
        dry_run=dry_run,
        force=force,
        include_optional=include_optional,
        listing_workers=listing_workers,
        estimate_workers=estimate_workers,
        sample_size=sample_size,
        settle_hours=settle_hours,
        selected_regions=regions,
        log_path=LOG_DIR / f"run_{timestamp_string()}.log",
        timestamp=timestamp_string(),
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
    return 0


def build_context(
    *,
    dry_run: bool,
    force: bool,
    include_optional: bool,
    listing_workers: int,
    estimate_workers: int,
    sample_size: int,
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
        sample_size=sample_size,
        settle_hours=settle_hours,
        selected_regions=parse_regions(list(regions) or None),
        log_path=LOG_DIR / f"{log_prefix}_{timestamp_string()}.log",
        timestamp=timestamp_string(),
        project=project,
    )


def assert_step_predecessors(ctx: Context, step: StepSpec) -> None:
    if ctx.force:
        return
    missing = [predecessor for predecessor in step.predecessors if not ctx.state_path(predecessor).exists()]
    if not missing:
        return
    raise RuntimeError(f"{step.action_id} requires these predecessor steps to be complete first: {', '.join(missing)}")


def invoke_single_step(ctx: Context, action_id: str) -> None:
    step = STEP_INDEX[action_id]
    print_summary(f"running 1 step; log: {ctx.log_path.relative_to(REPO_ROOT)}")
    print_summary(f"==> {step.action_id}: {step.description}")
    assert_step_predecessors(ctx, step)
    step.run(ctx)
    print_summary("completed selected steps")


def runtime_options(
    *,
    listing_workers: bool = False,
    estimate_workers: bool = False,
    sample_size: bool = False,
    settle_hours: bool = False,
    include_optional: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        options: list[Callable[[Callable[..., Any]], Callable[..., Any]]] = [
            click.option(
                "--region",
                "regions",
                multiple=True,
                help="Limit the command to one or more Marin storage regions such as `eu-west4`.",
            ),
            click.option(
                "--force",
                is_flag=True,
                help="Ignore cached markers and cached listings, and recompute the selected work.",
            ),
            click.option(
                "--dry-run",
                is_flag=True,
                help="Allow local artifact writes and read-only cloud inspection, but never apply remote mutations.",
            ),
            click.option(
                "--project",
                help=(
                    "Override the GCP project used for Cloud Storage API calls. "
                    "By default the command uses env vars or `gcloud config`."
                ),
            ),
        ]
        if include_optional:
            options.append(
                click.option(
                    "--include-optional",
                    is_flag=True,
                    help="Include optional cleanup actions such as clearing temporary holds.",
                )
            )
        if listing_workers:
            options.append(
                click.option(
                    "--listing-workers",
                    default=32,
                    show_default=True,
                    type=int,
                    help="Maximum concurrent listing-prefix fetches during wildcard resolution.",
                )
            )
        if sample_size:
            options.append(
                click.option(
                    "--sample-size",
                    default=3,
                    show_default=True,
                    type=int,
                    help="Number of copied prefixes to inspect per region when validating backup contents.",
                )
            )
        if estimate_workers:
            options.append(
                click.option(
                    "--estimate-workers",
                    default=32,
                    show_default=True,
                    type=int,
                    help="Maximum concurrent prefix scans when estimating backup size from STS specs.",
                )
            )
        if settle_hours:
            options.append(
                click.option(
                    "--settle-hours",
                    default=48,
                    show_default=True,
                    type=int,
                    help="Lifecycle settle window to record before delete-rule cleanup is allowed.",
                )
            )
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Run the one-off storage purge workflow as a click application with explicit prep, copy, and purge "
        "commands. Each subcommand owns one concrete step, writes its own artifacts, and can be inspected "
        "with `--help` before you run it."
    ),
)
def cli() -> None:
    pass


@cli.command(
    "plan",
    help=(
        "Show the ordered step list and whether a completion marker exists for each step in the selected regions. "
        "This command is read-only and is the fastest way to see where a partially completed workflow currently stands."
    ),
)
@click.option(
    "--region",
    "regions",
    multiple=True,
    help="Limit the displayed status to one or more Marin storage regions such as `eu-west4`.",
)
def plan_cli(regions: tuple[str, ...]) -> None:
    plan_command(parse_regions(list(regions) or None))


@cli.command(
    "run",
    help=(
        "Execute a contiguous slice of the ordered workflow using the same underlying steps as the explicit "
        "subcommands. Use this when you want the convenience of a range runner, but prefer the subcommands "
        "when you want one step at a time."
    ),
)
@runtime_options(
    listing_workers=True,
    estimate_workers=True,
    sample_size=True,
    settle_hours=True,
    include_optional=True,
)
@click.option(
    "--from",
    "from_action",
    type=click.Choice(sorted(STEP_INDEX)),
    help="Start from this step id in the ordered workflow.",
)
@click.option(
    "--to",
    "to_action",
    type=click.Choice(sorted(STEP_INDEX)),
    help="Stop after this step id in the ordered workflow.",
)
@click.option(
    "--only",
    "only_action",
    type=click.Choice(sorted(STEP_INDEX)),
    help="Run exactly one step id instead of a range.",
)
def run_cli(
    regions: tuple[str, ...],
    dry_run: bool,
    force: bool,
    include_optional: bool,
    listing_workers: int,
    estimate_workers: int,
    sample_size: int,
    settle_hours: int,
    project: str | None,
    from_action: str | None,
    to_action: str | None,
    only_action: str | None,
) -> None:
    run_command_mode(
        from_action=from_action,
        to_action=to_action,
        only_action=only_action,
        dry_run=dry_run,
        force=force,
        include_optional=include_optional,
        listing_workers=listing_workers,
        estimate_workers=estimate_workers,
        sample_size=sample_size,
        settle_hours=settle_hours,
        regions=parse_regions(list(regions) or None),
        project=project,
    )


@cli.group(
    help=(
        "Preparation commands derive or verify the inputs needed for copy and purge work without changing "
        "source-bucket data. These commands are the place to inspect listings, backup inputs, and bucket "
        "readiness before any remote mutations occur."
    )
)
def prep() -> None:
    pass


@cli.group(
    help=(
        "Copy commands create and validate the same-region backup jobs that protect the keep set before "
        "deletion starts. They operate on backup buckets and transfer jobs, not on destructive source-bucket "
        "lifecycle settings."
    )
)
def copy() -> None:
    pass


@cli.group(
    help=(
        "Purge commands operate on the source buckets after backups are in place and validated. "
        "These commands apply holds, freeze storage classes, perform lifecycle-based deletion, and then "
        "restore steady-state settings."
    )
)
def purge() -> None:
    pass


GROUPS: dict[str, click.Group] = {
    "prep": prep,
    "copy": copy,
    "purge": purge,
}


def register_step_command(group: click.Group, step: StepSpec) -> None:
    @runtime_options(
        listing_workers=step.listing_workers,
        estimate_workers=step.estimate_workers,
        sample_size=step.sample_size,
        settle_hours=step.settle_hours,
    )
    def command(
        regions: tuple[str, ...],
        dry_run: bool,
        force: bool,
        project: str | None,
        listing_workers: int = 32,
        estimate_workers: int = 32,
        sample_size: int = 3,
        settle_hours: int = 48,
    ) -> None:
        ctx = build_context(
            dry_run=dry_run,
            force=force,
            include_optional=False,
            listing_workers=listing_workers,
            estimate_workers=estimate_workers,
            sample_size=sample_size,
            settle_hours=settle_hours,
            regions=regions,
            log_prefix=step.action_id.replace(".", "__"),
            project=project,
        )
        invoke_single_step(ctx, step.action_id)

    command.__name__ = step.action_id.replace(".", "_").replace("-", "_")
    group.command(name=step.command_name, help=step.help_text)(command)


for step in STEPS:
    register_step_command(GROUPS[step.group_name], step)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
