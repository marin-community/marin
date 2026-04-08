#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated storage purge workflow: scan, summarize, compute, and delete.

Four steps run in order:
  1. scan      — List bucket objects into parquet segments
  2. summarize — Materialize per-directory summaries from objects
  3. compute   — Evaluate delete/protect rules, collapse, write CSV manifest
  4. delete    — Delete objects from manifest

Usage:
    uv run scripts/storage/cleanup.py plan
    uv run scripts/storage/cleanup.py run [--from STEP] [--to STEP] [--dry-run] [--workers N]
    uv run scripts/storage/cleanup.py scan [--force] [--workers N]
    uv run scripts/storage/cleanup.py summarize [--force]
    uv run scripts/storage/cleanup.py compute [--force]
    uv run scripts/storage/cleanup.py delete [--dry-run] [--workers N]
"""

from __future__ import annotations

import csv
import logging
import queue
import random
import shutil
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import click
from google.api_core.exceptions import DeadlineExceeded, NotFound, ServiceUnavailable, TooManyRequests
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from scripts.storage.db import (
    DEFAULT_CATALOG,
    GCS_DISCOUNT,
    REPO_ROOT,
    Context,
    StorageCatalog,
    _fetchall_dicts,
    continent_for_region,
    ensure_output_dirs,
    file_digest,
    human_bytes,
    materialize_dir_summary,
    plan_rows,
    print_summary,
    timestamp_string,
)
from scripts.storage.scan import (
    scan_objects,
    storage_client,
)

log = logging.getLogger(__name__)

N_LIST_WORKERS = 8
N_DELETE_WORKERS = 32
DELETE_BATCH_TIMEOUT = (10, 30)
DELETE_BATCH_MAX_ATTEMPTS = 9
DELETE_BATCH_INITIAL_BACKOFF = 2.0
DELETE_WORKER_QUEUE_TIMEOUT = 1.0

# ===========================================================================
# Step dataclass
# ===========================================================================


@dataclass(frozen=True)
class Step:
    name: str
    description: str
    runner: Callable[[Context], None]
    done: Callable[[StorageCatalog], bool]
    mutating: bool = False
    optional: bool = False


# ===========================================================================
# Done checks (file-based)
# ===========================================================================


def scan_done(catalog: StorageCatalog) -> bool:
    return any(catalog.objects_parquet_dir.glob("objects_*.parquet"))


def summarize_done(catalog: StorageCatalog) -> bool:
    return any(catalog.dir_summary_parquet_dir.glob("dir_summary_*.parquet"))


def compute_done(catalog: StorageCatalog) -> bool:
    return catalog.deletion_manifest_csv.exists()


# ===========================================================================
# Collapsing algorithm (from compute.py)
# ===========================================================================


@dataclass
class DirEntry:
    bucket: str
    prefix: str
    status: str  # "delete" or "keep"
    matched_rule: str
    standard_count: int
    standard_bytes: int
    nearline_count: int
    nearline_bytes: int
    coldline_count: int
    coldline_bytes: int
    archive_count: int
    archive_bytes: int

    @property
    def object_count(self) -> int:
        return self.standard_count + self.nearline_count + self.coldline_count + self.archive_count

    @property
    def total_bytes(self) -> int:
        return self.standard_bytes + self.nearline_bytes + self.coldline_bytes + self.archive_bytes

    @property
    def storage_class_breakdown(self) -> str:
        parts = []
        for name, count in [
            ("STANDARD", self.standard_count),
            ("NEARLINE", self.nearline_count),
            ("COLDLINE", self.coldline_count),
            ("ARCHIVE", self.archive_count),
        ]:
            if count > 0:
                parts.append(f"{name}:{count}")
        return ";".join(parts)

    @property
    def depth(self) -> int:
        return self.prefix.rstrip("/").count("/") + 1


def _parent_prefix(prefix: str) -> str | None:
    """Return the parent directory prefix, or None if already at root."""
    stripped = prefix.rstrip("/")
    idx = stripped.rfind("/")
    if idx < 0:
        return None
    return stripped[: idx + 1]


def _collapse_deletions(entries: list[DirEntry]) -> list[DirEntry]:
    """Collapse delete-marked directories bottom-up.

    If all children of a parent (within the same bucket) are "delete", replace
    them with a single entry for the parent with summed stats. Repeat until stable.
    """
    by_bucket: dict[str, list[DirEntry]] = defaultdict(list)
    for e in entries:
        by_bucket[e.bucket].append(e)

    result: list[DirEntry] = []
    for bucket, bucket_entries in sorted(by_bucket.items()):
        result.extend(_collapse_bucket(bucket, bucket_entries))
    return result


def _collapse_bucket(bucket: str, entries: list[DirEntry]) -> list[DirEntry]:
    """Collapse within a single bucket."""
    by_prefix: dict[str, DirEntry] = {e.prefix: e for e in entries}

    changed = True
    while changed:
        changed = False

        children_of: dict[str, list[str]] = defaultdict(list)
        for prefix in by_prefix:
            parent = _parent_prefix(prefix)
            if parent is not None:
                children_of[parent].append(prefix)

        for parent, child_prefixes in children_of.items():
            if len(child_prefixes) < 2:
                continue

            children = [by_prefix[cp] for cp in child_prefixes if cp in by_prefix]
            if len(children) != len(child_prefixes):
                continue
            if not all(c.status == "delete" for c in children):
                continue

            if parent in by_prefix and by_prefix[parent].status == "keep":
                continue

            merged = DirEntry(
                bucket=bucket,
                prefix=parent,
                status="delete",
                matched_rule=_most_common_rule(children),
                standard_count=sum(c.standard_count for c in children),
                standard_bytes=sum(c.standard_bytes for c in children),
                nearline_count=sum(c.nearline_count for c in children),
                nearline_bytes=sum(c.nearline_bytes for c in children),
                coldline_count=sum(c.coldline_count for c in children),
                coldline_bytes=sum(c.coldline_bytes for c in children),
                archive_count=sum(c.archive_count for c in children),
                archive_bytes=sum(c.archive_bytes for c in children),
            )

            if parent in by_prefix and by_prefix[parent].status == "delete":
                existing = by_prefix[parent]
                merged.standard_count += existing.standard_count
                merged.standard_bytes += existing.standard_bytes
                merged.nearline_count += existing.nearline_count
                merged.nearline_bytes += existing.nearline_bytes
                merged.coldline_count += existing.coldline_count
                merged.coldline_bytes += existing.coldline_bytes
                merged.archive_count += existing.archive_count
                merged.archive_bytes += existing.archive_bytes

            for cp in child_prefixes:
                del by_prefix[cp]
            by_prefix[parent] = merged
            changed = True

    return sorted(by_prefix.values(), key=lambda e: e.prefix)


def _most_common_rule(entries: list[DirEntry]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e.matched_rule] += 1
    return max(counts, key=counts.get)  # type: ignore[arg-type]


# ===========================================================================
# Query + manifest
# ===========================================================================

_STATUS_QUERY = """
    SELECT d.bucket, d.dir_prefix,
           d.standard_count, d.standard_bytes,
           d.nearline_count, d.nearline_bytes,
           d.coldline_count, d.coldline_bytes,
           d.archive_count, d.archive_bytes,
           CASE WHEN p.dir_prefix IS NOT NULL THEN 'keep'
                WHEN del.dir_prefix IS NOT NULL THEN 'delete'
                ELSE 'keep'
           END AS status,
           COALESCE(del.matched_rule, '') AS matched_rule
    FROM dir_summary d
    LEFT JOIN (
        SELECT DISTINCT d2.bucket, d2.dir_prefix
        FROM protect_rules p
        JOIN dir_summary d2
            ON d2.dir_prefix LIKE p.pattern
            AND (d2.bucket = p.bucket OR p.bucket = '*')
        WHERE d2.bucket = ?
    ) p USING (bucket, dir_prefix)
    LEFT JOIN (
        SELECT DISTINCT ON (bucket, dir_prefix) bucket, dir_prefix, matched_rule
        FROM (
            -- Prefix-only delete rules (storage_class IS NULL)
            SELECT d3.bucket, d3.dir_prefix, dr.pattern AS matched_rule
            FROM delete_rules dr
            JOIN dir_summary d3 ON d3.dir_prefix LIKE dr.pattern
            WHERE d3.bucket = ?
              AND dr.storage_class IS NULL

            UNION ALL

            -- Storage-class delete rules (conservative: only dirs with zero STANDARD objects
            -- where every present non-standard class is covered by a delete rule)
            SELECT d3.bucket, d3.dir_prefix, 'non-standard' AS matched_rule
            FROM dir_summary d3
            WHERE d3.bucket = ?
              AND d3.standard_count = 0
              AND (d3.nearline_count + d3.coldline_count + d3.archive_count) > 0
              AND (d3.nearline_count = 0 OR EXISTS (
                  SELECT 1 FROM delete_rules dr
                  WHERE dr.storage_class = 'NEARLINE' AND d3.dir_prefix LIKE dr.pattern
              ))
              AND (d3.coldline_count = 0 OR EXISTS (
                  SELECT 1 FROM delete_rules dr
                  WHERE dr.storage_class = 'COLDLINE' AND d3.dir_prefix LIKE dr.pattern
              ))
              AND (d3.archive_count = 0 OR EXISTS (
                  SELECT 1 FROM delete_rules dr
                  WHERE dr.storage_class = 'ARCHIVE' AND d3.dir_prefix LIKE dr.pattern
              ))
        )
    ) del USING (bucket, dir_prefix)
    WHERE d.bucket = ?
    ORDER BY d.dir_prefix
"""


def _compute_deletion_entries(conn: Any, bucket: str) -> list[DirEntry]:
    """Query dir_summary with delete/keep status and return delete-marked entries."""
    rows = _fetchall_dicts(conn.execute(_STATUS_QUERY, (bucket, bucket, bucket, bucket)))

    all_entries = [
        DirEntry(
            bucket=r["bucket"],
            prefix=r["dir_prefix"],
            status=r["status"],
            matched_rule=r["matched_rule"],
            standard_count=int(r["standard_count"] or 0),
            standard_bytes=int(r["standard_bytes"] or 0),
            nearline_count=int(r["nearline_count"] or 0),
            nearline_bytes=int(r["nearline_bytes"] or 0),
            coldline_count=int(r["coldline_count"] or 0),
            coldline_bytes=int(r["coldline_bytes"] or 0),
            archive_count=int(r["archive_count"] or 0),
            archive_bytes=int(r["archive_bytes"] or 0),
        )
        for r in rows
    ]

    collapsed = _collapse_deletions(all_entries)
    return [e for e in collapsed if e.status == "delete"]


CSV_COLUMNS = [
    "bucket",
    "prefix",
    "object_count",
    "total_bytes",
    "bytes_human",
    "storage_class_breakdown",
    "matched_rule",
]


def _write_manifest(entries: list[DirEntry], path: Any) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for e in entries:
            writer.writerow(
                {
                    "bucket": e.bucket,
                    "prefix": e.prefix,
                    "object_count": e.object_count,
                    "total_bytes": e.total_bytes,
                    "bytes_human": human_bytes(e.total_bytes),
                    "storage_class_breakdown": e.storage_class_breakdown,
                    "matched_rule": e.matched_rule,
                }
            )


def _print_deletion_summary(entries: list[DirEntry]) -> None:
    """Print a human-readable summary of the deletion manifest."""
    if not entries:
        print_summary("  no directories marked for deletion")
        return

    by_region: dict[str, list[DirEntry]] = defaultdict(list)
    for e in entries:
        region = e.bucket.removeprefix("marin-")
        by_region[region].append(e)

    prices_us = {"STANDARD": 0.020, "NEARLINE": 0.010, "COLDLINE": 0.004, "ARCHIVE": 0.0012}
    prices_eu = {"STANDARD": 0.023, "NEARLINE": 0.013, "COLDLINE": 0.006, "ARCHIVE": 0.0025}
    discount = 1.0 - GCS_DISCOUNT

    grand_prefixes = 0
    grand_objects = 0
    grand_bytes = 0
    grand_cost = 0.0

    for region in sorted(by_region):
        region_entries = by_region[region]
        continent = continent_for_region(region)
        prices = prices_us if continent == "US" else prices_eu

        region_objects = sum(e.object_count for e in region_entries)
        region_bytes = sum(e.total_bytes for e in region_entries)
        region_cost = 0.0
        for e in region_entries:
            for sc, byte_val in [
                ("STANDARD", e.standard_bytes),
                ("NEARLINE", e.nearline_bytes),
                ("COLDLINE", e.coldline_bytes),
                ("ARCHIVE", e.archive_bytes),
            ]:
                region_cost += byte_val / (1024**3) * prices[sc] * discount

        print_summary(
            f"  {region}: {len(region_entries)} prefixes, "
            f"{region_objects:,} objects, {human_bytes(region_bytes)}, "
            f"~${region_cost:,.2f}/mo"
        )
        grand_prefixes += len(region_entries)
        grand_objects += region_objects
        grand_bytes += region_bytes
        grand_cost += region_cost

    print_summary(
        f"  total: {grand_prefixes} prefixes, "
        f"{grand_objects:,} objects, {human_bytes(grand_bytes)}, "
        f"~${grand_cost:,.2f}/mo"
    )

    top = sorted(entries, key=lambda e: e.total_bytes, reverse=True)[:10]
    print_summary("  top 10 by size:")
    for e in top:
        print_summary(f"    {e.bucket}/{e.prefix}  {human_bytes(e.total_bytes)}  ({e.object_count:,} objects)")


# ===========================================================================
# Step runners
# ===========================================================================


def run_scan(ctx: Context) -> None:
    if ctx.force:
        # Wipe objects_parquet dir to re-scan from scratch
        parquet_dir = DEFAULT_CATALOG.objects_parquet_dir
        if parquet_dir.exists():
            shutil.rmtree(parquet_dir)
            parquet_dir.mkdir(parents=True, exist_ok=True)
    scan_objects(ctx)


def run_summarize(ctx: Context) -> None:
    if ctx.force:
        parquet_dir = DEFAULT_CATALOG.dir_summary_parquet_dir
        if parquet_dir.exists():
            shutil.rmtree(parquet_dir)
            parquet_dir.mkdir(parents=True, exist_ok=True)
    total_dirs = materialize_dir_summary(ctx.conn)
    print_summary(f"  materialized {total_dirs} directory summary rows")


def run_compute(ctx: Context) -> None:
    catalog = DEFAULT_CATALOG

    all_entries: list[DirEntry] = []
    for plan_row in plan_rows():
        bucket = plan_row["bucket"]
        print_summary(f"  computing deletion set for {bucket}...")
        entries = _compute_deletion_entries(ctx.conn, bucket)
        all_entries.extend(entries)

    manifest_path = catalog.deletion_manifest_csv
    _write_manifest(all_entries, manifest_path)
    print_summary(f"  wrote {len(all_entries)} prefixes to {manifest_path.relative_to(REPO_ROOT)}")

    # Write SHA-256 sidecar for fingerprint validation during delete
    sha_path = manifest_path.parent / (manifest_path.name + ".sha256")
    sha_path.write_text(file_digest(manifest_path))

    _print_deletion_summary(all_entries)


def _load_manifest(ctx: Context) -> dict[str, list[dict[str, str]]]:
    """Load the deletion manifest CSV and return full rows grouped by bucket.

    Validates the CSV fingerprint against the .sha256 sidecar if it exists.
    """
    catalog = DEFAULT_CATALOG
    manifest_path = catalog.deletion_manifest_csv
    if not manifest_path.exists():
        raise RuntimeError(
            f"Deletion manifest not found at {manifest_path.relative_to(REPO_ROOT)}. " "Run the compute step first."
        )

    sha_path = manifest_path.parent / (manifest_path.name + ".sha256")
    if sha_path.exists():
        expected = sha_path.read_text().strip()
        actual = file_digest(manifest_path)
        if expected != actual:
            raise RuntimeError(
                "Deletion manifest has been modified since the compute step ran. "
                "Re-run the compute step with --force to regenerate."
            )

    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_bucket[row["bucket"]].append(row)
    return dict(by_bucket)


def _raise_delete_worker_failure(bucket_name: str, failures: queue.Queue[Exception]) -> None:
    try:
        exc = failures.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"Delete failed for {bucket_name}; aborting remaining work.") from exc


def _queue_put_with_worker_checks(
    bucket_name: str,
    q: queue.Queue[str | None],
    item: str | None,
    stop_event: threading.Event,
    failures: queue.Queue[Exception],
) -> None:
    while True:
        _raise_delete_worker_failure(bucket_name, failures)
        if stop_event.is_set():
            raise RuntimeError(f"Delete aborted for {bucket_name} after a worker stopped.")
        try:
            q.put(item, timeout=DELETE_WORKER_QUEUE_TIMEOUT)
            return
        except queue.Full:
            continue


class _DeleteState:
    """Shared state for the delete pipeline: tracks recent prefixes/objects and deleted count.

    Duck-typed to the tqdm interface used by _delete_worker (update / set_postfix / .n).
    """

    RECENT_MAXLEN = 100

    def __init__(self, total_prefixes: int, total_objects: int) -> None:
        self.total_prefixes = total_prefixes
        self.total_objects = total_objects
        self.start_time = time.time()
        self._lock = threading.Lock()
        self.recent_prefixes: deque[str] = deque(maxlen=self.RECENT_MAXLEN)
        self.recent_objects: deque[str] = deque(maxlen=self.RECENT_MAXLEN)
        self.prefixes_done = 0
        self.n = 0  # deleted count; named .n for _delete_worker compatibility

    def record_prefix(self, prefix: str) -> None:
        with self._lock:
            self.recent_prefixes.append(prefix)
            self.prefixes_done += 1

    def record_object(self, name: str) -> None:
        with self._lock:
            self.recent_objects.append(name)

    def update(self, n: int) -> None:
        with self._lock:
            self.n += n

    def set_postfix(self, **_kwargs: object) -> None:
        pass

    def snapshot(self) -> tuple[list[str], list[str], int, int]:
        with self._lock:
            return list(self.recent_prefixes), list(self.recent_objects), self.prefixes_done, self.n


def _delete_worker(
    ctx: Context,
    bucket_name: str,
    q: queue.Queue[str | None],
    progress: _DeleteState,
    stop_event: threading.Event,
    failures: queue.Queue[Exception],
    batch_size: int = 100,
) -> None:
    client = storage_client(ctx)
    bucket_obj = client.bucket(bucket_name)
    batch: list[str] = []

    def flush() -> None:
        # Retry on throttling and transient 5xx / deadline failures with exponential backoff.
        for attempt in range(DELETE_BATCH_MAX_ATTEMPTS):
            try:
                with client.batch():
                    for name in batch:
                        bucket_obj.blob(name).delete(timeout=DELETE_BATCH_TIMEOUT)
                progress.update(len(batch))
                batch.clear()
                return
            except NotFound:
                # Some objects already deleted; treat as success for idempotency.
                progress.update(len(batch))
                batch.clear()
                return
            except (DeadlineExceeded, ServiceUnavailable, TooManyRequests) as exc:
                if attempt == DELETE_BATCH_MAX_ATTEMPTS - 1:
                    raise
                sleep = DELETE_BATCH_INITIAL_BACKOFF * (2**attempt) + random.uniform(0, 1)
                log.warning(
                    "%s from GCS batch delete for %d objects; sleeping %.1fs (attempt %d/%d)",
                    exc.__class__.__name__,
                    len(batch),
                    sleep,
                    attempt + 1,
                    DELETE_BATCH_MAX_ATTEMPTS,
                )
                time.sleep(sleep)

    try:
        while True:
            if stop_event.is_set():
                return
            try:
                name = q.get(timeout=DELETE_WORKER_QUEUE_TIMEOUT)
            except queue.Empty:
                continue
            if name is None:
                if batch:
                    flush()
                return
            progress.set_postfix(file=name, refresh=False)
            batch.append(name)
            if len(batch) >= batch_size:
                flush()
    except Exception as exc:
        stop_event.set()
        failures.put(exc)


def _listing_worker(
    ctx: Context,
    bucket_name: str,
    prefix_q: queue.Queue[str | None],
    object_q: queue.Queue[str | None],
    stop_event: threading.Event,
    failures: queue.Queue[Exception],
    state: _DeleteState,
) -> None:
    client = storage_client(ctx)
    try:
        while True:
            if stop_event.is_set():
                return
            try:
                prefix = prefix_q.get(timeout=DELETE_WORKER_QUEUE_TIMEOUT)
            except queue.Empty:
                continue
            if prefix is None:
                return
            state.record_prefix(prefix)
            for blob in client.list_blobs(bucket_name, prefix=prefix):
                if stop_event.is_set():
                    return
                state.record_object(blob.name)
                _queue_put_with_worker_checks(bucket_name, object_q, blob.name, stop_event, failures)
    except Exception as exc:
        stop_event.set()
        failures.put(exc)


_CONSOLE = Console(stderr=True)


def _render_delete_display(
    state: _DeleteState,
    prefix_q_size: int,
    object_q_size: int,
) -> Group:
    prefixes_snap, objects_snap, prefixes_done, deleted = state.snapshot()

    elapsed = time.time() - state.start_time
    rate = deleted / elapsed if elapsed > 1 else 0
    remaining = max(0, state.total_objects - deleted)
    if rate > 0:
        eta_s = int(remaining / rate)
        eta_str = f"{eta_s // 60}m {eta_s % 60}s"
    else:
        eta_str = "…"

    def make_item_table(items: list[str]) -> Table:
        t = Table(show_header=False, box=None, padding=(0, 0), expand=True)
        t.add_column("item", no_wrap=True, overflow="ellipsis")
        for item in items:
            t.add_row(item)
        return t

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(
        Panel(make_item_table(prefixes_snap[-20:]), title="Listing Prefixes"),
        Panel(make_item_table(objects_snap[-20:]), title="Objects Enumerated"),
    )

    status = Text(
        f" prefixes {prefixes_done}/{state.total_prefixes}"
        f"  prefix_q {prefix_q_size}"
        f"  object_q {object_q_size}"
        f"  deleted {deleted:,}"
        f"  {rate:,.0f} obj/s"
        f"  ETA {eta_str}",
        style="bold",
    )

    return Group(grid, status)


def _prefix_feeder(
    bucket_name: str,
    prefix_q: queue.Queue[str | None],
    rows: list[dict[str, str]],
    stop_event: threading.Event,
    failures: queue.Queue[Exception],
) -> None:
    try:
        for row in rows:
            _queue_put_with_worker_checks(bucket_name, prefix_q, row["prefix"], stop_event, failures)
        for _ in range(N_LIST_WORKERS):
            _queue_put_with_worker_checks(bucket_name, prefix_q, None, stop_event, failures)
    except Exception as exc:
        stop_event.set()
        failures.put(exc)


def run_delete(ctx: Context) -> None:
    """Delete objects listed in the pre-computed deletion manifest."""
    manifest = _load_manifest(ctx)

    if ctx.dry_run:
        for plan_row in plan_rows():
            rows = manifest.get(plan_row["bucket"], [])
            if not rows:
                continue
            total_count = sum(int(r["object_count"]) for r in rows)
            total_bytes = sum(int(r["total_bytes"]) for r in rows)
            print_summary(
                f"  would delete {total_count:,} objects ({human_bytes(total_bytes)}) from {plan_row['bucket']}"
            )
        return

    for plan_row in plan_rows():
        bucket_name = plan_row["bucket"]
        rows = manifest.get(bucket_name, [])
        if not rows:
            print_summary(f"  {bucket_name}: not in manifest, skipping")
            continue

        total_objects = sum(int(r["object_count"]) for r in rows)
        print_summary(
            f"  deleting from {bucket_name}: {total_objects:,} objects, {len(rows)} prefixes, "
            f"{N_LIST_WORKERS} list workers, {N_DELETE_WORKERS} delete workers"
        )

        # Shuffle prefixes to spread load across bucket keyspace rather than hammering one hotspot.
        shuffled_rows = list(rows)
        random.shuffle(shuffled_rows)

        prefix_q: queue.Queue[str | None] = queue.Queue(maxsize=256)
        object_q: queue.Queue[str | None] = queue.Queue(maxsize=10_000)
        stop_event = threading.Event()
        failures: queue.Queue[Exception] = queue.Queue()

        state = _DeleteState(total_prefixes=len(rows), total_objects=total_objects)
        feeder = threading.Thread(
            target=_prefix_feeder,
            args=(bucket_name, prefix_q, shuffled_rows, stop_event, failures),
            daemon=True,
        )
        list_workers = [
            threading.Thread(
                target=_listing_worker,
                args=(ctx, bucket_name, prefix_q, object_q, stop_event, failures, state),
                daemon=True,
            )
            for _ in range(N_LIST_WORKERS)
        ]
        delete_workers = [
            threading.Thread(
                target=_delete_worker,
                args=(ctx, bucket_name, object_q, state, stop_event, failures),
                daemon=True,
            )
            for _ in range(N_DELETE_WORKERS)
        ]
        for w in [feeder, *list_workers, *delete_workers]:
            w.start()
        try:
            with Live(console=_CONSOLE, refresh_per_second=1) as live:
                while any(w.is_alive() for w in [feeder, *list_workers]):
                    time.sleep(1.0)
                    _raise_delete_worker_failure(bucket_name, failures)
                    live.update(_render_delete_display(state, prefix_q.qsize(), object_q.qsize()))

                for _ in range(N_DELETE_WORKERS):
                    _queue_put_with_worker_checks(bucket_name, object_q, None, stop_event, failures)

                while any(w.is_alive() for w in delete_workers):
                    time.sleep(1.0)
                    _raise_delete_worker_failure(bucket_name, failures)
                    live.update(_render_delete_display(state, prefix_q.qsize(), object_q.qsize()))

            for w in [feeder, *list_workers, *delete_workers]:
                w.join()
            _raise_delete_worker_failure(bucket_name, failures)
        except Exception:
            stop_event.set()
            raise

        print_summary(f"  {bucket_name}: done")


# ===========================================================================
# Step registry
# ===========================================================================

STEPS: list[Step] = [
    Step("scan", "Scan objects into parquet", run_scan, scan_done),
    Step("summarize", "Materialize dir_summary from objects", run_summarize, summarize_done),
    Step("compute", "Compute deletion manifest CSV", run_compute, compute_done),
    Step("delete", "Delete objects from manifest", run_delete, lambda _: False, mutating=True),
]

STEP_INDEX: dict[str, Step] = {s.name: s for s in STEPS}
STEP_NAMES: list[str] = [s.name for s in STEPS]


# ===========================================================================
# Context builder
# ===========================================================================


def build_context(
    *,
    dry_run: bool,
    force: bool,
    scan_workers: int,
    project: str | None,
) -> Context:
    catalog = DEFAULT_CATALOG
    conn = ensure_output_dirs(catalog)
    return Context(
        conn=conn,
        db_lock=threading.Lock(),
        dry_run=dry_run,
        force=force,
        include_optional=False,
        scan_workers=scan_workers,
        settle_hours=0,
        log_path=catalog.log_dir / f"cleanup_{timestamp_string()}.log",
        timestamp=timestamp_string(),
        project=project,
    )


# ===========================================================================
# CLI helpers
# ===========================================================================


def _selected_steps(
    *,
    from_step: str | None,
    to_step: str | None,
) -> list[Step]:
    start = 0 if from_step is None else STEP_NAMES.index(from_step)
    end = len(STEPS) - 1 if to_step is None else STEP_NAMES.index(to_step)
    return STEPS[start : end + 1]


def _run_steps(ctx: Context, steps: list[Step]) -> None:
    for step in steps:
        tags = []
        if step.mutating:
            tags.append("mutating")
        if step.optional:
            tags.append("optional")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print_summary(f"==> {step.name}: {step.description}{tag_str}")

        step.runner(ctx)

    print_summary("completed selected steps")


# ===========================================================================
# CLI
# ===========================================================================


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Storage purge workflow: scan, summarize, compute deletion manifest, and delete objects.",
)
def cli() -> None:
    pass


@cli.command("plan", help="Show each step's status.")
def plan_cli() -> None:
    catalog = DEFAULT_CATALOG
    # Ensure dirs exist so we can check parquet files, but don't need gcloud for plan
    catalog.ensure_dirs()
    print("Storage purge steps:")
    for step in STEPS:
        tags = []
        if step.mutating:
            tags.append("mutating")
        if step.optional:
            tags.append("optional")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""

        try:
            status = "done" if step.done(catalog) else "pending"
        except Exception:
            status = "error"

        print(f"  {status:7}  {step.name:14} {step.description}{tag_str}")


@cli.command("run", help="Execute a contiguous slice of the ordered workflow.")
@click.option("--from", "from_step", type=click.Choice(STEP_NAMES), help="Start from this step.")
@click.option("--to", "to_step", type=click.Choice(STEP_NAMES), help="Stop after this step.")
@click.option("--dry-run", is_flag=True, help="Read-only mode: inspect but never mutate remote state.")
@click.option("--force", is_flag=True, help="Re-run steps even if already done.")
@click.option("--workers", "scan_workers", default=64, show_default=True, type=int, help="Concurrent workers.")
@click.option("--project", help="Override GCP project.")
def run_cli(
    from_step: str | None,
    to_step: str | None,
    dry_run: bool,
    force: bool,
    scan_workers: int,
    project: str | None,
) -> None:
    ctx = build_context(dry_run=dry_run, force=force, scan_workers=scan_workers, project=project)
    steps = _selected_steps(from_step=from_step, to_step=to_step)
    print_summary(f"running {len(steps)} steps; log: {ctx.log_path.relative_to(REPO_ROOT)}")
    _run_steps(ctx, steps)


# Individual step commands


@cli.command("scan", help="Scan bucket objects into parquet segments.")
@click.option("--force", is_flag=True, help="Wipe existing parquet and re-scan.")
@click.option("--workers", "scan_workers", default=64, show_default=True, type=int, help="Concurrent scan workers.")
@click.option("--project", help="Override GCP project.")
def scan_cli(force: bool, scan_workers: int, project: str | None) -> None:
    ctx = build_context(dry_run=False, force=force, scan_workers=scan_workers, project=project)
    print_summary(f"==> scan: {STEP_INDEX['scan'].description}")
    run_scan(ctx)
    print_summary("completed")


@cli.command("summarize", help="Materialize dir_summary from objects.")
@click.option("--force", is_flag=True, help="Wipe existing dir_summary and re-run.")
@click.option("--project", help="Override GCP project.")
def summarize_cli(force: bool, project: str | None) -> None:
    ctx = build_context(dry_run=False, force=force, scan_workers=1, project=project)
    print_summary(f"==> summarize: {STEP_INDEX['summarize'].description}")
    run_summarize(ctx)
    print_summary("completed")


@cli.command("compute", help="Compute deletion manifest CSV from rules.")
@click.option("--project", help="Override GCP project.")
def compute_cli(project: str | None) -> None:
    ctx = build_context(dry_run=False, force=False, scan_workers=1, project=project)
    print_summary(f"==> compute: {STEP_INDEX['compute'].description}")
    run_compute(ctx)
    print_summary("completed")


@cli.command("delete", help="Delete objects listed in the manifest.")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted.")
@click.option("--workers", "scan_workers", default=64, show_default=True, type=int, help="Concurrent delete workers.")
@click.option("--project", help="Override GCP project.")
def delete_cli(dry_run: bool, scan_workers: int, project: str | None) -> None:
    ctx = build_context(dry_run=dry_run, force=False, scan_workers=scan_workers, project=project)
    print_summary(f"==> delete: {STEP_INDEX['delete'].description}")
    run_delete(ctx)
    print_summary("completed")


if __name__ == "__main__":
    cli()
