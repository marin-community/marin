# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCS bucket scanning machinery: adaptive parallel scan of object listings into DuckDB.

Implements an adaptive two-level scan strategy: for each top-level prefix, a flat listing
is attempted first; if the prefix exceeds ADAPTIVE_SPLIT_THRESHOLD objects, it is split via
delimiter listing and children are recursively enqueued (up to ADAPTIVE_SCAN_MAX_DEPTH).
Results are streamed into an ObjectBuffer and flushed as sorted parquet segments.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import click
import google.auth
from google.cloud import storage
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from scripts.storage.db import (
    ADAPTIVE_SCAN_MAX_DEPTH,
    ADAPTIVE_SPLIT_THRESHOLD,
    DEFAULT_CATALOG,
    GCS_MAX_PAGE_SIZE,
    PLAN_FINGERPRINT,
    REPO_ROOT,
    Context,
    ObjectBuffer,
    ScanBuffer,
    ScannedObject,
    StepSpec,
    _BLOB_FIELDS,
    _fetchall_dicts,
    _fetchone_dict,
    buffer_prefix_scanned,
    buffer_split_cache,
    flush_metadata,
    load_split_cache,
    marker_matches,
    plan_rows,
    print_summary,
    read_split_cache,
    storage_class_id_map,
    write_marker,
)

log = logging.getLogger(__name__)

GCS_LIST_TIMEOUT = 120  # per-page timeout in seconds for list_blobs calls

# ---------------------------------------------------------------------------
# GCP helpers
# ---------------------------------------------------------------------------


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


def log_line(ctx: Context, message: str) -> None:
    with ctx.log_path.open("a") as f:
        f.write(message.rstrip() + "\n")


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


# ===========================================================================
# Scan machinery
# ===========================================================================


@dataclass(frozen=True)
class ScanEvent:
    """Event pushed from a scan worker to the main-thread event loop.

    Kinds:
      progress   — non-terminal status update (object_count is running total)
      objects    — non-terminal, one page of scanned objects (streamed incrementally)
      leaf_done  — terminal, prefix fully scanned (no objects attached)
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
        total = 0
        for page in client.list_blobs(
            bucket_name, delimiter="/", page_size=GCS_MAX_PAGE_SIZE, fields=_BLOB_FIELDS, timeout=GCS_LIST_TIMEOUT
        ).pages:
            page_items = [_blob_to_scanned(blob, bucket_name, sc_id_map) for blob in page]
            total += len(page_items)
            if page_items:
                _put("objects", objects=page_items, object_count=total)
        _put("leaf_done", object_count=total)
        return

    # Optimistic flat scan — read pages up to ADAPTIVE_SPLIT_THRESHOLD.
    # The probe is bounded by ADAPTIVE_SPLIT_THRESHOLD so accumulation is safe.
    iterator = client.list_blobs(
        bucket_name, prefix=prefix, page_size=GCS_MAX_PAGE_SIZE, fields=_BLOB_FIELDS, timeout=GCS_LIST_TIMEOUT
    )
    pages_iter = iterator.pages
    probe_objects: list[ScannedObject] = []
    is_small = False
    for page in pages_iter:
        page_items = [_blob_to_scanned(blob, bucket_name, sc_id_map) for blob in page]
        probe_objects.extend(page_items)
        if len(page_items) < GCS_MAX_PAGE_SIZE:
            is_small = True
            break
        if len(probe_objects) >= ADAPTIVE_SPLIT_THRESHOLD:
            break

    if is_small:
        if probe_objects:
            _put("objects", objects=probe_objects, object_count=len(probe_objects))
        _put("leaf_done", object_count=len(probe_objects))
        return

    if depth >= ADAPTIVE_SCAN_MAX_DEPTH:
        # Flush probe objects immediately, then stream remaining pages.
        total = len(probe_objects)
        if probe_objects:
            _put("objects", objects=probe_objects, object_count=total)
            del probe_objects
        for page in pages_iter:
            page_items = [_blob_to_scanned(blob, bucket_name, sc_id_map) for blob in page]
            total += len(page_items)
            if page_items:
                _put("objects", objects=page_items, object_count=total)
        _put("leaf_done", object_count=total)
        return

    # Large prefix, can still split — probe objects are discarded since the
    # delimiter listing below re-covers the same prefix.
    del probe_objects

    sub_iterator = client.list_blobs(
        bucket_name,
        prefix=prefix,
        delimiter="/",
        page_size=GCS_MAX_PAGE_SIZE,
        fields=_BLOB_FIELDS,
        timeout=GCS_LIST_TIMEOUT,
    )
    total = 0
    any_sub_prefixes = False
    for page in sub_iterator.pages:
        page_roots = [_blob_to_scanned(blob, bucket_name, sc_id_map) for blob in page]
        page_subs = list(page.prefixes)
        total += len(page_roots)
        if page_subs:
            any_sub_prefixes = True
            _put("split_page", objects=page_roots, object_count=total, sub_prefixes=page_subs)
        elif page_roots:
            _put("objects", objects=page_roots, object_count=total)

    if not any_sub_prefixes:
        _put("leaf_done", object_count=total)
    else:
        _put("split_done", object_count=total)


def _scan_worker_loop(
    ctx: Context,
    bucket_name: str,
    sc_id_map: dict[str, int],
    work_queue: queue.Queue[tuple[str | None, int]],
    event_queue: queue.Queue[ScanEvent],
) -> None:
    """Long-lived worker: pull (prefix, depth) from work_queue, scan, repeat."""
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


@dataclass
class WorkerSlot:
    """State of a single worker thread as seen by the display."""

    slot_id: int
    prefix: str = ""
    start_time: float = 0.0
    object_count: int = 0
    sub_prefix_count: int = 0


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
            BarColumn(bar_width=120),
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
        delta = event.object_count - slot.object_count
        if event.kind in ("progress", "objects"):
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
            self.total_objects += delta
            self._clear_slot(slot)
            self._progress.advance(self._task_id)
        elif event.kind == "error":
            self.prefixes_completed += 1
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


def _run_adaptive_scan(
    ctx: Context,
    bucket_name: str,
    sc_id_map: dict[str, int],
    pending: list[tuple[str, int]],
    num_workers: int,
    already_scanned: set[str],
    db_conn: Any,
    progress: ScanProgress,
    buf: ScanBuffer,
) -> None:
    """Two-queue adaptive scan: workers are self-scheduling."""
    work_queue: queue.Queue[tuple[str | None, int]] = queue.Queue()
    event_queue: queue.Queue[ScanEvent] = queue.Queue()
    in_flight = 0
    split_children: dict[str, list[str]] = {}
    split_cache = load_split_cache(db_conn)

    def _enqueue(prefix: str, depth: int) -> None:
        nonlocal in_flight
        in_flight += 1
        work_queue.put((prefix, depth))

    def _expand_cached_or_enqueue(prefix: str, depth: int) -> None:
        cached_children = read_split_cache(split_cache, bucket_name, prefix)
        if cached_children is not None:
            new_children = [sp for sp in cached_children if sp not in already_scanned]
            skipped = len(cached_children) - len(new_children)
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

        if event.kind == "objects":
            if event.objects:
                buf.objects.add(event.objects)
            progress.handle_event(event)
            return

        if event.kind == "split_page":
            if event.objects:
                buf.objects.add(event.objects)
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
            buffer_split_cache(buf, split_cache, bucket_name, event.prefix, sorted(all_children))
            flush_metadata(db_conn, buf)
            progress.handle_event(event)
            in_flight -= 1
            return

        if event.kind == "leaf_done":
            buffer_prefix_scanned(buf, bucket_name, event.prefix, event.object_count)
            flush_metadata(db_conn, buf)
            already_scanned.add(event.prefix)
            progress.handle_event(event)
            in_flight -= 1
            return

        if event.kind == "error":
            log.warning("scan failed for %s/%s — will retry on next run", bucket_name, event.prefix)
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
        buf.objects.flush(force=True)
        flush_metadata(db_conn, buf, force=True)
        for _ in worker_threads:
            work_queue.put(_SENTINEL)
        for t in worker_threads:
            t.join(timeout=5)


def scan_objects(ctx: Context, action: StepSpec) -> None:
    """Scan bucket objects into the DuckDB database for subsequent querying."""
    fingerprint = PLAN_FINGERPRINT
    if not ctx.force and marker_matches(ctx.conn, action.action_id, fingerprint):
        print_summary(f"skip {action.action_id}: marker is current")
        return

    sc_id_map = storage_class_id_map(ctx.conn)
    remote_summary: dict[str, Any] = {"project": resolved_project(ctx), "regions": {}}

    db_conn = ctx.conn
    for plan_row in plan_rows():
        region = plan_row["region"]
        bucket_name = plan_row["bucket"]

        print_summary(f"{action.action_id}: discovering top-level prefixes in {bucket_name}")
        client = storage_client(ctx)
        iterator = client.list_blobs(
            bucket_name, delimiter="/", fields="items(name),prefixes,nextPageToken", timeout=GCS_LIST_TIMEOUT
        )
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
                    "SELECT COUNT(*) as total FROM objects WHERE bucket = ?",
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
        buf = ScanBuffer(objects=ObjectBuffer(DEFAULT_CATALOG.objects_parquet_dir, db_conn))

        try:
            _run_adaptive_scan(ctx, bucket_name, sc_id_map, pending, workers, already_scanned, db_conn, progress, buf)
        finally:
            progress.stop()

        if progress.prefixes_expanded:
            n = progress.prefixes_expanded
            print_summary(f"{action.action_id}: {bucket_name}: {n} prefixes expanded via adaptive splitting")

        total_row = _fetchone_dict(
            db_conn.execute(
                "SELECT COUNT(*) as total FROM objects WHERE bucket = ?",
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

    write_marker(ctx.conn, action.action_id, fingerprint, dry_run=ctx.dry_run, extra={"remote_summary": remote_summary})
