# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror a wandb run into the refetchable cache (compute + I/O).

Two halves:

* :func:`mirror_run` — the synchronous worker. Streams the full history into
  unified-schema parquet shards (bounded memory), writes config/summary, mirrors
  the ``jax_profile`` artifact, and writes ``manifest.json`` LAST as the commit
  marker.
* :class:`MirrorManager` — runs :func:`mirror_run` on a background thread so the
  HTTP layer can return immediately (the controller proxy caps requests at 30s,
  and a cold mirror of a large run far exceeds that). Coalesces concurrent
  mirrors of the same run behind a per-run lock and exposes a pollable status.
"""

from __future__ import annotations

import glob
import logging
import os
import posixpath
import tarfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wandb

from buoy import cache
from buoy.config import HISTORY_ARTIFACT_TYPE, HISTORY_PAGE_ROWS, PROFILE_ARTIFACT_TYPES, BuoyConfig

logger = logging.getLogger("buoy.mirror")

# Callback the mirror invokes with a short human-readable stage string, surfaced
# by mirror_status so the SPA can show progress while a run is being fetched.
Progress = Callable[[str], None]

# A running run is re-mirrored every WATCH_INTERVAL while it is being viewed; the
# watcher stops once the run reaches a terminal state or no view has touched it
# within WATCH_IDLE_TIMEOUT (so an abandoned page doesn't refresh forever).
WATCH_INTERVAL = 30.0
WATCH_IDLE_TIMEOUT = 180.0
TERMINAL_STATES = frozenset({"finished", "crashed", "failed", "killed", "preempted"})


@dataclass(frozen=True)
class RunRef:
    entity: str
    project: str
    run_id: str

    @property
    def key(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


def history_columns(run: object) -> list[str]:
    """The numeric scalar metric columns to mirror, plus ``_step``.

    Derived from the run summary (one numeric value per logged metric), which
    pins a single schema for every history shard. Non-numeric, boolean, and
    wandb-internal keys are dropped — the same filter the per-page normalization
    applies, hoisted so every shard agrees on columns.
    """
    raw = getattr(run.summary, "_json_dict", None)
    summary = dict(raw) if raw is not None else dict(run.summary)  # type: ignore[arg-type]
    cols = ["_step"]
    for key, value in summary.items():
        if key.startswith("_"):  # wandb internals (_step, _runtime, _timestamp, ...)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        cols.append(key)
    return cols


def page_table(rows: list[dict], columns: list[str]) -> pa.Table:
    """Normalize one page of history rows to the fixed ``columns`` schema.

    ``_step`` is int64; every metric column is float64. Coercing to one explicit
    schema per page makes all shards mergeable with a plain concat at read time
    (the heterogeneous-row problem the design calls out).
    """
    df = pd.DataFrame(rows)
    if "_step" not in df.columns:
        df["_step"] = range(len(df))
    df = df.reindex(columns=columns)
    df["_step"] = pd.to_numeric(df["_step"], errors="coerce")
    df = df.dropna(subset=["_step"])
    metric_cols = [c for c in columns if c != "_step"]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_step"] = df["_step"].astype("int64")
    schema = pa.schema([("_step", pa.int64()), *[(c, pa.float64()) for c in metric_cols]])
    return pa.Table.from_pandas(df[columns], schema=schema, preserve_index=False)


def _mirror_history(run: object, prefix: str, columns: list[str], report: Progress) -> dict:
    """Mirror the run history, preferring the bulk parquet artifact over scan.

    Finished runs usually expose the full history as a single-parquet
    ``wandb-history`` artifact — a bulk download (~2s for 180 MB) vs minutes of
    ``scan_history`` pagination. Running runs (whose artifact is a stale periodic
    snapshot) and finished runs that never logged one fall back to scanning.
    """
    if getattr(run, "state", None) in TERMINAL_STATES:
        art = next((a for a in run.logged_artifacts() if a.type == HISTORY_ARTIFACT_TYPE), None)  # type: ignore[attr-defined]
        if art is not None:
            return _history_from_artifact(art, prefix, report)
    return _history_from_scan(run, prefix, columns, report)


def _history_from_artifact(art: object, prefix: str, report: Progress) -> dict:
    report("history: downloading parquet")
    local = art.download()  # type: ignore[attr-defined]
    files = sorted(glob.glob(os.path.join(local, "**", "*.parquet"), recursive=True))
    hdir = cache.history_dir(prefix)
    cache.clear_dir(hdir)  # replace any shards left by a previous scan mirror
    rows = 0
    numeric: set[str] = set()
    for i, path in enumerate(files):
        report("history: caching parquet")
        schema = pq.read_schema(path)
        rows += pq.read_metadata(path).num_rows
        for field in schema:
            if not field.name.startswith("_") and (pa.types.is_integer(field.type) or pa.types.is_floating(field.type)):
                numeric.add(field.name)
        cache.upload_file(path, posixpath.join(hdir, f"part-{i:05d}.parquet"))
    return {"parts": len(files), "rows": rows, "columns": sorted(numeric)}


def _history_from_scan(run: object, prefix: str, columns: list[str], report: Progress) -> dict:
    hdir = cache.history_dir(prefix)
    part = 0
    rows = 0
    batch: list[dict] = []

    def flush() -> None:
        nonlocal part, rows
        if not batch:
            return
        table = page_table(batch, columns)
        cache.write_parquet(posixpath.join(hdir, f"part-{part:05d}.parquet"), table)
        part += 1
        rows += table.num_rows
        batch.clear()
        report(f"history: {rows:,} steps")

    for row in run.scan_history():  # type: ignore[attr-defined]
        batch.append(row)
        if len(batch) >= HISTORY_PAGE_ROWS:
            flush()
    flush()
    return {"parts": part, "rows": rows, "columns": [c for c in columns if c != "_step"]}


def _profile_logdir(local: str) -> str:
    """Return the xprof logdir for a downloaded profile artifact.

    ``jax_profile`` ships it pre-unpacked (``plugins/profile/…`` at the root);
    ``profiler`` ships a single ``.tgz`` of the same tree, which we extract. Returns
    the directory that contains ``plugins/profile`` so xprof reads either uniformly.
    """
    tarballs = [e for e in os.listdir(local) if e.endswith((".tgz", ".tar.gz"))]
    if not tarballs:
        return local
    dest = os.path.join(local, "_unpacked")
    os.makedirs(dest, exist_ok=True)
    for name in tarballs:
        with tarfile.open(os.path.join(local, name)) as tar:
            tar.extractall(dest, filter="data")
    for root, _dirs, _files in os.walk(dest):
        if os.path.isdir(os.path.join(root, "plugins", "profile")):
            return root
    return dest


def _mirror_profile(run: object, prefix: str, existing: dict | None, report: Progress) -> dict | None:
    for art in run.logged_artifacts():  # type: ignore[attr-defined]
        if art.type not in PROFILE_ARTIFACT_TYPES:
            continue
        # Skip the (hundreds-of-MB) re-download when we already hold this exact
        # artifact version — critical for the running-run refresh loop.
        if existing and existing.get("artifact_name") == art.name and cache.exists(existing.get("logdir", "")):
            return existing
        mb = int((getattr(art, "size", 0) or 0) / 1e6)
        report(f"profile: downloading {mb} MB")
        local = art.download()
        report("profile: unpacking")
        logdir = _profile_logdir(local)
        safe = art.name.replace(":", "_").replace("/", "_")
        dst = posixpath.join(prefix, "artifacts", safe)
        report("profile: uploading to cache")
        cache.upload_tree(logdir, dst)
        return {"artifact_name": art.name, "logdir": dst, "size_bytes": int(getattr(art, "size", 0) or 0)}
    return None


def mirror_run(cfg: BuoyConfig, ref: RunRef, *, refresh: bool = False, on_progress: Progress | None = None) -> dict:
    """Synchronously mirror ``ref`` into the cache and return its manifest.

    Idempotent: a finished run whose ``manifest.json`` already exists is returned
    as-is unless ``refresh`` is set. A run that was still *running* when last
    mirrored is always re-fetched (its history was incomplete). ``on_progress`` is
    called with short stage strings for the UI.
    """
    report = on_progress or (lambda _msg: None)
    prefix = cache.run_prefix(cfg.cache_root, ref.entity, ref.project, ref.run_id)
    mpath = cache.manifest_path(prefix)
    existing = cache.read_json(mpath)
    if existing and not refresh and existing.get("state") != "running":
        return existing

    # Refresh overwrites shards in place and rewrites the manifest LAST, so the old
    # manifest stays the valid commit marker until the new one lands — reads of a
    # running run never see "not cached" mid-refresh. (A crash mid-refresh can leave
    # a brief shard/manifest skew for that one running run; a generation swap would
    # remove even that, tracked as a follow-up.)
    report("reading run metadata")
    api = wandb.Api()
    run = api.run(ref.key)
    columns = history_columns(run)
    history = _mirror_history(run, prefix, columns, report)
    report("writing config + summary")
    cache.write_json(posixpath.join(prefix, "config.json"), dict(run.config))
    summary_raw = getattr(run.summary, "_json_dict", None)
    cache.write_json(posixpath.join(prefix, "summary.json"), dict(summary_raw) if summary_raw is not None else {})
    profile = _mirror_profile(run, prefix, existing.get("profile") if existing else None, report)
    report("finalizing")

    author = getattr(run, "user", None)
    manifest = {
        "entity": ref.entity,
        "project": ref.project,
        "run_id": ref.run_id,
        "display_name": run.name,
        "state": run.state,
        "url": run.url,
        "user": getattr(author, "username", None) or getattr(author, "name", None),
        "created_at": str(getattr(run, "created_at", "") or ""),
        "notes": getattr(run, "notes", None),
        "tags": list(getattr(run, "tags", None) or []),
        "mirrored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "history": history,
        "profile": profile,
    }
    cache.write_json(mpath, manifest)  # written LAST = commit marker
    logger.info("mirrored %s: %d rows, profile=%s", ref.key, history["rows"], bool(profile))
    return manifest


@dataclass
class _State:
    state: str  # "running" | "done" | "error"
    error: str | None = None


class MirrorManager:
    """Runs mirrors on background threads with per-run coalescing + pollable state.

    Single-replica by design (the service runs ``replicas: 1``): in-process state
    is the whole truth. ``start`` is non-blocking; ``status`` is what the SPA
    polls until ``done``, then reads the cache.
    """

    def __init__(self, cfg: BuoyConfig) -> None:
        self._cfg = cfg
        self._guard = threading.Lock()
        self._states: dict[str, _State] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._watch_touch: dict[str, float] = {}
        self._watching: set[str] = set()
        self._progress: dict[str, str] = {}

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def _set_progress(self, key: str, msg: str) -> None:
        with self._guard:
            self._progress[key] = msg

    def start(self, ref: RunRef, *, refresh: bool = False) -> threading.Thread | None:
        """Start a background mirror; return the thread, or None if one was already running."""
        with self._guard:
            current = self._states.get(ref.key)
            if current and current.state == "running":
                return None
            self._states[ref.key] = _State("running")
            self._progress[ref.key] = "queued"
        thread = threading.Thread(target=self._worker, args=(ref, refresh), daemon=True)
        thread.start()
        return thread

    def _worker(self, ref: RunRef, refresh: bool) -> None:
        with self._lock_for(ref.key):
            try:
                mirror_run(self._cfg, ref, refresh=refresh, on_progress=lambda m: self._set_progress(ref.key, m))
                with self._guard:
                    self._states[ref.key] = _State("done")
            except Exception as exc:
                logger.exception("mirror failed for %s", ref.key)
                with self._guard:
                    self._states[ref.key] = _State("error", str(exc))

    def status(self, ref: RunRef) -> dict:
        with self._guard:
            current = self._states.get(ref.key)
            detail = self._progress.get(ref.key)
        if current and current.state == "running":
            return {"state": "running", "detail": detail}
        if current and current.state == "error":
            return {"state": "error", "error": current.error}
        prefix = cache.run_prefix(self._cfg.cache_root, ref.entity, ref.project, ref.run_id)
        if cache.read_manifest(prefix) is not None:
            return {"state": "done"}
        return {"state": "absent"}

    def touch_running(self, ref: RunRef, manifest: dict) -> threading.Thread | None:
        """Keep the cache fresh for a running run while a page is viewing it.

        Each view bumps a last-touch time and ensures a single background watcher
        is re-mirroring the run every ``WATCH_INTERVAL`` until it reaches a terminal
        state — or until no view has touched it within ``WATCH_IDLE_TIMEOUT`` (so a
        closed tab stops the refresh loop). Returns the watcher thread when one is
        started, else None (finished run, or a watcher is already live).
        """
        if manifest.get("state") != "running":
            return None
        with self._guard:
            self._watch_touch[ref.key] = time.monotonic()
            if ref.key in self._watching:
                return None
            self._watching.add(ref.key)
        thread = threading.Thread(target=self._watch_worker, args=(ref,), daemon=True)
        thread.start()
        return thread

    def _watch_worker(self, ref: RunRef) -> None:
        try:
            while True:
                time.sleep(WATCH_INTERVAL)
                with self._guard:
                    idle = time.monotonic() - self._watch_touch.get(ref.key, 0.0)
                if idle > WATCH_IDLE_TIMEOUT:
                    logger.info("watch idle %.0fs, stopping for %s", idle, ref.key)
                    return
                try:
                    with self._lock_for(ref.key):
                        manifest = mirror_run(self._cfg, ref, refresh=True)
                except Exception:
                    logger.exception("watch refresh failed for %s", ref.key)
                    continue
                if manifest.get("state") in TERMINAL_STATES:
                    logger.info("run %s reached %s; stopping watch", ref.key, manifest.get("state"))
                    return
        finally:
            with self._guard:
                self._watching.discard(ref.key)
