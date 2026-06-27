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

import logging
import posixpath
import threading
import time
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
import wandb

from buoy import cache
from buoy.config import HISTORY_PAGE_ROWS, PROFILE_ARTIFACT_TYPE, BuoyConfig

logger = logging.getLogger("buoy.mirror")


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


def _mirror_history(run: object, prefix: str, columns: list[str]) -> dict:
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

    for row in run.scan_history():  # type: ignore[attr-defined]
        batch.append(row)
        if len(batch) >= HISTORY_PAGE_ROWS:
            flush()
    flush()
    return {"parts": part, "rows": rows, "columns": [c for c in columns if c != "_step"]}


def _mirror_profile(run: object, prefix: str) -> dict | None:
    for art in run.logged_artifacts():  # type: ignore[attr-defined]
        if art.type != PROFILE_ARTIFACT_TYPE:
            continue
        local = art.download()
        safe = art.name.replace(":", "_").replace("/", "_")
        dst = posixpath.join(prefix, "artifacts", safe)
        cache.upload_tree(local, dst)
        return {"artifact_name": art.name, "logdir": dst, "size_bytes": int(getattr(art, "size", 0) or 0)}
    return None


def mirror_run(cfg: BuoyConfig, ref: RunRef, *, refresh: bool = False) -> dict:
    """Synchronously mirror ``ref`` into the cache and return its manifest.

    Idempotent: a finished run whose ``manifest.json`` already exists is returned
    as-is unless ``refresh`` is set. A run that was still *running* when last
    mirrored is always re-fetched (its history was incomplete).
    """
    prefix = cache.run_prefix(cfg.cache_root, ref.entity, ref.project, ref.run_id)
    mpath = cache.manifest_path(prefix)
    existing = cache.read_json(mpath)
    if existing and not refresh and existing.get("state") != "running":
        return existing

    # Re-mirroring overwrites shards in place, so drop the old commit marker first:
    # mid-refresh (or after a crash) readers then see "not cached" and re-fetch,
    # never a manifest pointing at half-updated shards.
    if existing is not None:
        cache.delete(mpath)

    api = wandb.Api()
    run = api.run(ref.key)
    columns = history_columns(run)
    history = _mirror_history(run, prefix, columns)
    cache.write_json(posixpath.join(prefix, "config.json"), dict(run.config))
    summary_raw = getattr(run.summary, "_json_dict", None)
    cache.write_json(posixpath.join(prefix, "summary.json"), dict(summary_raw) if summary_raw is not None else {})
    profile = _mirror_profile(run, prefix)

    manifest = {
        "entity": ref.entity,
        "project": ref.project,
        "run_id": ref.run_id,
        "display_name": run.name,
        "state": run.state,
        "url": run.url,
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

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def start(self, ref: RunRef, *, refresh: bool = False) -> threading.Thread | None:
        """Start a background mirror; return the thread, or None if one was already running."""
        with self._guard:
            current = self._states.get(ref.key)
            if current and current.state == "running":
                return None
            self._states[ref.key] = _State("running")
        thread = threading.Thread(target=self._worker, args=(ref, refresh), daemon=True)
        thread.start()
        return thread

    def _worker(self, ref: RunRef, refresh: bool) -> None:
        with self._lock_for(ref.key):
            try:
                mirror_run(self._cfg, ref, refresh=refresh)
                with self._guard:
                    self._states[ref.key] = _State("done")
            except Exception as exc:
                logger.exception("mirror failed for %s", ref.key)
                with self._guard:
                    self._states[ref.key] = _State("error", str(exc))

    def status(self, ref: RunRef) -> dict:
        with self._guard:
            current = self._states.get(ref.key)
        if current and current.state == "running":
            return {"state": "running"}
        if current and current.state == "error":
            return {"state": "error", "error": current.error}
        prefix = cache.run_prefix(self._cfg.cache_root, ref.entity, ref.project, ref.run_id)
        if cache.read_manifest(prefix) is not None:
            return {"state": "done"}
        return {"state": "absent"}

    def refresh_if_running(self, ref: RunRef, manifest: dict) -> None:
        """Re-mirror in the background if the cached run was still running."""
        if manifest.get("state") == "running":
            self.start(ref, refresh=True)
