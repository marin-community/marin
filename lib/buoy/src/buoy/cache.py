# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The refetchable run cache: an fsspec-addressed tree (GCS in prod, a local dir
in tests) under ``<cache_root>/<entity>/<project>/<run_id>/``.

Layout::

    history/part-00000.parquet   # unified-schema history shards (one frame split)
    config.json
    summary.json
    artifacts/<name>/...         # the jax_profile logdir, mirrored verbatim
    manifest.json                # written LAST — its presence == fully cached

All writers go through here so the rest of the service never touches fsspec
directly. ``manifest.json`` is the commit marker: a reader that finds it can
trust every referenced file is present.
"""

from __future__ import annotations

import json
import math
import os
import posixpath

import pandas as pd
import pyarrow.parquet as pq
from rigging.filesystem import url_to_fs

MANIFEST_NAME = "manifest.json"


def run_prefix(cache_root: str, entity: str, project: str, run_id: str) -> str:
    return posixpath.join(cache_root, entity, project, run_id)


def manifest_path(prefix: str) -> str:
    return posixpath.join(prefix, MANIFEST_NAME)


def history_dir(prefix: str) -> str:
    return posixpath.join(prefix, "history")


def _fs(path: str):
    # rigging.url_to_fs wraps GCS in CrossRegionGuardedFS, so the tree put/get of
    # hundreds-of-MB profiles can't silently cross regions.
    fs, _ = url_to_fs(path)
    return fs


def exists(path: str) -> bool:
    return _fs(path).exists(path)


def read_json(path: str) -> dict | None:
    fs = _fs(path)
    if not fs.exists(path):
        return None
    with fs.open(path, "r") as handle:
        return json.load(handle)


def _json_safe(obj: object) -> object:
    """Replace non-finite floats (NaN/±Inf — common in wandb summaries) with None.

    Python's ``json`` emits them as bare ``NaN``/``Infinity`` tokens, which are
    invalid JSON: the browser's ``JSON.parse`` rejects them, breaking the summary/
    config panes. Normalize to null so every cached document is strict JSON.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def write_json(path: str, obj: dict) -> None:
    fs = _fs(path)
    fs.makedirs(posixpath.dirname(path), exist_ok=True)
    with fs.open(path, "w") as handle:
        json.dump(_json_safe(obj), handle, default=str, allow_nan=False)


def write_parquet(path: str, table) -> None:
    fs = _fs(path)
    fs.makedirs(posixpath.dirname(path), exist_ok=True)
    with fs.open(path, "wb") as handle:
        pq.write_table(table, handle)


def read_manifest(prefix: str) -> dict | None:
    return read_json(manifest_path(prefix))


def read_history(prefix: str, columns: list[str]) -> pd.DataFrame:
    """Read selected columns from the history shards into one frame.

    Only the requested columns are read off disk. Shards may differ in columns
    (incremental appends for a running run can introduce a later metric that
    earlier shards lack), so each shard contributes whichever requested columns
    it has and the concat aligns the rest to NaN.
    """
    hdir = history_dir(prefix)
    fs = _fs(hdir)
    parts = sorted(fs.glob(posixpath.join(hdir, "part-*.parquet")))
    frames: list[pd.DataFrame] = []
    for part in parts:
        with fs.open(part, "rb") as handle:
            pf = pq.ParquetFile(handle)
            available = [c for c in columns if c in pf.schema_arrow.names]
            if available:
                frames.append(pf.read(columns=available).to_pandas())
    if not frames:
        return pd.DataFrame(columns=columns)
    # Reindex so a requested column absent from every shard (e.g. a metric newly in
    # summary but not yet in any history row) is present and all-NaN, not missing.
    return pd.concat(frames, ignore_index=True).reindex(columns=columns)


def upload_tree(local_dir: str, dst_prefix: str) -> None:
    """Recursively copy a local directory into the cache (GCS or local)."""
    fs = _fs(dst_prefix)
    fs.makedirs(dst_prefix, exist_ok=True)
    fs.put(local_dir.rstrip("/") + "/", dst_prefix.rstrip("/") + "/", recursive=True)


def download_tree(src_prefix: str, local_dir: str) -> str:
    """Recursively materialize a cache subtree to a local directory; return it."""
    fs = _fs(src_prefix)
    os.makedirs(local_dir, exist_ok=True)
    fs.get(src_prefix.rstrip("/") + "/", local_dir.rstrip("/") + "/", recursive=True)
    return local_dir


def upload_file(local_path: str, dst: str) -> None:
    fs = _fs(dst)
    fs.makedirs(posixpath.dirname(dst), exist_ok=True)
    fs.put_file(local_path, dst)


def clear_dir(path: str) -> None:
    fs = _fs(path)
    if fs.exists(path):
        fs.rm(path, recursive=True)
