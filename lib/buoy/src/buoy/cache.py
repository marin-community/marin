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


def write_json(path: str, obj: dict) -> None:
    fs = _fs(path)
    fs.makedirs(posixpath.dirname(path), exist_ok=True)
    with fs.open(path, "w") as handle:
        json.dump(obj, handle, default=str)


def write_parquet(path: str, table) -> None:
    fs = _fs(path)
    fs.makedirs(posixpath.dirname(path), exist_ok=True)
    with fs.open(path, "wb") as handle:
        pq.write_table(table, handle)


def read_manifest(prefix: str) -> dict | None:
    return read_json(manifest_path(prefix))


def read_history(prefix: str, columns: list[str]) -> pd.DataFrame:
    """Read selected columns from the history shards into one frame.

    All shards share a unified schema (see mirror.py), so a plain concat is
    correct; only the requested columns are read off disk.
    """
    hdir = history_dir(prefix)
    fs = _fs(hdir)
    parts = sorted(fs.glob(posixpath.join(hdir, "part-*.parquet")))
    frames: list[pd.DataFrame] = []
    for part in parts:
        with fs.open(part, "rb") as handle:
            frames.append(pq.read_table(handle, columns=columns).to_pandas())
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


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
