# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filesystem helpers for alternating RL manifests and pickled payloads."""

from __future__ import annotations

import os
import pickle
from typing import Any

from iris.marin_fs import url_to_fs
from levanter.utils.fsspec_utils import expand_glob


def write_pickle(path: str, value: Any) -> None:
    """Write one pickle atomically to any fsspec-backed filesystem."""
    fs, fs_path = url_to_fs(path)
    parent = os.path.dirname(fs_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    tmp_path = f"{fs_path}.tmp"
    if fs.exists(tmp_path):
        fs.rm(tmp_path)
    with fs.open(tmp_path, "wb") as handle:
        pickle.dump(value, handle)
    if fs.exists(fs_path):
        fs.rm(fs_path)
    fs.mv(tmp_path, fs_path)


def read_pickle(path: str) -> Any:
    """Read one pickle from any fsspec-backed filesystem."""
    fs, fs_path = url_to_fs(path)
    with fs.open(fs_path, "rb") as handle:
        return pickle.load(handle)


def glob_paths(pattern: str) -> list[str]:
    """Return sorted glob matches for one filesystem pattern."""
    return sorted(expand_glob(pattern))
