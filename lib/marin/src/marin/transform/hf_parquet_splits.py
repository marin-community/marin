# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read downloaded HuggingFace parquet shards by split.

Staging jobs that bridge HF datasets into raw text (e.g. the LM-eval and
structured-text PPL slices) read parquet shards that a download step laid out
on an fsspec filesystem. These helpers locate the shards for a requested
``split`` (optionally scoped to a ``subset``) and stream their rows.
"""

import os
import posixpath
from collections.abc import Iterable
from typing import Any

from datasets import load_dataset
from rigging.filesystem import StoragePath, url_to_fs


def _parquet_file_matches_split(path: str, split: str) -> bool:
    filename = os.path.basename(path)
    if not filename.endswith(".parquet"):
        return False
    return filename == f"{split}.parquet" or filename.startswith(f"{split}-")


def find_split_parquet_files(input_path: str, split: str, subset: str | None) -> list[str]:
    """Find downloaded HF parquet files for ``split`` under an fsspec path.

    When ``subset`` is given and a matching subdirectory exists, the scan is
    restricted to that subset so sibling subsets sharing the same split name
    are not pulled in; otherwise it falls back to the whole tree.
    """
    fs, root = url_to_fs(input_path)
    roots: list[str] = []
    if subset and subset != "default":
        subset_root = posixpath.join(root, subset)
        if fs.exists(subset_root):
            roots.append(subset_root)
    if not roots:
        roots.append(root)

    matches: list[str] = []
    for candidate_root in roots:
        if fs.isfile(candidate_root):
            candidates = [candidate_root]
            selected = [path for path in candidates if path.endswith(".parquet")]
        else:
            candidates = list(fs.find(candidate_root, withdirs=False))
            selected = [path for path in candidates if _parquet_file_matches_split(path, split)]
        matches.extend(selected)

    if not matches:
        raise FileNotFoundError(f"No parquet files found for split {split!r} under {input_path}")

    scheme = StoragePath(input_path).scheme
    prefix = f"{scheme}://" if scheme and scheme != "file" else ""
    return [f"{prefix}{path}" for path in sorted(set(matches))]


def load_hf_split_iterable(input_path: str, split: str, subset: str | None) -> Iterable[dict[str, Any]]:
    """Stream rows from the downloaded HF parquet shards for ``split``."""
    data_files = find_split_parquet_files(input_path, split, subset)
    dataset = load_dataset("parquet", data_files={split: data_files}, split=split, streaming=True)
    return dataset
