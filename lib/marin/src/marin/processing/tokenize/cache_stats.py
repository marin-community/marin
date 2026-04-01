# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

from rigging.filesystem import open_url, url_to_fs

SplitName = Literal["train", "validation"]
_STATS_FILE_NAME = ".stats.json"


@dataclass(frozen=True)
class TokenizedCacheStats:
    """Element and token counts recorded for a tokenized cache split."""

    total_elements: int
    total_tokens: int


def tokenized_cache_stats_path(cache_root: str, split: SplitName) -> str:
    """Return the `.stats.json` path for a tokenized cache split."""
    fs, fs_path = url_to_fs(cache_root)
    stats_fs_path = str(PurePath(fs_path) / split / _STATS_FILE_NAME)
    stats_path = fs.unstrip_protocol(stats_fs_path)
    if stats_path.startswith("file://"):
        return stats_fs_path

    return stats_path


def read_tokenized_cache_stats(cache_root: str, split: SplitName) -> TokenizedCacheStats:
    """Read tokenized cache stats for one split."""
    stats_path = tokenized_cache_stats_path(cache_root, split)
    try:
        with open_url(stats_path, mode="r") as f:
            stats = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Tokenized cache stats not found at {stats_path}") from exc

    total_elements = stats.get("total_elements")
    total_tokens = stats.get("total_tokens")
    if not isinstance(total_elements, int) or total_elements < 0:
        raise ValueError(f"Invalid tokenized cache stats at {stats_path}: expected non-negative total_elements.")
    if not isinstance(total_tokens, int) or total_tokens < 0:
        raise ValueError(f"Invalid tokenized cache stats at {stats_path}: expected non-negative total_tokens.")

    return TokenizedCacheStats(total_elements=total_elements, total_tokens=total_tokens)
