# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-split statistics for a tokenized cache.

Each tokenized dataset split gets a small ``.stats.json`` sidecar recording that
split's document and token counts. The file lives next to the split's Levanter
cache at ``<cache_root>/<split>/.stats.json`` and is written at the end of
tokenization by ``store_builder.write_stats_json`` -- ``total_tokens`` comes from
the cache ledger's ``input_ids`` field count and ``total_elements`` from its row
count.

This module is the read side of that contract: it defines the path convention
(``tokenized_cache_stats_path``), the typed record (``TokenizedCacheStats``), and
a validating reader (``read_tokenized_cache_stats``). Consumers read these counts
to size training runs without rescanning the cache (e.g. resolving step or epoch
counts from a split's token total).
"""

import json
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

from rigging.filesystem import StoragePath, url_to_fs

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
        stats = json.loads(StoragePath(stats_path).read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Tokenized cache stats not found at {stats_path}") from exc

    total_elements = stats.get("total_elements")
    total_tokens = stats.get("total_tokens")
    if not isinstance(total_elements, int) or total_elements < 0:
        raise ValueError(f"Invalid tokenized cache stats at {stats_path}: expected non-negative total_elements.")
    if not isinstance(total_tokens, int) or total_tokens < 0:
        raise ValueError(f"Invalid tokenized cache stats at {stats_path}: expected non-negative total_tokens.")

    return TokenizedCacheStats(total_elements=total_elements, total_tokens=total_tokens)
