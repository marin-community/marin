# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize bounded short/long replay streams from pinned Snowball S3 shards."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from levanter.store.cache import write_levanter_cache
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import StoragePath

LENGTH_THRESHOLD = 65_536
READ_BATCH_ROWS = 128
WRITE_BATCH_ROWS = 1024


@dataclass(frozen=True)
class ReplaySubsetConfig:
    source_shards: tuple[str, ...]
    output_path: str
    short_tokens: int
    long_tokens: int
    tokenizer: str


@dataclass(frozen=True)
class SelectedRows:
    source: str
    offsets_sha256: str
    short: np.ndarray
    long: np.ndarray
    short_tokens: int
    long_tokens: int


def select_length_strata(lengths: np.ndarray, short_budget: int, long_budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Take whole documents in source order, crossing each positive budget once."""
    selections = []
    for mask, budget in ((lengths <= LENGTH_THRESHOLD, short_budget), (lengths > LENGTH_THRESHOLD, long_budget)):
        indices = np.flatnonzero(mask)
        if budget <= 0:
            indices = indices[:0]
        else:
            cumulative = np.cumsum(lengths[indices], dtype=np.int64)
            count = int(np.searchsorted(cumulative, budget, side="left")) + 1
            indices = indices[:count]
        selections.append(indices)
    return selections[0], selections[1]


def select_replay_rows(config: ReplaySubsetConfig) -> list[SelectedRows]:
    remaining_short, remaining_long = config.short_tokens, config.long_tokens
    selected = []
    for source in config.source_shards:
        store = JaggedArrayStore.open(str(StoragePath(source) / "input_ids"), mode="r", dtype=np.int32)
        count = len(store)
        offsets = np.asarray(store.offsets[: count + 1].read().result())
        lengths = np.diff(np.concatenate((np.array([0], dtype=np.int64), offsets[1:])))
        assert np.all(lengths > 0), source
        short, long = select_length_strata(lengths, remaining_short, remaining_long)
        short_tokens, long_tokens = int(lengths[short].sum()), int(lengths[long].sum())
        selected.append(
            SelectedRows(source, hashlib.sha256(offsets.tobytes()).hexdigest(), short, long, short_tokens, long_tokens)
        )
        remaining_short -= short_tokens
        remaining_long -= long_tokens
        if remaining_short <= 0 and remaining_long <= 0:
            break
    if remaining_short > 0 or remaining_long > 0:
        raise ValueError(f"Replay sources exhausted: short deficit={remaining_short}, long deficit={remaining_long}")
    return selected


def _records(selected: list[SelectedRows], *, long_documents: bool) -> Iterator[dict[str, np.ndarray]]:
    for shard in selected:
        indices = shard.long if long_documents else shard.short
        if len(indices) == 0:
            continue
        store = JaggedArrayStore.open(str(StoragePath(shard.source) / "input_ids"), mode="r", dtype=np.int32)
        # A long row can contain hundreds of millions of tokens. Keep at most
        # one such document in each read/write batch without changing its data.
        batch_rows = 1 if long_documents else READ_BATCH_ROWS
        for start in range(0, len(indices), batch_rows):
            for tokens in store.get_batch_sync(indices[start : start + batch_rows].tolist()):
                yield {"input_ids": tokens}


def prepare_replay_subset(config: ReplaySubsetConfig) -> dict[str, str]:
    """Write only the bounded subset; source shards must already be region-local."""
    if not config.source_shards or config.short_tokens < 0 or config.long_tokens < 0:
        raise ValueError("Replay subset requires source shards and nonnegative token budgets")
    if len(set(config.source_shards)) != len(config.source_shards):
        raise ValueError("Replay source shards must be unique")
    selected = select_replay_rows(config)
    outputs = {}
    for name, long_documents, budget in (
        ("lte_64k", False, config.short_tokens),
        ("gt_64k", True, config.long_tokens),
    ):
        if budget == 0:
            continue
        output = str(StoragePath(config.output_path) / name)
        write_levanter_cache(
            _records(selected, long_documents=long_documents),
            output,
            metadata={"tokenizer": config.tokenizer, "length_threshold": LENGTH_THRESHOLD},
            batch_size=1 if long_documents else WRITE_BATCH_ROWS,
            max_pending_batches=1,
        )
        outputs[name] = output
    audit = [
        {
            "source": shard.source,
            "offsets_sha256": shard.offsets_sha256,
            "short_rows": len(shard.short),
            "long_rows": len(shard.long),
            "short_tokens": shard.short_tokens,
            "long_tokens": shard.long_tokens,
            "short_indices_sha256": hashlib.sha256(shard.short.tobytes()).hexdigest(),
            "long_indices_sha256": hashlib.sha256(shard.long.tobytes()).hexdigest(),
        }
        for shard in selected
    ]
    (StoragePath(config.output_path) / "selection.json").write_text(json.dumps(audit, indent=2) + "\n")
    return outputs
