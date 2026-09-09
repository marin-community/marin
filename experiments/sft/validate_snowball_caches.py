# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the immutable prebuilt caches used by the Snowball agentic stages."""

import json
import math
from dataclasses import dataclass

import numpy as np
from levanter.store.cache import CacheLedger, TreeCache
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from rigging.filesystem import prefix_join

from experiments.datasets.grug_a2b_agentic_sft_eot import _CACHE_SOURCE as OPENCODE_CACHE
from experiments.sft.configs.snowball_lce_final import _NEMOTRON_CACHE_SOURCE as NEMOTRON_CACHE

_SEQ_LEN = 32_768
_BATCH_SIZE = 64
_OPENCODE_EPOCHS = 5
_EXPECTED_OPENCODE_STEPS = 1_888
_VOCAB_SIZE = 128_256


@dataclass(frozen=True)
class CacheSpec:
    name: str
    root: str
    mask_key: str
    mask_dtype: np.dtype
    expected_steps: int | None = None
    epochs: int | None = None


def _validate(spec: CacheSpec) -> dict[str, int | str]:
    split_path = prefix_join(spec.root, "train")
    stats = read_tokenized_cache_stats(spec.root, "train")
    ledger = CacheLedger.load(split_path)
    if not ledger.is_finished:
        raise ValueError(f"{spec.name}: cache ledger is unfinished")
    if ledger.total_num_rows != stats.total_elements:
        raise ValueError(
            f"{spec.name}: ledger has {ledger.total_num_rows} rows but stats report {stats.total_elements}"
        )
    if ledger.field_counts.get("input_ids") != stats.total_tokens:
        raise ValueError(
            f"{spec.name}: ledger has {ledger.field_counts.get('input_ids')} tokens but stats report "
            f"{stats.total_tokens}"
        )

    exemplar = {
        "input_ids": np.zeros((0,), dtype=np.int32),
        spec.mask_key: np.zeros((0,), dtype=spec.mask_dtype),
    }
    cache = TreeCache.load(split_path, exemplar)
    if len(cache) == 0:
        raise ValueError(f"{spec.name}: cache is empty")

    for index in (0, len(cache) - 1):
        record = cache[index]
        tokens = np.asarray(record["input_ids"])
        mask = np.asarray(record[spec.mask_key])
        if tokens.ndim != 1 or mask.ndim != 1 or tokens.shape != mask.shape:
            raise ValueError(f"{spec.name}[{index}]: incompatible token/mask shapes {tokens.shape} and {mask.shape}")
        if tokens.size == 0:
            raise ValueError(f"{spec.name}[{index}]: empty record")
        if np.min(tokens) < 0 or np.max(tokens) >= _VOCAB_SIZE:
            raise ValueError(f"{spec.name}[{index}]: token outside [0, {_VOCAB_SIZE})")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{spec.name}[{index}]: mask contains values other than zero and one")

    resolved_steps = 0
    if spec.epochs is not None:
        resolved_steps = math.ceil(spec.epochs * stats.total_tokens / (_SEQ_LEN * _BATCH_SIZE))
        if resolved_steps != spec.expected_steps:
            raise ValueError(
                f"{spec.name}: {spec.epochs} epochs resolve to {resolved_steps} updates, expected {spec.expected_steps}"
            )

    return {
        "name": spec.name,
        "rows": stats.total_elements,
        "tokens": stats.total_tokens,
        "resolved_steps": resolved_steps,
    }


def main() -> None:
    specs = (
        CacheSpec(
            name="opencode-fixed-eot",
            root=OPENCODE_CACHE,
            mask_key="assistant_mask",
            mask_dtype=np.dtype(np.float32),
            expected_steps=_EXPECTED_OPENCODE_STEPS,
            epochs=_OPENCODE_EPOCHS,
        ),
        CacheSpec(
            name="nemotron-terminal",
            root=NEMOTRON_CACHE,
            mask_key="assistant_masks",
            mask_dtype=np.dtype(np.int32),
        ),
    )
    print(json.dumps([_validate(spec) for spec in specs], sort_keys=True))


if __name__ == "__main__":
    main()
