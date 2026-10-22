# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check prepared native-token caches before allocating training accelerators."""

import json

import numpy as np
from levanter.store.cache import TreeCache
from rigging.filesystem.storage_path import prefix_join

READ_BATCH_SIZE = 8


def verify_cache(cache_dir: str, report: dict) -> None:
    """Require prepared row counts, context lengths, binary masks and supervision totals."""
    cache = TreeCache.load(
        prefix_join(cache_dir, "train"),
        {"input_ids": np.zeros(0, dtype=np.int32), "loss_weights": np.zeros(0, dtype=np.float32)},
    )
    expected_rows = report["training_rows"] + report["padding_rows"]
    if len(cache) != expected_rows:
        raise ValueError(f"Wrong cache row count: {len(cache)} != {expected_rows}")
    supervised = 0
    for start in range(0, expected_rows, READ_BATCH_SIZE):
        for row in cache.get_batch_sync(list(range(start, min(start + READ_BATCH_SIZE, expected_rows)))):
            if len(row["input_ids"]) != report["max_sequence_tokens"]:
                raise ValueError("Cache sequence length differs from training context")
            weights = row["loss_weights"]
            if weights.shape != row["input_ids"].shape or not np.isin(weights, [0, 1]).all():
                raise ValueError("Cache loss mask is malformed")
            supervised += int(weights.sum())
    if supervised != report["supervised_tokens"]:
        raise ValueError("Cache changed the supervised token count")
    print(json.dumps({"verified_cache_rows": len(cache), "supervised_tokens": supervised}), flush=True)
