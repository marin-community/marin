# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for consolidated metadata copy using a shared ts.Transaction (#4100)."""

import asyncio
import copy
import operator
import os
import tempfile

import jax
import numpy as np
from levanter.store.cache import (
    CacheLedger,
    _consolidate_metadata,
    _expose_cache_rows,
    _extend_cache_with_other_cache,
    consolidate_shard_caches,
)
from levanter.store.tree_store import TreeStore

NUM_SHARDS = 8
ROWS_PER_SHARD = 32
ROW_WIDTH = 16

# rank-1 only (no shapes metadata)
EXEMPLAR_FLAT = {"input_ids": np.array([0], dtype=np.int32)}

# multi-field with a rank-2 leaf (triggers shapes metadata)
EXEMPLAR_SHAPED = {
    "input_ids": np.array([0], dtype=np.int32),
    "spans": np.zeros((0, 2), dtype=np.int32),
}


def _build_and_consolidate(exemplar, make_row) -> TreeStore:
    """Build shards, copy data + metadata, return the merged store."""
    with tempfile.TemporaryDirectory(prefix="levanter-test-consolidate-") as tmpdir:
        shard_root = os.path.join(tmpdir, "shards")
        os.makedirs(shard_root)

        data_offset_tree = jax.tree.map(lambda _: 0, exemplar)
        total_rows = 0
        shard_infos = []

        for i in range(NUM_SHARDS):
            shard_path = os.path.join(shard_root, f"shard_{i}")
            store = TreeStore.open(exemplar, shard_path, mode="w", cache_metadata=True)
            store.extend([make_row(i) for _ in range(ROWS_PER_SHARD)])

            shard_infos.append(
                {
                    "path": shard_path,
                    "row_offset": total_rows,
                    "data_offset_tree": copy.deepcopy(data_offset_tree),
                    "ledger": CacheLedger(total_num_rows=ROWS_PER_SHARD, shard_rows={}, is_finished=True),
                }
            )
            total_rows += ROWS_PER_SHARD

            this_offsets = jax.tree.map(lambda x: x.data_size, store.tree)
            data_offset_tree = jax.tree.map(operator.add, data_offset_tree, this_offsets)

        dest_path = os.path.join(tmpdir, "dest")
        TreeStore.open(exemplar, dest_path, mode="w", cache_metadata=True)

        for info in shard_infos:
            asyncio.run(
                _extend_cache_with_other_cache(
                    dest_path,
                    info["path"],
                    exemplar,
                    info["data_offset_tree"],
                    info["row_offset"],
                )
            )
        asyncio.run(_consolidate_metadata(dest_path, exemplar, shard_infos))
        _expose_cache_rows(dest_path, exemplar, total_rows)

        merged = TreeStore.open(exemplar, dest_path, mode="r", cache_metadata=True)
        assert len(merged) == NUM_SHARDS * ROWS_PER_SHARD

        for i, info in enumerate(shard_infos):
            row = merged[info["row_offset"]]
            assert row["input_ids"][0] == i, f"shard {i} data mismatch"

        return merged


def test_consolidate_metadata_flat():
    """Round-trip with a single rank-1 field (no shapes metadata)."""

    def make_row(shard_index):
        return {"input_ids": np.full((ROW_WIDTH,), shard_index, dtype=np.int32)}

    _build_and_consolidate(EXEMPLAR_FLAT, make_row)


def test_consolidate_metadata_shaped():
    """Round-trip with multiple fields including rank-2 (exercises shapes metadata)."""

    def make_row(shard_index):
        return {
            "input_ids": np.full((ROW_WIDTH,), shard_index, dtype=np.int32),
            "spans": np.full((3, 2), shard_index, dtype=np.int32),
        }

    merged = _build_and_consolidate(EXEMPLAR_SHAPED, make_row)
    row = merged[0]
    assert row["spans"].shape == (3, 2)


def _build_shard_cache(shard_path: str, exemplar, rows: list[dict]) -> None:
    """Build a shard cache directory with data and a serialized ledger."""
    store = TreeStore.open(exemplar, shard_path, mode="w", cache_metadata=True)
    store.extend(rows)
    _expose_cache_rows(shard_path, exemplar, len(rows))
    ledger = CacheLedger(
        total_num_rows=len(rows),
        shard_rows={os.path.basename(shard_path): len(rows)},
        is_finished=True,
        finished_shards=[os.path.basename(shard_path)],
        field_counts={},
    )
    ledger._serialize_and_commit(shard_path)


def test_consolidate_shard_caches_end_to_end():
    """Call consolidate_shard_caches directly, exercising the threaded pre-pass and Zephyr data copy."""
    with tempfile.TemporaryDirectory(prefix="levanter-test-consolidate-e2e-") as tmpdir:
        shard_paths = []
        for i in range(NUM_SHARDS):
            shard_path = os.path.join(tmpdir, f"shard_{i}")
            rows = [{"input_ids": np.full((ROW_WIDTH,), i, dtype=np.int32)} for _ in range(ROWS_PER_SHARD)]
            _build_shard_cache(shard_path, EXEMPLAR_FLAT, rows)
            shard_paths.append(shard_path)

        dest_path = os.path.join(tmpdir, "merged")
        ledger = consolidate_shard_caches(shard_paths, dest_path, EXEMPLAR_FLAT, copy_max_workers=1)

        assert ledger.total_num_rows == NUM_SHARDS * ROWS_PER_SHARD
        assert ledger.is_finished

        merged = TreeStore.open(EXEMPLAR_FLAT, dest_path, mode="r", cache_metadata=True)
        assert len(merged) == NUM_SHARDS * ROWS_PER_SHARD

        for i in range(NUM_SHARDS):
            row = merged[i * ROWS_PER_SHARD]
            assert row["input_ids"][0] == i, f"shard {i} data mismatch"
