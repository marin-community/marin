# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for zephyr/shuffle.py.

Tests the scatter write/read roundtrip, per-shard stats, needs_external_sort,
and multi-segment schema evolution — all without spinning up a full coordinator.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from zephyr.plan import deterministic_hash, _arrow_reduce_gen
from zephyr.shuffle import (
    ScatterParquetIterator,
    ScatterShard,
    _ZEPHYR_SORT_KEY,
    _build_scatter_shard_from_manifest,
    make_envelope_batch,
    _write_parquet_scatter,
    _write_scatter_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key(item):
    return item["k"]


def _target(key, num_shards):
    return deterministic_hash(key) % num_shards


def _build_shard(tmp_path, items, num_output_shards=4, source_shard=0):
    """Write a scatter file and manifest; return (manifest_path, seg_paths)."""
    parquet_path = str(tmp_path / f"shard-{source_shard:04d}.parquet")
    list_shard = _write_parquet_scatter(
        iter(items),
        source_shard=source_shard,
        parquet_path=parquet_path,
        key_fn=_key,
        num_output_shards=num_output_shards,
    )
    seg_paths = list(list_shard)
    manifest_path = str(tmp_path / "scatter_metadata")
    _write_scatter_manifest(seg_paths, manifest_path)
    return manifest_path, seg_paths


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_scatter_roundtrip(tmp_path):
    """All items written via scatter are recovered when reading all shards."""
    num_shards = 4
    items = [{"k": i % 4, "v": i} for i in range(40)]
    manifest_path, _ = _build_shard(tmp_path, items, num_output_shards=num_shards)

    # Collect all items across all shards
    recovered = []
    for shard_idx in range(num_shards):
        shard = _build_scatter_shard_from_manifest(manifest_path, shard_idx)
        recovered.extend(list(shard))

    assert sorted(recovered, key=lambda x: x["v"]) == sorted(items, key=lambda x: x["v"])


def test_scatter_each_shard_gets_correct_items(tmp_path):
    """Items are routed to shards by deterministic_hash(key) % num_shards."""
    num_shards = 4
    items = [{"k": i % 4, "v": i} for i in range(40)]
    manifest_path, _ = _build_shard(tmp_path, items, num_output_shards=num_shards)

    for shard_idx in range(num_shards):
        shard = _build_scatter_shard_from_manifest(manifest_path, shard_idx)
        recovered = sorted(list(shard), key=lambda x: x["v"])
        expected = sorted([x for x in items if _target(x["k"], num_shards) == shard_idx], key=lambda x: x["v"])
        assert recovered == expected, f"shard {shard_idx} mismatch"


def test_scatter_roundtrip_sorted_chunks(tmp_path):
    """Each chunk iterator from get_iterators() yields items sorted by key."""
    items = [{"k": i % 2, "v": i} for i in range(20)]
    manifest_path, _ = _build_shard(tmp_path, items, num_output_shards=2)

    for shard_idx in range(2):
        shard = _build_scatter_shard_from_manifest(manifest_path, shard_idx)
        for chunk_iter in shard.get_iterators():
            chunk = list(chunk_iter)
            keys = [_key(x) for x in chunk]
            assert keys == sorted(keys), f"chunk for shard {shard_idx} not sorted"


# ---------------------------------------------------------------------------
# Per-shard max_chunk_rows — no cross-shard contamination
# ---------------------------------------------------------------------------


def test_max_chunk_rows_per_shard(tmp_path):
    """max_row_group_rows reflects only the target shard's row groups, not a global max.

    We create a skewed distribution: one key has many items (large row group),
    another has few (small row group). Verify each shard's stat is accurate.
    """
    num_shards = 4
    # k=3 → shard 0 (deterministic_hash(3) % 4 == 0): send 500 items
    # k=0 → shard 1 (deterministic_hash(0) % 4 == 1): send 2 items
    items = [{"k": 3, "v": i} for i in range(500)]
    items += [{"k": 0, "v": i + 1000} for i in range(2)]

    manifest_path, _ = _build_shard(tmp_path, items, num_output_shards=num_shards)

    shard0 = _build_scatter_shard_from_manifest(manifest_path, 0)  # k=3, 500 items
    shard1 = _build_scatter_shard_from_manifest(manifest_path, 1)  # k=0, 2 items

    assert shard0.max_row_group_rows == 500
    assert shard1.max_row_group_rows == 2, (
        f"shard1 max_chunk_rows={shard1.max_row_group_rows}, expected 2; "
        "contamination from shard0's large row group would show 500"
    )


# ---------------------------------------------------------------------------
# needs_external_sort
# ---------------------------------------------------------------------------


def test_needs_external_sort_triggers(tmp_path):
    """needs_external_sort returns True when estimated memory exceeds the budget."""
    shard = ScatterShard(
        iterators=[
            ScatterParquetIterator(
                path="gs://fake/path.parquet",
                shard_idx=0,
                chunk_count=1000,
                is_pickled=False,
                filesystem=pa.fs.LocalFileSystem(),
            )
        ],
        max_row_group_rows=1000,
        avg_item_bytes=1000.0,
    )
    # 1000 chunks * 1000 rows * 1000 bytes = 1 GB > 0.5 * 1 GB
    assert shard.needs_external_sort(memory_limit=1024**3)


def test_needs_external_sort_below_threshold(tmp_path):
    """needs_external_sort returns False for a small shard well within budget."""
    items = [{"k": 0, "v": i} for i in range(5)]
    manifest_path, _ = _build_shard(tmp_path, items, num_output_shards=1)
    # k=0 → deterministic_hash(0) % 1 = 0
    shard = _build_scatter_shard_from_manifest(manifest_path, 0)
    assert not shard.needs_external_sort(memory_limit=32 * 1024**3)


def test_needs_external_sort_empty_shard():
    """needs_external_sort returns False for a shard with no chunks."""
    shard = ScatterShard(iterators=[], max_row_group_rows=100_000, avg_item_bytes=200.0)
    assert not shard.needs_external_sort(memory_limit=32 * 1024**3)


# ---------------------------------------------------------------------------
# Schema evolution (multi-segment)
# ---------------------------------------------------------------------------


def test_schema_evolution_multi_segment(tmp_path):
    """Items with evolving schemas across shards create multiple segments.

    k=1 → shard 0 (hash(1)%2=0): score is int  — written first (shard 0)
    k=0 → shard 1 (hash(0)%2=1): score is None — written second (shard 1)

    int64 != null triggers schema evolution and a second segment.
    All items are still recovered correctly.
    """
    # k=1 items → shard 0, score is int (non-null)
    items = [{"k": 1, "v": i, "score": i} for i in range(5)]
    # k=0 items → shard 1, score is None
    items += [{"k": 0, "v": i + 100, "score": None} for i in range(5)]

    parquet_path = str(tmp_path / "shard-0000.parquet")
    list_shard = _write_parquet_scatter(
        iter(items),
        source_shard=0,
        parquet_path=parquet_path,
        key_fn=_key,
        num_output_shards=2,
    )
    seg_paths = list(list_shard)
    assert len(seg_paths) >= 2, "expected multiple segments from schema evolution"

    manifest_path = str(tmp_path / "scatter_metadata")
    _write_scatter_manifest(seg_paths, manifest_path)

    recovered = []
    for shard_idx in range(2):
        shard = _build_scatter_shard_from_manifest(manifest_path, shard_idx)
        recovered.extend(list(shard))

    assert sorted(recovered, key=lambda x: x["v"]) == sorted(items, key=lambda x: x["v"])


# ---------------------------------------------------------------------------
# avg_item_bytes stat
# ---------------------------------------------------------------------------


def test_avg_item_bytes_written(tmp_path):
    """avg_item_bytes is written to the manifest and recovered in ScatterShard."""
    items = [{"k": 0, "v": i} for i in range(20)]
    manifest_path, _ = _build_shard(tmp_path, items, num_output_shards=1)
    shard = _build_scatter_shard_from_manifest(manifest_path, 0)
    assert shard.avg_item_bytes > 0, "avg_item_bytes should be populated from manifest"


# ---------------------------------------------------------------------------
# Pickle roundtrip
# ---------------------------------------------------------------------------


def test_scatter_parquet_iterator_pickle_roundtrip(tmp_path):
    """ScatterParquetIterator with is_pickled=True round-trips non-Arrow-serializable items."""
    items = [frozenset([1, 2]), frozenset([3, 4, 5])]
    batch = make_envelope_batch(
        items,
        shard_idx=0,
        chunk_idx=0,
        key_values=[0, 1],
        sort_values=None,
        pickled=True,
    )

    path = str(tmp_path / "test.parquet")
    pq.write_table(pa.Table.from_batches([batch]), path)

    it = ScatterParquetIterator(
        path=path,
        shard_idx=0,
        chunk_count=1,
        is_pickled=True,
        filesystem=pa.fs.LocalFileSystem(),
    )
    chunks = [list(chunk_iter) for chunk_iter in it.get_chunk_iterators()]
    assert len(chunks) == 1
    assert chunks[0] == items


# ---------------------------------------------------------------------------
# Sort key column
# ---------------------------------------------------------------------------


def test_scatter_writes_sort_key_column(tmp_path):
    """Parquet files contain a _zephyr_sort_key column matching key_fn values."""
    num_shards = 2
    items = [{"k": i % 3, "v": i} for i in range(20)]
    parquet_path = str(tmp_path / "shard-0000.parquet")
    list_shard = _write_parquet_scatter(
        iter(items),
        source_shard=0,
        parquet_path=parquet_path,
        key_fn=_key,
        num_output_shards=num_shards,
    )
    seg_paths = list(list_shard)
    for seg_path in seg_paths:
        table = pq.read_table(seg_path)
        assert _ZEPHYR_SORT_KEY in table.column_names
        for row in table.to_pylist():
            # In flat mode, user fields are top-level columns alongside _zephyr_* metadata
            assert row[_ZEPHYR_SORT_KEY] == row["k"]


def test_get_chunk_tables_returns_arrow(tmp_path):
    """get_chunk_tables yields pa.Table instances with sort key columns."""
    items = [{"k": i % 2, "v": i} for i in range(20)]
    manifest_path, _ = _build_shard(tmp_path, items, num_output_shards=1)
    shard = _build_scatter_shard_from_manifest(manifest_path, 0)
    tables = []
    for it in shard.iterators:
        tables.extend(list(it.get_chunk_tables()))
    assert len(tables) > 0
    for t in tables:
        assert isinstance(t, pa.Table)
        assert _ZEPHYR_SORT_KEY in t.column_names


def test_scatter_with_combiner(tmp_path):
    """Scatter with combiner_fn produces correct combined output."""
    # Duplicate keys — combiner keeps only one per key
    items = [{"k": i % 3, "v": i} for i in range(30)]

    def combiner(key, items_iter):
        return [next(items_iter)]

    parquet_path = str(tmp_path / "shard-0000.parquet")
    list_shard = _write_parquet_scatter(
        iter(items),
        source_shard=0,
        parquet_path=parquet_path,
        key_fn=_key,
        num_output_shards=1,
        combiner_fn=combiner,
    )
    seg_paths = list(list_shard)
    manifest_path = str(tmp_path / "scatter_metadata")
    _write_scatter_manifest(seg_paths, manifest_path)

    shard = _build_scatter_shard_from_manifest(manifest_path, 0)
    recovered = list(shard)
    # With combiner keeping first, we get at most 3 unique keys (0,1,2)
    keys = {item["k"] for item in recovered}
    assert keys == {0, 1, 2}
    assert len(recovered) == 3


# ---------------------------------------------------------------------------
# external_sort_merge
# ---------------------------------------------------------------------------


def test_arrow_merge_produces_same_results(tmp_path):
    """Arrow merge-sort produces correct grouped results."""
    num_shards = 4
    items = [{"k": i % 7, "v": i} for i in range(200)]
    manifest_path, _ = _build_shard(tmp_path, items, num_output_shards=num_shards)

    def keep_first(key, items_iter):
        return next(items_iter)

    for shard_idx in range(num_shards):
        shard = _build_scatter_shard_from_manifest(manifest_path, shard_idx)
        arrow_results = sorted(list(_arrow_reduce_gen(shard, keep_first)), key=lambda x: x["v"])

        # Verify: each result should be a valid item from the original set
        for result in arrow_results:
            assert result in items, f"unexpected item {result} in shard {shard_idx}"

        # Verify: each key appears at most once (keep_first deduplication)
        keys_seen = [result["k"] for result in arrow_results]
        assert len(keys_seen) == len(set(keys_seen)), f"duplicate keys in shard {shard_idx}"


def test_arrow_external_sort_roundtrip(tmp_path):
    """Arrow external sort produces correctly sorted output."""
    from zephyr.external_sort import external_sort_merge

    tables = [
        pa.table({"a": [1, 4, 7], _ZEPHYR_SORT_KEY: [1, 4, 7]}),
        pa.table({"a": [2, 5, 8], _ZEPHYR_SORT_KEY: [2, 5, 8]}),
        pa.table({"a": [3, 6, 9], _ZEPHYR_SORT_KEY: [3, 6, 9]}),
    ]
    sort_dir = str(tmp_path / "ext_sort")

    result = list(
        external_sort_merge(
            iter(tables),
            sort_keys=[(_ZEPHYR_SORT_KEY, "ascending")],
            merge_key=lambda row: row[_ZEPHYR_SORT_KEY],
            external_sort_dir=sort_dir,
        )
    )
    # Result rows contain all columns including _zephyr_sort_key; extract "a" values
    a_values = [row["a"] for row in result]
    assert a_values == list(range(1, 10))


def test_arrow_external_sort_cleans_up(tmp_path):
    """Arrow external sort run files are deleted after merge."""
    from zephyr.external_sort import external_sort_merge

    tables = [pa.table({"val": [i], _ZEPHYR_SORT_KEY: [i]}) for i in range(10)]
    sort_dir = str(tmp_path / "ext_sort")

    list(
        external_sort_merge(
            iter(tables),
            sort_keys=[(_ZEPHYR_SORT_KEY, "ascending")],
            merge_key=lambda row: row[_ZEPHYR_SORT_KEY],
            external_sort_dir=sort_dir,
        )
    )
    import os

    if os.path.exists(sort_dir):
        remaining = os.listdir(sort_dir)
        assert remaining == [], f"run files should be deleted, found: {remaining}"


def test_needs_external_sort_zero_memory():
    """needs_external_sort returns False when memory_limit is 0 (unknown)."""
    shard = ScatterShard(
        iterators=[
            ScatterParquetIterator(
                path="gs://fake/path.parquet",
                shard_idx=0,
                chunk_count=1000,
                is_pickled=False,
                filesystem=pa.fs.LocalFileSystem(),
            )
        ],
        max_row_group_rows=1000,
        avg_item_bytes=1000.0,
    )
    assert not shard.needs_external_sort(memory_limit=0)
