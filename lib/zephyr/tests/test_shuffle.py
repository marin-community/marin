# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for zephyr/shuffle.py.

Covers the scatter write/read roundtrip, per-shard stats, and external sort —
without spinning up a full coordinator.
"""

from collections import OrderedDict

import polars as pl
from zephyr.external_sort import EXTERNAL_SORT_FAN_IN, _dataframe_to_items, external_sort_merge
from zephyr.shuffle import (
    _SORT_KEY_COL,
    ScatterReader,
    ScatterWriter,
    _items_to_dataframe,
    _write_scatter,
    deterministic_hash,
)


def _read_shard(shard: ScatterReader) -> list:
    frames = list(shard.get_iterators())
    if not frames:
        return []
    combined = pl.concat([f.collect() for f in frames], how="diagonal_relaxed")
    return list(_dataframe_to_items(combined))


def _key(item):
    return item["k"]


def _target(key, num_shards):
    return deterministic_hash(key) % num_shards


def _build_shard(tmp_path, items, num_output_shards=4, source_shard=0):
    """Write a scatter file + sidecar; return scatter_paths for direct reducer reads."""
    data_path = str(tmp_path / f"shard-{source_shard:04d}.shuffle")
    list_shard = _write_scatter(
        iter(items),
        source_shard=source_shard,
        data_path=data_path,
        key_fn=_key,
        num_output_shards=num_output_shards,
    )
    scatter_paths = list(list_shard)
    return scatter_paths


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_scatter_roundtrip(tmp_path):
    """All items written via scatter are recovered when reading all shards."""
    num_shards = 4
    items = [{"k": i % 4, "v": i} for i in range(40)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=num_shards)

    recovered = []
    for shard_idx in range(num_shards):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        recovered.extend(_read_shard(shard))

    assert sorted(recovered, key=lambda x: x["v"]) == sorted(items, key=lambda x: x["v"])


def test_scatter_each_shard_gets_correct_items(tmp_path):
    """Items are routed to shards by deterministic_hash(key) % num_shards."""
    num_shards = 4
    items = [{"k": i % 4, "v": i} for i in range(40)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=num_shards)

    for shard_idx in range(num_shards):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        recovered = sorted(_read_shard(shard), key=lambda x: x["v"])
        expected = sorted([x for x in items if _target(x["k"], num_shards) == shard_idx], key=lambda x: x["v"])
        assert recovered == expected, f"shard {shard_idx} mismatch"


def test_scatter_roundtrip_sorted_chunks(tmp_path):
    """Each chunk iterator from get_iterators() yields items sorted by key."""
    items = [{"k": i % 2, "v": i} for i in range(20)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=2)

    for shard_idx in range(2):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        for lf in shard.get_iterators():
            chunk = list(_dataframe_to_items(lf.collect()))
            keys = [_key(x) for x in chunk]
            assert keys == sorted(keys), f"chunk for shard {shard_idx} not sorted"


# ---------------------------------------------------------------------------
# Per-shard stats — no cross-shard contamination
# ---------------------------------------------------------------------------


def test_max_chunk_rows_per_shard(tmp_path):
    """max_chunk_rows reflects only the target shard's chunks, not a global max."""
    num_shards = 4
    items = [{"k": 3, "v": i} for i in range(500)]
    items += [{"k": 0, "v": i + 1000} for i in range(2)]

    scatter_paths = _build_shard(tmp_path, items, num_output_shards=num_shards)

    big_shard = ScatterReader.from_sidecars(scatter_paths, _target(3, num_shards))
    small_shard = ScatterReader.from_sidecars(scatter_paths, _target(0, num_shards))

    assert big_shard.max_chunk_rows == 500
    assert small_shard.max_chunk_rows == 2, (
        f"small_shard max_chunk_rows={small_shard.max_chunk_rows}, expected 2; "
        "contamination from the large chunk would show 500"
    )


# ---------------------------------------------------------------------------
# needs_external_sort
# ---------------------------------------------------------------------------


def test_needs_external_sort_triggers(tmp_path):
    fake_path = str(tmp_path / "fake.shuffle")
    shard = ScatterReader(
        files=[(fake_path, 1000)],
        target_shard=0,
        max_chunk_rows=1000,
        avg_item_bytes=1000.0,
    )
    # 1000 chunks * 1000 rows * 1000 bytes = 1 GB > 0.5 * 1 GB
    assert shard.needs_external_sort(memory_limit=1024**3)


def test_needs_external_sort_below_threshold(tmp_path):
    items = [{"k": 0, "v": i} for i in range(5)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=1)
    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    assert not shard.needs_external_sort(memory_limit=32 * 1024**3)


def test_needs_external_sort_empty_shard():
    shard = ScatterReader(files=[], target_shard=0, max_chunk_rows=100_000, avg_item_bytes=200.0)
    assert not shard.needs_external_sort(memory_limit=32 * 1024**3)


# ---------------------------------------------------------------------------
# avg_item_bytes
# ---------------------------------------------------------------------------


def test_avg_item_bytes_written(tmp_path):
    items = [{"k": 0, "v": i} for i in range(20)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=1)
    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    assert shard.avg_item_bytes > 0


def test_merge_sorted_chunks_basic(tmp_path):
    """merge_sorted_chunks yields grouped (key, items) across all chunks."""
    items = [
        {"k": "a", "v": 1},
        {"k": "b", "v": 2},
        {"k": "a", "v": 3},
        {"k": "b", "v": 4},
    ]
    # Force two chunks by writing twice
    data_path = str(tmp_path / "shard-0000.shuffle")
    writer = ScatterWriter(data_path=data_path, key_fn=_key, num_output_shards=1)
    writer.write(_items_to_dataframe(items[:2], _key, None, 1))
    writer.write(_items_to_dataframe(items[2:], _key, None, 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    groups = {k: list(items) for k, items in shard.merge_sorted_chunks(key_fn=_key)}

    assert len(groups) == 2
    assert sorted([item["v"] for item in groups["a"]]) == [1, 3]
    assert sorted([item["v"] for item in groups["b"]]) == [2, 4]


def test_merge_sorted_chunks_secondary_sort(tmp_path):
    """merge_sorted_chunks respects sort_fn for ordering within groups."""
    items = [
        {"k": "a", "ts": 10, "v": 1},
        {"k": "a", "ts": 5, "v": 2},
    ]
    # Write as two separate chunks
    data_path = str(tmp_path / "shard-0000.shuffle")
    writer = ScatterWriter(data_path=data_path, key_fn=_key, num_output_shards=1)
    writer.write(_items_to_dataframe([items[0]], _key, lambda x: x["ts"], 1))
    writer.write(_items_to_dataframe([items[1]], _key, lambda x: x["ts"], 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    # Sort by ts
    groups = {k: list(items) for k, items in shard.merge_sorted_chunks(key_fn=_key, sort_fn=lambda x: x["ts"])}

    assert len(groups["a"]) == 2
    assert [item["v"] for item in groups["a"]] == [2, 1]  # 5 comes before 10


def test_scatter_with_combiner(tmp_path):
    """ScatterWriter applies combiner_fn during flushes."""
    items = [
        {"k": "a", "v": 1},
        {"k": "a", "v": 2},
    ]

    def sum_combiner(key, items):
        yield {"k": key, "v": sum(i["v"] for i in items)}

    data_path = str(tmp_path / "shard-0000.shuffle")
    writer = ScatterWriter(data_path=data_path, key_fn=_key, num_output_shards=1, combiner_fn=sum_combiner)
    writer.write(_items_to_dataframe(items, _key, None, 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    recovered = _read_shard(shard)
    assert len(recovered) == 1
    assert recovered[0] == {"k": "a", "v": 3}


def test_merge_sorted_chunks_external_trigger(tmp_path):
    """merge_sorted_chunks successfully spills to disk when budget is exceeded."""
    from unittest.mock import patch

    from iris.env_resources import TaskResources

    items = [{"k": i, "v": i} for i in range(10)]
    data_path = str(tmp_path / "shard-0000.shuffle")
    # Write many small chunks
    writer = ScatterWriter(data_path=data_path, key_fn=_key, num_output_shards=1)
    for i in range(10):
        writer.write(_items_to_dataframe([items[i]], _key, None, 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)

    # Force external sort by mocking a tiny memory limit
    external_dir = tmp_path / "sort_work"
    external_dir.mkdir()

    with patch("iris.env_resources.TaskResources.from_environment") as mock_res:
        # 1 byte memory limit will trigger external sort
        mock_res.return_value = TaskResources(memory_bytes=1, cpu_cores=1, gpu_count=0, tpu_count=0)
        groups = {k: list(it) for k, it in shard.merge_sorted_chunks(key_fn=_key, external_sort_dir=str(external_dir))}

    assert len(groups) == 10
    assert sorted(groups.keys()) == list(range(10))


def test_scatter_null_keys(tmp_path):
    """Items with None keys are handled correctly."""
    items = [{"k": None, "v": 1}, {"k": None, "v": 2}]
    num_shards = 2
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=num_shards)

    # Both should go to the same shard
    shard_idx = deterministic_hash(None) % num_shards
    shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
    recovered = _read_shard(shard)
    assert len(recovered) == 2


def test_scatter_empty_input(tmp_path):
    """Scatter handles zero items gracefully."""
    scatter_paths = _build_shard(tmp_path, [], num_output_shards=1)
    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    assert _read_shard(shard) == []
    assert list(shard.merge_sorted_chunks(key_fn=_key)) == []


def test_scatter_all_non_serializable(tmp_path):
    """Everything works even if NO items are Arrow-serializable."""
    items = [
        {"k": "a", "v": frozenset([1])},
        {"k": "b", "v": frozenset([2])},
    ]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=1)
    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    recovered = sorted(_read_shard(shard), key=lambda x: x["k"])
    assert recovered == items


# ---------------------------------------------------------------------------
# Mixed-type values (dropped Parquet's schema constraint)
# ---------------------------------------------------------------------------


def test_scatter_handles_arbitrary_python_objects(tmp_path):
    """Values that are not Arrow-friendly (frozenset, mixed None/int) round-trip."""
    items = [
        {"k": 0, "v": frozenset([1, 2, 3])},
        {"k": 0, "v": frozenset([4, 5])},
        {"k": 1, "v": None},
        {"k": 1, "v": frozenset([6])},
    ]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=2)

    recovered = []
    for shard_idx in range(2):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        recovered.extend(_read_shard(shard))

    def _ord(x):
        return (x["k"], repr(x["v"]))

    assert sorted(recovered, key=_ord) == sorted(items, key=_ord)


# ---------------------------------------------------------------------------
# Byte-budget flushing
# ---------------------------------------------------------------------------


def test_scatter_byte_budget_flushes_mid_write(tmp_path):
    """A tiny byte budget forces flushes during write, not only at close."""
    num_shards = 2
    items = [{"k": i % num_shards, "v": i} for i in range(200)]
    data_path = str(tmp_path / "shard-0000.shuffle")

    # Budget of 1 byte forces a flush on every batch after the first.
    writer = ScatterWriter(
        data_path=data_path,
        key_fn=_key,
        num_output_shards=num_shards,
        buffer_limit_bytes=1,
    )
    batch_size = 10
    for i in range(0, len(items), batch_size):
        batch = _items_to_dataframe(items[i : i + batch_size], _key, None, num_shards)
        writer.write(batch)
    writer.close()

    # Multiple chunks must have been written (not just the close-time flush).
    scatter_paths = [data_path]
    total_chunks = sum(ScatterReader.from_sidecars(scatter_paths, s).total_chunks for s in range(num_shards))
    assert total_chunks > 2, f"expected >2 chunks with 1-byte budget, got {total_chunks}"


def test_scatter_estimate_tracks_skewed_items(tmp_path):
    """Write-time EMA sampling catches large late items and triggers mid-write flushes."""
    num_shards = 1
    data_path = str(tmp_path / "shard-0000.shuffle")

    # Start with tiny items, then switch to large items. With a frozen estimate
    # the budget check would never fire for the large items. With EMA updates it
    # should: _item_bytes_estimate rises and eventually exceeds budget / rows.
    small_items = [{"k": 0, "v": "x"} for _ in range(50)]
    large_items = [{"k": 0, "v": "y" * 50_000} for _ in range(10)]

    # Budget large enough that small items alone never flush, but one large
    # item should push the estimate over threshold quickly.
    budget = 10_000  # 10 KB — well under 10 * 50 KB large items
    writer = ScatterWriter(
        data_path=data_path,
        key_fn=_key,
        num_output_shards=num_shards,
        buffer_limit_bytes=budget,
    )
    # Write small items as one batch, large items one-at-a-time so EMA adapts
    # after each large item (same closed-loop behavior as the old per-item path).
    writer.write(_items_to_dataframe(small_items, _key, None, num_shards))
    for item in large_items:
        writer.write(_items_to_dataframe([item], _key, None, num_shards))
    writer.close()

    # All items must survive the skewed flush pattern.
    scatter_paths = [data_path]
    recovered = _read_shard(ScatterReader.from_sidecars(scatter_paths, 0))
    all_items = small_items + large_items
    assert sorted(recovered, key=lambda x: x["v"]) == sorted(all_items, key=lambda x: x["v"])

    # The estimate must have been updated: mid-write flushes should have fired
    # for the large items (not just at close).
    assert writer._mid_write_flushes > 0, "expected mid-write flushes for large items"


def test_scatter_estimate_adapts_to_gradual_drift(tmp_path):
    """Write-time EMA bounds peak buffered rows even when item sizes grow gradually."""
    num_shards = 1
    data_path = str(tmp_path / "shard-0000.shuffle")

    # Items grow linearly from ~100 B to ~100 KB across 200 records.
    # If all 200 were buffered at once the real RSS would be ~10 MB.
    n_items = 200
    items = [{"k": 0, "v": "x" * (100 + i * 500)} for i in range(n_items)]

    # 500 KB budget. With a frozen first-batch estimate the budget check would
    # let all items accumulate. With EMA adaptation across batches the estimate
    # tracks the growing sizes and flushes before peak RSS reaches the budget.
    budget = 500_000
    writer = ScatterWriter(
        data_path=data_path,
        key_fn=_key,
        num_output_shards=num_shards,
        buffer_limit_bytes=budget,
    )
    # Write in small batches so EMA has chances to adapt as item sizes grow.
    batch_size = 10
    for i in range(0, n_items, batch_size):
        batch = _items_to_dataframe(items[i : i + batch_size], _key, None, num_shards)
        writer.write(batch)
    writer.close()

    scatter_paths = [data_path]
    recovered = _read_shard(ScatterReader.from_sidecars(scatter_paths, 0))
    assert sorted(recovered, key=lambda x: x["v"]) == sorted(items, key=lambda x: x["v"])

    assert writer._mid_write_flushes > 0, "expected mid-write flushes as item sizes grew"
    assert writer._peak_buffer_rows < n_items, (
        f"peak_buffer_rows={writer._peak_buffer_rows} should be < {n_items}; "
        "a frozen estimate lets all items accumulate before close()"
    )


def test_scatter_byte_budget_preserves_all_items(tmp_path):
    """Items are not lost or duplicated when byte-budget flushes fire mid-write."""
    num_shards = 3
    items = [{"k": i % num_shards, "v": i} for i in range(300)]
    scatter_paths = _build_shard(
        tmp_path,
        items,
        num_output_shards=num_shards,
    )

    recovered = []
    for shard_idx in range(num_shards):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        recovered.extend(_read_shard(shard))

    assert sorted(recovered, key=lambda x: x["v"]) == sorted(items, key=lambda x: x["v"])


# ---------------------------------------------------------------------------
# external_sort_merge
# ---------------------------------------------------------------------------


def _make_sorted_frame(values: list[int]) -> pl.LazyFrame:
    """Build a sorted LazyFrame by _SORT_KEY_COL for use in external sort tests."""
    return pl.DataFrame(
        {"v": values, _SORT_KEY_COL: [OrderedDict([("key", v), ("sort_value", None)]) for v in values]}
    ).lazy()


def test_external_sort_merge_streaming(tmp_path):
    frames = [_make_sorted_frame([1, 4, 7]), _make_sorted_frame([2, 5, 8]), _make_sorted_frame([3, 6, 9])]
    rows = list(external_sort_merge(iter(frames), sort_key=_SORT_KEY_COL, external_sort_dir=str(tmp_path)))
    result = [row["v"] for row in rows]
    assert result == list(range(1, 10))


def test_external_sort_merge_single_batch(tmp_path):
    frames = [_make_sorted_frame([i]) for i in range(10)]
    rows = list(external_sort_merge(iter(frames), sort_key=_SORT_KEY_COL, external_sort_dir=str(tmp_path)))
    result = [row["v"] for row in rows]
    assert result == list(range(10))


def test_external_sort_merge_cleans_up(tmp_path):
    frames = [_make_sorted_frame([i]) for i in range(EXTERNAL_SORT_FAN_IN + 1)]
    list(external_sort_merge(iter(frames), sort_key=_SORT_KEY_COL, external_sort_dir=str(tmp_path)))
    assert list(tmp_path.iterdir()) == [], "run files should be deleted after merge"


def test_external_sort_multi_chunk_source_shard(tmp_path):
    """external_sort_merge merges interleaved chunks from the same source shard correctly.
    """
    data_path = str(tmp_path / "shard-0000.shuffle")
    # buffer_limit_bytes=1 forces a flush after every write() call.
    writer = ScatterWriter(
        data_path=data_path,
        key_fn=_key,
        num_output_shards=1,
        buffer_limit_bytes=1,
    )
    # Flush 1 → chunk-0 sorted: [key=1, key=3]
    writer.write(_items_to_dataframe([{"k": 3, "v": "a"}, {"k": 1, "v": "b"}], _key, None, 1))
    # Flush 2 → chunk-1 sorted: [key=2]
    # Interleaved ranges: bundling both into one scan_parquet gives [1, 3, 2], not [1, 2, 3].
    writer.write(_items_to_dataframe([{"k": 2, "v": "c"}], _key, None, 1))
    writer.close()

    shard = ScatterReader.from_sidecars([data_path], target_shard=0)
    assert shard.total_chunks >= 2, "test requires multiple chunks; flush did not fire"

    external_dir = tmp_path / "sort_work"
    external_dir.mkdir()

    rows = list(
        external_sort_merge(
            shard.get_iterators(),
            sort_key=_SORT_KEY_COL,
            external_sort_dir=str(external_dir),
        )
    )
    assert [r["k"] for r in rows] == [1, 2, 3]


def test_mixed_schema_corruption(tmp_path):
    """Verify that mixing Arrow-friendly and Arrow-hostile items in the same shard works."""
    data_path = str(tmp_path / "mixed.shuffle")

    def _key(x):
        return x["k"]

    num_shards = 1

    writer = ScatterWriter(data_path=data_path, key_fn=_key, num_output_shards=num_shards)

    friendly_items = [{"k": 1, "v": i} for i in range(5)]
    writer.write(_items_to_dataframe(friendly_items, _key, None, num_shards))

    hostile_items = [{"k": 1, "v": frozenset([1, 2, 3])}]
    writer.write(_items_to_dataframe(hostile_items, _key, None, num_shards))

    writer.close()

    reader = ScatterReader.from_sidecars([data_path], target_shard=0)
    frames = list(reader.get_iterators())
    combined = pl.concat([f.collect() for f in frames], how="diagonal_relaxed")
    recovered = list(_dataframe_to_items(combined))

    expected = friendly_items + hostile_items
    assert recovered == expected


def test_mixed_schema_with_combiner(tmp_path):
    """Verify that combiners work correctly with mixed-schema buffers."""
    data_path = str(tmp_path / "combined.shuffle")

    def _key(x):
        return x["k"]

    num_shards = 1

    def _combiner(key, items):
        return list(items)

    writer = ScatterWriter(data_path=data_path, key_fn=_key, num_output_shards=num_shards, combiner_fn=_combiner)

    friendly_items = [{"k": 1, "v": i} for i in range(5)]
    writer.write(_items_to_dataframe(friendly_items, _key, None, num_shards))

    hostile_items = [{"k": 1, "v": frozenset([1, 2, 3])}]
    writer.write(_items_to_dataframe(hostile_items, _key, None, num_shards))

    writer.close()

    reader = ScatterReader.from_sidecars([data_path], target_shard=0)
    frames = list(reader.get_iterators())
    combined = pl.concat([f.collect() for f in frames], how="diagonal_relaxed")
    recovered = list(_dataframe_to_items(combined))

    expected = friendly_items + hostile_items
    assert recovered == expected
