# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for zephyr/shuffle.py.

Covers the scatter write/read roundtrip, per-shard stats, and external sort —
without spinning up a full coordinator.
"""

from zephyr.plan import deterministic_hash
from zephyr.shuffle import (
    ScatterFileIterator,
    ScatterReader,
    _FRAME_FORMAT_ARROW,
    _FRAME_FORMAT_PICKLE,
    _write_chunk_frame,
    _write_scatter,
)


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
        recovered.extend(list(shard))

    assert sorted(recovered, key=lambda x: x["v"]) == sorted(items, key=lambda x: x["v"])


def test_scatter_each_shard_gets_correct_items(tmp_path):
    """Items are routed to shards by deterministic_hash(key) % num_shards."""
    num_shards = 4
    items = [{"k": i % 4, "v": i} for i in range(40)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=num_shards)

    for shard_idx in range(num_shards):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        recovered = sorted(list(shard), key=lambda x: x["v"])
        expected = sorted([x for x in items if _target(x["k"], num_shards) == shard_idx], key=lambda x: x["v"])
        assert recovered == expected, f"shard {shard_idx} mismatch"


def test_scatter_roundtrip_sorted_chunks(tmp_path):
    """Each chunk iterator from get_iterators() yields items sorted by key."""
    items = [{"k": i % 2, "v": i} for i in range(20)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=2)

    for shard_idx in range(2):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        for chunk_iter in shard.get_iterators():
            chunk = list(chunk_iter)
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


def test_needs_external_sort_triggers():
    shard = ScatterReader(
        iterators=[ScatterFileIterator(path="gs://fake/path.shuffle", chunks=tuple((i, 1) for i in range(1000)))],
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
    shard = ScatterReader(iterators=[], max_chunk_rows=100_000, avg_item_bytes=200.0)
    assert not shard.needs_external_sort(memory_limit=32 * 1024**3)


# ---------------------------------------------------------------------------
# avg_item_bytes
# ---------------------------------------------------------------------------


def test_avg_item_bytes_written(tmp_path):
    items = [{"k": 0, "v": i} for i in range(20)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=1)
    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    assert shard.avg_item_bytes > 0


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
        recovered.extend(list(shard))

    def _ord(x):
        return (x["k"], repr(x["v"]))

    assert sorted(recovered, key=_ord) == sorted(items, key=_ord)


# ---------------------------------------------------------------------------
# ScatterFileIterator low-level
# ---------------------------------------------------------------------------


def test_scatter_file_iterator_multiple_chunks(tmp_path):
    """Multiple frames in one file are read in declared order."""
    chunk_a = [{"i": i} for i in range(5)]
    chunk_b = [{"i": i} for i in range(100, 103)]

    frame_a = _write_chunk_frame(chunk_a)
    frame_b = _write_chunk_frame(chunk_b)

    path = str(tmp_path / "two-chunks.shuffle")
    with open(path, "wb") as f:
        f.write(frame_a)
        f.write(frame_b)

    it = ScatterFileIterator(path=path, chunks=((0, len(frame_a)), (len(frame_a), len(frame_b))))
    chunks = [list(c) for c in it.get_chunk_iterators()]
    assert chunks == [chunk_a, chunk_b]


# ---------------------------------------------------------------------------
# Frame format: Arrow IPC vs pickle fallback
# ---------------------------------------------------------------------------


def test_write_chunk_frame_uses_arrow_for_plain_dicts():
    """Plain dict items with primitive values use the Arrow IPC format tag."""
    items = [{"k": i, "v": float(i)} for i in range(10)]
    frame = _write_chunk_frame(items)
    assert frame[0:1] == _FRAME_FORMAT_ARROW, "expected Arrow IPC format tag for Arrow-compatible dicts"


def test_write_chunk_frame_falls_back_to_pickle_for_frozensets():
    """Items containing frozensets cannot be Arrow-encoded and use pickle."""
    items = [{"k": 0, "v": frozenset([1, 2, 3])}]
    frame = _write_chunk_frame(items)
    assert frame[0:1] == _FRAME_FORMAT_PICKLE, "expected pickle format tag for frozenset values"


def test_arrow_roundtrip_end_to_end(tmp_path):
    """Items written via the Arrow path round-trip correctly through scatter."""
    items = [{"k": i % 3, "v": i, "label": f"item-{i}"} for i in range(30)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=3)

    recovered = []
    for shard_idx in range(3):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        recovered.extend(list(shard))

    assert sorted(recovered, key=lambda x: x["v"]) == sorted(items, key=lambda x: x["v"])


# ---------------------------------------------------------------------------
# external_sort_merge
# ---------------------------------------------------------------------------


def test_external_sort_merge_streaming(tmp_path):
    from zephyr.external_sort import external_sort_merge

    iters = [iter([1, 4, 7]), iter([2, 5, 8]), iter([3, 6, 9])]
    result = list(external_sort_merge(iter(iters), merge_key=lambda x: x, external_sort_dir=str(tmp_path)))
    assert result == list(range(1, 10))


def test_external_sort_merge_single_batch(tmp_path):
    from zephyr.external_sort import external_sort_merge

    iters = [iter([i]) for i in range(10)]
    result = list(external_sort_merge(iter(iters), merge_key=lambda x: x, external_sort_dir=str(tmp_path)))
    assert result == list(range(10))


def test_external_sort_merge_cleans_up(tmp_path):
    from zephyr.external_sort import EXTERNAL_SORT_FAN_IN, external_sort_merge

    iters = [iter([i]) for i in range(EXTERNAL_SORT_FAN_IN + 1)]
    list(external_sort_merge(iter(iters), merge_key=lambda x: x, external_sort_dir=str(tmp_path)))
    assert list(tmp_path.iterdir()) == [], "run files should be deleted after merge"
