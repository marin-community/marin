# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for zephyr/shuffle.py.

Covers the scatter write/read roundtrip, per-shard stats, and external sort —
without spinning up a full coordinator.
"""

import itertools
import re
from collections import OrderedDict
from collections.abc import Iterator
from unittest.mock import patch

import cloudpickle
import polars as pl
import pytest
from iris.env_resources import TaskResources
from zephyr.external_sort import external_sort_merge
from zephyr.runners import _InProcessWorkerContext
from zephyr.shard_keys import deterministic_hash
from zephyr.shuffle import (
    _PAYLOAD_COL,
    _SORT_KEY_COL,
    ScatterReader,
    ScatterWriter,
    _dataframe_to_items,
    _items_to_dataframe,
    _write_scatter,
)
from zephyr.worker_context import _worker_ctx_var


@pytest.fixture(autouse=True)
def mock_worker_ctx():
    """Provide a dummy worker context so ScatterWriter can resolve num_workers."""
    ctx = _InProcessWorkerContext(chunk_prefix="test", execution_id="test", stage_name="test", num_workers=1)
    token = _worker_ctx_var.set(ctx)
    yield
    _worker_ctx_var.reset(token)


def _read_shard(shard: ScatterReader) -> list:
    frames = shard.get_frames()
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
    items = [{"k": i % 2, "v": i} for i in range(20)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=2)

    for shard_idx in range(2):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        for lf in shard.get_frames():
            chunk = list(_dataframe_to_items(lf.collect()))
            keys = [_key(x) for x in chunk]
            assert keys == sorted(keys), f"chunk for shard {shard_idx} not sorted"


def test_avg_item_bytes_written(tmp_path):
    items = [{"k": 0, "v": i} for i in range(20)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=1)
    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    assert shard.avg_item_bytes > 0


def test_merge_sorted_chunks_basic(tmp_path):
    """merge_sorted_chunks k-way merges all chunks into one globally sorted stream."""
    items = [
        {"k": "a", "v": 1},
        {"k": "b", "v": 2},
        {"k": "a", "v": 3},
        {"k": "b", "v": 4},
    ]
    # Force two chunks by writing twice
    data_path = str(tmp_path / "shard-0000/scatter/")
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0)
    writer.write(_items_to_dataframe(items[:2], _key, None, 1))
    writer.write(_items_to_dataframe(items[2:], _key, None, 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    merged = list(shard.merge_sorted_chunks(external_sort_dir=str(tmp_path)))

    assert [_key(item) for item in merged] == ["a", "a", "b", "b"]
    assert [item["v"] for item in merged] == [1, 3, 2, 4]


def test_merge_sorted_chunks_secondary_sort(tmp_path):
    """Secondary sort is encoded at write time; merge preserves total order."""
    items = [
        {"k": "a", "ts": 10, "v": 1},
        {"k": "a", "ts": 5, "v": 2},
    ]
    # Write as two separate chunks
    data_path = str(tmp_path / "shard-0000/scatter/")
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0)
    writer.write(_items_to_dataframe([items[0]], _key, lambda x: x["ts"], 1))
    writer.write(_items_to_dataframe([items[1]], _key, lambda x: x["ts"], 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    merged = list(shard.merge_sorted_chunks(external_sort_dir=str(tmp_path)))

    assert len(merged) == 2
    assert [item["v"] for item in merged] == [2, 1]  # ts=5 comes before ts=10


def test_scatter_with_combiner(tmp_path):
    """ScatterWriter applies combiner_fn during flushes."""
    items = [
        {"k": "a", "v": 1},
        {"k": "a", "v": 2},
    ]

    def sum_combiner(key, items):
        yield {"k": key, "v": sum(i["v"] for i in items)}

    data_path = str(tmp_path / "shard-0000/scatter/")
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0, combiner_fn=sum_combiner)
    writer.write(_items_to_dataframe(items, _key, None, 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    recovered = _read_shard(shard)
    assert len(recovered) == 1
    assert recovered[0] == {"k": "a", "v": 3}


def test_merge_sorted_chunks_external_trigger(tmp_path):
    """merge_sorted_chunks successfully spills to disk when budget is exceeded."""
    items = [{"k": i, "v": i} for i in range(10)]
    data_path = str(tmp_path / "shard-0000/scatter/")
    # Write many small chunks
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0)
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
        merged = list(shard.merge_sorted_chunks(external_sort_dir=str(external_dir)))

    assert len(merged) == 10
    assert [item["k"] for item in merged] == list(range(10))


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
    assert list(shard.merge_sorted_chunks(external_sort_dir=str(tmp_path))) == []


def test_scatter_key_fn_must_be_serializable(tmp_path):
    """key_fn must be serializable to the Polars sort-key column."""
    items = [
        {"tags": {1, 2}, "v": 0},
        {"tags": {3}, "v": 1},
    ]

    data_path = str(tmp_path / "shard-0000.shuffle")
    with pytest.raises(ValueError, match=re.escape("key_fn must return an Arrow-serializable object.")):
        list(
            _write_scatter(
                iter(items),
                source_shard=0,
                data_path=data_path,
                key_fn=lambda item: frozenset(item["tags"]),
                num_output_shards=2,
            )
        )


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
        {
            _PAYLOAD_COL: pl.Series([cloudpickle.dumps({"v": v}) for v in values], dtype=pl.Binary),
            _SORT_KEY_COL: [OrderedDict([("key", v), ("sort_value", None)]) for v in values],
        }
    ).lazy()


def _external_sort_items(
    batches: Iterator[pl.LazyFrame],
    *,
    sort_key: str,
    external_sort_dir: str,
    fan_in: int,
    shard: int,
) -> list:
    merged = external_sort_merge(
        batches,
        sort_key=sort_key,
        external_sort_dir=external_sort_dir,
        fan_in=fan_in,
        shard=shard,
    )
    return list(itertools.chain.from_iterable(map(_dataframe_to_items, merged)))


def test_external_sort_merge_streaming(tmp_path):
    frames = [_make_sorted_frame([1, 4, 7]), _make_sorted_frame([2, 5, 8]), _make_sorted_frame([3, 6, 9])]
    rows = _external_sort_items(
        frames,
        sort_key=_SORT_KEY_COL,
        external_sort_dir=str(tmp_path),
        fan_in=4,
        shard=0,
    )
    result = [row["v"] for row in rows]
    assert result == list(range(1, 10))


def test_external_sort_merge_single_batch(tmp_path):
    frames = [_make_sorted_frame([i]) for i in range(10)]
    rows = _external_sort_items(
        frames,
        sort_key=_SORT_KEY_COL,
        external_sort_dir=str(tmp_path),
        fan_in=10,
        shard=0,
    )
    result = [row["v"] for row in rows]
    assert result == list(range(10))


def test_external_sort_merge_cleans_up(tmp_path):
    fan_in = 4
    frames = [_make_sorted_frame([i]) for i in range(fan_in + 1)]
    list(
        external_sort_merge(
            frames,
            sort_key=_SORT_KEY_COL,
            external_sort_dir=str(tmp_path),
            fan_in=fan_in,
            shard=0,
        )
    )
    assert list(tmp_path.iterdir()) == [], "run files should be deleted after merge"


def test_external_sort_merge_across_source_shards(tmp_path):
    """external_sort_merge correctly merges interleaved keys from multiple source shards."""
    # Shard 0 writes keys [1, 3], shard 1 writes key [2].  The merge must produce [1, 2, 3].
    paths = []
    for shard_idx, items in [(0, [{"k": 3, "v": "a"}, {"k": 1, "v": "b"}]), (1, [{"k": 2, "v": "c"}])]:
        data_path = f"{tmp_path}/shard-{shard_idx:04d}/scatter/"
        paths.append(data_path)
        writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=shard_idx)
        writer.write(_items_to_dataframe(items, _key, None, 1))
        writer.close()

    shard = ScatterReader.from_sidecars(paths, target_shard=0)
    assert shard.total_chunks == 2, "expected one Parquet file per source shard"

    external_dir = tmp_path / "sort_work"
    external_dir.mkdir()

    rows = _external_sort_items(
        shard.get_frames(),
        sort_key=_SORT_KEY_COL,
        external_sort_dir=str(external_dir),
        fan_in=4,
        shard=0,
    )
    assert [r["k"] for r in rows] == [1, 2, 3]
