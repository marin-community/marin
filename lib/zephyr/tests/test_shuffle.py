# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for zephyr/shuffle.py.

Covers the scatter write/read roundtrip, per-shard stats, and external sort —
without spinning up a full coordinator.
"""

import io
import itertools
import os
from collections import OrderedDict
from unittest.mock import patch

import cloudpickle
import fsspec
import polars as pl
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.local import LocalFileSystem
from iris.env_resources import TaskResources
from rigging.filesystem.storage_path import StoragePath
from zephyr import memory_budget
from zephyr.runners import _InProcessWorkerContext
from zephyr.shard_keys import deterministic_hash
from zephyr.shuffle import (
    _PAYLOAD_COL,
    _SCATTER_MAX_ROW_GROUPS_PER_CHUNK,
    _SHARD_COL,
    _SORT_KEY_COL,
    ScatterReader,
    ScatterWriter,
    _dataframe_to_items,
    _items_to_dataframe,
    _merge_sorted_frames,
    _read_sidecar_slices_parallel,
    _write_scatter,
)
from zephyr.worker_context import _worker_ctx_var


@pytest.fixture(autouse=True)
def mock_worker_ctx():
    """Provide a dummy worker context for ScatterWriter."""
    ctx = _InProcessWorkerContext(chunk_prefix="test", execution_id="test", stage_name="test")
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


def test_scatter_reader_uses_virtual_hosted_coreweave_endpoint(monkeypatch):
    path = "s3://marin-us-east-02a/execution/stage0/shard-0000/scatter/c0000.parquet"
    calls = []
    frame = pl.DataFrame(
        {
            _PAYLOAD_COL: [cloudpickle.dumps({"k": "a"})],
            _SHARD_COL: [0],
            _SORT_KEY_COL: [OrderedDict([("key", b"a"), ("sort_value", None)])],
        }
    ).lazy()

    def scan_parquet(scan_path, *, storage_options):
        calls.append((scan_path, storage_options))
        return frame

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://cwlota.com")
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.setattr(pl, "scan_parquet", scan_parquet)

    reader = ScatterReader(files=[("source", [path])], target_shard=0, avg_item_bytes=1.0)
    rows = reader.get_frames()[0].collect().to_dicts()

    assert len(rows) == 1
    assert calls == [
        (
            path,
            {
                "aws_endpoint_url": "http://marin-us-east-02a.cwlota.com",
                "aws_virtual_hosted_style_request": "true",
            },
        )
    ]


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
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0, sort_fn=lambda x: x["ts"])
    writer.write(_items_to_dataframe([items[0]], _key, lambda x: x["ts"], 1))
    writer.write(_items_to_dataframe([items[1]], _key, lambda x: x["ts"], 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    merged = list(shard.merge_sorted_chunks(external_sort_dir=str(tmp_path)))

    assert len(merged) == 2
    assert [item["v"] for item in merged] == [2, 1]  # ts=5 comes before ts=10


def test_scatter_sort_fn_null_batch_then_concrete(tmp_path):
    """Flush must handle a batch with all-None sort_values followed by one with ints.

    When sort_fn returns None for every item in one batch, Polars infers Null dtype
    for the sort_value field of _SORT_KEY_COL.  The next batch with concrete int
    values produces Int64.  pl.concat in _flush must use vertical_relaxed, or it
    raises a SchemaError on the struct field type mismatch.
    """

    def sort_fn(x):
        return x.get("priority")  # returns None when key absent

    data_path = str(tmp_path / "shard-0000/scatter/")
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0, sort_fn=sort_fn)

    # All-None sort values → _SORT_KEY_COL.sort_value: Null
    frame_null = _items_to_dataframe([{"k": "a", "v": 0}, {"k": "a", "v": 1}], _key, sort_fn, 1)
    # Concrete int sort values → _SORT_KEY_COL.sort_value: Int64
    frame_concrete = _items_to_dataframe(
        [{"k": "a", "v": 2, "priority": 5}, {"k": "a", "v": 3, "priority": 3}], _key, sort_fn, 1
    )

    writer.write(frame_null)
    writer.write(frame_concrete)
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    recovered = _read_shard(shard)
    assert sorted(x["v"] for x in recovered) == [0, 1, 2, 3]


def test_merge_sorted_chunks_cross_shard_null_sort_value(tmp_path):
    """merge_sorted_chunks must handle source shards with differing sort_value dtypes.

    Source shard 0 writes items where sort_fn returns None for every item, so its
    parquet file stores _SORT_KEY_COL.sort_value as Null dtype.  Source shard 1
    writes items where sort_fn returns ints, so its file stores Int64.  When the
    reducer calls merge_sorted_chunks across both files, pl.merge_sorted sees
    incompatible struct schemas and raises SchemaError.
    """

    def sort_fn(x):
        return x.get("priority")

    data_path_0 = str(tmp_path / "shard-0000/scatter/")
    writer_0 = ScatterWriter(data_path=data_path_0, key_fn=_key, source_shard=0, sort_fn=sort_fn)
    writer_0.write(_items_to_dataframe([{"k": "a", "v": 0}, {"k": "a", "v": 1}], _key, sort_fn, 1))
    paths_0 = list(writer_0.close())

    data_path_1 = str(tmp_path / "shard-0001/scatter/")
    writer_1 = ScatterWriter(data_path=data_path_1, key_fn=_key, source_shard=1, sort_fn=sort_fn)
    writer_1.write(
        _items_to_dataframe([{"k": "a", "v": 2, "priority": 5}, {"k": "a", "v": 3, "priority": 3}], _key, sort_fn, 1)
    )
    paths_1 = list(writer_1.close())

    shard = ScatterReader.from_sidecars(paths_0 + paths_1, target_shard=0)
    # Currently raises SchemaError: struct field sort_value is Null in shard 0's
    # file but Int64 in shard 1's file; pl.merge_sorted requires identical schemas.
    merged = list(shard.merge_sorted_chunks(external_sort_dir=str(tmp_path / "sort")))
    assert sorted(x["v"] for x in merged) == [0, 1, 2, 3]


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
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0)
    writer._flush_threshold_bytes = 0
    for i in range(10):
        writer.write(_items_to_dataframe([items[i]], _key, None, 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)

    external_dir = tmp_path / "sort_work"
    external_dir.mkdir()

    with patch("zephyr.shuffle.memory_budget.read_merge_fan_in", return_value=2):
        merged = list(shard.merge_sorted_chunks(external_sort_dir=str(external_dir)))

    assert len(merged) == 10
    assert [item["k"] for item in merged] == list(range(10))


def test_merge_sorted_chunks_skips_empty_target_shard(tmp_path):
    key = "only-key"
    populated_shard = _target(key, 2)
    empty_shard = 1 - populated_shard
    scatter_paths = _build_shard(tmp_path, [{"k": key, "v": 1}], num_output_shards=2)
    reader = ScatterReader.from_sidecars(scatter_paths, empty_shard)

    assert reader.total_chunks > 0
    assert reader.shard_payload_bytes == 0
    with patch(
        "zephyr.shuffle.memory_budget.read_merge_fan_in",
        side_effect=AssertionError("empty target shards do not need memory planning"),
    ):
        assert list(reader.merge_sorted_chunks(external_sort_dir=str(tmp_path / "sort"))) == []


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
    """key_fn must return a msgpack-serializable value."""

    class _Unserializable:
        pass

    items = [{"v": 0}, {"v": 1}]
    data_path = str(tmp_path / "shard-0000.shuffle")
    with pytest.raises(ValueError, match="key_fn must return a msgpack-serializable object"):
        list(
            _write_scatter(
                iter(items),
                source_shard=0,
                data_path=data_path,
                key_fn=lambda item: _Unserializable(),
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


def test_scatter_bounds_parquet_row_groups(tmp_path):
    num_targets = 1024
    data_path = f"{tmp_path}/shard-0000/scatter/"
    frame = pl.DataFrame(
        {
            _PAYLOAD_COL: pl.Series(
                [cloudpickle.dumps({"k": target}) for target in range(num_targets)],
                dtype=pl.Binary,
            ),
            _SHARD_COL: pl.Series(range(num_targets), dtype=pl.Int32),
            _SORT_KEY_COL: [
                OrderedDict([("key", target.to_bytes(4, "big")), ("sort_value", None)]) for target in range(num_targets)
            ],
        }
    )
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0)
    writer.write(frame)
    writer.close()

    parquet = pq.ParquetFile(f"{data_path}c0000.parquet")
    assert parquet.metadata.num_row_groups <= _SCATTER_MAX_ROW_GROUPS_PER_CHUNK

    reader = ScatterReader(files=[(data_path, [f"{data_path}c0000.parquet"])], target_shard=513, avg_item_bytes=1)
    assert _read_shard(reader) == [{"k": 513}]


def test_scatter_auto_flush_uses_task_memory_budget(tmp_path):
    """A multiplexed shard flushes against its task budget, not the worker pod limit."""
    items = [{"k": 0, "v": i} for i in range(100)]
    frame = _items_to_dataframe(items, _key, None, 1)
    frame_bytes = int(frame.estimated_size())
    task_memory_bytes = int(
        (memory_budget.FIXED_OVERHEAD_WRITE_BYTES + 0.5 * memory_budget.R_WRITE * frame_bytes)
        / memory_budget.SAFETY_FRACTION_WRITE
    )
    ctx = _InProcessWorkerContext(
        chunk_prefix="test",
        execution_id="test",
        stage_name="test",
        task_memory_bytes=task_memory_bytes,
    )
    token = _worker_ctx_var.set(ctx)
    try:
        with (
            patch("zephyr.shuffle.TaskResources.from_environment") as environment_resources,
            patch("zephyr.shuffle._process_rss_bytes", return_value=0),
        ):
            environment_resources.return_value = TaskResources(
                memory_bytes=1024**3,
                cpu_cores=1,
                gpu_count=0,
                tpu_count=0,
            )
            writer = ScatterWriter(data_path=str(tmp_path / "scatter"), key_fn=_key, source_shard=0)
            writer.write(frame)
            writer.write(frame)
            scatter_paths = list(writer.close())
    finally:
        _worker_ctx_var.reset(token)

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    assert shard.total_chunks == 2
    assert sorted(row["v"] for row in _read_shard(shard)) == sorted([*range(100), *range(100)])


def test_read_merge_fan_in_accounts_for_process_baseline():
    inputs = {
        "avg_item_bytes": 200,
        "total_chunks": 10_000,
        "shard_payload_bytes": 512 * 2**20,
        "polars_threads": 8,
    }
    low_baseline_fan_in = memory_budget.read_merge_fan_in(4 * 2**30, 128 * 2**20, **inputs)
    high_baseline_fan_in = memory_budget.read_merge_fan_in(4 * 2**30, 1 * 2**30, **inputs)

    assert high_baseline_fan_in < low_baseline_fan_in


def test_read_merge_fan_in_accounts_for_thread_shards_and_total_payload():
    inputs = {
        "task_memory_bytes": 8 * 2**30,
        "baseline_rss_bytes": 256 * 2**20,
        "avg_item_bytes": 1_000,
        "total_chunks": 100,
    }

    low_thread_fan_in = memory_budget.read_merge_fan_in(
        **inputs,
        shard_payload_bytes=2 * 2**30,
        polars_threads=2,
    )
    high_thread_fan_in = memory_budget.read_merge_fan_in(
        **inputs,
        shard_payload_bytes=2 * 2**30,
        polars_threads=30,
    )
    high_payload_fan_in = memory_budget.read_merge_fan_in(
        **inputs,
        shard_payload_bytes=4 * 2**30,
        polars_threads=2,
    )

    assert high_thread_fan_in < low_thread_fan_in
    assert high_payload_fan_in < low_thread_fan_in


def test_read_merge_fan_in_saturates_at_minimum():
    fan_in = memory_budget.read_merge_fan_in(
        2 * 2**30,
        512 * 2**20,
        avg_item_bytes=64 * 2**10,
        total_chunks=2,
        shard_payload_bytes=1 * 2**30,
        polars_threads=2,
    )

    assert fan_in == memory_budget.MIN_MERGE_FAN_IN


def test_read_merge_fan_in_rejects_direct_merge_for_incident_shape():
    fan_in = memory_budget.read_merge_fan_in(
        16 * 2**30,
        350 * 2**20,
        avg_item_bytes=175.4,
        total_chunks=191,
        shard_payload_bytes=4.41 * 2**30,
        polars_threads=2,
    )

    assert fan_in < 191


def test_read_merge_fan_in_preserves_direct_merge_when_streaming_batch_covers_payload():
    fan_in = memory_budget.read_merge_fan_in(
        60 * 2**30,
        200 * 2**20,
        avg_item_bytes=356_637,
        total_chunks=66,
        shard_payload_bytes=10.6 * 2**30,
        polars_threads=30,
    )

    assert fan_in == 66


# ---------------------------------------------------------------------------
# _merge_sorted_frames
# ---------------------------------------------------------------------------


def _make_sorted_frame(values: list[int]) -> pl.LazyFrame:
    """Build a sorted LazyFrame by _SORT_KEY_COL for use in external sort tests."""
    return pl.DataFrame(
        {
            _PAYLOAD_COL: pl.Series([cloudpickle.dumps({"v": v}) for v in values], dtype=pl.Binary),
            _SORT_KEY_COL: [OrderedDict([("key", v), ("sort_value", None)]) for v in values],
        }
    ).lazy()


def _merge_sorted_frames_items(
    frames: list[pl.LazyFrame],
    *,
    sort_key: str,
    external_sort_dir: str,
    fan_in: int,
    shard: int,
) -> list:
    merged = _merge_sorted_frames(
        frames,
        sort_key=sort_key,
        external_sort_dir=external_sort_dir,
        fan_in=fan_in,
        shard=shard,
    )
    return list(itertools.chain.from_iterable(map(_dataframe_to_items, merged)))


def test_merge_sorted_frames_streaming(tmp_path):
    frames = [_make_sorted_frame([1, 4, 7]), _make_sorted_frame([2, 5, 8]), _make_sorted_frame([3, 6, 9])]
    rows = _merge_sorted_frames_items(
        frames,
        sort_key=_SORT_KEY_COL,
        external_sort_dir=str(tmp_path),
        fan_in=4,
        shard=0,
    )
    result = [row["v"] for row in rows]
    assert result == list(range(1, 10))


def test_merge_sorted_frames_single_batch(tmp_path):
    frames = [_make_sorted_frame([i]) for i in range(10)]
    rows = _merge_sorted_frames_items(
        frames,
        sort_key=_SORT_KEY_COL,
        external_sort_dir=str(tmp_path),
        fan_in=10,
        shard=0,
    )
    result = [row["v"] for row in rows]
    assert result == list(range(10))


def test_merge_sorted_frames_cleans_up(tmp_path):
    fan_in = 4
    frames = [_make_sorted_frame([i]) for i in range(fan_in + 1)]
    list(
        _merge_sorted_frames(
            frames,
            sort_key=_SORT_KEY_COL,
            external_sort_dir=str(tmp_path),
            fan_in=fan_in,
            shard=0,
        )
    )
    assert list(tmp_path.iterdir()) == [], "run files should be deleted after merge"


def test_merge_sorted_frames_limits_every_pass_fan_in(tmp_path, monkeypatch):
    """No single merge_sorted call combines more than fan_in frames, across every pass."""
    fan_in = 3
    frames = [_make_sorted_frame([value]) for value in range(20)]

    call_sizes: list[int] = []
    real_merge_sorted = pl.merge_sorted

    def spy_merge_sorted(inputs, *args, **kwargs):
        call_sizes.append(len(inputs))
        return real_merge_sorted(inputs, *args, **kwargs)

    monkeypatch.setattr(pl, "merge_sorted", spy_merge_sorted)

    rows = _merge_sorted_frames_items(
        frames,
        sort_key=_SORT_KEY_COL,
        external_sort_dir=str(tmp_path),
        fan_in=fan_in,
        shard=0,
    )

    assert [row["v"] for row in rows] == list(range(20))
    assert list(tmp_path.iterdir()) == []
    assert len(call_sizes) > 1, f"expected multiple merge_sorted calls for 20 frames at fan_in={fan_in}"
    assert all(size <= fan_in for size in call_sizes), f"a merge_sorted call exceeded fan_in={fan_in}: {call_sizes}"


def test_merge_sorted_frames_reads_coreweave_spills_with_virtual_host_addressing(monkeypatch):
    frames = [_make_sorted_frame([value]) for value in range(5)]
    spill_bytes: dict[str, bytes] = {}
    scan_calls: list[tuple[str, dict[str, str]]] = []
    real_scan_parquet = pl.scan_parquet

    class SpillBuffer(io.BytesIO):
        def __init__(self, path: StoragePath):
            super().__init__()
            self.path = str(path)

        def __exit__(self, *_exc: object) -> None:
            spill_bytes[self.path] = self.getvalue()
            self.close()

    def open_spill(path: StoragePath, mode: str = "rb", **_kwargs) -> SpillBuffer:
        assert mode == "wb"
        return SpillBuffer(path)

    def scan_parquet(path: str, *, storage_options: dict[str, str]) -> pl.LazyFrame:
        scan_calls.append((path, storage_options))
        return real_scan_parquet(io.BytesIO(spill_bytes[path]))

    def remove_spill(path: StoragePath) -> None:
        spill_bytes.pop(str(path))

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://cwlota.com")
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.setattr(StoragePath, "mkdirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(StoragePath, "open", open_spill)
    monkeypatch.setattr(StoragePath, "rm", remove_spill)
    monkeypatch.setattr(pl, "scan_parquet", scan_parquet)

    rows = _merge_sorted_frames_items(
        frames,
        sort_key=_SORT_KEY_COL,
        external_sort_dir="s3://marin-us-east-02a/calibration/spill",
        fan_in=2,
        shard=0,
    )

    assert [row["v"] for row in rows] == list(range(5))
    assert spill_bytes == {}
    assert scan_calls
    assert all(
        storage_options
        == {
            "aws_endpoint_url": "http://marin-us-east-02a.cwlota.com",
            "aws_virtual_hosted_style_request": "true",
        }
        for _, storage_options in scan_calls
    )


def test_scatter_removes_partial_dir_on_write_failure(tmp_path):
    """A local shard whose item stream fails mid-write leaves no partial directory."""
    data_path = f"{tmp_path}/shard-0000/scatter/"

    def failing_items():
        yield {"k": "a"}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _write_scatter(failing_items(), source_shard=0, data_path=data_path, key_fn=_key, num_output_shards=2)

    assert not os.path.exists(data_path), "failed shard left a partial scatter directory"


def test_merge_sorted_frames_across_source_shards(tmp_path):
    """_merge_sorted_frames correctly merges interleaved keys from multiple source shards."""
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

    rows = _merge_sorted_frames_items(
        shard.get_frames(),
        sort_key=_SORT_KEY_COL,
        external_sort_dir=str(external_dir),
        fan_in=4,
        shard=0,
    )
    assert [r["k"] for r in rows] == [1, 2, 3]


class _CountingFileSystem(LocalFileSystem):
    """Counts how many filesystem instances a read builds."""

    protocol = "counting"
    clients_built = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).clients_built += 1

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(str(path).removeprefix("counting://"))


def test_sidecar_reads_build_one_client(tmp_path):
    """Reading many sidecars concurrently builds one client (#8402).

    fsspec keys its instance cache on the calling thread, so resolving inside
    a pool worker builds a client, and a connection pool, per thread.
    """
    fsspec.register_implementation("counting", _CountingFileSystem, clobber=True)
    paths = []
    for shard_idx in range(8):
        data_path = f"{tmp_path}/shard-{shard_idx:04d}/scatter/"
        writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=shard_idx)
        writer.write(_items_to_dataframe([{"k": shard_idx}], _key, None, 1))
        writer.close()
        paths.append(f"counting://{data_path}")

    _CountingFileSystem.clear_instance_cache()
    _CountingFileSystem.clients_built = 0

    slices = _read_sidecar_slices_parallel(paths, target_shard=0)

    assert [s.path for s in slices] == paths
    assert _CountingFileSystem.clients_built == 1
