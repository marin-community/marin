# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for zephyr/shuffle.py.

Covers the scatter write/read roundtrip, per-shard stats, and external sort —
without spinning up a full coordinator.
"""

import os
import tempfile
from unittest.mock import patch
from urllib.parse import urlparse

import cloudpickle
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datafusion import col
from datafusion.object_store import LocalFileSystem as DataFusionLocalFileSystem
from fsspec.implementations.local import LocalFileSystem
from iris.env_resources import TaskResources
from zephyr.parquet_scan import datafusion_context, register_object_stores
from zephyr.runners import _InProcessWorkerContext
from zephyr.shard_keys import deterministic_hash
from zephyr.shuffle import (
    _PAYLOAD_COL,
    _SCATTER_MAX_ROW_GROUPS_PER_CHUNK,
    _SHARD_COL,
    _SORT_KEY_COL,
    ScatterReader,
    ScatterWriter,
    _items_to_table,
    _merge_sorted_frames,
    _Sidecar,
    _table_to_items,
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
    with tempfile.TemporaryDirectory(prefix="zephyr-test-sort-") as sort_dir:
        return list(shard.merge_sorted_chunks(sort_dir))


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
    stores = []

    class FakeS3:
        def __init__(self, **kwargs):
            stores.append((kwargs, os.environ.get("AWS_VIRTUAL_HOSTED_STYLE_REQUEST")))

    class FakeContext:
        def __init__(self):
            self.registrations = []

        def register_object_store(self, scheme, store, host=None):
            self.registrations.append((scheme, store, host))

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://cwlota.com")
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.setattr("zephyr.parquet_scan.AmazonS3", FakeS3)

    context = FakeContext()
    register_object_stores(context, [path])

    assert stores == [
        (
            {
                "bucket_name": "marin-us-east-02a",
                "endpoint": "http://marin-us-east-02a.cwlota.com",
                "allow_http": True,
            },
            "true",
        )
    ]
    assert context.registrations[0][0] == "s3://"
    assert context.registrations[0][2] == "marin-us-east-02a"
    assert "AWS_VIRTUAL_HOSTED_STYLE_REQUEST" not in os.environ


def test_datafusion_context_disables_native_disk_spill(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(pa, "cpu_count", lambda: 1)
    context = datafusion_context(memory_limit_bytes=1_000_000)
    batches = [pa.record_batch({"value": range(start + 10_000, start, -1)}) for start in range(0, 1_000_000, 10_000)]
    frame = context.create_dataframe([batches]).sort(col("value").sort())

    error = None
    try:
        frame.collect()
    except Exception as exc:
        error = exc

    assert error is not None
    assert list(tmp_path.iterdir()) == []


def test_datafusion_context_rejects_nonpositive_target_partitions():
    with pytest.raises(ValueError, match="target_partitions must be positive"):
        datafusion_context(target_partitions=0)


def test_scatter_roundtrip_sorted_chunks(tmp_path):
    items = [{"k": i % 2, "v": i} for i in range(20)]
    scatter_paths = _build_shard(tmp_path, items, num_output_shards=2)

    for shard_idx in range(2):
        shard = ScatterReader.from_sidecars(scatter_paths, shard_idx)
        context = datafusion_context()
        for frame in shard.get_frames(context):
            chunk = list(_table_to_items(frame.to_arrow_table()))
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
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0, num_output_shards=1)
    writer.write(_items_to_table(items[:2], _key, None, 1))
    writer.write(_items_to_table(items[2:], _key, None, 1))
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
    writer = ScatterWriter(
        data_path=data_path, key_fn=_key, source_shard=0, num_output_shards=1, sort_fn=lambda x: x["ts"]
    )
    writer.write(_items_to_table([items[0]], _key, lambda x: x["ts"], 1))
    writer.write(_items_to_table([items[1]], _key, lambda x: x["ts"], 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    merged = list(shard.merge_sorted_chunks(external_sort_dir=str(tmp_path)))

    assert len(merged) == 2
    assert [item["v"] for item in merged] == [2, 1]  # ts=5 comes before ts=10


def test_scatter_sort_fn_null_batch_then_concrete(tmp_path):
    """Flush must handle a batch with all-None sort_values followed by one with ints.

    When sort_fn returns None for every item in one batch, Arrow infers Null dtype
    for the sort_value field of _SORT_KEY_COL.  The next batch with concrete int
    values produces Int64. The flush must promote the nested field before sorting.
    """

    def sort_fn(x):
        return x.get("priority")  # returns None when key absent

    data_path = str(tmp_path / "shard-0000/scatter/")
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0, num_output_shards=1, sort_fn=sort_fn)

    # All-None sort values → _SORT_KEY_COL.sort_value: Null
    table_null = _items_to_table([{"k": "a", "v": 0}, {"k": "a", "v": 1}], _key, sort_fn, 1)
    # Concrete int sort values → _SORT_KEY_COL.sort_value: Int64
    table_concrete = _items_to_table(
        [{"k": "a", "v": 2, "priority": 5}, {"k": "a", "v": 3, "priority": 3}], _key, sort_fn, 1
    )

    writer.write(table_null)
    writer.write(table_concrete)
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    recovered = _read_shard(shard)
    assert sorted(x["v"] for x in recovered) == [0, 1, 2, 3]


def test_merge_sorted_chunks_cross_shard_null_sort_value(tmp_path):
    """merge_sorted_chunks must handle source shards with differing sort_value dtypes.

    Source shard 0 writes items where sort_fn returns None for every item, so its
    parquet file stores _SORT_KEY_COL.sort_value as Null dtype.  Source shard 1
    writes items where sort_fn returns ints, so its file stores Int64. The
    reducer must unify those schemas before DataFusion merges them.
    """

    def sort_fn(x):
        return x.get("priority")

    data_path_0 = str(tmp_path / "shard-0000/scatter/")
    writer_0 = ScatterWriter(data_path=data_path_0, key_fn=_key, source_shard=0, num_output_shards=1, sort_fn=sort_fn)
    writer_0.write(_items_to_table([{"k": "a", "v": 0}, {"k": "a", "v": 1}], _key, sort_fn, 1))
    paths_0 = list(writer_0.close())

    data_path_1 = str(tmp_path / "shard-0001/scatter/")
    writer_1 = ScatterWriter(data_path=data_path_1, key_fn=_key, source_shard=1, num_output_shards=1, sort_fn=sort_fn)
    writer_1.write(
        _items_to_table([{"k": "a", "v": 2, "priority": 5}, {"k": "a", "v": 3, "priority": 3}], _key, sort_fn, 1)
    )
    paths_1 = list(writer_1.close())

    shard = ScatterReader.from_sidecars(paths_0 + paths_1, target_shard=0)
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
    writer = ScatterWriter(
        data_path=data_path, key_fn=_key, source_shard=0, num_output_shards=1, combiner_fn=sum_combiner
    )
    writer.write(_items_to_table(items, _key, None, 1))
    scatter_paths = list(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    recovered = _read_shard(shard)
    assert len(recovered) == 1
    assert recovered[0] == {"k": "a", "v": 3}


def test_merge_sorted_chunks_external_runs_use_gcs_urls(tmp_path, monkeypatch):
    items = [{"k": i, "v": i} for i in range(10)]
    scatter_paths = []
    for i in range(10):
        data_path = str(tmp_path / f"shard-{i:04d}/scatter/")
        writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=i, num_output_shards=1)
        writer.write(_items_to_table([items[i]], _key, None, 1))
        scatter_paths.extend(writer.close())

    shard = ScatterReader.from_sidecars(scatter_paths, 0)

    remote_root = tmp_path / "gcs"

    def remote_path(url: str):
        parsed = urlparse(url)
        assert (parsed.scheme, parsed.netloc) == ("gs", "bucket")
        return remote_root / parsed.path.lstrip("/")

    class FakeStoragePath:
        def __init__(self, url: str):
            self.url = url

        def __truediv__(self, name: str):
            return FakeStoragePath(f"{self.url.rstrip('/')}/{name}")

        def __str__(self) -> str:
            return self.url

        def mkdirs(self) -> None:
            remote_path(self.url).mkdir(parents=True, exist_ok=True)

        def rm(self) -> None:
            remote_path(self.url).unlink()

    monkeypatch.setattr("zephyr.shuffle.StoragePath", FakeStoragePath)
    monkeypatch.setattr(
        "zephyr.parquet_scan.GoogleCloud",
        lambda bucket_name: DataFusionLocalFileSystem(prefix=str(remote_root)),
    )

    # Force Zephyr's explicit external sort with a payload estimate above the merge budget.
    shard.shard_payload_bytes = 1_000_000

    with patch("iris.env_resources.TaskResources.from_environment") as mock_res:
        mock_res.return_value = TaskResources(memory_bytes=1_000_000, cpu_cores=1, gpu_count=0, tpu_count=0)
        merged = shard.merge_sorted_chunks(external_sort_dir="gs://bucket/external-sort/")
        first = next(merged)
        active_runs = [path for path in remote_root.rglob("*") if path.is_file()]
        remaining = list(merged)

    assert active_runs
    assert [item["k"] for item in [first, *remaining]] == list(range(10))
    assert [path for path in remote_root.rglob("*") if path.is_file()] == []


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
    table = pa.table(
        {
            _PAYLOAD_COL: pa.array(
                [cloudpickle.dumps({"k": target}) for target in range(num_targets)],
                type=pa.binary(),
            ),
            _SHARD_COL: pa.array(range(num_targets), type=pa.int32()),
            _SORT_KEY_COL: pa.StructArray.from_arrays(
                [
                    pa.array([target.to_bytes(4, "big") for target in range(num_targets)], type=pa.binary()),
                    pa.nulls(num_targets),
                ],
                names=["key", "sort_value"],
            ),
        }
    )
    writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=0, num_output_shards=num_targets)
    writer.write(table)
    scatter_paths = list(writer.close())

    parquet = pq.ParquetFile(f"{data_path}c0004.parquet")
    assert parquet.metadata.num_row_groups <= _SCATTER_MAX_ROW_GROUPS_PER_CHUNK

    reader = ScatterReader.from_sidecars(scatter_paths, target_shard=513)
    assert reader.total_chunks == 1
    assert _read_shard(reader) == [{"k": 513}]


def test_scatter_auto_flush_uses_task_memory_budget(tmp_path):
    """A multiplexed shard flushes against its task budget, not the worker pod limit."""
    items = [{"k": 0, "v": i} for i in range(10_000)]
    table = _items_to_table(items, _key, None, 1)
    task_memory_bytes = 128 * 1024**2
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
            patch("zephyr.shuffle._SCATTER_FLUSH_MEMORY_FRACTION", 0.001),
        ):
            environment_resources.return_value = TaskResources(
                memory_bytes=1024**3,
                cpu_cores=1,
                gpu_count=0,
                tpu_count=0,
            )
            writer = ScatterWriter(data_path=str(tmp_path / "scatter"), key_fn=_key, source_shard=0, num_output_shards=1)
            writer.write(table)
            writer.write(table)
            scatter_paths = list(writer.close())
    finally:
        _worker_ctx_var.reset(token)

    shard = ScatterReader.from_sidecars(scatter_paths, 0)
    assert shard.total_chunks == 2
    assert sorted(row["v"] for row in _read_shard(shard)) == sorted([*range(10_000), *range(10_000)])


# ---------------------------------------------------------------------------
# bounded sorted merge
# ---------------------------------------------------------------------------


def _make_sorted_frame(context, values: list[int]):
    """Build a DataFusion frame from values already ordered by the sort key."""
    table = pa.table(
        {
            _PAYLOAD_COL: pa.array([cloudpickle.dumps({"v": v}) for v in values], type=pa.binary()),
            _SORT_KEY_COL: pa.StructArray.from_arrays(
                [pa.array(values, type=pa.int64()), pa.nulls(len(values))],
                names=["key", "sort_value"],
            ),
        }
    )
    return context.from_arrow(table)


def _merge_sorted_items(
    context,
    frames,
    *,
    sort_key: str,
    external_sort_dir: str,
    fan_in: int,
    shard: int,
) -> list:
    merged = _merge_sorted_frames(
        context=context,
        frames=list(frames),
        sort_key=sort_key,
        external_sort_dir=external_sort_dir,
        fan_in=fan_in,
        shard=shard,
    )
    return [row for batch in merged for row in _table_to_items(batch)]


def test_merge_sorted_frames_streaming(tmp_path):
    context = datafusion_context()
    frames = [
        _make_sorted_frame(context, [1, 4, 7]),
        _make_sorted_frame(context, [2, 5, 8]),
        _make_sorted_frame(context, [3, 6, 9]),
    ]
    rows = _merge_sorted_items(
        context,
        frames,
        sort_key=_SORT_KEY_COL,
        external_sort_dir=str(tmp_path),
        fan_in=4,
        shard=0,
    )
    result = [row["v"] for row in rows]
    assert result == list(range(1, 10))


def test_merge_sorted_frames_single_batch(tmp_path):
    context = datafusion_context()
    frames = [_make_sorted_frame(context, [i]) for i in range(10)]
    rows = _merge_sorted_items(
        context,
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
    context = datafusion_context()
    frames = [_make_sorted_frame(context, [i]) for i in range(fan_in + 1)]
    list(
        _merge_sorted_frames(
            context=context,
            frames=frames,
            sort_key=_SORT_KEY_COL,
            external_sort_dir=str(tmp_path),
            fan_in=fan_in,
            shard=0,
        )
    )
    assert list(tmp_path.iterdir()) == [], "run files should be deleted after merge"


def test_merge_sorted_frames_multi_pass_preserves_order(tmp_path):
    context = datafusion_context()
    frames = [_make_sorted_frame(context, [value]) for value in range(20)]
    rows = _merge_sorted_items(
        context,
        frames,
        sort_key=_SORT_KEY_COL,
        external_sort_dir=str(tmp_path),
        fan_in=2,
        shard=0,
    )

    assert [row["v"] for row in rows] == list(range(20))
    assert list(tmp_path.iterdir()) == []


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
    # Shard 0 writes keys [1, 3], shard 1 writes key [2].  The merge must produce [1, 2, 3].
    paths = []
    for shard_idx, items in [(0, [{"k": 3, "v": "a"}, {"k": 1, "v": "b"}]), (1, [{"k": 2, "v": "c"}])]:
        data_path = f"{tmp_path}/shard-{shard_idx:04d}/scatter/"
        paths.append(data_path)
        writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=shard_idx, num_output_shards=1)
        writer.write(_items_to_table(items, _key, None, 1))
        writer.close()

    shard = ScatterReader.from_sidecars(paths, target_shard=0)
    assert shard.total_chunks == 2, "expected one Parquet file per source shard"

    external_dir = tmp_path / "sort_work"
    external_dir.mkdir()

    context = datafusion_context()
    rows = _merge_sorted_items(
        context,
        shard.get_frames(context),
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
        writer = ScatterWriter(data_path=data_path, key_fn=_key, source_shard=shard_idx, num_output_shards=1)
        writer.write(_items_to_table([{"k": shard_idx}], _key, None, 1))
        writer.close()
        paths.append(f"counting://{data_path}")

    _CountingFileSystem.clear_instance_cache()
    _CountingFileSystem.clients_built = 0

    sidecars = _Sidecar.read_all(paths)

    assert [sidecar.path for sidecar in sidecars] == paths
    assert _CountingFileSystem.clients_built == 1
