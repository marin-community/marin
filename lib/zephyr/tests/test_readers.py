# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for parquet reader (load_parquet)."""


import itertools
from contextlib import closing

import msgspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard as zstd
from zephyr import counters
from zephyr.dataset import LoadFileOp
from zephyr.expr import ColumnExpr, CompareExpr, LiteralExpr, col
from zephyr.input_file import DEFAULT_FILE_PATH_COLUMN, InputFileSpec
from zephyr.plan import compose_map
from zephyr.readers import (
    SUPPORTED_EXTENSIONS,
    compute_parquet_splits,
    load_file,
    load_file_batch,
    load_jsonl,
    load_parquet,
    load_parquet_batch,
    load_vortex,
)
from zephyr.worker_context import InputProgress
from zephyr.writers import write_vortex_file


def _write_test_parquet(path: str, records: list[dict], row_group_size: int = 2) -> None:
    """Write a parquet file with small row groups for testing."""
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, row_group_size=row_group_size)


RECORDS = [{"id": i, "name": f"row{i}", "score": float(i * 10)} for i in range(10)]


@pytest.mark.parametrize("loader,extension", [(load_parquet, "parquet"), (load_jsonl, "jsonl"), (load_vortex, "vortex")])
def test_nested_lookup_reads_preserve_main_input_progress(tmp_path, worker_context, loader, extension):
    main_path = str(tmp_path / "main.parquet")
    _write_test_parquet(main_path, [{"id": i} for i in range(100)], row_group_size=25)
    lookup_path = tmp_path / f"lookup.{extension}"
    records = [{"id": 123}]
    if extension == "parquet":
        _write_test_parquet(str(lookup_path), records)
    elif extension == "vortex":
        write_vortex_file(records, str(lookup_path))
    else:
        lookup_path.write_bytes(msgspec.json.encode(records[0]))

    with closing(compose_map([LoadFileOp("parquet")])(iter([main_path]))) as reader:
        assert [row["id"] for row in itertools.islice(reader, 26)] == list(range(26))
        expected = InputProgress(completed=25, total=100)
        assert worker_context.get_runner_snapshot().input_progress == expected
        with closing(loader(str(lookup_path))) as lookup:
            assert next(lookup) == records[0]
            assert worker_context.get_runner_snapshot().input_progress == expected
        assert list(loader(str(lookup_path))) == records
        assert worker_context.get_runner_snapshot().input_progress == expected
        assert len(list(reader)) == 74
    assert worker_context.get_runner_snapshot().input_progress == InputProgress(completed=100, total=100)
    assert worker_context.get_counters()[counters.RECORDS_IN] == 100


@pytest.mark.parametrize("loader", [load_parquet, load_parquet_batch])
def test_parquet_reader_reports_cheap_row_range_total(tmp_path, loader):
    path = str(tmp_path / "progress.parquet")
    _write_test_parquet(path, [{"id": i} for i in range(12)], row_group_size=4)
    reader = loader(InputFileSpec(path=path, row_start=2, row_end=10, filter_expr=col("id") >= 5))
    rows = list(reader)
    assert reader.total_rows == 8
    if loader is load_parquet_batch:
        rows = [row for batch in rows for row in batch.to_pylist()]
    assert rows == [{"id": i} for i in range(5, 10)]


@pytest.mark.parametrize("filter_expr", [None, col("id") >= 5])
def test_vortex_reader_reports_row_range_total(tmp_path, filter_expr):
    path = str(tmp_path / "progress.vortex")
    write_vortex_file([{"id": i} for i in range(12)], path)
    reader = load_vortex(InputFileSpec(path=path, row_start=2, row_end=10, filter_expr=filter_expr))
    assert list(reader) == [{"id": i} for i in range(2, 10) if filter_expr is None or i >= 5]
    assert reader.total_rows == 8


def test_input_progress_resets_between_jsonl_files(tmp_path, worker_context):
    pipeline = compose_map([LoadFileOp("jsonl")])
    paths = []
    for size in (5, 2):
        path = tmp_path / f"input-{size}.jsonl"
        path.write_bytes(b"\n".join(msgspec.json.encode({"id": index}) for index in range(size)))
        paths.append(str(path))
    with closing(pipeline(iter(paths))) as reader:
        assert [row["id"] for row in itertools.islice(reader, 6)] == [0, 1, 2, 3, 4, 0]
        assert worker_context.get_runner_snapshot().input_progress == InputProgress(completed=0, total=None)
        assert list(reader) == [{"id": 1}]
    assert worker_context.get_runner_snapshot().input_progress == InputProgress(completed=2, total=None)
    assert worker_context.get_counters()[counters.RECORDS_IN] == 7


def test_load_parquet_plain(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    result = list(load_parquet(path))
    assert result == RECORDS


def test_load_parquet_columns(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    spec = InputFileSpec(path=path, columns=["id", "name"])
    result = list(load_parquet(spec))
    assert result == [{"id": r["id"], "name": r["name"]} for r in RECORDS]


@pytest.mark.parametrize(
    "row_start,row_end",
    [(2, 7), (2, None), (None, 7)],
)
def test_load_parquet_row_range(tmp_path, row_start, row_end):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=3)

    spec = InputFileSpec(path=path, row_start=row_start, row_end=row_end)
    result = list(load_parquet(spec))
    assert [r["id"] for r in result] == list(range(10))[row_start:row_end]


def test_load_parquet_filter(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    spec = InputFileSpec(
        path=path,
        filter_expr=CompareExpr(op="ge", left=ColumnExpr(name="score"), right=LiteralExpr(value=50.0)),
    )
    result = list(load_parquet(spec))
    assert all(r["score"] >= 50.0 for r in result)
    assert [r["id"] for r in result] == [5, 6, 7, 8, 9]


def test_load_parquet_filter_and_row_range(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=3)

    spec = InputFileSpec(
        path=path,
        row_start=1,
        row_end=8,
        filter_expr=CompareExpr(op="ge", left=ColumnExpr(name="score"), right=LiteralExpr(value=50.0)),
    )
    result = list(load_parquet(spec))
    # rows 1-7, then filtered to score >= 50 → ids 5, 6, 7
    assert [r["id"] for r in result] == [5, 6, 7]


def test_load_parquet_filter_on_unprojected_column(tmp_path):
    """Filter can reference columns not in the projection."""
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)

    spec = InputFileSpec(
        path=path,
        columns=["id", "name"],
        filter_expr=CompareExpr(op="ge", left=ColumnExpr(name="score"), right=LiteralExpr(value=50.0)),
    )
    result = list(load_parquet(spec))
    assert [r["id"] for r in result] == [5, 6, 7, 8, 9]
    assert all(set(r.keys()) == {"id", "name"} for r in result)


def test_load_parquet_empty(tmp_path):
    path = str(tmp_path / "empty.parquet")
    table = pa.Table.from_pylist([], schema=pa.schema([("id", pa.int64())]))
    pq.write_table(table, path)

    result = list(load_parquet(path))
    assert result == []


def test_compute_parquet_splits_single(tmp_path):
    """File smaller than approx_shard_bytes returns one split covering all rows."""
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=5)

    splits = compute_parquet_splits(path, approx_shard_bytes=256 * 1024 * 1024)
    assert splits == [(0, len(RECORDS))]


def test_compute_parquet_splits_multiple(tmp_path):
    """File whose row groups exceed approx_shard_bytes returns multiple splits."""
    path = str(tmp_path / "data.parquet")
    # Write 10 row groups of 1 row each with a large-ish payload so we can
    # force splits at a small byte threshold.
    records = [{"id": i, "payload": "x" * 1000} for i in range(10)]
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, row_group_size=1)

    pf = pq.ParquetFile(path)
    single_rg_bytes = pf.metadata.row_group(0).total_byte_size
    # Threshold just above one row group forces a split after every row group.
    threshold = single_rg_bytes + 1

    splits = compute_parquet_splits(path, approx_shard_bytes=threshold)
    assert len(splits) > 1
    # Splits must be contiguous and cover all rows.
    assert splits[0][0] == 0
    assert splits[-1][1] == 10
    for (_, end), (start, _) in itertools.pairwise(splits):
        assert end == start


def test_compute_parquet_splits_row_ranges_are_readable(tmp_path):
    """Each split returned by compute_parquet_splits can be read back via load_parquet."""
    path = str(tmp_path / "data.parquet")
    records = [{"id": i} for i in range(20)]
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=2)

    pf = pq.ParquetFile(path)
    single_rg_bytes = pf.metadata.row_group(0).total_byte_size
    splits = compute_parquet_splits(path, approx_shard_bytes=single_rg_bytes * 3 + 1)

    all_ids = []
    for row_start, row_end in splits:
        spec = InputFileSpec(path=path, row_start=row_start, row_end=row_end)
        all_ids.extend(r["id"] for r in load_parquet(spec))

    assert sorted(all_ids) == list(range(20))


@pytest.mark.parametrize("row_start,row_end", [(None, None), (2, 3)])
def test_load_parquet_filter_skips_corrupt_unrelated_payload(tmp_path, row_start, row_end):
    """Footer pruning must avoid reading payloads belonging to other components."""
    path = tmp_path / "data.parquet"
    records = [
        {"component": "left", "payload": "unrelated-a"},
        {"component": "left", "payload": "unrelated-b"},
        {"component": "right", "payload": "selected-a"},
        {"component": "right", "payload": "selected-b"},
    ]
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=2, compression="zstd")
    payload = pq.ParquetFile(path).metadata.row_group(0).column(1)
    with path.open("r+b") as stream:
        stream.seek(payload.dictionary_page_offset)
        stream.write(b"\x00" * payload.total_compressed_size)
    with pytest.raises(OSError):
        list(load_parquet(str(path)))
    predicate = col("component") == "right"
    spec = InputFileSpec(
        path=str(path), columns=["payload"], filter_expr=predicate, row_start=row_start, row_end=row_end
    )
    expected = [{"payload": "selected-a"}]
    if row_end is None:
        expected.append({"payload": "selected-b"})
    assert list(load_parquet(spec)) == expected
    assert compute_parquet_splits(str(path), 1, filter_expr=predicate) == [(2, 4)]
    assert compute_parquet_splits(str(path), 1, filter_expr=col("component") == "absent") == []


def test_load_parquet_batch_returns_record_batches(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=4)

    batches = list(load_parquet_batch(path))
    assert all(isinstance(b, pa.RecordBatch) for b in batches)
    # All rows present across batches
    all_rows = [row for b in batches for row in b.to_pylist()]
    assert sorted(all_rows, key=lambda r: r["id"]) == RECORDS


def test_load_parquet_batch_consistent_with_load_parquet(tmp_path):
    """load_parquet_batch + to_pylist must equal load_parquet."""
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=3)

    spec = InputFileSpec(path=path, row_start=2, row_end=8)
    via_batch = [row for b in load_parquet_batch(spec) for row in b.to_pylist()]
    via_dict = list(load_parquet(spec))
    assert via_batch == via_dict


@pytest.mark.parametrize("ext", [".jsonl.zst", ".jsonl.zstd"])
def test_load_jsonl_decompresses_zstd_extensions(tmp_path, ext):
    """Both the ``.zst`` and ``.zstd`` extensions must be zstd-decompressed.

    ``.zstd`` is advertised in SUPPORTED_EXTENSIONS; a genuinely-compressed
    file written under it must decode, not be read back as raw bytes.
    """
    path = str(tmp_path / f"data{ext}")
    encoder = msgspec.json.Encoder()
    raw = b"".join(encoder.encode(r) + b"\n" for r in RECORDS)
    with open(path, "wb") as f:
        f.write(zstd.ZstdCompressor().compress(raw))

    assert path.endswith(SUPPORTED_EXTENSIONS)
    assert list(load_jsonl(path)) == RECORDS


def test_load_file_honors_explicit_format_over_extension(tmp_path):
    """An explicit ``format`` picks the reader, whatever the file is named.

    ``Dataset.load_parquet`` and friends record the format on the spec; a
    corpus of extension-less shards (Spark ``part-00000``) must still read.
    """
    path = str(tmp_path / "part-00000")
    _write_test_parquet(path, RECORDS)

    assert list(load_file(InputFileSpec(path=path, format="parquet"))) == RECORDS
    with pytest.raises(ValueError, match="Unsupported extension"):
        list(load_file(path))


def test_load_file_explicit_format_beats_a_misleading_extension(tmp_path):
    """A ``.parquet``-named JSONL file reads as JSONL when the spec says so."""
    path = str(tmp_path / "data.parquet")
    encoder = msgspec.json.Encoder()
    with open(path, "wb") as f:
        f.write(b"".join(encoder.encode(r) + b"\n" for r in RECORDS))

    assert list(load_file(InputFileSpec(path=path, format="jsonl"))) == RECORDS


def test_load_file_batch_honors_explicit_parquet_format(tmp_path):
    path = str(tmp_path / "part-00000")
    _write_test_parquet(path, RECORDS)

    spec = InputFileSpec(path=path, format="parquet")
    assert [row for b in load_file_batch(spec) for row in b.to_pylist()] == RECORDS
    with pytest.raises(RuntimeError, match="only supports Parquet"):
        list(load_file_batch(path))


@pytest.mark.parametrize("file_format", ["parquet", "jsonl", "vortex"])
def test_load_file_project_only_injected_path(tmp_path, file_format):
    path = str(tmp_path / f"data.{file_format}")
    records = [{"id": 1, "payload": "first"}, {"id": 2, "payload": "second"}]
    if file_format == "parquet":
        _write_test_parquet(path, records)
    elif file_format == "vortex":
        write_vortex_file(records, path)
    else:
        with open(path, "wb") as f:
            for record in records:
                f.write(msgspec.json.encode(record) + b"\n")

    spec = InputFileSpec(path=path, columns=[DEFAULT_FILE_PATH_COLUMN])
    assert list(load_file(spec, include_file_paths=True)) == [{DEFAULT_FILE_PATH_COLUMN: path}] * len(records)


def test_load_file_batch_project_only_injected_path(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS, row_group_size=3)

    spec = InputFileSpec(path=path, columns=[DEFAULT_FILE_PATH_COLUMN], row_start=2, row_end=7)
    rows = [row for batch in load_file_batch(spec, include_file_paths=True) for row in batch.to_pylist()]
    assert rows == [{DEFAULT_FILE_PATH_COLUMN: path}] * 5


def test_load_file_preserves_projection_order_with_injected_path(tmp_path):
    path = str(tmp_path / "data.parquet")
    _write_test_parquet(path, RECORDS)
    columns = [DEFAULT_FILE_PATH_COLUMN, "id"]
    spec = InputFileSpec(path=path, columns=columns)

    records = list(load_file(spec, include_file_paths=True))
    batches = list(load_file_batch(spec, include_file_paths=True))

    assert all(list(record) == columns for record in records)
    assert all(batch.schema.names == columns for batch in batches)
    assert [record["id"] for record in records] == list(range(len(RECORDS)))
    assert [row for batch in batches for row in batch.to_pylist()] == records


def test_load_file_vortex_project_only_injected_path_with_range_and_filter(tmp_path):
    path = str(tmp_path / "data.vortex")
    write_vortex_file(RECORDS, path)

    spec = InputFileSpec(
        path=path,
        columns=[DEFAULT_FILE_PATH_COLUMN],
        row_start=2,
        row_end=8,
        filter_expr=CompareExpr(op="ge", left=ColumnExpr(name="score"), right=LiteralExpr(value=50.0)),
    )
    assert list(load_file(spec, include_file_paths=True)) == [{DEFAULT_FILE_PATH_COLUMN: path}] * 3
