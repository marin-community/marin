# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.counts import (
    count_json_elements,
    count_jsonl_rows,
    count_parquet_rows,
    count_source_files,
    count_source_rows,
    count_table_rows,
    count_task_packages,
)
from taskcompendium.pipeline.inputs import SourceFormat


def test_parquet_footer_counts_rows_across_row_groups(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"n": list(range(7))}), path, row_group_size=3)
    assert count_parquet_rows(StoragePath(str(path))) == 7


def test_jsonl_counts_non_empty_lines_like_the_reader(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3}\n')
    assert count_jsonl_rows(StoragePath(str(path))) == 3


def test_jsonl_counts_gzipped_lines(tmp_path):
    path = tmp_path / "rows.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"a": 1}\n{"a": 2}\n'))
    assert count_jsonl_rows(StoragePath(str(path))) == 2


def test_json_counts_top_level_array_elements(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    assert count_json_elements(StoragePath(str(path))) == 2


def test_json_rejects_a_non_array_document(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError, match="record array"):
        count_json_elements(StoragePath(str(path)))


def test_task_packages_count_directories_holding_task_toml(tmp_path):
    (tmp_path / "task-a").mkdir()
    (tmp_path / "task-a" / "task.toml").write_text("")
    (tmp_path / "task-b").mkdir()
    (tmp_path / "task-b" / "task.toml").write_text("")
    (tmp_path / "task-b" / "nested").mkdir()
    (tmp_path / "task-b" / "nested" / "task.toml").write_text("")
    (tmp_path / "not-a-package").mkdir()
    assert count_task_packages(StoragePath(str(tmp_path))) == 2


def test_table_rows_sum_parquet_shard_footers(tmp_path):
    pq.write_table(pa.table({"n": list(range(3))}), tmp_path / "shard-00000.parquet")
    pq.write_table(pa.table({"n": list(range(4))}), tmp_path / "shard-00001.parquet")
    (tmp_path / "notes.txt").write_text("not a shard")
    assert count_table_rows(StoragePath(str(tmp_path))) == 7


def test_memory_filesystem_counts_like_a_local_one():
    filesystem = StoragePath("memory://counts-bucket/rows.parquet")
    with filesystem.open("wb") as handle:
        pq.write_table(pa.table({"n": list(range(5))}), handle)
    assert count_parquet_rows(filesystem) == 5


def test_source_rows_dispatches_per_format(tmp_path):
    parquet = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"n": list(range(2))}), parquet)
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    document = tmp_path / "rows.json"
    document.write_text(json.dumps([{"a": 1}]))
    assert count_source_rows(StoragePath(str(parquet)), SourceFormat.PARQUET) == 2
    assert count_source_rows(StoragePath(str(jsonl)), SourceFormat.JSONL) == 3
    assert count_source_rows(StoragePath(str(document)), SourceFormat.JSON) == 1
    assert count_source_rows(StoragePath(str(parquet)), SourceFormat.GENERATED) is None
    assert count_source_rows(StoragePath(str(parquet)), SourceFormat.CSV) is None


def test_source_files_counts_each_declared_file(tmp_path):
    first = tmp_path / "rows-00000.parquet"
    pq.write_table(pa.table({"n": list(range(2))}), first)
    second = tmp_path / "rows-00001.parquet"
    pq.write_table(pa.table({"n": list(range(3))}), second)
    counts = count_source_files(
        {"rows-00000.parquet": StoragePath(str(first)), "rows-00001.parquet": StoragePath(str(second))},
        SourceFormat.PARQUET,
    )
    assert counts == {"rows-00000.parquet": 2, "rows-00001.parquet": 3}
