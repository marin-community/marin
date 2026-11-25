# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.counts import count_json_elements, count_jsonl_rows, count_parquet_rows


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


def test_memory_filesystem_counts_like_a_local_one():
    filesystem = StoragePath("memory://counts-bucket/rows.parquet")
    with filesystem.open("wb") as handle:
        pq.write_table(pa.table({"n": list(range(5))}), handle)
    assert count_parquet_rows(filesystem) == 5
