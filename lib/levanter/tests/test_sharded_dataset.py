# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import warnings

import pyarrow as pa
import pyarrow.parquet as pq

from levanter.data.sharded_datasource import (
    ParquetDataSource,
    TextUrlDataSource,
    _mk_shard_name_mapping,
    _sniff_format_for_dataset,
)


def test_sniff_format_for_json():
    # this tests where some people use ".json" to mean a jsonlines file
    # and others use it to mean a json file

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(b'[{"text": "hello world"}, {"text": "hello world!"]')
        f.flush()
        assert _sniff_format_for_dataset(f.name) == ".json"

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(b'{"text": "hello world"}\n{"text": "hello world!"}\n')
        f.flush()
        assert _sniff_format_for_dataset(f.name) == ".jsonl"

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(b'{\n"ids": [1, 2, 3]\n}\n')
        f.flush()
        assert _sniff_format_for_dataset(f.name) == ".json"


def test_sniff_format_for_parquet():

    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        table = pa.table({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        pq.write_table(table, f.name)
        f.flush()

        assert _sniff_format_for_dataset(f.name) == ".parquet"


def test_basic_parquet_datasource_read_row():

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as f:
        # Create a simple dataset
        data = {"column1": ["value1", "value2", "value3"], "column2": [10, 20, 30]}
        table = pa.Table.from_pydict(data)
        pq.write_table(table, f.name)

        datasource = ParquetDataSource([os.path.abspath(f.name)])

        assert len(datasource.shard_names) == 1, "Expected only one shard"
        shard_name = datasource.shard_names[0]

        # sanity check: Read data starting from row 1
        row_data = list(datasource.open_shard_at_row(shard_name=shard_name, row=1))

        # Verify the output
        assert len(row_data) == 2  # We expect 2 rows starting from index 1
        assert row_data[0]["column1"] == "value2"
        assert row_data[0]["column2"] == 20
        assert row_data[1]["column1"] == "value3"
        assert row_data[1]["column2"] == 30


def test_text_url_data_source_parquet():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as f:
        data = {
            "text": ["line1", "line2", "line3", "line4", "line5", "line6"],
            "column2": [10, 20, 30, 40, 50, 60],
        }
        table = pa.Table.from_pydict(data)
        pq.write_table(table, f.name)

        datasource = TextUrlDataSource([os.path.abspath(f.name)], text_key="text")

        assert len(datasource.shard_names) == 1, "Expected only one shard"
        shard_name = datasource.shard_names[0]

        # Read data starting from row 2
        row_data = list(datasource.open_shard_at_row(shard_name=shard_name, row=2))

        # Verify the output
        expected_texts = ["line3", "line4", "line5", "line6"]
        assert len(row_data) == len(expected_texts), f"Expected {len(expected_texts)} rows starting from index 2"
        assert row_data == expected_texts, f"Expected texts {expected_texts}, got {row_data}"


def test_shard_name_mapping_pairs_each_url_with_its_own_existence(tmp_path):
    # A glob spec, a named-and-present literal, and a named-but-absent literal in one
    # call: every shard must appear in the mapping, and only the absent one may be
    # reported missing.
    (tmp_path / "a.jsonl").write_text("{}\n")
    (tmp_path / "b.jsonl").write_text("{}\n")
    (tmp_path / "named.jsonl").write_text("{}\n")
    absent = str(tmp_path / "absent.jsonl")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mapping = _mk_shard_name_mapping([str(tmp_path / "?.jsonl"), str(tmp_path / "named.jsonl"), absent])

    assert sorted(mapping.values()) == sorted(
        [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), str(tmp_path / "named.jsonl"), absent]
    )
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert absent in messages[0]
    assert "a.jsonl" not in messages[0]
    assert "named.jsonl" not in messages[0]
