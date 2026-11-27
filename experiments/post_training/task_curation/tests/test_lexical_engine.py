# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DatakitDecontamination behavior: attribute marks through the eval hash index."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.task_curation.lexical_engine import DatakitDecontamination

MARK_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("partition_id", pa.int64()),
        ("contaminated", pa.bool_()),
        ("max_overlap", pa.float64()),
        ("matched_hashes", pa.list_(pa.uint64())),
    ]
)
FLAG_FIELDS = ("contaminated", "max_overlap", "matched_hashes", "matched_eval_ids", "matched_eval_families")


def write_mark_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=MARK_SCHEMA), path)


def read_flags(directory: Path) -> dict[str, dict]:
    flags = {}
    for file in sorted(directory.glob("*.parquet")):
        for row in pq.read_table(file).to_pylist():
            flags[row["id"]] = {field: row[field] for field in FLAG_FIELDS}
    return flags


def engine(tmp_path: Path) -> DatakitDecontamination:
    # The attribution path never touches the bloom; the hash index is read only
    # when a mark carries matched hashes, so a fixture corpus version suffices.
    return DatakitDecontamination(
        bloom_dir=str(tmp_path / "bloom"),
        global_drop_set_dir=str(tmp_path / "global"),
        eval_corpus_version="fixture-corpus-v1",
    )


def test_attribute_writes_eval_ids_and_families_for_matched_hashes(tmp_path):
    marks = tmp_path / "marks"
    write_mark_file(
        marks / "part-00000.parquet",
        [{"id": "task-a", "partition_id": 0, "contaminated": True, "max_overlap": 1.0, "matched_hashes": [7]}],
    )
    write_mark_file(
        marks / "part-00001.parquet",
        [{"id": "task-b", "partition_id": 1, "contaminated": False, "max_overlap": 0.0, "matched_hashes": []}],
    )
    index = tmp_path / "index" / "eval_hash_index.parquet"
    index.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"hash": 7, "eval_id": "asdiv-test-0"}],
            schema=pa.schema([("hash", pa.uint64()), ("eval_id", pa.string())]),
        ),
        index,
    )
    flags_dir = tmp_path / "flags"

    rows = engine(tmp_path)._attribute(str(marks / "*.parquet"), str(flags_dir), str(index))

    assert rows == 2
    flags = read_flags(flags_dir)
    assert flags["task-a"]["contaminated"] is True
    assert flags["task-a"]["matched_eval_ids"] == ["asdiv-test-0"]
    assert flags["task-a"]["matched_eval_families"] == ["asdiv"]
    assert flags["task-b"]["contaminated"] is False
    assert flags["task-b"]["matched_eval_ids"] == []


def test_attribute_keeps_a_flag_file_for_a_zero_row_mark_partition(tmp_path):
    marks = tmp_path / "marks"
    write_mark_file(marks / "part-00000.parquet", [])
    flags_dir = tmp_path / "flags"

    rows = engine(tmp_path)._attribute(str(marks / "*.parquet"), str(flags_dir), str(tmp_path / "unused-index.parquet"))

    assert rows == 0
    files = sorted(flags_dir.glob("*.parquet"))
    assert len(files) == 1
    assert pq.read_table(files[0]).num_rows == 0
    # The zero-row file carries the typed flag schema so downstream readers see one
    # output per population partition.
    assert pq.read_schema(files[0]).names == [
        "id",
        "max_overlap",
        "contaminated",
        "matched_hashes",
        "matched_eval_ids",
        "matched_eval_families",
    ]
