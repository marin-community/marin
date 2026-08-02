# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from build_manifest import PositionOrder, block_sample_positions  # noqa: E402
from extend_fast_student_manifest import (  # noqa: E402
    EXPANSION_VERSION,
    assigned_source_names,
    extended_source_table,
    fixed_global_positions,
    reusable_source_result,
    selected_extension_table,
)


def base_table() -> pa.Table:
    rows = []
    for index in range(512):
        rows.append(
            {
                "id": f"eval-{index}",
                "input_path": "s3://bucket/source.parquet",
                "input_row_group": 0,
                "input_row_in_group": index,
                "source": "source",
                "source_category": "standard",
                "split": "eval",
                "eval_rank": index,
                "train_rank": -1,
                "in_750k": False,
                "in_3m": False,
                "raw_characters": 4,
                "raw_sha256": f"eval-sha-{index}",
                "normalized_sha256": f"eval-normalized-{index}",
                "text": "eval",
            }
        )
    for index in range(3):
        rows.append(
            {
                "id": f"train-{index}",
                "input_path": "s3://bucket/source.parquet",
                "input_row_group": 0,
                "input_row_in_group": 512 + index,
                "source": "source",
                "source_category": "standard",
                "split": "train",
                "eval_rank": -1,
                "train_rank": index,
                "in_750k": index == 0,
                "in_3m": True,
                "raw_characters": 5,
                "raw_sha256": f"train-sha-{index}",
                "normalized_sha256": f"train-normalized-{index}",
                "text": "train",
            }
        )
    return pa.Table.from_pylist(rows)


def selection_positions(row_count: int) -> np.ndarray:
    return block_sample_positions(
        np.random.default_rng(42),
        100,
        row_count,
        excluded={1, 2, 10, 11},
        block_size=8,
        position_order=PositionOrder.SELECTION,
    )


def test_block_sample_positions_is_deterministic_disjoint_and_nested() -> None:
    excluded = {1, 2, 10, 11}
    first = selection_positions(30)
    second = selection_positions(30)
    larger = selection_positions(60)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(larger[: len(first)], first)
    assert len(first) == len(set(first.tolist())) == 30
    assert set(first.tolist()).isdisjoint(excluded)


def test_block_sample_positions_supports_dense_remainder() -> None:
    selected = block_sample_positions(
        np.random.default_rng(42),
        10,
        4,
        excluded={0, 1, 2, 3, 4, 5},
        block_size=2,
        position_order=PositionOrder.SELECTION,
    )

    assert set(selected.tolist()) == {6, 7, 8, 9}


def test_block_sample_positions_rejects_invalid_exclusions() -> None:
    with pytest.raises(ValueError, match="outside"):
        block_sample_positions(np.random.default_rng(42), 10, 1, excluded={10})
    with pytest.raises(ValueError, match="available"):
        block_sample_positions(np.random.default_rng(42), 10, 7, excluded={0, 1, 2, 3})


def test_fixed_global_positions_maps_groups_and_files(tmp_path: Path) -> None:
    parquet_path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"id": list(range(6))}), parquet_path, row_group_size=2)
    uri = f"file://{parquet_path}"
    table = pa.Table.from_pylist(
        [
            {"input_path": uri, "input_row_group": 0, "input_row_in_group": 1},
            {"input_path": uri, "input_row_group": 2, "input_row_in_group": 0},
        ]
    )

    positions = fixed_global_positions(table, None, [(str(parquet_path), 6)], "file")

    assert positions == {1, 4}
    extension = block_sample_positions(
        np.random.default_rng(42),
        6,
        3,
        excluded=positions,
        block_size=2,
        position_order=PositionOrder.SELECTION,
    )
    assert set(extension.tolist()).isdisjoint(positions)


def test_fixed_global_positions_rejects_unknown_and_duplicate_coordinates(tmp_path: Path) -> None:
    parquet_path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"id": list(range(4))}), parquet_path, row_group_size=2)
    uri = f"file://{parquet_path}"
    unknown = pa.Table.from_pylist([{"input_path": uri, "input_row_group": 2, "input_row_in_group": 0}])
    duplicate = pa.Table.from_pylist(
        [
            {"input_path": uri, "input_row_group": 0, "input_row_in_group": 1},
            {"input_path": uri, "input_row_group": 0, "input_row_in_group": 1},
        ]
    )

    with pytest.raises(ValueError, match="unknown input row group"):
        fixed_global_positions(unknown, None, [(str(parquet_path), 4)], "file")
    with pytest.raises(ValueError, match="duplicate input positions"):
        fixed_global_positions(duplicate, None, [(str(parquet_path), 4)], "file")


def test_extended_source_table_preserves_base_rows() -> None:
    base = base_table()
    extension = [
        {
            "id": "new-0",
            "raw_text": "new text",
            "input_path": "s3://bucket/source.parquet",
            "input_row_group": 1,
            "input_row_in_group": 0,
        },
        {
            "id": "new-1",
            "raw_text": "more text",
            "input_path": "s3://bucket/source.parquet",
            "input_row_group": 1,
            "input_row_in_group": 1,
        },
        {
            "id": "new-2",
            "raw_text": "third text",
            "input_path": "s3://bucket/source.parquet",
            "input_row_group": 1,
            "input_row_in_group": 2,
        },
        {
            "id": "new-3",
            "raw_text": "fourth text",
            "input_path": "s3://bucket/source.parquet",
            "input_row_group": 1,
            "input_row_in_group": 3,
        },
    ]

    output = extended_source_table("source", base, extension, {"10m": 5, "30m": 7})

    assert output.select(base.column_names).slice(0, len(base)).equals(base)
    assert output["train_rank"].to_pylist()[-4:] == [3, 4, 5, 6]
    assert pc.sum(output["in_10m"]).as_py() == 5
    assert pc.sum(output["in_30m"]).as_py() == 7
    assert pc.sum(output["in_3m"]).as_py() == 3


def test_extended_source_table_rejects_non_nested_target() -> None:
    with pytest.raises(ValueError, match="smaller than the fixed base"):
        extended_source_table("source", base_table(), [], {"10m": 2})


def test_reusable_source_result_requires_exact_inputs() -> None:
    result = {"output_url": "s3://bucket/output.parquet"}
    report = {
        "expansion_version": EXPANSION_VERSION,
        "base_manifest_sha256": "base-sha",
        "rung": "10m",
        "rung_quotas": {"10m": 100},
        "input_snapshot_sha256": "input-sha",
        "source_result": result,
    }

    assert reusable_source_result(report, "base-sha", "10m", {"10m": 100}, "input-sha") == result
    assert reusable_source_result(report, "different-sha", "10m", {"10m": 100}, "input-sha") is None
    assert reusable_source_result(report, "base-sha", "10m", {"10m": 101}, "input-sha") is None
    assert reusable_source_result(report, "base-sha", "10m", {"10m": 100}, "different-input") is None


def test_assigned_source_names_partitions_all_sources() -> None:
    sources = ["d", "b", "e", "a", "c"]

    shards = [assigned_source_names(sources, index, 3) for index in range(3)]

    assert shards == [["a", "d"], ["b", "e"], ["c"]]
    assert sorted(source for shard in shards for source in shard) == sorted(sources)


def test_selected_extension_table_is_independent_of_read_chunk_size(tmp_path: Path) -> None:
    parquet_path = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [f"row-{index}" for index in range(520)],
                "text": [f"raw text {index}" for index in range(520)],
            }
        ),
        parquet_path,
        row_group_size=520,
    )
    uri = f"file://{parquet_path}"
    base = base_table()
    base = base.set_column(
        base.schema.get_field_index("input_path"),
        "input_path",
        pa.array([uri] * len(base)),
    )
    arguments = (None, [(str(parquet_path), 520)], "source", base, {"10m": 5, "30m": 7}, "file")

    chunked, chunked_counts = selected_extension_table(*arguments, read_chunk_rows=2)
    single, single_counts = selected_extension_table(*arguments, read_chunk_rows=100)

    assert chunked.equals(single)
    assert chunked_counts == single_counts == {uri: 4}
    assert len(chunked) == 519
    assert chunked["train_rank"].to_pylist()[-4:] == [3, 4, 5, 6]
