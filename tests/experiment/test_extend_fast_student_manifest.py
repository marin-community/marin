# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from extend_fast_student_manifest import (  # noqa: E402
    EXPANSION_VERSION,
    extended_source_table,
    reusable_source_result,
    sample_positions_excluding,
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


def test_sample_positions_excluding_is_deterministic_and_disjoint() -> None:
    excluded = {1, 2, 10, 11}
    first = sample_positions_excluding(np.random.default_rng(42), 100, 30, excluded)
    second = sample_positions_excluding(np.random.default_rng(42), 100, 30, excluded)

    np.testing.assert_array_equal(first, second)
    assert len(first) == len(set(first.tolist())) == 30
    assert set(first.tolist()).isdisjoint(excluded)


def test_sample_positions_excluding_supports_dense_remainder() -> None:
    selected = sample_positions_excluding(np.random.default_rng(42), 10, 4, {0, 1, 2, 3, 4, 5})

    np.testing.assert_array_equal(selected, np.asarray([6, 7, 8, 9]))


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
    ]

    output = extended_source_table("source", base, extension, target_quota=5)

    assert output.select(base.column_names).slice(0, len(base)).equals(base)
    assert output["train_rank"].to_pylist()[-2:] == [3, 4]
    assert pc.sum(output["in_expanded_rung"]).as_py() == 5
    assert pc.sum(output["in_3m"]).as_py() == 3


def test_extended_source_table_rejects_non_nested_target() -> None:
    with np.testing.assert_raises_regex(ValueError, "extension count"):
        extended_source_table("source", base_table(), [], target_quota=2)


def test_reusable_source_result_requires_exact_inputs() -> None:
    result = {"output_url": "s3://bucket/output.parquet"}
    report = {
        "expansion_version": EXPANSION_VERSION,
        "base_manifest_sha256": "base-sha",
        "rung": "10m",
        "quota": 100,
        "source_result": result,
    }

    assert reusable_source_result(report, "base-sha", "10m", 100) == result
    assert reusable_source_result(report, "different-sha", "10m", 100) is None
    assert reusable_source_result(report, "base-sha", "10m", 101) is None
