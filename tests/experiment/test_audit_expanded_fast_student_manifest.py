# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from audit_expanded_fast_student_manifest import source_metrics  # noqa: E402


def source_table(training_rows: int, ten_million_rows: int) -> pa.Table:
    rows = []
    for index in range(512):
        rows.append(
            {
                "input_path": "s3://bucket/source.parquet",
                "input_row_group": 0,
                "input_row_in_group": index,
                "raw_sha256": f"eval-{index}",
                "normalized_sha256": f"eval-normalized-{index}",
                "split": "eval",
                "eval_rank": index,
                "train_rank": -1,
                "in_10m": False,
                "in_30m": False,
            }
        )
    for index in range(training_rows):
        rows.append(
            {
                "input_path": "s3://bucket/source.parquet",
                "input_row_group": 1,
                "input_row_in_group": index,
                "raw_sha256": f"train-{index}",
                "normalized_sha256": f"train-normalized-{index}",
                "split": "train",
                "eval_rank": -1,
                "train_rank": index,
                "in_10m": index < ten_million_rows,
                "in_30m": True,
            }
        )
    return pa.Table.from_pylist(rows)


def write_source_tables(tmp_path: Path) -> tuple[Path, Path]:
    ten_path = tmp_path / "10m.parquet"
    thirty_path = tmp_path / "30m.parquet"
    pq.write_table(source_table(4, 4).drop(["in_30m"]), ten_path)
    pq.write_table(source_table(7, 4), thirty_path)
    return ten_path, thirty_path


def test_source_metrics_accepts_exact_nested_rows(tmp_path: Path) -> None:
    ten_path, thirty_path = write_source_tables(tmp_path)

    metrics = source_metrics(str(ten_path), str(thirty_path), 4, 7)

    assert metrics == {
        "evaluation_rows": 512,
        "train_10m_rows": 4,
        "train_30m_rows": 7,
        "total_30m_rows": 519,
    }


def test_source_metrics_rejects_changed_nested_row(tmp_path: Path) -> None:
    ten_path, thirty_path = write_source_tables(tmp_path)
    thirty = pq.read_table(thirty_path)
    raw_hashes = thirty["raw_sha256"].to_pylist()
    raw_hashes[513] = "changed"
    changed = thirty.set_column(
        thirty.schema.get_field_index("raw_sha256"),
        "raw_sha256",
        pa.array(raw_hashes),
    )
    pq.write_table(changed, thirty_path)

    with pytest.raises(ValueError, match="exact 10M rows differ"):
        source_metrics(str(ten_path), str(thirty_path), 4, 7)
