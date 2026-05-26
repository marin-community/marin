# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.posttrain.acecode_v2_verified_sft import (
    ACECODE_V2_HF_ID,
    ACECODE_V2_REVISION,
    ACECODE_V2_SOURCE_COLUMNS,
    ACECODE_V2_SOURCE_LABELS,
    AceCodeV2Split,
    AceCodeV2View,
    acecode_v2_source_row,
    acecode_v2_split,
    materialize_prompt_test_views,
    prompt_test_view_rows,
    source_balanced_rows,
)
from experiments.posttrain.instruction_datasets import INSTRUCTION_DATASET_NAME_TO_CONFIG


def _sample_row(row_id: str = "sample-acecode-v2-row", source: str = "oss") -> dict:
    return {
        "id": row_id,
        "source": source,
        "question": "Write a function `add_one(x)` that returns `x + 1`.",
        "tests": ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
    }


def _normalized_rows_per_source(rows_per_source: int) -> list[dict]:
    rows = []
    for source in ACECODE_V2_SOURCE_LABELS:
        for index in range(rows_per_source):
            rows.append(acecode_v2_source_row(_sample_row(row_id=f"{source}-{index}", source=source)))
    return rows


def _write_raw_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _read_view_rows(output_path: Path, view: AceCodeV2View) -> list[dict]:
    table = pq.read_table(output_path / view.value / "data-00000-of-00001.parquet")
    return table.to_pylist()


def test_source_row_normalization_preserves_prompt_tests_and_provenance():
    row = acecode_v2_source_row(_sample_row())

    assert row["id"] == "sample-acecode-v2-row"
    assert row["source"] == "oss"
    assert row["split"] == "train"
    assert row["question"] == "Write a function `add_one(x)` that returns `x + 1`."
    assert row["tests"] == ["assert add_one(1) == 2", "assert add_one(-1) == 0"]
    assert row["num_tests"] == 2
    assert row["question_sha256"] == hashlib.sha256(row["question"].encode("utf-8")).hexdigest()
    expected_test_hash_payload = json.dumps(row["tests"], sort_keys=True, separators=(",", ":"))
    expected_test_hash = hashlib.sha256(expected_test_hash_payload.encode("utf-8")).hexdigest()
    assert row["test_sha256"] == expected_test_hash
    assert row["upstream_dataset"] == ACECODE_V2_HF_ID
    assert row["upstream_revision"] == ACECODE_V2_REVISION


def test_source_row_normalization_preserves_prompt_whitespace():
    source_row = _sample_row()
    source_row["question"] = "\nWrite a function `add_one(x)`.\n"

    row = acecode_v2_source_row(source_row)

    assert row["question"] == "\nWrite a function `add_one(x)`.\n"
    assert row["question_sha256"] == hashlib.sha256(row["question"].encode("utf-8")).hexdigest()


def test_split_assignment_is_hash_based_and_stable():
    assert acecode_v2_split("train-0") == AceCodeV2Split.TRAIN
    assert acecode_v2_split("validation-0") == AceCodeV2Split.VALIDATION
    assert acecode_v2_split("validation-0") == acecode_v2_split("validation-0")


def test_small_smoke_view_is_source_balanced_and_train_only():
    rows = _normalized_rows_per_source(rows_per_source=40)

    smoke_rows = prompt_test_view_rows(rows, AceCodeV2View.SMALL_SMOKE_300, small_smoke_per_source=3)

    assert len(smoke_rows) == 9
    assert Counter(row["source"] for row in smoke_rows) == {source: 3 for source in ACECODE_V2_SOURCE_LABELS}
    assert {row["split"] for row in smoke_rows} == {"train"}
    assert smoke_rows == prompt_test_view_rows(rows, AceCodeV2View.SMALL_SMOKE_300, small_smoke_per_source=3)


def test_source_balanced_rows_rejects_empty_source_sample_size():
    rows = _normalized_rows_per_source(rows_per_source=1)

    with pytest.raises(ValueError, match="per_source must be positive"):
        source_balanced_rows(rows, per_source=0, split=AceCodeV2Split.TRAIN)


def test_materialize_prompt_test_views_writes_real_parquet_and_manifest(tmp_path: Path):
    raw_path = tmp_path / "raw"
    output_path = tmp_path / "processed"
    raw_rows = [
        _sample_row(row_id=f"{source}-{index}", source=source)
        for source in ACECODE_V2_SOURCE_LABELS
        for index in range(40)
    ]
    _write_raw_parquet(raw_path / "data" / "train-00000-of-00001.parquet", raw_rows)

    materialize_prompt_test_views(
        input_path=str(raw_path),
        output_path=str(output_path),
        small_smoke_per_source=2,
    )

    normalized_rows = [acecode_v2_source_row(row) for row in raw_rows]
    train_rows = _read_view_rows(output_path, AceCodeV2View.TRAIN_ALL_BUT_HOLDOUT)
    smoke_rows = _read_view_rows(output_path, AceCodeV2View.SMALL_SMOKE_300)
    manifest = json.loads((output_path / "manifest.json").read_text())

    assert set(train_rows[0]) == set(ACECODE_V2_SOURCE_COLUMNS)
    assert len(train_rows) == sum(row["split"] == "train" for row in normalized_rows)
    assert len(smoke_rows) == 2 * len(ACECODE_V2_SOURCE_LABELS)
    assert Counter(row["source"] for row in smoke_rows) == {source: 2 for source in ACECODE_V2_SOURCE_LABELS}
    assert manifest["upstream_dataset"] == ACECODE_V2_HF_ID
    assert manifest["upstream_revision"] == ACECODE_V2_REVISION
    assert manifest["views"]["train_all_but_holdout"]["rows"] == len(train_rows)
    assert manifest["views"]["small_smoke_300"]["rows"] == len(smoke_rows)


def test_materialize_validation_view_writes_holdout_rows(tmp_path: Path):
    raw_path = tmp_path / "raw"
    output_path = tmp_path / "processed"
    raw_rows = [
        _sample_row(row_id="evol-train-candidate-0", source="evol"),
        _sample_row(row_id="evol-validation-candidate-47", source="evol"),
        _sample_row(row_id="oss-train-candidate-0", source="oss"),
        _sample_row(row_id="oss-validation-candidate-1", source="oss"),
        _sample_row(row_id="stack_python_fns-train-candidate-0", source="stack_python_fns"),
        _sample_row(row_id="stack_python_fns-validation-candidate-42", source="stack_python_fns"),
    ]
    _write_raw_parquet(raw_path / "data" / "train-00000-of-00001.parquet", raw_rows)

    materialize_prompt_test_views(
        input_path=str(raw_path),
        output_path=str(output_path),
        views=(AceCodeV2View.VALIDATION_5PCT_SOURCE_STRATIFIED,),
    )

    validation_rows = _read_view_rows(output_path, AceCodeV2View.VALIDATION_5PCT_SOURCE_STRATIFIED)
    manifest = json.loads((output_path / "manifest.json").read_text())

    assert {row["id"] for row in validation_rows} == {
        "evol-validation-candidate-47",
        "oss-validation-candidate-1",
        "stack_python_fns-validation-candidate-42",
    }
    assert {row["split"] for row in validation_rows} == {"validation"}
    assert manifest["views"]["validation_5pct_source_stratified"]["rows"] == len(validation_rows)


def test_raw_acecode_v2_is_not_registered_as_instruction_sft():
    assert ACECODE_V2_HF_ID not in INSTRUCTION_DATASET_NAME_TO_CONFIG
