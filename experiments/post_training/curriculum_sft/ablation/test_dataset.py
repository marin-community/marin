# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow.dataset as ds

from experiments.post_training.curriculum_sft.ablation.dataset import (
    GENERATION_FILENAME,
    MaterializeDatasetConfig,
    materialize_dataset,
)
from experiments.post_training.curriculum_sft.ablation.matrix import (
    AblationCell,
    CurriculumCondition,
    GenerationSpec,
)


def test_materialize_dataset_produces_rendered_parquet(tmp_path):
    cell = AblationCell(CurriculumCondition.TASK_ONLY, GenerationSpec.WEAK, accepted_examples=1)
    payload = {
        "task_id": "example",
        "issuer": "Fictional issuer",
        "question": "Revenue is 80 and operating cost is 100. What is gross profit?",
        "facts": {"revenue": 100, "operating_cost": 80},
        "answer": {"gross_profit": 20, "margin_bps": 2000},
        "evidence": ["disclosure.revenue", "disclosure.operating_cost"],
    }
    ledger = {
        "batch_id": "batch-test",
        "cells": [
            {
                "cell": cell.name,
                "tasks": [payload],
                "requested": 1,
                "accepted": 1,
                "unique_accepted": 1,
                "format_rate": 1.0,
                "arithmetic_rate": 1.0,
                "evidence_rate": 1.0,
                "replicates": 1,
            }
        ],
    }
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    (generation_root / GENERATION_FILENAME).write_text(json.dumps(ledger))

    result = materialize_dataset(
        MaterializeDatasetConfig(
            generation_root=str(generation_root),
            output_path=str(tmp_path / "output"),
            cell=cell,
        )
    )

    table = ds.dataset(result.main_output_dir, format="parquet").to_table()
    assert table.column_names == ["id", "text", "source_id"]
    assert table.num_rows == 1
    [text] = table["text"].to_pylist()
    assert "revenue of 100" in text
    assert "operating cost of 80" in text
    assert "Revenue is 80" not in text
