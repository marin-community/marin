# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow.parquet as parquet

from experiments.post_training.cat_count_canary.data import TRAIN_FILENAME, VALIDATION_FILENAME
from experiments.post_training.task_sessions.validation import ValidationDataConfig, write_validation_data


def test_validation_rows_preserve_task_classes_and_answers_in_parquet(tmp_path: Path) -> None:
    write_validation_data(ValidationDataConfig(str(tmp_path), rows=6))

    answers = ["1", "12", "3", "30", "5", "56", "7", "20"]
    for filename, count in ((TRAIN_FILENAME, 6), (VALIDATION_FILENAME, 8)):
        rows = parquet.read_table(tmp_path / filename).to_pylist()
        assert len(rows) == count
        assert {row["env_class"] for row in rows} == {"cat_count", "gsm8k_multi_turn"}
        assert [row["reward_spec"]["ground_truth"] for row in rows] == answers[:count]
