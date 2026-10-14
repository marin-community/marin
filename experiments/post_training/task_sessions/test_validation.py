# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow.parquet as parquet
import pytest
from taskcompendium.grader import grader_config
from taskcompendium.models import TaskSpec

from experiments.post_training.cat_count_canary.data import TRAIN_FILENAME, VALIDATION_FILENAME
from experiments.post_training.task_sessions.validation import (
    MATH_SOURCE,
    ValidationDataConfig,
    build_run,
    write_validation_data,
)


def test_validation_parquet_preserves_task_classes_answers_and_splits(tmp_path: Path) -> None:
    write_validation_data(ValidationDataConfig(str(tmp_path), rows=6))

    answers = ["1", "12", "3", "30", "5", "56", "7", "20"]
    for filename, count, split in ((TRAIN_FILENAME, 6, "train"), (VALIDATION_FILENAME, 8, "validation")):
        rows = parquet.read_table(tmp_path / filename).to_pylist()
        assert len(rows) == count
        tasks = [TaskSpec.model_validate_json(row["task_spec"]) for row in rows]
        extras = [grader_config(task)["extras"] for task in tasks]
        assert MATH_SOURCE in {task.source.dataset for task in tasks}
        assert [str(extra["reward_spec"]["ground_truth"]) for extra in extras] == answers[:count]
        assert {extra["extra_info"]["split"] for extra in extras} == {split}


@pytest.mark.parametrize("mode", ["whole", "step"])
def test_validation_changed_steps_do_not_reuse_cached_output(mode: str) -> None:
    first = build_run(mode=mode, version="2026.10.12", steps=10)
    unchanged = build_run(mode=mode, version="2026.10.12", steps=10)
    changed = build_run(mode=mode, version="2026.10.12", steps=20)

    assert first.path("s3://validation") == unchanged.path("s3://validation")
    assert first.path("s3://validation") != changed.path("s3://validation")
