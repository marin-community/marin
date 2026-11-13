# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pytest

from experiments.post_training.long_loop.eval_runtime.selection import selected_rollouts


def test_published_repetitions_survive_selection_and_missing_duplicate_attempts_fail():
    unique = [
        {
            "task_id": f"task-{i}",
            "benchmark": "aime24",
            "agent": "single-turn",
            "repetitions": 3,
            "task_spec": json.dumps({"source": {"dataset": "zwhe99/aime24"}}),
            "task_rollout_spec": "{}",
        }
        for i in range(2)
    ]
    expanded = [{**row, "repetition_index": rep} for row in unique for rep in range(3)]
    table, receipt = selected_rollouts(pa.Table.from_pylist(unique), pa.Table.from_pylist(expanded), 1)
    assert table.num_rows == 3
    assert receipt["unique_tasks"] == 1
    assert receipt["attempts_per_data_source"] == {"zwhe99/aime24": 3}
    assert [(x["dataset_row_uid"], x["repetition_index"]) for x in receipt["identities"]] == [
        ("0", 0),
        ("1", 1),
        ("2", 2),
    ]
    for bad, message in ((expanded[1:], "incomplete"), ([expanded[0], *expanded], "duplicates")):
        with pytest.raises(ValueError, match=message):
            selected_rollouts(pa.Table.from_pylist(unique), pa.Table.from_pylist(bad), 1)


def test_benchmark_filter_selects_only_the_requested_paths():
    unique = [
        {
            "task_id": f"{benchmark}-{i}",
            "benchmark": benchmark,
            "agent": "pi" if benchmark == "tb2-recovery" else "single-turn",
            "repetitions": 1,
            "task_spec": json.dumps({"source": {"dataset": benchmark}}),
            "task_rollout_spec": "{}",
        }
        for benchmark in ("aime24", "tb2-recovery")
        for i in range(2)
    ]
    expanded = [{**row, "repetition_index": 0} for row in unique]
    unique_table, expanded_table = pa.Table.from_pylist(unique), pa.Table.from_pylist(expanded)
    table, receipt = selected_rollouts(unique_table, expanded_table, 2, frozenset({"tb2-recovery"}))
    assert set(table["benchmark"].to_pylist()) == {"tb2-recovery"}
    assert receipt["tasks_per_benchmark"] == {"tb2-recovery": 2}
    assert selected_rollouts(unique_table, expanded_table, 2)[1]["paths"] == 2
    for bad in (frozenset(), frozenset({"not-a-path"})):
        with pytest.raises(ValueError, match="not supported paths"):
            selected_rollouts(unique_table, expanded_table, 2, bad)
