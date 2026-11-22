# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.post_training.long_loop.eval_runtime.audit_results import audit


def test_same_total_cannot_hide_missing_benchmark_repetitions(tmp_path):
    directory = tmp_path / "exports/dumped_evals/eval_only"
    directory.mkdir(parents=True)
    row = {
        "data_source": "aime24",
        "verification": {"status": "verified", "score": 1},
        "response_token_ids": [1, 2],
        "stop_reason": "stop",
    }
    output = directory / "results.jsonl"
    output.write_text("\n".join(json.dumps(row) for _ in range(3)))
    expected = {"aime24": 2, "gpqa-diamond": 1}
    wrong = audit(str(tmp_path), expected_rollouts=3, expected_by_data_source=expected)
    assert wrong["completed_rollouts"] == 3
    assert not wrong["all_verified"]
    output.write_text("\n".join(json.dumps(item) for item in [row, row, {**row, "data_source": "gpqa-diamond"}]))
    corrected = audit(str(tmp_path), expected_rollouts=3, expected_by_data_source=expected)
    assert corrected["all_verified"]


def test_duplicate_attempt_cannot_replace_missing_identity(tmp_path):
    directory = tmp_path / "exports/dumped_evals/eval_only"
    directory.mkdir(parents=True)
    row = {
        "data_source": "dataset",
        "uid": "0",
        "verification": {"status": "verified", "score": 1},
        "response_token_ids": [1],
        "stop_reason": "stop",
    }
    output = directory / "results.jsonl"
    expected = [{"data_source": "dataset", "dataset_row_uid": str(i)} for i in range(2)]
    output.write_text("\n".join(json.dumps(row) for _ in range(2)))
    assert not audit(str(tmp_path), 2, {"dataset": 2}, expected)["all_verified"]
    output.write_text("\n".join(json.dumps({**row, "uid": str(i)}) for i in range(2)))
    assert audit(str(tmp_path), 2, {"dataset": 2}, expected)["all_verified"]
