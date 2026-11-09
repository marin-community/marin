# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.long_loop.bixbench import merge_results, prepare_subset


def test_subset_preserves_original_task_and_environment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    table = pa.table({"benchmark": ["other", "bixbench-pi"], "task": ["other-task", "original-task"]})
    pq.write_table(table, source / "rollouts.parquet")
    (source / "environment.json").write_text(json.dumps({"data_uri": str(source), "runtime_pin": "original"}))
    output = tmp_path / "subset"
    prepare_subset(output=str(output), source=str(source))
    assert pq.read_table(output / "rollouts.parquet").to_pylist() == [table.to_pylist()[1]]
    assert json.loads((output / "environment.json").read_text())["runtime_pin"] == "original"


def write_results(root, names, failed=()):
    target = root / "exports/dumped_evals/eval_only"
    target.mkdir(parents=True)
    for name in names:
        row = {
            "uid": "1",
            "prompt_token_ids": [3, 7],
            "response_token_ids": [8],
            "stop_reason": "stop",
            "verification": {"status": "error" if name in failed else "verified", "score": 0.0},
        }
        (target / (name + ".jsonl")).write_text(json.dumps(row) + "\n")
    return target


def test_merge_preserves_nineteen_results_and_rejects_changed_task(tmp_path):
    original = tmp_path / "original"
    repair = tmp_path / "repair"
    names = ["bixbench-pi"] + [f"benchmark-{i}" for i in range(19)]
    before = write_results(original, names, failed=["bixbench-pi"])
    after = write_results(repair, ["bixbench-pi"])
    # Dataset-local indices change in the subset; the exact prompt must not.
    repaired = json.loads((after / "bixbench-pi.jsonl").read_text())
    repaired["uid"] = "0"
    (after / "bixbench-pi.jsonl").write_text(json.dumps(repaired) + "\n")
    output = tmp_path / "merged"
    merge_results(output=str(output), original=str(original), repair=str(repair))
    for name in names[1:]:
        assert (output / "exports/dumped_evals/eval_only" / (name + ".jsonl")).read_bytes() == (
            before / (name + ".jsonl")
        ).read_bytes()
    assert json.loads((output / "evaluation.json").read_text())["all_verified"]
    repaired["prompt_token_ids"] = [99]
    (after / "bixbench-pi.jsonl").write_text(json.dumps(repaired))
    with pytest.raises(ValueError, match="task identity"):
        merge_results(output=str(tmp_path / "bad"), original=str(original), repair=str(repair))
