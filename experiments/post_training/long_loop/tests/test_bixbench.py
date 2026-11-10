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
            "data_source": name,
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
    for root in (original, repair):
        pq.write_table(pa.table({"benchmark": ["bixbench-pi"], "task": ["same-task"]}), root / "rollouts.parquet")
        config = {
            "artifacts": {},
            "run": {},
            "iris": {"job_name": root.name},
            "ray": {"rendezvous_dir": "local", "log_dir": "local"},
            "inputs": {
                "validation_data": [{"uri": str(root), "relative_path": "rollouts.parquet"}],
                "model": {"uri": "pinned-model", "tokenizer_revision": "pinned-tokenizer", "identity": root.name},
            },
            "runtime": {"task_env": {"JUDGE_BASE_URL": "run-endpoint"}, "image": "pinned-image"},
            "skyrl": {
                "data": {"val_data": []},
                "terminal_bench_config": {"installed_agent": {"output_path": "local"}},
                "environment": {"task_sessions": {"remote_judge": {"base_url": "endpoint", "model": "judge-pin"}}},
                "trainer": {
                    "export_path": "local",
                    "ckpt_path": "local",
                    "run_name": root.name,
                    **{
                        role: {
                            "model": {
                                "revision": "adopted",
                                "tokenizer_revision": "adopted",
                                "source_identity": "adopted",
                            }
                        }
                        for role in ("policy", "ref")
                    },
                },
                "generator": {
                    "trajectory_retention": {
                        "run_id": root.name,
                        "model_source_identity": "adopted",
                        "output_path": "local",
                    }
                },
            },
        }
        (root / "terminal.json").write_text(json.dumps({"config": config}))
    # A repaired trajectory may reach more turns than the failed original.
    repaired = json.loads((after / "bixbench-pi.jsonl").read_text())
    repaired["uid"] = "0"
    repaired["prompt_token_ids"] = [3, 7, 11, 12]
    (after / "bixbench-pi.jsonl").write_text(json.dumps(repaired) + "\n")
    output = tmp_path / "merged"
    merge_results(output=str(output), original=str(original), repair=str(repair))
    for name in names[1:]:
        assert (output / "exports/dumped_evals/eval_only" / (name + ".jsonl")).read_bytes() == (
            before / (name + ".jsonl")
        ).read_bytes()
    assert json.loads((output / "evaluation.json").read_text())["all_verified"]
    config["runtime"]["image"] = "wrong-image"
    (repair / "terminal.json").write_text(json.dumps({"config": config}))
    with pytest.raises(ValueError, match="model/runtime"):
        merge_results(output=str(tmp_path / "bad-runtime"), original=str(original), repair=str(repair))
    pq.write_table(pa.table({"benchmark": ["bixbench-pi"], "task": ["different-task"]}), repair / "rollouts.parquet")
    with pytest.raises(ValueError, match="task identity"):
        merge_results(output=str(tmp_path / "bad"), original=str(original), repair=str(repair))
