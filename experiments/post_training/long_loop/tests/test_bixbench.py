# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.long_loop.bixbench import merge_results, merge_verifier_recovery, prepare_subset


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


@pytest.fixture
def verifier_recovery(tmp_path):
    original, selection, recovery, source = [
        tmp_path / name for name in ("original", "selection", "recovery", "attempt")
    ]
    native = write_results(original, ["bixbench-pi", "other"], failed=["bixbench-pi"])
    row = json.loads((native / "bixbench-pi.jsonl").read_text())
    row["exception_type"] = "VerifierTimeoutError"
    row["verification"].update(score=None, passed=None, diagnostics={})
    (native / "bixbench-pi.jsonl").write_text(json.dumps(row) + "\n")
    (native / "aggregated_results.jsonl").write_text('{"native_coverage": 0.5}\n')
    (original / "terminal.json").write_text(json.dumps({"result": {"state": "succeeded"}}))
    (original / "resolved-launch.yaml").write_text('{"original_model": "checkpoint20"}')
    selection.mkdir()
    (selection / "selection.json").write_text(
        json.dumps(
            {
                "attempts": 2,
                "attempts_per_data_source": {"bixbench-pi": 1, "other": 1},
                "identities": [
                    {
                        "task_id": "bixbench-pi/task",
                        "dataset_row_uid": "1",
                        "data_source": "bixbench-pi",
                        "repetition_index": 0,
                    },
                    {"task_id": "other/task", "dataset_row_uid": "1", "data_source": "other", "repetition_index": 0},
                ],
            }
        )
    )
    agent = source / "trials/attempt/agent"
    agent.mkdir(parents=True)
    answer = b"<answer>original answer</answer>\n"
    trajectory = json.dumps(
        {
            "steps": [
                {
                    "tool_calls": [{"tool_call_id": "call1", "arguments": {"command": "cat /workspace/answer.txt"}}],
                    "observation": {
                        "results": [
                            {"source_call_id": "call1", "content": answer.decode(), "extra": {"is_error": False}}
                        ]
                    },
                }
            ]
        }
    ).encode()
    (agent / "trajectory.json").write_bytes(trajectory)
    (source / "identity.json").write_text(
        json.dumps({"task_id": "bixbench-pi/task", "attempt_id": "attempt", "session_id": "eval-None-0--1_0"})
    )
    (agent.parent / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {"exception_type": "VerifierTimeoutError"},
                "verifier_result": None,
                "config": {"verifier": {"override_timeout_sec": 300}},
            }
        )
    )
    tests = source / "task/tests"
    tests.mkdir(parents=True)
    (tests / "llm_judge.py").write_bytes(b"frozen verifier")
    (tests / "ground_truth.json").write_bytes(b"frozen target")
    recovery.mkdir()
    result = {
        "status": "completed",
        "returncode": 0,
        "candidate_resampled": False,
        "original_results_mutated": False,
        "task_id": "bixbench-pi/task",
        "original_attempt": "attempt",
        "source": str(source),
        "source_trajectory_sha256": hashlib.sha256(trajectory).hexdigest(),
        "verifier_result": {"rewards": {"reward": 0}},
        "verifier_timeout_seconds": 300,
        "judge_model": "same judge",
        "judge_revision": "b" * 40,
    }
    payloads = {
        "result.json": json.dumps(result).encode(),
        "grade.json": b'{"rewards":{"reward":0}}',
        "answer.txt": answer,
        **{name: (tests / name).read_bytes() for name in ("llm_judge.py", "ground_truth.json")},
    }
    for name, raw in payloads.items():
        (recovery / name).write_bytes(raw)
    manifest = json.dumps(
        {
            "files": [
                {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                for name, raw in payloads.items()
            ]
        }
    ).encode()
    (recovery / "manifest.json").write_bytes(manifest)
    return {
        "original": str(original),
        "selection": str(selection),
        "recovery": str(recovery),
        "recovery_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "accepted_unscored_attempts": [],
    }


def test_verifier_only_view_preserves_native_failure_and_sampled_evidence(tmp_path, verifier_recovery):
    original = tmp_path / "original"
    native = original / "exports/dumped_evals/eval_only"
    before = {path.name: path.read_bytes() for path in native.glob("*.jsonl")}
    output = tmp_path / "merged"
    merge_verifier_recovery(output=str(output), **verifier_recovery)
    merge_verifier_recovery(output=str(output), **verifier_recovery)
    assert {path.name: path.read_bytes() for path in native.glob("*.jsonl")} == before
    view = output / "exports/dumped_evals/eval_only"
    assert (view / "other.jsonl").read_bytes() == before["other.jsonl"]
    assert not (view / "aggregated_results.jsonl").exists()
    assert (output / "original-native/aggregated_results.jsonl").read_bytes() == before["aggregated_results.jsonl"]
    recovered = json.loads((view / "bixbench-pi.jsonl").read_text())
    original_row = json.loads(before["bixbench-pi.jsonl"])
    assert recovered["response_token_ids"] == original_row["response_token_ids"]
    assert recovered["verification"]["score"] == 0
    assert recovered["verifier_recovery"]["original_verification"] == original_row["verification"]
    assert recovered["exception_type"] == "VerifierTimeoutError"
    assert json.loads((output / "evaluation.json").read_text())["all_verified"]
    for name in ("terminal.json", "resolved-launch.yaml"):
        assert (output / name).read_bytes() == (original / name).read_bytes()


@pytest.mark.parametrize("changed", ["identity", "verifier"])
def test_verifier_recovery_rejects_wrong_attempt_or_changed_frozen_grader(tmp_path, verifier_recovery, changed):
    if changed == "identity":
        path = tmp_path / "attempt/identity.json"
        identity = json.loads(path.read_text())
        path.write_text(json.dumps({**identity, "session_id": "eval-None-0--another_0"}))
        message = "another selected attempt"
    else:
        (tmp_path / "attempt/task/tests/llm_judge.py").write_text("changed verifier")
        message = "original frozen verifier"
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=message):
        merge_verifier_recovery(output=str(output), **verifier_recovery)
    assert not output.exists()


def test_verifier_recovery_preserves_only_explicitly_accepted_unscored_attempts(tmp_path, verifier_recovery):
    native = tmp_path / "original/exports/dumped_evals/eval_only/other.jsonl"
    row = json.loads(native.read_text())
    row["exception_type"] = "VerifierTimeoutError"
    row["verification"].update(status="error", score=None, passed=None)
    native.write_text(json.dumps(row) + "\n")
    original_bytes = native.read_bytes()
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="additional unscored"):
        merge_verifier_recovery(output=str(output), **verifier_recovery)
    assert not output.exists()
    accepted = [{"data_source": "other", "uid": "1", "exception_type": "VerifierTimeoutError"}]
    merge_verifier_recovery(output=str(output), **{**verifier_recovery, "accepted_unscored_attempts": accepted})
    report = json.loads((output / "evaluation.json").read_text())
    assert report["completed_rollouts"] == 2
    assert report["verified_rollouts"] == 1
    assert not report["all_verified"]
    assert report["unscored_attempts"] == report["accepted_unscored_attempts"] == accepted
    assert (output / "exports/dumped_evals/eval_only/other.jsonl").read_bytes() == original_bytes
    assert native.read_bytes() == original_bytes
