# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import runpy
from pathlib import Path

import fsspec
import pytest
import yaml

combine = runpy.run_path(str(Path(__file__).with_name("combine.py")))["combine"]
validate_shards = runpy.run_path(str(Path(__file__).with_name("combine.py")))["validate_shards"]
partition = runpy.run_path(str(Path(__file__).with_name("shard.py")))["partition"]
freeze_shards = runpy.run_path(str(Path(__file__).with_name("combine.py")))["freeze_shards"]


def test_recovery_coverage_uses_full_cohort_and_excludes_transport_errors():
    retained = [{"task_name": str(i), "reward": 0, "exception_type": None} for i in range(437)]
    manifest = {"retained_trials": retained, "recovery_tasks": [str(i) for i in range(437, 500)]}
    recovered = [{"task_name": str(i), "reward": 1, "exception_type": None} for i in range(437, 450)]
    transport = {
        "task_name": "450",
        "reward": 0,
        "exception_type": "NonZeroAgentExitCodeError",
        "agent_logs": [{"terminal_error_classes": ["httpx.ReadTimeout"]}],
    }
    result = combine(manifest, retained, [*recovered, transport])
    assert result["n_scored"] == 450
    assert result["n_unscored"] == 1
    assert result["mean_reward"] == pytest.approx(13 / 450)
    assert result["sealable"]
    assert len(result["pending_tasks"]) == 49
    assert not combine(manifest, retained, [*recovered[:-1], transport])["sealable"]
    with pytest.raises(ValueError, match="duplicate"):
        combine(manifest, retained, [*recovered, recovered[0]])


def test_sharded_recovery_preserves_scores_and_assigns_unscored_tasks_once():
    retained = [{"task_name": str(i), "reward": i % 2, "exception_type": None} for i in range(100)]
    manifest = {"retained_trials": retained, "recovery_tasks": [str(i) for i in range(100, 500)]}
    recovered = [{"task_name": str(i), "reward": 1, "exception_type": None} for i in range(100, 130)]
    recovered.append({"task_name": "130", "reward": 0, "exception_type": "AgentSetupTimeoutError"})
    report = {"manifest": manifest, **combine(manifest, retained, recovered)}
    config = {
        "datasets": [{"name": "swebench-verified", "task_names": manifest["recovery_tasks"]}],
        "agent": {"max_input_tokens": 131072},
    }
    next_manifest, shards = partition(report, config, 3)
    assert len(next_manifest["retained_trials"]) == 130
    assignments = [name for shard in shards for name in shard["datasets"][0]["task_names"]]
    assert len(assignments) == len(set(assignments)) == 370
    assert "130" in assignments
    shard_results = [{"task_name": name, "reward": 0, "exception_type": None} for name in assignments]
    final = combine(next_manifest, next_manifest["retained_trials"], shard_results)
    assert final["n_scored"] == 500
    assert final["mean_reward"] == pytest.approx(80 / 500)

    base = copy.deepcopy(config)
    base["datasets"][0]["task_names"] = next_manifest["recovery_tasks"]
    overlapping = copy.deepcopy(shards)
    overlapping[1]["datasets"][0]["task_names"].append(assignments[0])
    with pytest.raises(ValueError, match="overlapping"):
        validate_shards(base, overlapping, next_manifest["recovery_tasks"])
    with pytest.raises(ValueError, match="complete"):
        validate_shards(base, shards[:-1], next_manifest["recovery_tasks"])
    drifted = copy.deepcopy(shards)
    drifted[0]["agent"]["max_input_tokens"] = 32768
    with pytest.raises(ValueError, match="settings"):
        validate_shards(base, drifted, next_manifest["recovery_tasks"])


def test_freeze_shards_reads_each_root_and_rejects_wrong_runtime(tmp_path, monkeypatch):
    fs = fsspec.filesystem("memory")
    prefix = str(tmp_path)
    monkeypatch.setitem(freeze_shards.__globals__, "filesystem_for", lambda url: (fs, url.removeprefix("s3://")))
    retained_payload = json.dumps({"task_name": "0", "verifier_result": {"rewards": {"reward": 1}}}).encode()
    retained_path = prefix + "/retained/result.json"
    fs.pipe(retained_path, retained_payload)
    retained = [
        {
            "task_name": "0",
            "reward": 1,
            "exception_type": None,
            "source_uri": "s3://" + retained_path,
            "sha256": hashlib.sha256(retained_payload).hexdigest(),
        }
    ]
    tasks = [str(i) for i in range(1, 500)]
    config = {"datasets": [{"task_names": tasks}], "agent": {"name": "mini-swe-agent"}}
    manifest = {
        "model": "test-model",
        "retained_trials": retained,
        "recovery_tasks": tasks,
        "recovery_runtime": {"pin": "fixed"},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_path.with_name("swebench-verified-recovery.yaml").write_text(yaml.safe_dump(config))
    shards = []
    for index in range(2):
        shard_config = copy.deepcopy(config)
        shard_config["datasets"][0]["task_names"] = tasks[index::2]
        path = tmp_path / f"shard-{index}.yaml"
        path.write_text(yaml.safe_dump(shard_config))
        root = prefix + f"/shard-{index}"
        fs.pipe(root + "/harbor_resume_identity.json", json.dumps({"runtime": {"pin": "fixed"}}).encode())
        fs.pipe(root + "/harbor_jobs/job/config.json", json.dumps(shard_config).encode())
        fs.pipe(
            root + "/harbor_jobs/job/task__run/result.json",
            json.dumps({"task_name": tasks[index], "verifier_result": {"rewards": {"reward": index}}}).encode(),
        )
        shards.append({"config": str(path), "results": "s3://" + root})
    report = freeze_shards(manifest_path, shards, None)
    assert report["n_scored"] == 3
    assert report["mean_reward"] == pytest.approx(2 / 3)
    assert len(report["pending_tasks"]) == 497
    assert [s["results"] for s in report["shards"]] == [s["results"] for s in shards]
    fs.pipe(prefix + "/shard-1/harbor_resume_identity.json", b'{"runtime":{"pin":"wrong"}}')
    with pytest.raises(ValueError, match="runtime"):
        freeze_shards(manifest_path, shards, None)
