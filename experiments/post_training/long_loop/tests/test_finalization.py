# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.skyrl import ResolvedModelLocator, SkyRLOutputPaths, SkyRLRun, SkyRLRunConfig, run_skyrl

from experiments.post_training.long_loop.finalization import after_checkpoint, record_finalization
from experiments.post_training.long_loop.graph import execute_local_graph


@pytest.mark.parametrize("state", ["failed", "killed"])
@pytest.mark.parametrize("checkpoint", [True, False])
def test_terminal_rl_queues_same_finalization_graph_or_records_no_checkpoint(tmp_path, monkeypatch, state, checkpoint):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    records = []
    monkeypatch.setattr("marin.rl.skyrl.record_rollout_run", records.append)
    output = SkyRLOutputPaths(
        *(str(tmp_path / name) for name in ("checkpoints", "exports", "attempts", "resolved.yaml", "terminal.json"))
    )
    model = ResolvedModelLocator("model", "pin", "/tmp/model", "tokenizer", "revision")
    config = SkyRLRunConfig(
        yaml.safe_dump({"run": {"finalization": "any_terminal"}}), "rl", "attempt", model, output, True, None
    )
    result = {
        "state": "failed",
        "iris_job_id": "/user/rl",
        "iris_job_state": state,
        "failure": "GroupAdmissionStalledError: no admitted groups",
        "finalization": "completed" if checkpoint else "skipped_no_checkpoint",
        "model": (
            None
            if not checkpoint
            else {
                "policy_export_uri": str(tmp_path / "exports/global_step_16/policy"),
                "global_step": 16,
                "tokenizer_uri": "tokenizer",
                "tokenizer_revision": "revision",
                "checkpoint_root": output.checkpoint_root,
                "terminal_manifest_uri": output.terminal_manifest_uri,
            }
        ),
    }

    def launch(command):
        Path(output.terminal_manifest_uri).write_text(json.dumps({"result": result}))
        return subprocess.CompletedProcess(command, 1, json.dumps(result), "")

    monkeypatch.setattr("marin.rl.skyrl._run_launcher", launch)
    trained = ArtifactStep("rl", "2026.11.13", SkyRLRun, run_skyrl, lambda _ctx: config)
    finalized = apply("finalize", record_finalization, version="2026.11.13", output=OUT, trained=trained)
    dispatched = []

    def dispatch(*, output, previous):
        dispatched.append(Path(output).parent.name)
        (Path(output) / "completed.json").write_text(json.dumps({"previous": previous}))

    previous = finalized
    for name in ("evaluation", "report", "archive"):
        previous = after_checkpoint(apply(name, dispatch, version="2026.11.13", output=OUT, previous=previous), trained)
    execute_local_graph(previous)

    assert records[0].status == "failed"
    manifest = json.loads((tmp_path / "finalize/2026.11.13/finalization.json").read_text())
    assert manifest["rl_terminal"]["iris_job_state"] == state
    assert manifest["rl_terminal"]["failure"] == result["failure"]
    if checkpoint:
        assert dispatched == ["evaluation", "report", "archive"]
        assert manifest["selected_step"] == 16
    else:
        assert not dispatched
        skipped = json.loads((tmp_path / "archive/2026.11.13/finalization-skipped.json").read_text())
        assert skipped["reason"] == "no_committed_best_checkpoint"
        assert skipped["rl_terminal"]["iris_job_state"] == state


def test_failed_export_does_not_look_like_no_checkpoint(tmp_path, monkeypatch):
    config = SkyRLRunConfig(
        yaml.safe_dump({"run": {"finalization": "any_terminal"}}),
        "run",
        "attempt",
        ResolvedModelLocator("model", "pin", "/tmp/model", "tokenizer", "revision"),
        SkyRLOutputPaths(*(str(tmp_path / name) for name in ("ckpt", "hf", "attempts", "config", "terminal"))),
        True,
        None,
    )
    response = {
        "state": "failed",
        "iris_job_state": "failed",
        "iris_job_id": "/user/rl",
        "failure": "RL stalled",
        "finalization": "failed",
        "finalization_failure": "export allocation failed",
    }
    monkeypatch.setattr("marin.rl.skyrl.record_rollout_run", lambda _record: None)
    monkeypatch.setattr(
        "marin.rl.skyrl._run_launcher", lambda command: subprocess.CompletedProcess(command, 1, json.dumps(response), "")
    )
    with pytest.raises(RuntimeError, match="RL stalled"):
        run_skyrl(config)
