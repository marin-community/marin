# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from marin.rl.skyrl import ResolvedModelLocator, SkyRLOutputPaths, SkyRLRun, SkyRLRunConfig

from experiments.post_training.inventory import loop_result


@pytest.fixture
def run_config(tmp_path):
    export = tmp_path / "exports"
    checkpoints = tmp_path / "checkpoints"
    export.mkdir()
    checkpoints.mkdir()
    return loop_result.LoopRunConfig(
        skyrl=SkyRLRunConfig(
            launch_config_yaml=yaml.safe_dump(
                {
                    "skyrl": {"data": {"expected_eval_rows": 2}, "generator": {"eval_n_samples_per_prompt": 1}},
                    "inputs": {"train_data": [{"identity": "training@1"}], "validation_data": [{"identity": "panel@1"}]},
                }
            ),
            run_id="fixture/run",
            attempt_id="attempt-1",
            model=ResolvedModelLocator("model://fixture", "initializer@1", "local-model", "tokenizer", "revision"),
            output=SkyRLOutputPaths(str(checkpoints), str(export), str(tmp_path / "attempts"), "config", "terminal"),
            export_hf=False,
            draft_checkpoint_root=None,
        ),
        label="Fixture loop",
        expected_steps=2,
    )


def write_observations(config, scenario):
    phase = {"eval/all/num_attempted": 2, "eval/all/num_scored": 2, "eval/all/avg_verifier_score": 0.0}
    baseline = phase | ({"eval/all/num_scored": 1} if scenario == "missing_grade" else {})
    steps = [
        {
            "step": step,
            "metrics": {
                "policy/policy_update_steps": 1,
                "policy/skipped_steps": 0,
                "policy/raw_grad_norm": 0.1,
                "policy/policy_loss": 0.01,
            },
        }
        for step in (1, 2)
    ]
    if scenario == "nonfinite_loss":
        steps[0]["metrics"]["policy/policy_loss"] = float("nan")
    if scenario == "missing_step":
        steps.pop()
    if scenario == "skipped_extra_step":
        steps.append({"step": 3, "metrics": {"policy/policy_update_steps": 0, "policy/skipped_steps": 1}})
    start = datetime.now(UTC) - (timedelta(days=1) if scenario == "stale" else timedelta())
    data = {
        "schema_version": 1,
        "expected_steps": 2,
        "started_at": start.isoformat(),
        "steps": steps,
        "evaluations": {"0": baseline, "2": phase},
        "wandb_url": "https://wandb.ai/example/project/runs/fixture",
    }
    Path(config.skyrl.output.export_root, "loop-evidence.json").write_text(json.dumps(data))
    if scenario != "missing_checkpoint":
        Path(config.skyrl.output.checkpoint_root, "latest_ckpt_global_step.txt").write_text("2")


@pytest.mark.parametrize(
    "scenario",
    ["complete", "missing_grade", "missing_step", "skipped_extra_step", "missing_checkpoint", "stale", "nonfinite_loss"],
)
def test_terminal_summary_requires_native_success_complete_evidence_and_checkpoint(monkeypatch, run_config, scenario):
    def train(config):
        write_observations(run_config, scenario)
        return SkyRLRun(
            path="terminal",
            hf_model_uri=None,
            global_step=None,
            tokenizer_uri="tokenizer",
            tokenizer_revision="revision",
            checkpoint_root=config.output.checkpoint_root,
            draft_checkpoint_root=None,
            terminal_manifest_uri="terminal",
            iris_job_id="fixture-job",
        )

    monkeypatch.setattr(loop_result, "run_skyrl", train)
    if scenario == "complete":
        result = loop_result.run_loop(run_config)
        assert result.iris_job_id == "fixture-job"
    else:
        with pytest.raises(RuntimeError, match="loop evidence is incomplete"):
            loop_result.run_loop(run_config)
    summary = json.loads(Path(run_config.skyrl.output.export_root, "loop-result.json").read_text())
    assert summary["status"] == ("succeeded" if scenario == "complete" else "failed")
    assert summary["evaluation"]["protocol_sha256"] is None
    assert summary["evaluation"]["task_set_sha256"] is None
    assert summary["gpu_hours"] is None
    assert summary["date_basis"] == "launch_started"
    assert summary["timing_scope"] == "loop_launcher_to_terminal"
    if scenario == "stale":
        assert summary["baseline"] is None
        assert summary["final"] is None
    elif scenario == "missing_grade":
        assert summary["baseline"]["scored"] == 1
    else:
        assert summary["baseline"]["reward"] == 0.0


def test_training_error_survives_summary_persistence_without_fabricated_scores(monkeypatch, run_config):
    failure = RuntimeError("remote launcher failed with private arguments")

    def train(config):
        raise failure

    monkeypatch.setattr(loop_result, "run_skyrl", train)
    with pytest.raises(RuntimeError) as caught:
        loop_result.run_loop(run_config)
    assert caught.value is failure
    summary = json.loads(Path(run_config.skyrl.output.export_root, "loop-result.json").read_text())
    assert summary["status"] == "failed"
    assert summary["baseline"] is summary["final"] is None
    assert "private arguments" not in json.dumps(summary)
