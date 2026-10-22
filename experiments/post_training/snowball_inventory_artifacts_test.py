# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fray.current_client import set_current_client
from fray.local_backend import LocalClient, LocalJobHandle
from marin.rl.skyrl import ResolvedModelLocator, SkyRLOutputPaths, SkyRLRun, SkyRLRunConfig

from experiments.post_training import snowball_inventory_compare as comparison
from experiments.post_training import snowball_inventory_evaluate as evaluation
from experiments.post_training import snowball_inventory_smoke as inventory
from experiments.post_training.snowball_inventory_panel import FAMILY_COUNTS


class IsolatedPreflightClient(LocalClient):
    """Execute a submitted callable with only the environment sent to its child."""

    def __init__(self):
        self.submissions = 0

    def submit(self, request, adopt_existing=True):
        self.submissions += 1
        entry = request.entrypoint.callable_entrypoint
        assert entry is not None
        with patch.dict(os.environ, request.environment.env_vars, clear=True):
            entry.callable(*entry.args, **entry.kwargs)
        future = Future()
        future.set_result(None)
        return LocalJobHandle("isolated-preflight", future)


@pytest.fixture
def launch(tmp_path):
    document = {
        "skyrl": {
            "environment": {
                "task_sessions": {
                    "remote_judge": {"api_key_env": "JUDGE_API_KEY"},
                    "daytona": {"api_key_env": "DAYTONA_EVAL_API_KEY"},
                }
            },
            "trainer": {"max_steps": 50},
        }
    }
    return SkyRLRunConfig(
        launch_config_yaml=yaml.safe_dump(document),
        run_id="fixture",
        attempt_id="attempt",
        model=ResolvedModelLocator("model", "manifest", "model", "tokenizer", "revision"),
        output=SkyRLOutputPaths(
            checkpoint_root=str(tmp_path / "checkpoints"),
            export_root=str(tmp_path / "exports"),
            attempts_root=str(tmp_path / "attempts"),
            resolved_config_uri=str(tmp_path / "resolved-launch.yaml"),
            terminal_manifest_uri=str(tmp_path / "terminal.json"),
        ),
        export_hf=True,
        draft_checkpoint_root=None,
    )


def completed_run(launch):
    return SkyRLRun(
        path=str(Path(launch.output.terminal_manifest_uri).parent),
        hf_model_uri=str(Path(launch.output.export_root) / "policy"),
        global_step=yaml.safe_load(launch.launch_config_yaml)["skyrl"]["trainer"]["max_steps"],
        tokenizer_uri=launch.model.tokenizer_uri,
        tokenizer_revision=launch.model.tokenizer_revision,
        checkpoint_root=launch.output.checkpoint_root,
        draft_checkpoint_root=None,
        terminal_manifest_uri=launch.output.terminal_manifest_uri,
        iris_job_id="fixture-job",
    )


@pytest.mark.parametrize(
    "kind,score",
    [
        ("inventory", 1.0),
        ("evaluation", 1.0),
        ("recoverable_evaluation", 1.0),
        ("evaluation", None),
        ("recoverable_evaluation", None),
    ],
)
def test_nested_preflight_credentials_preserve_evaluation_diagnostics(kind, score, launch, tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "test-judge-key")
    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "test-eval-key")
    monkeypatch.setenv("HF_TOKEN", "test-hf-key")
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setenv("UNRELATED_SECRET", "must-stay-in-coordinator")
    evidence = tmp_path / "child-admission.json"

    def admitted():
        # This fake grader boundary fails if either credential is lost by Fray.
        assert os.environ["JUDGE_API_KEY"] == "test-judge-key"
        assert os.environ["DAYTONA_EVAL_API_KEY"] == "test-eval-key"
        assert "UNRELATED_SECRET" not in os.environ
        evidence.write_text(json.dumps({"judge_probe": "passed", "verifier_probe": "passed"}))

    def inventory_preflight(*args):
        admitted()

    def evaluation_preflight(config):
        admitted()
        report = {
            "launch_config_yaml": config.launch.launch_config_yaml,
            "dataset_uids": {"uid": "task"},
            "tasks": [{"task_id": "task", "evaluation_family": "calendar", "contract_id": "contract"}],
            "model_manifest_identity": "manifest",
            "runtime_tree_hash": "fixture-runtime",
            "protocol": {},
            "pairing_contract_field": "source_contract_id",
        }
        (tmp_path / "preflight.json").write_text(json.dumps(report))

    def trained(config):
        assert evidence.exists()
        return completed_run(config)

    def read_runtime(action, request, **kwargs):
        assert evidence.exists()
        if action == "generation-state":
            return {"complete": True}
        return {
            "outcomes": [{"task_id": "task", "score": score, "stop_reason": "stop"}],
            "source_dumps": [],
            "recovery": {"attempted": 0},
        }

    monkeypatch.setattr(inventory, "verify_pool_runtime", inventory_preflight)
    monkeypatch.setattr(inventory, "run_skyrl", trained)
    monkeypatch.setattr(evaluation, "preflight_evaluation", evaluation_preflight)
    monkeypatch.setattr(evaluation, "run_skyrl", trained)
    monkeypatch.setattr(evaluation, "runtime_operation", read_runtime)
    client = IsolatedPreflightClient()
    with set_current_client(client):
        if kind == "inventory":
            result = inventory.run_inventory(inventory.InventoryRunConfig(launch, "pool-sha"))
            assert result.global_step == 50
        else:
            config = evaluation.EvaluationConfig(
                launch,
                "panel-sha",
                str(tmp_path),
                recovery=evaluation.SavedAnswerRecovery(600, 900) if kind == "recoverable_evaluation" else None,
            )
            runner = evaluation.run_evaluation if kind == "evaluation" else evaluation.run_recoverable_evaluation
            if kind == "evaluation" and score is None:
                with pytest.raises(ValueError):
                    runner(config)
            else:
                result = runner(config)
                assert result.complete == (score is not None)
            report = json.loads((tmp_path / "report.json").read_text())
            assert report["families"]["calendar"]["scored"] == (0 if score is None else 1)
            assert report["complete"] == (score is not None)
            assert report["tasks"][0]["score"] == score
            if kind == "recoverable_evaluation":
                assert report["recovery"] == {"attempted": 0}
                assert report["required_score_coverage"] == 1.0
    assert client.submissions == 1
    assert json.loads(evidence.read_text()) == {"judge_probe": "passed", "verifier_probe": "passed"}


@pytest.mark.parametrize("max_steps", [50, 75])
@pytest.mark.parametrize("missing_final_grade", [False, True])
def test_periodic_comparison_persists_fixed_pairing_and_coverage_before_return(
    max_steps, missing_final_grade, launch, tmp_path, monkeypatch
):
    document = yaml.safe_load(launch.launch_config_yaml)
    document["skyrl"]["trainer"]["max_steps"] = max_steps
    launch = replace(launch, launch_config_yaml=yaml.safe_dump(document))
    steps = tuple(range(0, max_steps + 1, 25))
    rows = [
        {
            "task_id": f"{family}-{index}",
            "evaluation_family": family,
            "source_contract_id": f"original-{family}-{index}",
            "dataset_uid": f"uid-{family}-{index}",
        }
        for family, count in FAMILY_COUNTS.items()
        for index in range(count)
    ]
    probe = {
        "status": "passed",
        "validation_family_counts": FAMILY_COUNTS,
        "validation_tasks": rows,
        "validation_sha256": "panel-sha",
    }

    def trained(config):
        (tmp_path / "probe").mkdir()
        (tmp_path / "probe" / "report.json").write_text(json.dumps(probe))
        return completed_run(config.training)

    def read_saved_results(action, request):
        assert action == "results"
        step = request["evaluation_step"]
        return {
            "outcomes": [
                {
                    "task_id": row["task_id"],
                    "score": None if missing_final_grade and step == max_steps and index == 0 else 1.0,
                    "stop_reason": "stop",
                }
                for index, row in enumerate(reversed(rows))
            ],
            "source_dumps": [{"uri": f"retained-step-{step}.jsonl", "sha256": f"dump-{step}"}],
        }

    monkeypatch.setattr(comparison, "run_inventory", trained)
    monkeypatch.setattr(comparison, "runtime_operation", read_saved_results)
    artifact = comparison.comparison_step(
        comparison.Initializer.SFT,
        version="2026.10.22",
        model_uri=str(tmp_path / "initializer"),
        pool_uri=str(tmp_path / "pool"),
        pool_version="2026.10.22",
        pool_sha256="pool-sha",
        validation_uri=str(tmp_path / "panel"),
        validation_version="2026.10.22",
        validation_sha256="panel-sha",
        judge_base_url="https://judge.example/v1",
        judge_model="fixture-judge",
        max_steps=max_steps,
    )
    config = inventory.InventoryRunConfig(launch, "pool-sha")
    if missing_final_grade:
        with pytest.raises(ValueError):
            artifact.run(config)
    else:
        assert artifact.run(config).global_step == max_steps
    reports = [json.loads((tmp_path / "evaluation" / f"step-{step}" / "report.json").read_text()) for step in steps]
    assert len({report["pairing_identity"] for report in reports}) == 1
    for step, report in zip(steps, reports, strict=True):
        assert report["step"] == step
        assert report["panel_sha256"] == "panel-sha"
        assert {family: row["expected"] for family, row in report["families"].items()} == FAMILY_COUNTS
        assert [row["task_id"] for row in report["tasks"]] == [row["task_id"] for row in rows]
        assert report["source_dumps"] == [{"uri": f"retained-step-{step}.jsonl", "sha256": f"dump-{step}"}]
        assert report["complete"] == (not missing_final_grade or step != max_steps)
    assert reports[-1]["aggregate_fixed_denominator_mean"] == pytest.approx(909 / 910 if missing_final_grade else 1.0)
