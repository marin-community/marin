# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
import yaml
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from experiments.post_training.snowball_inventory_evaluate import EvaluationOutcome, PanelTask, main, summarize_panel


@pytest.fixture
def panel():
    return [
        PanelTask("calendar-a", "calendar", "contract-a"),
        PanelTask("calendar-b", "calendar", "contract-b"),
        PanelTask("native-c", "structural", "contract-c"),
    ]


def test_missing_grade_remains_in_family_and_aggregate_denominators(panel):
    report = summarize_panel(
        panel,
        [
            EvaluationOutcome("calendar-a", 1.0, "stop", "verified"),
            EvaluationOutcome("native-c", None, "length", "unavailable", "ProviderRateLimitError"),
        ],
    )

    assert report["complete"] is False
    assert report["aggregate_fixed_denominator_mean"] == pytest.approx(1 / 3)
    assert report["macro_family_mean"] == 0.25
    assert report["families"]["calendar"]["fixed_denominator_mean"] == 0.5
    assert report["families"]["calendar"]["coverage"] == 0.5
    assert report["families"]["structural"]["scored"] == 0
    assert report["families"]["structural"]["length_stops"] == 1
    assert report["families"]["structural"]["failure_categories"] == {"ProviderRateLimitError": 1}
    assert report["families"]["calendar"]["failure_categories"] == {}
    assert [row["attempted"] for row in report["tasks"]] == [True, False, True]


def test_pairing_identity_does_not_depend_on_model_scores_or_result_order(panel):
    baseline = summarize_panel(panel, [EvaluationOutcome(task.task_id, 0.0, "length") for task in panel])
    checkpoint = summarize_panel(panel, [EvaluationOutcome(task.task_id, 1.0, "stop") for task in reversed(panel)])

    assert baseline["pairing_identity"] == checkpoint["pairing_identity"]
    assert baseline["complete"] is True
    assert baseline["aggregate_fixed_denominator_mean"] == 0
    assert baseline["families"]["calendar"]["length_stops"] == 2
    assert checkpoint["aggregate_fixed_denominator_mean"] == 1
    assert [row["task_id"] for row in checkpoint["tasks"]] == [task.task_id for task in panel]


def test_duplicate_outcomes_cannot_inflate_fixed_panel_score(panel):
    with pytest.raises(ValueError, match="repeated evaluation task"):
        summarize_panel(panel, [EvaluationOutcome("calendar-a", 1.0, "stop")] * 2)


@pytest.mark.parametrize("reply_tokens,window_tokens", [(None, 20480), (8192, 24576)])
def test_reference_launch_matches_requested_budget_and_keeps_complete_panel(reply_tokens, window_tokens):
    arguments = dict(
        model_uri="s3://marin-us-east-02a/test/verified-reference-export",
        model_version="2026.10.20.1",
        panel_uri="s3://marin-us-east-02a/test/frozen-reference-panel",
        panel_version="2026.10.20.1",
        panel_sha256="a" * 64,
        training_config_uri=None,
        checkpoint_step=None,
    )
    if reply_tokens is not None:
        arguments["max_new_tokens"] = reply_tokens
    with build_context(BuildContext(versions=VersionCodex(default="2026.10.20.901"))):
        step = main.callback.__wrapped__(**arguments)
    config = step.build_config(
        StepContext.for_run(
            output_path="s3://marin-us-east-02a/test/reference-evaluation",
            prefix="s3://marin-us-east-02a/test",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )
    document = yaml.safe_load(config.launch.launch_config_yaml)
    recipe = document["skyrl"]
    assert config.max_new_tokens == (reply_tokens or 4096)
    assert recipe["context_budget"] == {
        "request_window_tokens": window_tokens,
        "max_new_tokens_per_turn": reply_tokens or 4096,
        "max_turns": 1,
    }
    assert document["inputs"]["model"]["uri"] == arguments["model_uri"]
    assert document["inputs"]["validation_data"][0]["uri"] == arguments["panel_uri"]
    assert document["inputs"]["train_data"] == []
    assert recipe["trainer"]["eval_num_prompts"] is None
    assert recipe["trainer"]["evaluation_runner"] == "task"
    assert recipe["generator"]["eval_n_samples_per_prompt"] == 1
    assert recipe["trajectory_runner"]["max_concurrent_tasks"] == 16


def test_runtime_subprocess_isolates_parent_python_environment_and_inherits_judge(tmp_path, monkeypatch):
    from experiments.post_training.snowball_inventory_evaluate import runtime_operation  # noqa: PLC0415

    executable = tmp_path / "uv"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "result=Path(sys.argv[sys.argv.index('--result')+1])\n"
        "request=Path(sys.argv[sys.argv.index('--request')+1])\n"
        "result.write_text(json.dumps({'request':json.loads(request.read_text()),"
        "'parent_python_leaked':any(k in os.environ for k in "
        "['PYTHONPATH','PYTHONHOME','UV_PROJECT_ENVIRONMENT','VIRTUAL_ENV']),"
        "'judge_inherited':os.environ.get('JUDGE_API_KEY')=='test-judge-key'}))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    for name in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        monkeypatch.setenv(name, str(tmp_path / "wrong-parent-environment"))
    monkeypatch.setenv("JUDGE_API_KEY", "test-judge-key")

    report = runtime_operation("preflight", {"panel_sha256": "pinned-panel"})

    assert report == {
        "request": {"panel_sha256": "pinned-panel"},
        "parent_python_leaked": False,
        "judge_inherited": True,
    }


def test_scoring_retry_reuses_first_launch_despite_fresh_artifact_attempt_id(tmp_path, monkeypatch):
    from marin.execution.lazy import ArtifactStep  # noqa: PLC0415
    from marin.training.training import LevanterCheckpoint  # noqa: PLC0415

    from experiments.post_training import snowball_inventory_evaluate as evaluator  # noqa: PLC0415

    model = ArtifactStep.adopt("tests/baseline", "2026.10.21", str(tmp_path / "model"), kind=LevanterCheckpoint)
    panel = ArtifactStep.adopt("tests/panel", "2026.10.21", str(tmp_path / "panel"))
    with build_context(BuildContext(versions=VersionCodex(default="2026.10.21"))):
        step = evaluator.recoverable_evaluation_step(
            name="tests/recoverable-score",
            version="2026.10.21",
            model=model,
            panel=panel,
            panel_sha256="a" * 64,
            family_counts={"structural": 1},
            tokenizer_uri="fixture/model",
            tokenizer_revision="b" * 40,
            recovery=evaluator.SavedAnswerRecovery(10, 20),
            judge_base_url="https://judge.invalid/v1",
            judge_model="fixture-judge",
            relative_path="",
        )
    context = StepContext.for_run(
        output_path=str(tmp_path / "score"),
        prefix=str(tmp_path),
        runtime_args=step.runtime_args,
        deps=step.deps,
    )
    first, rebuilt = step.build_config(context), step.build_config(context)
    assert first.launch.attempt_id != rebuilt.launch.attempt_id
    inference = []
    scoring = []
    admitted_document = None

    def run_inference(config):
        nonlocal admitted_document
        inference.append(config.attempt_id)
        admitted_document = yaml.safe_load(config.launch_config_yaml)

    def runtime(action, request, **kwargs):
        document = yaml.safe_load(request["launch_config_yaml"])
        if action == "preflight":
            return {
                "launch_config_yaml": request["launch_config_yaml"],
                "dataset_uids": {"0": "paired"},
                "tasks": [{"task_id": "paired", "evaluation_family": "structural", "contract_id": "source"}],
                "model_manifest_identity": "exact-model",
                "protocol": "fixed",
                "pairing_contract_field": "source",
            }
        if action == "generation-state":
            if admitted_document is not None:
                assert document == admitted_document  # The real terminal reader requires exact equality.
            return {"complete": admitted_document is not None}
        assert action == "recover-results"
        scoring.append(1)
        if len(scoring) == 1:
            raise OSError("CPU coordinator failed after successful inference")
        return {"outcomes": [{"task_id": "paired", "score": 1.0, "stop_reason": "stop"}], "recovery": {}}

    monkeypatch.setattr(evaluator, "remote", lambda function, **kwargs: function)
    monkeypatch.setattr(evaluator, "runtime_operation", runtime)
    monkeypatch.setattr(evaluator, "run_skyrl", run_inference)
    with pytest.raises(OSError, match="coordinator failed"):
        step.run(first)
    completed = step.run(rebuilt)
    assert completed.complete
    assert inference == [first.launch.attempt_id] and len(scoring) == 2
    preflight = json.loads((tmp_path / "score" / "preflight.json").read_text())
    assert yaml.safe_load(preflight["launch_config_yaml"])["run"]["attempt_id"] == first.launch.attempt_id
    with pytest.raises(ValueError, match="different source, model, panel"):
        step.run(replace(rebuilt, max_new_tokens=4096))
