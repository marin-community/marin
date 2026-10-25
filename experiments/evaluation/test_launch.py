# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line evaluation launch and relaunch behavior."""

import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from marin.evaluation.evalchemy import runtime
from marin.evaluation.hardware import Platform
from marin.evaluation.records import RunStatus
from marin.evaluation.runner import EvaluationBatch, _record
from marin.execution.step_runner import StepRunner

import experiments.evaluation.launch as launch_module
from experiments.evaluation.cli import cli
from experiments.evaluation.evals import EvalchemyDefinition
from experiments.evaluation.launch import LaunchSpec, build_evaluation_steps, prepare_evaluation_batch
from experiments.evaluation.models import models


@pytest.fixture
def submitted_batches(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "local-test-key")
    monkeypatch.setenv("DAYTONA_API_KEY", "local-test-key")
    batches: list[EvaluationBatch] = []

    class Client:
        def submit(self, *, entrypoint, **_kwargs):
            _, args, _ = entrypoint.resolve()
            batches.append(args[0])
            return SimpleNamespace(job_id="/test/eval")

    @contextmanager
    def client(**_kwargs):
        yield Client()

    monkeypatch.setattr("experiments.evaluation.cli.open_iris_client", client)
    return batches


def _command(tmp_path: Path) -> list[str]:
    return [
        "launch",
        "--model",
        "qwen3-8b",
        "--evals",
        "gsm8k-smoke,arc-easy",
        "--platform",
        "gpu",
        "--accelerator",
        "H100x1",
        "--records-prefix",
        str(tmp_path),
        "--run-id",
        "repeatable-evals",
        "--no-wait",
    ]


@pytest.mark.parametrize("explicit_run_id", [True, False])
def test_relaunch_reuses_completed_steps_and_preserves_partial_archive(
    tmp_path, submitted_batches, explicit_run_id, monkeypatch
):
    runner = CliRunner()
    command = _command(tmp_path)
    if not explicit_run_id:
        index = command.index("--run-id")
        del command[index : index + 2]
    first = runner.invoke(cli, command)
    assert first.exit_code == 0, first.output
    batch = submitted_batches[-1]
    # Records from earlier launches can have IDs derived from a different runtime revision.
    batch = replace(
        batch,
        evaluations=tuple(
            replace(
                evaluation,
                identity=replace(
                    evaluation.identity,
                    run_id=f"saved-eval-{index}",
                    output_dir=str(tmp_path / f"saved-eval-{index}" / "results"),
                ),
            )
            for index, evaluation in enumerate(batch.evaluations)
        ),
    )
    executed = []

    def execute(evaluation, _output_path):
        executed.append(evaluation.identity.run_id)
        path = _record(batch, evaluation.identity, RunStatus.SUCCEEDED, None, {}, {}, {})
        return {"record_path": path}

    steps = build_evaluation_steps(batch, execute)
    StepRunner().run(steps[:1])
    pending = batch.evaluations[1]
    _record(batch, pending.identity, RunStatus.ARTIFACT_FAILED, "interrupted", {}, {}, {})
    partial = Path(pending.identity.output_dir) / "saved-request.json"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"response":"already generated"}')
    executed.clear()
    monkeypatch.setattr(launch_module, "evalchemy_runtime_source_hash", lambda: "a" * 64)

    second = runner.invoke(cli, [*command, "--run-id", batch.group_id, "--instances", "8", "--max-concurrent", "8"])
    assert second.exit_code == 0, second.output
    restored = submitted_batches[-1]
    StepRunner().run(build_evaluation_steps(restored, execute))

    assert executed == [pending.identity.run_id]
    assert restored.evaluations[1].identity.output_dir == pending.identity.output_dir
    assert partial.read_text() == '{"response":"already generated"}'


@pytest.mark.parametrize("changed_args", [["--limit", "2"], ["--model", "qwen3-0.6b"]])
def test_relaunch_changed_scoring_settings_use_distinct_steps(tmp_path, submitted_batches, changed_args):
    runner = CliRunner()
    command = _command(tmp_path)
    first = runner.invoke(cli, command)
    assert first.exit_code == 0, first.output

    changed = runner.invoke(cli, [*command, *changed_args])

    assert changed.exit_code == 0, changed.output
    original = {evaluation.identity.run_id for evaluation in submitted_batches[0].evaluations}
    revised = {evaluation.identity.run_id for evaluation in submitted_batches[1].evaluations}
    assert original.isdisjoint(revised)


@pytest.mark.parametrize("cache_key", [None, "pipeline-artifact"])
def test_relaunch_changed_request_concurrency_preserves_identity(tmp_path, cache_key):
    config = tmp_path / "custom.yaml"
    config.write_text("tasks: [gsm8k]\nextra_model_args:\n  num_concurrent: 16\n")
    model = models()["qwen3-8b"]
    spec = LaunchSpec(
        model=model,
        evals=(),
        evalchemy_definitions=(EvalchemyDefinition(name="custom", config_path=config),),
        harbor_definitions=(),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=None,
        records_prefix=str(tmp_path / "records"),
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=0,
        cache_key=cache_key,
        run_id="repeatable-evals" if cache_key is None else None,
    )
    before = prepare_evaluation_batch(spec)
    executed = []

    def execute(evaluation, _output_path):
        executed.append(evaluation.identity.run_id)
        return {"finished": evaluation.identity.run_id}

    StepRunner().run(build_evaluation_steps(before, execute))
    executed.clear()

    config.write_text("tasks: [gsm8k]\nextra_model_args:\n  num_concurrent: 64\n")
    tuned = replace(model.serve, max_num_batched_tokens=8192, max_num_seqs=128, gpu_memory_utilization=0.9)
    after = prepare_evaluation_batch(replace(spec, model=replace(model, serve=tuned)))
    StepRunner().run(build_evaluation_steps(after, execute))

    assert executed == []


@pytest.fixture
def runtime_checkout(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    for directory in runtime._RUNTIME_DIRECTORIES:
        source = root / directory / "source.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n")
    for filename in runtime._RUNTIME_FILES:
        source = root / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("runtime input\n")
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    monkeypatch.setattr(runtime, "find_project_root", lambda _start: root)
    return root


def _pipeline_spec(tmp_path):
    return LaunchSpec(
        model=models()["qwen3-8b"],
        evals=("gsm8k-smoke", "arc-easy"),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=1,
        records_prefix=str(tmp_path / "records"),
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=0,
        cache_key="caller-owned-immutable-artifact",
        version="2026.10.19",
    )


@pytest.mark.parametrize("dependency", ["lib/finestore/src/source.py", "lib/rigging/src/source.py"])
def test_pipeline_dependency_update_reuses_completed_step_and_partial_archive(tmp_path, runtime_checkout, dependency):
    spec = _pipeline_spec(tmp_path)
    before = prepare_evaluation_batch(spec)
    executed = []

    def execute(evaluation, _output_path):
        executed.append(evaluation.identity.run_id)
        return {"finished": evaluation.identity.run_id}

    StepRunner().run(build_evaluation_steps(before, execute)[:1])
    pending = before.evaluations[1]
    _record(before, pending.identity, RunStatus.ARTIFACT_FAILED, "interrupted", {}, {}, {})
    partial = Path(pending.identity.output_dir) / "saved-request.json"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"response":"already generated"}')
    executed.clear()

    source = runtime_checkout / dependency
    source.write_text(source.read_text() + "dependency update\n")
    after = prepare_evaluation_batch(spec)
    StepRunner().run(build_evaluation_steps(after, execute))

    assert after.group_id == before.group_id
    assert after.evaluations[0].identity.eval_runtime != before.evaluations[0].identity.eval_runtime
    assert executed == [pending.identity.run_id]
    assert after.evaluations[1].identity.output_dir == pending.identity.output_dir
    assert partial.read_text() == '{"response":"already generated"}'


@pytest.mark.parametrize("filename", ["vendor/evalchemy/eval/source.py", "vendor/evalchemy/uv.lock"])
def test_pipeline_evaluator_update_uses_new_evaluation_steps(tmp_path, runtime_checkout, filename):
    spec = _pipeline_spec(tmp_path)
    before = prepare_evaluation_batch(spec)
    executed = []

    def execute(evaluation, _output_path):
        executed.append(evaluation.identity.run_id)
        return {"finished": evaluation.identity.run_id}

    StepRunner().run(build_evaluation_steps(before, execute))
    executed.clear()
    source = runtime_checkout / filename
    source.write_text(source.read_text() + "evaluator update\n")
    after = prepare_evaluation_batch(spec)
    StepRunner().run(build_evaluation_steps(after, execute))

    assert after.group_id == before.group_id
    assert {evaluation.identity.run_id for evaluation in before.evaluations}.isdisjoint(executed)
    assert sorted(executed) == sorted(evaluation.identity.run_id for evaluation in after.evaluations)


@pytest.mark.parametrize(
    "benchmark,changed", [("financebench", False), ("olympiadbench", False), ("sotopia-hard", True)]
)
def test_policy_cohort_change_invalidates_only_sotopia(tmp_path, submitted_batches, benchmark, changed):
    campaign = Path(__file__).parent / "campaigns" / "eval-campaign-09-25-unlabeled"
    cohort = tmp_path / "cohort.json"
    cohort.write_text((campaign / "sotopia-baselines.json").read_text())
    config = tmp_path / "finalization.yaml"
    judge = campaign / "ablations/olympiadbench_judge/MiniMaxAI-MiniMax-M3-MXFP8-262k.yaml"
    config.write_text(f"judge_model_config: {judge.resolve()}\nsotopia_baselines: {cohort}\n")
    command = _command(tmp_path)
    index = command.index("--evals")
    del command[index : index + 2]
    command.extend(["--policy-finalization-config", str(config)])
    if benchmark == "sotopia-hard":
        command.extend(["--harbor-config", str(campaign / "harbor-configs/sotopia-hard.yaml")])
    else:
        command.extend(["--evals", benchmark])
    runner = CliRunner()
    first = runner.invoke(cli, command)
    assert first.exit_code == 0, first.output
    batch = submitted_batches[-1]
    evaluation = batch.evaluations[0]
    _record(batch, evaluation.identity, RunStatus.SUCCEEDED, None, {}, {}, {})
    original = json.loads(evaluation.identity.eval_ref.postprocessing["marin-10197"])
    sources = json.loads(cohort.read_text())
    sources["models"][0]["results_path"] += "-replacement"
    cohort.write_text(json.dumps(sources))
    second = runner.invoke(cli, command)
    assert second.exit_code == 0, second.output
    revised = submitted_batches[-1].evaluations[0]
    assert (revised.identity.run_id != evaluation.identity.run_id) is changed
    frozen = json.loads(revised.identity.eval_ref.postprocessing["marin-10197"])
    assert (frozen != original) is changed
    if benchmark != "sotopia-hard":
        assert "JUDGE_API_KEY" not in revised.secret_env_keys


@pytest.mark.parametrize("platform,accelerator", [("tpu", "v5litepod-4"), ("gpu", "GB200x1")])
def test_policy_judge_rejects_incompatible_placement_before_submission(
    tmp_path, submitted_batches, platform, accelerator
):
    config = Path(__file__).parent / "campaigns/eval-campaign-09-25-unlabeled/finalization.yaml"
    command = [
        *_command(tmp_path),
        "--evals",
        "financebench",
        "--policy-finalization-config",
        str(config),
        "--platform",
        platform,
        "--accelerator",
        accelerator,
    ]
    result = CliRunner().invoke(cli, command)
    assert result.exit_code != 0
    assert submitted_batches == []


def test_policy_snapshot_uses_project_python_and_preserves_cohort_sources(tmp_path):
    campaign = Path(__file__).parent / "campaigns/eval-campaign-09-25-unlabeled"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    system_python = bin_dir / "python3"
    system_python.write_text("#!/bin/sh\necho 'system Python has no project dependencies' >&2\nexit 1\n")
    system_python.chmod(0o755)
    snapshots = tmp_path / "snapshots"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; CONFIG_SNAPSHOT_DIR="$2"; ' 'snapshot_launch_configs --policy-finalization-config "$3"',
            "snapshot",
            str(campaign / "common.sh"),
            str(snapshots),
            str(campaign / "finalization.yaml"),
        ],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "UV_NO_SYNC": "1",
            "MARIN_DIR": str(Path(__file__).resolve().parents[2]),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    snapshot = snapshots / result.stdout.strip()
    manifest = json.loads((snapshot / "launch.json").read_text())
    assert {row["kind"] for row in manifest["files"]} == {
        "policy-finalization-config",
        "judge_model_config",
        "sotopia_baselines",
    }
    saved_cohort = next(snapshot.glob("*-sotopia-baselines.json"))
    assert saved_cohort.read_bytes() == (campaign / "sotopia-baselines.json").read_bytes()
