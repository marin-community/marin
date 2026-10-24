# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable identities across command-line evaluation relaunches."""

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
