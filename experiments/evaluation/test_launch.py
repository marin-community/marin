# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable identities across command-line evaluation relaunches."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from marin.evaluation.runner import EvaluationBatch
from marin.execution.step_runner import StepRunner

from experiments.evaluation.cli import cli
from experiments.evaluation.launch import build_evaluation_steps


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
def test_relaunch_reuses_completed_steps_and_preserves_partial_archive(tmp_path, submitted_batches, explicit_run_id):
    runner = CliRunner()
    command = _command(tmp_path)
    if not explicit_run_id:
        index = command.index("--run-id")
        del command[index : index + 2]
    first = runner.invoke(cli, command)
    assert first.exit_code == 0, first.output
    batch = submitted_batches[-1]
    executed = []

    def execute(evaluation, _output_path):
        executed.append(evaluation.identity.run_id)
        return {"finished": evaluation.identity.run_id}

    steps = build_evaluation_steps(batch, execute)
    StepRunner().run(steps[:1])
    pending = batch.evaluations[1]
    partial = Path(pending.identity.output_dir) / "saved-request.json"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"response":"already generated"}')
    executed.clear()

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


def test_relaunch_changed_request_concurrency_preserves_identity(tmp_path, submitted_batches):
    config = tmp_path / "custom.yaml"
    config.write_text("tasks: [gsm8k]\nextra_model_args:\n  num_concurrent: 16\n")
    command = [*_command(tmp_path), "--evalchemy-config", str(config)]
    runner = CliRunner()
    first = runner.invoke(cli, command)
    assert first.exit_code == 0, first.output

    config.write_text("tasks: [gsm8k]\nextra_model_args:\n  num_concurrent: 64\n")
    second = runner.invoke(cli, command)

    assert second.exit_code == 0, second.output
    before = submitted_batches[0].evaluations[-1]
    after = submitted_batches[1].evaluations[-1]
    assert after.identity.run_id == before.identity.run_id
    assert after.identity.output_dir == before.identity.output_dir
