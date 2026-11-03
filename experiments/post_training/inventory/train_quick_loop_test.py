# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protect distinct cached experiment identities at the public arm-selection CLI."""

from click.testing import CliRunner

from experiments.post_training.inventory.train_quick_loop import main


def test_public_arm_cli_assigns_distinct_training_and_report_identities(monkeypatch, tmp_path):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    submissions = []
    monkeypatch.setattr(
        "marin.rl.cli._submit_or_run", lambda module, handles, concurrency, env: submissions.append(handles)
    )
    version = "2026.11.02"
    for arm in ("a", "b", "c", "d"):
        result = CliRunner().invoke(main, ["--version", version, "--arm", arm, "--run"])
        assert result.exit_code == 0, result.output
    reports = [report for [report] in submissions]
    trainings = [training for report in reports for training in report.deps]
    assert len({report.name for report in reports}) == 4
    assert len({training.name for training in trainings}) == 4
    assert {report.version for report in reports} == {version}
    assert {training.version for training in trainings} == {version}
