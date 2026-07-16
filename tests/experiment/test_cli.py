# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for the experiment CLI bootstrap (``marin.experiment.cli``).

Drives the shared version/run options through Click's ``CliRunner`` end to end: deferred versions
resolve from ``--version`` / ``--override``, the plan flags mutable (rebuilding) artifacts, and a
misspelled or malformed override is rejected before anything runs.
"""

import os

import click
import pytest
from click.testing import CliRunner
from marin.execution.lazy import OUT, apply
from marin.experiment.cli import build_options, experiment_main


def _noop(**_kwargs):
    return None


def _build_two():
    """A dataset handle and a model handle depending on it, both deferring their version."""
    data = apply("data/toy", _noop, x=1)
    model = apply("models/toy", _noop, data=data)
    return model


@pytest.fixture
def runner():
    return CliRunner()


def test_version_is_required(runner):
    result = runner.invoke(experiment_main(_build_two), [])
    assert result.exit_code != 0
    assert "version" in result.output.lower()


def test_plan_resolves_deferred_versions_from_default(runner):
    result = runner.invoke(experiment_main(_build_two), ["--version", "2026.07.16"])
    assert result.exit_code == 0, result.output
    assert "data/toy@2026.07.16" in result.output
    assert "models/toy@2026.07.16" in result.output


def test_override_steers_a_single_artifact_by_name(runner):
    result = runner.invoke(
        experiment_main(_build_two),
        ["--version", "2026.07.16", "--override", "data/toy=2026.06.28"],
    )
    assert result.exit_code == 0, result.output
    assert "data/toy@2026.06.28" in result.output
    assert "models/toy@2026.07.16" in result.output


def test_plan_flags_mutable_versions_that_will_rebuild(runner):
    # A mutable (dev) resolution is a rebuild-every-run hazard; the plan must surface it, and a
    # calendar version must not be flagged.
    dev = runner.invoke(experiment_main(_build_two), ["--version", "dev"])
    assert dev.exit_code == 0, dev.output
    assert "rebuild" in dev.output.lower()
    assert "data/toy@dev" in dev.output

    calver = runner.invoke(experiment_main(_build_two), ["--version", "2026.07.16"])
    assert "rebuild" not in calver.output.lower()


def test_unused_override_is_rejected(runner):
    # A typo'd or pin-shadowed override silently no-ops; the run must refuse rather than proceed
    # under the default version.
    result = runner.invoke(
        experiment_main(_build_two),
        ["--version", "2026.07.16", "--override", "data/typo=2026.06.28"],
    )
    assert result.exit_code != 0
    assert "data/typo" in result.output


def test_malformed_and_invalid_overrides_are_rejected(runner):
    missing_eq = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--override", "noequals"])
    assert missing_eq.exit_code != 0
    assert "NAME=VERSION" in missing_eq.output

    bad_version = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--override", "data/toy=v1"])
    assert bad_version.exit_code != 0
    assert "calendar version" in bad_version.output


def test_bad_default_version_is_a_click_error_not_a_traceback(runner):
    result = runner.invoke(experiment_main(_build_two), ["--version", "v1"])
    assert result.exit_code != 0
    assert "calendar version" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_build_options_composes_with_experiment_options(runner):
    @click.command()
    @click.option("--flavor", default="a")
    @build_options
    def cmd(flavor):
        return apply(f"x/{flavor}", _noop, x=1)

    result = runner.invoke(cmd, ["--flavor", "z", "--version", "2026.07.16"])
    assert result.exit_code == 0, result.output
    assert "x/z@2026.07.16" in result.output


def _write_marker(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "done"), "w") as f:
        f.write("built")


def test_run_builds_the_handles_plan_does_not(runner, tmp_path, monkeypatch):
    # The plan/run toggle: --run executes the step (a marker lands in its output dir); the default
    # plan builds nothing.
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    def build():
        return apply("scratch/marker", _write_marker, out=OUT)

    marker = tmp_path / "scratch" / "marker" / "dev" / "done"

    plan = runner.invoke(experiment_main(build), ["--version", "dev"])
    assert plan.exit_code == 0, plan.output
    assert not marker.exists()

    built = runner.invoke(experiment_main(build), ["--version", "dev", "--run"])
    assert built.exit_code == 0, built.output
    assert marker.exists()
