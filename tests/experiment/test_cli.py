# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for the experiment CLI bootstrap (``marin.experiment.cli``).

These cover the CLI's own logic — deferred versions and overrides reaching the builders, the
mutable-version flag in the plan, the unused-override guard, and the plan/run toggle. They do not
re-test Click's option parsing.
"""

import os

import pytest
from click.testing import CliRunner
from marin.execution.lazy import OUT, apply
from marin.experiment.cli import experiment_main


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


def test_version_and_overrides_reach_deferred_builders(runner):
    # The core wiring: --version supplies the default, --override steers one artifact by name, and
    # both flow through to the resolved handle graph.
    result = runner.invoke(
        experiment_main(_build_two),
        ["--version", "2026.07.16", "--override", "data/toy=2026.06.28"],
    )
    assert result.exit_code == 0, result.output
    assert "data/toy@2026.06.28" in result.output  # overridden by name
    assert "models/toy@2026.07.16" in result.output  # took the default


def test_plan_flags_artifacts_that_resolved_to_a_mutable_version(runner):
    # A mutable resolution rebuilds on every run; the plan must mark it, and a calendar version must
    # not be marked.
    dev = runner.invoke(experiment_main(_build_two), ["--version", "dev"])
    assert dev.exit_code == 0, dev.output
    assert "mutable" in dev.output

    calver = runner.invoke(experiment_main(_build_two), ["--version", "2026.07.16"])
    assert "mutable" not in calver.output


def test_unused_override_is_rejected(runner):
    # A typo'd or pin-shadowed override silently no-ops, so a run could proceed under the wrong
    # version; build_options must refuse instead.
    result = runner.invoke(
        experiment_main(_build_two),
        ["--version", "2026.07.16", "--override", "data/typo=2026.06.28"],
    )
    assert result.exit_code != 0
    assert "data/typo" in result.output


def test_duplicate_override_for_one_name_is_rejected(runner):
    result = runner.invoke(
        experiment_main(_build_two),
        ["--version", "dev", "--override", "data/toy=dev", "--override", "data/toy=2026.06.28"],
    )
    assert result.exit_code != 0
    assert "data/toy" in result.output


def _write_marker(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "done"), "w") as f:
        f.write("built")


def test_run_executes_the_build_and_plan_does_not(runner, tmp_path, monkeypatch):
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
