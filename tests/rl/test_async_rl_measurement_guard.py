# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml
from click.testing import CliRunner

from experiments.post_training import async_rl


@pytest.mark.parametrize("runner", ["sync", "async"])
def test_actual_cli_binds_guard_and_explicit_retry_without_scientific_changes(runner, monkeypatch):
    executions = []
    build = async_rl.build_experiment

    def capture(*args, **kwargs):
        result = build(*args, **kwargs)
        executions.append(result[0].runtime_args["skyrl_execution"])
        return result

    monkeypatch.setattr(async_rl, "build_experiment", capture)
    options = [
        "--version",
        "2026.09.09.247",
        "--runner",
        runner,
        "--scale",
        "screening",
        "--completion",
        "metrics",
        "--stage",
        "rl",
        "--updates",
        "8",
        "--eval-updates",
        "8",
    ]
    original = CliRunner().invoke(async_rl.main, options)
    uri = "s3://marin-us-east-02a/marin/users/ahmad/diagnostics/guard.json"
    guarded = CliRunner().invoke(async_rl.main, [*options, "--measurement-guard-uri", uri, "--max-retries", "1"])
    assert original.exit_code == guarded.exit_code == 0, original.output + guarded.output
    before, after = (json.loads(result.output) for result in [original, guarded])
    assert [execution.max_retries for execution in executions] == [0, 1]
    a, b = (yaml.safe_load(value["request"]["config_yaml"]) for value in [before, after])
    assert "measurement_guard_uri" not in a["trainer"]
    assert b["trainer"].pop("measurement_guard_uri") == uri
    assert a == b
    assert before["request"]["run_id"] != after["request"]["run_id"]


@pytest.mark.parametrize("uri", ["", "s3://bucket", "s3://bucket/key?version=x", "/tmp/marker"])
def test_invalid_guard_uri_fails_during_recipe_resolution(uri):
    with pytest.raises(ValueError, match="explicit S3 object"):
        async_rl.training_config(
            async_rl.Runner.SYNC, async_rl.Scale.SCREENING, spans=True, staleness=0, measurement_guard_uri=uri
        )
