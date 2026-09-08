# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.lazy import StepContext

from experiments.post_training import async_snowball


def test_explicit_snowball_rows_reach_data_writer_and_identity():
    args = dict(
        version="2026.09.08.34", scale=async_snowball.Scale.CADENCE_GATE, timeout_seconds=1350, completion="metrics"
    )
    baseline = async_snowball.build_experiment(**args)
    selected = async_snowball.build_experiment(**args, train_rows=128, validation_rows=128, dataloader_workers=0)
    data = selected.deps[1]
    config = data.build_config(StepContext.for_fingerprint(data.runtime_args, data.deps))
    assert isinstance(config, async_snowball.SnowballDataConfig)
    assert config.train_rows == 128 and config.validation_rows == 128
    assert selected.fingerprint() != baseline.fingerprint()
    request = selected.build_config(StepContext.for_fingerprint(selected.runtime_args, selected.deps)).request
    assert yaml.safe_load(request.config_yaml)["data"]["num_workers"] == 0


def test_snowball_row_and_worker_defaults_are_cli_equivalent():
    args = ["--version", "2026.09.08.34", "--scale", "cadence-gate", "--completion", "metrics"]
    default = CliRunner().invoke(async_snowball.main, args)
    explicit = CliRunner().invoke(async_snowball.main, [*args, "--train-rows", "1024", "--validation-rows", "128"])
    assert default.exit_code == explicit.exit_code == 0
    assert json.loads(default.output) == json.loads(explicit.output)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_snowball_rejects_invalid_loader_workers(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        async_snowball.training_config(async_snowball.Scale.CADENCE_GATE, dataloader_workers=value)


def test_snowball_receiver_readback_requires_trace_and_preserves_other_recipe_values():
    scale = async_snowball.Scale.CADENCE_GATE
    before = yaml.safe_load(async_snowball.training_config(scale, publication_stage_timing=True))
    after = yaml.safe_load(
        async_snowball.training_config(scale, publication_stage_timing=True, publication_receiver_state=True)
    )
    assert after["generator"].pop("publication_receiver_state") is True
    assert after == before
    with pytest.raises(ValueError, match="requires weight-sync stage timing"):
        async_snowball.training_config(scale, publication_receiver_state=True)


def test_snowball_receiver_readback_cli_reaches_native_request():
    args = ["--version", "2026.09.08.42", "--scale", "cadence-gate", "--completion", "metrics"]
    result = CliRunner().invoke(
        async_snowball.main, [*args, "--publication-stage-timing", "--publication-receiver-state"]
    )
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(json.loads(result.output)["request"]["config_yaml"])
    assert config["generator"]["publication_receiver_state"] is True
