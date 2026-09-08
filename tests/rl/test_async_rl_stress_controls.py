# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.lazy import materialized_config

from experiments.post_training import async_rl


def config(**kwargs):
    return async_rl.training_config(async_rl.Runner.ASYNC, async_rl.Scale.SCREENING, spans=True, staleness=8, **kwargs)


def test_stress_defaults_preserve_config_and_eval():
    original = config()
    assert config(generation_workers=None, training_ignore_eos=False) == original
    assert config(generation_workers=64) == original
    control = yaml.safe_load(original)
    stress = yaml.safe_load(config(generation_workers=512, training_ignore_eos=True))
    assert stress["trainer"]["fully_async"]["num_parallel_generation_workers"] == 512
    assert stress["generator"]["sampling_params"].pop("ignore_eos") is True
    stress["trainer"]["fully_async"]["num_parallel_generation_workers"] = 64
    assert stress == control


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "512"])
def test_stress_workers_reject_invalid_values(workers):
    with pytest.raises(ValueError, match="positive integer"):
        config(generation_workers=workers)


def test_stress_ignore_eos_requires_boolean():
    with pytest.raises(ValueError, match="boolean"):
        config(training_ignore_eos=1)


def test_stress_cli_reaches_native_request():
    result = CliRunner().invoke(
        async_rl.main,
        [
            "--version",
            "2026.09.08.47",
            "--scale",
            "screening",
            "--stage",
            "rl",
            "--completion",
            "metrics",
            "--screening-steps",
            "20",
            "--eval-interval",
            "20",
            "--staleness",
            "8",
            "--max-num-seqs",
            "4",
            "--train-rows",
            "2048",
            "--generation-workers",
            "512",
            "--training-ignore-eos",
            "--eval-response-tokens",
            "1024",
        ],
    )
    assert result.exit_code == 0, result.output
    settings = yaml.safe_load(json.loads(result.output)["request"]["config_yaml"])
    assert settings["generator"]["max_num_seqs"] == 4
    assert settings["trainer"]["fully_async"]["num_parallel_generation_workers"] == 512
    assert settings["generator"]["sampling_params"]["ignore_eos"] is True
    assert settings["generator"]["eval_sampling_params"] == {"max_generate_length": 1024}


def test_scheduler_cap_preserves_other_training_and_eval_settings():
    control = yaml.safe_load(config())
    limited = yaml.safe_load(config(max_num_seqs=4))
    assert limited["generator"].pop("max_num_seqs") == 4
    assert limited == control
    assert config(max_num_seqs=None) == config()


def test_training_pool_is_bound_to_data_and_native_request_identity():
    kwargs = dict(
        version="2026.09.08.48",
        cluster="cw-us-east-02a",
        runner=async_rl.Runner.ASYNC,
        scale=async_rl.Scale.SCREENING,
        completion="metrics",
    )
    control, _ = async_rl.build_experiment(**kwargs)
    explicit, _ = async_rl.build_experiment(**kwargs, train_rows=1024)
    extended, _ = async_rl.build_experiment(**kwargs, train_rows=2048)
    prefix = "s3://marin-us-east-02a/marin"
    assert control.fingerprint() == explicit.fingerprint()
    assert control.deps[1].fingerprint() != extended.deps[1].fingerprint()
    assert control.fingerprint() != extended.fingerprint()
    original = materialized_config(control.deps[1], prefix)
    larger = materialized_config(extended.deps[1], prefix)
    assert original.train_rows == 1024 and larger.train_rows == 2048
    assert original.validation_rows == larger.validation_rows == 128
    assert (
        materialized_config(control, prefix).request.config_yaml
        == materialized_config(extended, prefix).request.config_yaml
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_scheduler_cap_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        config(max_num_seqs=value)
