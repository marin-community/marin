# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml
from click.testing import CliRunner

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
            "--generation-workers",
            "512",
            "--training-ignore-eos",
            "--eval-response-tokens",
            "1024",
        ],
    )
    assert result.exit_code == 0, result.output
    settings = yaml.safe_load(json.loads(result.output)["request"]["config_yaml"])
    assert settings["trainer"]["fully_async"]["num_parallel_generation_workers"] == 512
    assert settings["generator"]["sampling_params"]["ignore_eos"] is True
    assert settings["generator"]["eval_sampling_params"] == {"max_generate_length": 1024}
