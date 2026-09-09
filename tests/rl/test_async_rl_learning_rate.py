# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml
from click.testing import CliRunner

from experiments.post_training import async_rl


@pytest.mark.parametrize("lr", [0, -1, float("nan"), float("inf"), True, "1e-6"])
def test_invalid_learning_rate_rejected(lr):
    with pytest.raises(ValueError, match="finite and positive"):
        async_rl.training_config(async_rl.Runner.ASYNC, async_rl.Scale.SCREENING, spans=True, staleness=8, lr=lr)


def test_actual_type_a_cli_changes_only_learning_rate_and_identity():
    options = [
        "--version",
        "2026.09.09.230",
        "--runner",
        "async",
        "--scale",
        "screening",
        "--stage",
        "rl",
        "--completion",
        "metrics",
        "--updates",
        "96",
        "--eval-updates",
        "32",
        "--staleness",
        "8",
        "--weight-sync-interval",
        "1",
        "--generation-workers",
        "64",
        "--first-token-admission",
        "--no-kl-loss",
    ]
    default = CliRunner().invoke(async_rl.main, options)
    low = CliRunner().invoke(async_rl.main, [*options, "--lr", "1e-6"])
    assert default.exit_code == low.exit_code == 0, default.output + low.output
    before, after = (json.loads(result.output)["request"] for result in (default, low))
    original, modified = (yaml.safe_load(request["config_yaml"]) for request in (before, after))
    assert original["trainer"]["policy"]["optimizer_config"]["lr"] == 2e-6
    assert modified["trainer"]["policy"]["optimizer_config"]["lr"] == 1e-6
    assert before["run_id"] != after["run_id"]
    modified["trainer"]["policy"]["optimizer_config"]["lr"] = 2e-6
    assert modified == original
    assert original["trainer"]["fully_async"]["first_token_admission"] is True
    assert original["trainer"]["fully_async"]["num_parallel_generation_workers"] == 64
    assert original["trainer"]["max_steps"] == 96
