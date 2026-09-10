# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import yaml
from click.testing import CliRunner

from experiments.post_training import async_rl


def test_admission_cli_preserves_default_and_isolates_delay_configuration():
    options = [
        "--version",
        "2026.09.09.251",
        "--runner",
        "async",
        "--scale",
        "screening",
        "--stage",
        "rl",
        "--completion",
        "metrics",
        "--updates",
        "8",
        "--eval-updates",
        "8",
        "--staleness",
        "6",
        "--weight-sync-interval",
        "4",
        "--first-token-admission",
        "--epoch-seeded-shuffle",
        "--dataloader-workers",
        "0",
        "--no-kl-loss",
    ]
    runner = CliRunner()
    responses = [
        runner.invoke(async_rl.main, options + extra)
        for extra in [
            [],
            ["--admission-order", "fifo", "--injected-delay-max", "0"],
            ["--max-buffered-groups", "256"],
            ["--max-buffered-groups", "256", "--injected-delay-max", "3"],
            ["--max-buffered-groups", "256", "--admission-order", "freshest_first"],
        ]
    ]
    assert all(result.exit_code == 0 for result in responses), [result.output for result in responses]
    requests = [json.loads(result.output)["request"] for result in responses]
    assert requests[0] == requests[1]
    configs = [yaml.safe_load(request["config_yaml"]) for request in requests]
    assert configs[3]["trainer"]["fully_async"]["injected_delay_max_steps"] == 3
    assert configs[3]["trainer"]["fully_async"]["max_buffered_groups"] == 256
    assert configs[4]["trainer"]["fully_async"]["admission_order"] == "freshest_first"
    configs[3]["trainer"]["fully_async"].pop("injected_delay_max_steps")
    configs[4]["trainer"]["fully_async"].pop("admission_order")
    assert configs[2] == configs[3] == configs[4]
    configs[2]["trainer"]["fully_async"].pop("max_buffered_groups")
    assert configs[2] == configs[0]
    assert len({request["run_id"] for request in requests[2:]}) == 3


def test_delay_cli_rejects_missing_native_token_stamp_and_sync_mode():
    runner = CliRunner()
    for runner_name, extra in [("async", []), ("sync", ["--first-token-admission"])]:
        result = runner.invoke(
            async_rl.main,
            [
                "--version",
                "2026.09.09.251",
                "--runner",
                runner_name,
                "--scale",
                "screening",
                "--stage",
                "rl",
                "--staleness",
                "6",
                "--injected-delay-max",
                "3",
                *extra,
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
