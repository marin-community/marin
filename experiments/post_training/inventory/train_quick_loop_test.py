# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build each public arm through the actual CLI without allocating workers."""

import pytest
from click.testing import CliRunner

from experiments.post_training.inventory.train_quick_loop import main


@pytest.mark.parametrize("arm", ["a", "b", "c", "d"])
def test_public_arm_cli_plans_training_and_terminal_publication(arm, monkeypatch, tmp_path):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    result = CliRunner().invoke(main, ["--version", "2026.11.02", "--arm", arm])
    assert result.exit_code == 0, result.output
    assert f"snowball-inventory-batch-fixed-{arm}-report@2026.11.02" in result.output
    assert "rl-data/task-selection/snowball-100k@2026.10.23.2" in result.output
