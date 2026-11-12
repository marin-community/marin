# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shlex

from click.testing import CliRunner

from experiments.data_belay.__main__ import RUNS_ROOT, cli


def test_train_launches_the_dense_baseline_under_the_data_belay_run_root():
    result = CliRunner().invoke(cli, ["train", "--rung", "d512", "--seed", "0"])
    assert result.exit_code == 0, result.output
    command = shlex.split(result.output)
    assert command[command.index("--size") + 1] == "d512"
    assert "--dense" in command
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--output-path") + 1] == f"{RUNS_ROOT}/dbelay-main-d512-control-s0"
