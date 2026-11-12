# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shlex

from click.testing import CliRunner

from experiments.data_belay.__main__ import RUNS_ROOT, cli
from experiments.grug.fast_track import launch


def test_the_launcher_plans_the_train_command_at_the_baseline_budget_under_the_data_belay_run_root():
    printed = CliRunner().invoke(cli, ["train", "--rung", "d512", "--seed", "0"])
    assert printed.exit_code == 0, printed.output
    command = shlex.split(printed.output)
    args = command[command.index("experiments.grug.fast_track.launch") + 1 :]
    args.remove("--submit")

    planned = CliRunner().invoke(launch.main, args, standalone_mode=False)

    assert planned.exit_code == 0, planned.output
    assert f"override_output_path='{RUNS_ROOT}/dbelay-main-d512-control-s0'" in planned.output
    assert '"num_train_steps": 690' in planned.output
