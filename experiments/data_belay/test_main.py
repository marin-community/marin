# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shlex

import click
import pytest
from click.testing import CliRunner

from experiments.data_belay.__main__ import RUNS_ROOT, cli, require_dilution_references
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


def test_report_refuses_a_share_without_dilution_references_and_prints_the_commands_that_build_them():
    with pytest.raises(click.ClickException) as refused:
        require_dilution_references("fineweb-sample-10bt-p50-d497e2", 0.5, "d768")
    message = refused.value.message
    assert "--against" in message
    for label, source in [("dk-climblab-ja", "climblab-ja"), ("dk-finepdfs-arb_arab", "finepdfs/arb_Arab")]:
        assert f"prepare-datakit --label {label} --source {source} --share 0.5 --submit" in message
    assert "seeds 0, 1, 2 of each at d768" in message


@pytest.mark.parametrize(("candidate", "share"), [("fineweb-sample-10bt-p20-cb6b5a", 0.2), ("control", 0.0)])
def test_report_judges_a_share_with_references_and_the_control_which_gives_up_no_share(candidate, share):
    require_dilution_references(candidate, share, "d512")
