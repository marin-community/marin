# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shlex

import click
import pytest
from click.testing import CliRunner

from experiments.data_belay.__main__ import RUNS_ROOT, cli, reference_commands, require_dilution_references
from experiments.data_belay.results import DILUTION_REFERENCE_SOURCES
from experiments.grug.fast_track import launch


def test_the_launcher_parses_the_train_command_as_a_baseline_run_under_the_data_belay_run_root():
    printed = CliRunner().invoke(cli, ["train", "--rung", "d512", "--seed", "0"])
    assert printed.exit_code == 0, printed.output
    command = shlex.split(printed.output)
    args = command[command.index("experiments.grug.fast_track.launch") + 1 :]

    params = launch.main.make_context("launch", args).params

    assert params["output_path"] == f"{RUNS_ROOT}/dbelay-main-d512-control-s0"
    assert (params["size"], params["dense"], params["seed"]) == ("d512", True, 0)
    # No step or batch override: the run trains at the dense baseline's budget.
    assert params["num_steps"] is None and params["batch_size"] is None


def test_the_reference_commands_prepare_every_dilution_reference_at_the_share():
    prepared = [
        cli.commands["prepare-datakit"].make_context("prepare-datakit", args[1:]).params
        for args in reference_commands(0.5)
    ]

    assert {(p["label"], p["sources"]) for p in prepared} == {(k, (v,)) for k, v in DILUTION_REFERENCE_SOURCES.items()}
    assert all(p["share"] == 0.5 for p in prepared)


def test_report_refuses_a_share_without_dilution_references():
    with pytest.raises(click.ClickException):
        require_dilution_references("fineweb-sample-10bt-p50-d497e2", 0.5, "d768")


@pytest.mark.parametrize(("candidate", "share"), [("fineweb-sample-10bt-p20-cb6b5a", 0.2), ("control", 0.0)])
def test_report_judges_a_share_with_references_and_the_control_which_gives_up_no_share(candidate, share):
    require_dilution_references(candidate, share, "d512")
