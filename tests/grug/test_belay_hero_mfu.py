# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay hero-mfu track: the stack's arguments parse in the launcher, its PGLE profile is checked in, and the
routing acceptance rule accepts mfl-mp-qall's measured routing and rejects unhealthy routing."""

from pathlib import Path

import pytest

from experiments.grug.fast_track import launch
from experiments.grug.fast_track.belay_hero_mfu import PGLE_DIR, STACKS, RunCheck, stack_args

REPO = Path(__file__).parents[2]


@pytest.mark.parametrize("seed_autotune", [False, True])
def test_stack_args_parse_in_the_launcher(seed_autotune: bool) -> None:
    args = [*stack_args("mfl-mp-qall", seed_autotune=seed_autotune), "--run-id", "belay-X0000-control"]
    context = launch.main.make_context("launch", args)
    assert context.params["num_steps"] == 30
    assert (REPO / PGLE_DIR / STACKS["mfl-mp-qall"].pgle_profile).is_file()


def test_acceptance_rule() -> None:
    qall = RunCheck("mfl-mp-qall", 11.465, lb_mean=8.0003, lb_layer_max=8.001, drops_max=0.0, bias=(-0.192, 0.121))
    assert qall.accepted
    assert not RunCheck("x", 11.4, 8.0003, 8.001, 0.0, (-0.5, 0.1)).accepted
    assert not RunCheck("x", 11.4, 8.0003, 8.001, 2e-5, (-0.1, 0.1)).accepted
    assert not RunCheck("x", 11.4, 8.004, 8.006, 0.0, (-0.1, 0.1)).accepted
