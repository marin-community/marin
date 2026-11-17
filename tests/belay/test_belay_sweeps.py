# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.belay.__main__ import sweep_members_left, sweep_verdict
from experiments.belay.config import TRACKS, Track
from experiments.belay.results import Verdict
from experiments.belay.submit import is_sweep_member, parse_sweep, registry_of, sweep_members


def test_a_sweep_makes_one_variant_per_value_on_top_of_the_shared_settings() -> None:
    sweep = parse_sweep("--model-set", "qk_mult=22.6, 15,30")
    members = sweep_members("X0229", "QK logit scale", [("--model-set", "mla_q_norm=false")], sweep)
    assert [m.experiment for m in members] == ["X0229v1", "X0229v2", "X0229v3"]
    assert members[1].variant == "QK logit scale, qk_mult=15"
    assert members[1].settings == [("--model-set", "mla_q_norm=false"), ("--model-set", "qk_mult=15")]
    # Sweep members are their idea's experiments, like ladder rungs, and not mistaken for rungs.
    assert {registry_of(m.experiment) for m in members} == {"X0229"}
    assert [is_sweep_member(e) for e in ("X0229v2", "X0229", "X0229d768")] == [True, False, False]


@pytest.mark.parametrize("text", ["qk_mult=22.6", "qk_mult=1,2,3,4,5", "qk_mult=1,1", "=1,2"])
def test_a_sweep_needs_two_to_four_distinct_values(text: str) -> None:
    with pytest.raises(SystemExit):
        parse_sweep("--model-set", text)


def test_a_swept_field_is_not_also_a_shared_setting() -> None:
    sweep = parse_sweep("--model-set", "qk_mult=15,30")
    with pytest.raises(SystemExit, match="swept"):
        sweep_members("X0229", "cap", [("--model-set", "qk_mult=22.6")], sweep)


def trial(tid: str, run: str, delta: float | None = None, variant: str = "") -> dict:
    result = {"loss": [3.0], "delta": [delta]} if delta is not None else {}
    return {"id": tid, "rung": "speedrun", "wandb": [run], "variant": variant, **result}


def test_a_sweep_waits_for_values_still_running_or_unrecorded() -> None:
    trials = [trial("t001", "belay-X0229v1-seed0", -0.01), trial("t002", "belay-X0229v2-seed0")]
    assert sweep_members_left(trials, "X0229v1", live=set()) == ["X0229v2"]
    # v2 recorded at equal data, but its equal-time stage is still going (a live probe): the sweep waits.
    trials[1] = trial("t002", "belay-X0229v2-seed0", 0.0)
    assert sweep_members_left(trials, "X0229v1", live={"X0229v2"}) == ["X0229v2"]
    assert sweep_members_left(trials, "X0229v1", live=set()) == []


def test_a_finished_sweep_is_judged_by_its_best_value_using_each_values_latest_trial() -> None:
    trials = [
        trial("t001", "belay-X0229v1-seed0", -0.008, "cap, qk_mult=22.6"),  # equal data, superseded below
        trial("t002", "belay-X0229v2-seed0", 0.001, "cap, qk_mult=15"),
        trial("t003", "belay-X0229v1-seed0-t", -0.002, "cap, qk_mult=22.6"),  # v1's equal-time result
        trial("t004", "belay-X0229v3-seed0", -0.006, "cap, qk_mult=30"),
    ]
    overall, status = sweep_verdict(trials, TRACKS[Track.SPEEDRUN])
    assert overall is Verdict.WIN
    assert "best cap, qk_mult=30" in status
