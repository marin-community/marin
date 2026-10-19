# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.sft_shootout_rows.arms import Attempt, match_plain_to_relay, relay_subsample
from experiments.post_training.sft_shootout_rows.build import plain_family_b


def episode(family: str, number: int, supervised: int) -> Attempt:
    return Attempt(f"{family}-task-{number}", family, f"{family}-trial-{number}", 0, 0, supervised, supervised * 3)


def test_relay_subsample_holds_each_family_to_its_own_budget():
    episodes = [episode("tmax", n, 100 + n) for n in range(40)] + [episode("swesmith", n, 50 + n) for n in range(40)]
    chosen, totals = relay_subsample(episodes, {"tmax": 1_000, "swesmith": 700})
    for family, budget in (("tmax", 1_000), ("swesmith", 700)):
        picked = [a.supervised for a in chosen if a.family == family]
        assert totals[family] == sum(picked) >= budget
        assert totals[family] - picked[-1] < budget


def test_relay_subsample_of_one_family_matches_the_full_build():
    episodes = [episode("tmax", n, 100 + n) for n in range(40)] + [episode("swesmith", n, 50 + n) for n in range(40)]
    alone, _ = relay_subsample([a for a in episodes if a.family == "tmax"], {"tmax": 1_000})
    both, _ = relay_subsample(episodes, {"tmax": 1_000, "swesmith": 700})
    assert alone == [a for a in both if a.family == "tmax"]


def test_relay_subsample_takes_a_short_family_whole():
    episodes = [episode("swesmith", n, 10) for n in range(5)]
    chosen, totals = relay_subsample(episodes, {"swesmith": 1_000})
    assert len(chosen) == 5 and totals == {"swesmith": 50}


def test_match_plain_to_relay_cuts_only_the_short_family():
    chosen = {
        a.task_id: a
        for a in [episode("tmax", n, 100) for n in range(10)] + [episode("swesmith", n, 100) for n in range(10)]
    }
    kept = match_plain_to_relay(chosen, {"tmax": 1_000, "swesmith": 1_000}, {"tmax": 1_200, "swesmith": 450})
    assert sum(a.supervised for a in kept.values() if a.family == "tmax") == 1_000
    assert sum(a.supervised for a in kept.values() if a.family == "swesmith") == 500


def test_plain_family_b_excludes_replay_and_sums_to_b():
    plain = {
        "B_supervised_tokens": 30,
        "splits": {"epoch-0": {"supervised_tokens": {"replay": 8, "swesmith": 10, "tmax": 20}}},
    }
    assert plain_family_b(plain) == {"swesmith": 10, "tmax": 20}
