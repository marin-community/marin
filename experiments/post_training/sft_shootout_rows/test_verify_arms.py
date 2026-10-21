# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.sft_shootout_rows.verify_arms import ArmCheck, check_relay

PLAIN = {
    "B_supervised_tokens": 30,
    "splits": {"epoch-0": {"supervised_tokens": {"swesmith": 10, "tmax": 20, "replay": 7}}},
}


def test_a_full_relay_arm_with_replay_rows_verifies():
    episodes = [("a", "swesmith", 6), ("b", "swesmith", 5), ("c", "tmax", 21)]
    index = [
        {"epoch": epoch, "trial_id": trial, "family": family, "supervised_tokens": tokens}
        for epoch in range(3)
        for trial, family, tokens in episodes
    ]
    # Replay rows carry no trial id.
    index += [{"epoch": epoch, "family": "replay", "byte_offset": 0, "supervised_tokens": 7} for epoch in range(3)]
    epoch = {"agentic": {"a": 1, "b": 1, "c": 1}, "supervised": {"swesmith": 11, "tmax": 21, "replay": 7}}
    manifest = {
        "relay_component": False,
        "B_supervised_tokens": 30,
        "relay_families": {"swesmith": {"achieved": 11}, "tmax": {"achieved": 21}},
        "relay_pool": {"swesmith": {"supervised_tokens": 40}, "tmax": {"supervised_tokens": 50}},
    }
    selection = [{"trial_id": trial} for trial, _, _ in episodes]
    arm = ArmCheck(manifest, index, {0: epoch, 1: epoch, 2: epoch}, {}, selection)
    report = check_relay("relay", arm, PLAIN)
    assert report["swesmith"] == {"plain_b": 10, "achieved": 11, "delta": 1, "episodes": 2}
    assert report["tmax"] == {"plain_b": 20, "achieved": 21, "delta": 1, "episodes": 1}
