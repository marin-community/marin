# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from experiments.post_training.tmax_reserve.pilot import apt_usage, pass_at_k


def _trial(root, name, keystrokes, screen):
    agent = root / name / "attempts" / "000" / "agent"
    agent.mkdir(parents=True)
    step = {
        "tool_calls": [{"arguments": {"keystrokes": keystrokes}}],
        "observation": {"results": [{"content": screen}]},
    }
    (agent / "trajectory.json").write_text(json.dumps({"steps": [{"source": "user"}, step]}))
    return root / name


def test_apt_usage_counts_apt_commands_and_mirror_downloads(tmp_path):
    trials = [
        _trial(
            tmp_path,
            "fetched",
            "apt-get update && apt-get -y install curl\n",
            "Get:1 http://archive.ubuntu.com/ubuntu jammy InRelease",
        ),
        _trial(tmp_path, "blocked", "sudo apt install jq\n", "Err:1 http://archive.ubuntu.com/ubuntu jammy InRelease"),
        _trial(tmp_path, "none", "pytest -q tests/\n", "1 passed"),
        _trial(tmp_path, "mention", "echo 'apt-cache policy is not an install'\n", ""),
    ]
    assert apt_usage(trials) == {"trials_scanned": 4, "ran_apt": 2, "fetched_from_mirror": 1}


def test_pass_at_k_ignores_unscored_trials():
    trials = [
        {"task": "a", "reward": 1.0, "exception": None},
        {"task": "a", "reward": 0.0, "exception": None},
        {"task": "b", "reward": 0.0, "exception": None},
        {"task": "b", "reward": None, "exception": "AgentTimeoutError"},
    ]
    result = pass_at_k(trials)
    assert result["tasks"] == {"a": {"attempts": 2, "passes": 1}, "b": {"attempts": 1, "passes": 0}}
    assert result["mean_pass_at_1"] == 0.25 and result["pass_at_k"] == 0.5
