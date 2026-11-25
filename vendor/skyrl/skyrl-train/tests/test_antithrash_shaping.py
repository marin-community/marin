# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the I3 antithrash reward shaping: scope, counting, cap, and metrics."""

import json

import pytest
from skyrl_train.config.reward_shaping import (
    ANTITHRASH_CAP,
    ANTITHRASH_PER_REPEAT_PENALTY,
    AntithrashScope,
    parse_trajectory_reward_shaping_config,
)
from skyrl_train.trajectory_runners.trajectory_reward_shaping import (
    antithrash_repeats,
    reward_shaping_metrics,
    shape_trajectory_rewards,
)


def assistant(command):
    content = json.dumps({"commands": [{"keystrokes": command, "duration": 1.0}]})
    return {"role": "assistant", "content": content}


def tool_call(name, arguments):
    return {"role": "assistant", "tool_calls": [{"id": "0", "type": "function", "function": {"name": name, "arguments": arguments}}]}


def wait_message():
    return {"role": "assistant", "content": '{"commands": [{"keystrokes": "", "duration": 10.0}]}'}


def output(messages, *, reward=1.0):
    """One whole-trajectory row whose conversation is ``messages`` and whose outcome is ``reward``."""
    return {
        "response_ids": [[1]],
        "loss_masks": [[1]],
        "rewards": [reward],
        "evidence_messages": [messages],
        "stop_reasons": ["stop"],
        "error_treatments": [None],
        "is_last_step": None,
    }


def config(**antithrash):
    return {"enabled": True, "antithrash": {"enabled": True, **antithrash}}


def test_config_defaults_match_the_design():
    parsed = parse_trajectory_reward_shaping_config({})
    assert parsed.antithrash.enabled is False
    assert parsed.antithrash.per_repeat_penalty == ANTITHRASH_PER_REPEAT_PENALTY == 0.02
    assert parsed.antithrash.cap == ANTITHRASH_CAP == 0.1
    assert parsed.antithrash.applies_to is AntithrashScope.ALL


def test_config_rejects_a_boolean_scope_and_unknown_keys():
    with pytest.raises(ValueError, match="applies_to"):
        parse_trajectory_reward_shaping_config({"antithrash": {"applies_to": True}})
    with pytest.raises(ValueError, match="unknown antithrash"):
        parse_trajectory_reward_shaping_config({"antithrash": {"per_repeat": 0.02}})
    with pytest.raises(ValueError, match="positive"):
        parse_trajectory_reward_shaping_config({"antithrash": {"enabled": True, "per_repeat_penalty": 0.0}})
    with pytest.raises(ValueError, match="cap"):
        parse_trajectory_reward_shaping_config({"antithrash": {"enabled": True, "cap": 0.01}})


def test_repeats_count_maximal_runs_minus_one_per_run():
    messages = [assistant("cat a.txt")] * 4 + [assistant("ls")] + [assistant("cat a.txt")] * 3
    assert antithrash_repeats(messages) == 5
    # A wait between two identical actions is transparent: the run continues across it.
    assert antithrash_repeats([assistant("cat a.txt"), wait_message(), assistant("cat a.txt")]) == 1
    assert antithrash_repeats([assistant("cat a.txt"), assistant("ls"), assistant("cat a.txt")]) == 0
    assert antithrash_repeats(None) == 0
    assert antithrash_repeats([tool_call("bash", '{"command": "ls"}')] * 2) == 1


def test_correct_only_scope_charges_only_full_credit():
    shaped = output([assistant("cat a.txt")] * 3, reward=1.0)
    shape_trajectory_rewards(shaped, config(applies_to="correct_only"))
    assert shaped["reward_shaping_components"][0]["antithrash"] == -0.04

    failing = output([assistant("cat a.txt")] * 3, reward=0.0)
    shape_trajectory_rewards(failing, config(applies_to="correct_only"))
    assert failing["reward_shaping_components"][0]["antithrash"] == 0.0

    # The all scope charges the failing trajectory too.
    shape_trajectory_rewards(failing, config(applies_to="all"))
    assert failing["reward_shaping_components"][0]["antithrash"] == -0.04


def test_cap_bounds_the_charge_and_zero_repeats_charge_nothing():
    shaped = output([assistant("cat a.txt")] * 9, reward=1.0)
    shape_trajectory_rewards(shaped, config())  # eight repeats at 0.02 = 0.16, capped at 0.1
    assert shaped["reward_shaping_components"][0]["antithrash"] == -0.1

    clean = output([assistant("one"), assistant("two"), assistant("three")], reward=1.0)
    shape_trajectory_rewards(clean, config())
    assert clean["reward_shaping_components"][0]["antithrash"] == 0.0


def test_disabled_config_never_charges():
    shaped = output([assistant("cat a.txt")] * 4, reward=1.0)
    shape_trajectory_rewards(shaped, {"enabled": True, "antithrash": {"enabled": False}})
    assert shaped["reward_shaping_components"][0]["antithrash"] == 0.0


def test_shaping_without_evidence_messages_stays_defined():
    shaped = {"response_ids": [[1]], "loss_masks": [[1]], "rewards": [1.0]}
    shape_trajectory_rewards(shaped, config())
    assert shaped["reward_shaping_components"][0]["antithrash"] == 0.0


def test_metrics_report_fired_fraction_and_mean_penalty():
    charged = output([assistant("cat a.txt")] * 3, reward=1.0)
    clean = output([assistant("one"), assistant("two")], reward=1.0)
    for batch in (charged, clean):
        shape_trajectory_rewards(batch, config(applies_to="correct_only"))
    metrics = reward_shaping_metrics(
        trajectory_components=[batch["reward_shaping_components"][0] for batch in (charged, clean)],
        shaped_totals=[1.0, 1.0],
        trajectory_lengths=[1.0, 1.0],
        trajectory_stops=[None, None],
        trajectory_outcomes=[1.0, 1.0],
        trajectory_loop_advantages=[0.0, 0.0],
        trajectory_loop_token_counts=[0.0, 0.0],
        charged_loop_advantages=[],
        loop_incidence=[False, False],
        reasoning_span_token_penalties=[0.0, 0.0],
        reasoning_span_incidence=[False, False],
    )
    assert metrics["generate/reward_shaping/antithrash_fired_fraction"] == 0.5
    assert metrics["generate/reward_shaping/antithrash_mean_penalty"] == -0.02


def test_the_charge_lands_on_the_optimization_reward_not_the_outcome():
    shaped = output([assistant("cat a.txt")] * 3, reward=1.0)
    shape_trajectory_rewards(shaped, config())
    assert shaped["unshaped_rewards"] == [1.0]
    assert shaped["rewards"] == [0.96]
