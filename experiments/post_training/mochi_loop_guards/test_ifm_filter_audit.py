# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the report-only IFM-style filter audit over SFT packs."""

import json

from experiments.post_training.mochi_loop_guards.ifm_filter_audit import (
    fire_rules,
    pack_actions,
    wait_after_unchanged_output,
)


def user(text):
    return {"role": "user", "content": text}


def command(command_text, duration=1.0):
    return {
        "role": "assistant",
        "content": json.dumps({"commands": [{"keystrokes": command_text, "duration": duration}]}),
    }


def wait_message():
    return {"role": "assistant", "content": '{"commands": [{"keystrokes": "", "duration": 10.0}]}'}


def tool_call(name, arguments):
    return {
        "role": "assistant",
        "tool_calls": [{"id": "0", "type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def conversation(*turns):
    """Interleave assistant turns with one observation each."""
    messages = []
    for turn in turns:
        messages.append(turn)
        messages.append(user("New Terminal Output:\nsame"))
    return messages


def test_pack_actions_split_assistant_and_observation():
    triples = pack_actions(conversation(command("ls"), command("ls")))
    assert len(triples) == 2
    assert triples[0][0] == triples[1][0]
    assert triples[0][1] == triples[1][1]
    assert not triples[0][2]


def test_run8_fires_on_eight_identical_commands():
    fired = fire_rules(conversation(*[command("cat a.txt")] * 8))
    assert "run8" in fired
    assert fire_rules(conversation(*[command("cat a.txt")] * 7)) == []


def test_frequent_fires_on_ten_copies_with_a_run():
    fired = fire_rules(conversation(*([command("cat a.txt")] * 10 + [command("ls")] * 4)))
    assert "frequent" in fired
    # Ten copies without a three-copy run do not fire.
    scattered = []
    for index in range(10):
        scattered += [command("cat a.txt"), command(f"probe {index}")]
    assert fire_rules(conversation(*scattered)) == []


def test_terminus8_needs_identical_messages_and_a_repeated_observation():
    fired = fire_rules(conversation(*[command("cat a.txt")] * 8))
    # Eight identical commands and one identical observation cover the run.
    assert "terminus8" in fired


def test_waits_after_unchanged_output_are_counted():
    turns = [command("cat a.txt"), wait_message(), wait_message(), wait_message()]
    messages = []
    for turn in turns:
        messages += [turn, user("unchanged")]
    # Turn 1 has no earlier observation to be "unchanged" against; turns 2 and 3 do.
    assert wait_after_unchanged_output(messages) == 2
    # Changed output between turns resets the count.
    messages = []
    for index, turn in enumerate(turns):
        messages += [turn, user(f"output {index}")]
    assert wait_after_unchanged_output(messages) == 0
