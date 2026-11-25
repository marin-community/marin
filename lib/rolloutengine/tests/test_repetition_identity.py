# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared action and observation identity in ``repetition``."""

import json

from rolloutengine.repetition import (
    ActionPair,
    action_identity,
    canonical_arguments,
    detect_stall,
    duplicate_tool_call_fraction,
    maximal_runs,
    parse_commands,
    repeat_turns,
    wait_only,
)


def terminus_message(*commands: dict) -> dict:
    return {"role": "assistant", "content": json.dumps({"analysis": "a", "plan": "p", "commands": list(commands)})}


def tool_message(*calls: tuple[str, dict]) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": str(index), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for index, (name, args) in enumerate(calls)
        ],
    }


def user_observation(text: str) -> tuple[dict, ...]:
    return ({"role": "user", "content": text},)


def test_terminus_identity_reads_keystrokes_not_reasoning():
    """Analysis and plan churn and a changed wait duration keep one action identical."""
    first = terminus_message({"keystrokes": "cat a.txt", "duration": 1.0})
    churned = json.dumps(
        {
            "analysis": "completely different reasoning",
            "plan": "new plan",
            "commands": [{"keystrokes": "cat a.txt", "duration": 9.0}],
        }
    )
    assert action_identity(first) == action_identity({"role": "assistant", "content": churned})


def test_terminus_whitespace_normalization():
    assert action_identity(terminus_message({"keystrokes": "cat   a.txt", "duration": 1.0})) == action_identity(
        terminus_message({"keystrokes": "cat a.txt", "duration": 1.0})
    )


def test_tool_call_identity_is_name_and_canonical_arguments():
    reordered = tool_message(("shell", {"command": "ls -la", "timeout": 5}))
    flipped = tool_message(("shell", {"timeout": 5, "command": "ls -la"}))
    assert action_identity(reordered) == action_identity(flipped)
    assert action_identity(reordered) != action_identity(tool_message(("shell", {"command": "ls -l"})))


def test_parallel_tool_batch_is_one_action_in_call_order():
    single = tool_message(("shell", {"command": "a"}), ("shell", {"command": "b"}))
    reversed_batch = tool_message(("shell", {"command": "b"}), ("shell", {"command": "a"}))
    assert action_identity(single) != action_identity(reversed_batch)


def test_plain_text_replies_identify_by_normalized_content():
    assert action_identity({"role": "assistant", "content": "Answer:  12"}) == action_identity(
        {"role": "assistant", "content": "Answer: 12"}
    )


def test_parse_commands_reads_json_and_xml_variants():
    json_batch = parse_commands(json.dumps({"commands": [{"keystrokes": "ls", "duration": 2.0}]}))
    assert json_batch is not None and json_batch.keystrokes == ("ls",)
    # A Terminus reply without a commands list is a plain reply, not a command batch.
    assert parse_commands(json.dumps({"analysis": "a", "plan": "p"})) is None
    assert parse_commands("the answer is 12") is None


def test_wait_only_empty_keystrokes():
    assert wait_only(terminus_message({"keystrokes": "", "duration": 10.0}))
    assert wait_only(terminus_message())
    assert not wait_only(terminus_message({"keystrokes": "cat a.txt", "duration": 1.0}))
    # One real command in a batch makes the whole batch an action.
    assert not wait_only(terminus_message({"keystrokes": "", "duration": 5.0}, {"keystrokes": "ls", "duration": 1.0}))


def test_wait_only_shell_sleep_and_poll():
    assert wait_only(tool_message(("shell", {"command": "sleep 30"})))
    assert wait_only(tool_message(("shell", {"command": "ps aux"})))
    assert wait_only(tool_message(("shell", {"command": "wait"})))
    assert not wait_only(tool_message(("shell", {"command": "sleep 30 && cat a.txt"})))
    assert not wait_only(tool_message(("shell", {"command": "cat a.txt"})))


def test_observations_stored_as_user_turns_are_compared_exactly():
    """The #9259 blind spot: Terminus observations are user turns and enter the pair identity."""
    action = terminus_message({"keystrokes": "cat a.txt", "duration": 1.0})
    same_output = ActionPair.of(action, user_observation("New Terminal Output:\nhello"))
    changed_output = ActionPair.of(action, user_observation("New Terminal Output:\ngoodbye"))
    assert same_output.action == changed_output.action
    assert same_output.observation != changed_output.observation
    assert detect_stall([same_output, changed_output], 2) is False


def test_detect_stall_needs_full_window_and_non_wait_pairs():
    action = terminus_message({"keystrokes": "cat a.txt", "duration": 1.0})
    pair = ActionPair.of(action, user_observation("same"))
    assert detect_stall([pair, pair], 3) is False
    assert detect_stall([pair, pair, pair], 3) is True
    wait = ActionPair.of(terminus_message({"keystrokes": "", "duration": 5.0}), user_observation("same"))
    assert detect_stall([pair, pair, wait], 3) is False
    assert detect_stall([wait, wait, wait], 3) is False


def test_stall_ignores_a_wait_inside_the_window():
    action = terminus_message({"keystrokes": "cat a.txt", "duration": 1.0})
    pair = ActionPair.of(action, user_observation("same"))
    wait = ActionPair.of(terminus_message({"keystrokes": "", "duration": 5.0}), user_observation("same"))
    # wait, act, act, act: the exemption of the wait does not hide the three identical actions.
    assert detect_stall([wait, pair, pair, pair], 3) is True


def test_non_adjacent_repeats_are_not_a_run():
    assert repeat_turns(["a", "b", "a"]) == 0
    assert repeat_turns(["a", "a", "b", "a", "a", "a"]) == 3


def test_maximal_runs_counts_runs_not_pairs():
    assert maximal_runs(["a", "a", "b", "b", "b", "c"]) == [2, 3, 1]
    assert maximal_runs([]) == []


def test_retries_with_changed_replies_break_a_stall():
    """A retry whose observation differs is work, not a stall, even when the action repeats."""
    action = tool_message(("shell", {"command": "pytest -x"}))
    failing = ActionPair.of(action, ({"role": "tool", "content": "1 failed"},))
    still_failing = ActionPair.of(action, ({"role": "tool", "content": "1 failed, 1 more"},))
    assert detect_stall([failing, still_failing], 2) is False


def test_final_tool_call_participates_in_identity():
    final = tool_message(("submit", {"answer": "12"}))
    assert action_identity(final).startswith("tool")
    assert not wait_only(final)


def test_seven_versus_eight_copies():
    copies = ["a"] * 7 + ["b"]
    assert repeat_turns(copies) == 6
    assert repeat_turns(["a"] * 8) == 7


def test_duplicate_tool_call_fraction_uses_canonical_pairs():
    messages = [
        tool_message(("shell", {"command": "cat a"})),
        tool_message(("shell", {"command": "cat a"}), ("read", {"path": "b"})),
        tool_message(("shell", {"command": "cat a"})),
    ]
    # Four calls, two repeats of an earlier (name, arguments) pair.
    assert duplicate_tool_call_fraction(messages) == 0.5
    # Key order and JSON whitespace do not make a new call; the command text itself is exact.
    reordered = tool_message(("shell", {"timeout": 5, "command": "cat a"}))
    assert duplicate_tool_call_fraction([tool_message(("shell", {"command": "cat a", "timeout": 5})), reordered]) == 0.5
    assert duplicate_tool_call_fraction([tool_message(("shell", {"command": "cat  a"}))]) == 0.0
    assert duplicate_tool_call_fraction([{"role": "assistant", "content": "done"}]) is None


def test_canonical_arguments_handles_strings_and_objects():
    assert canonical_arguments('{"b": 1, "a": 2}') == canonical_arguments({"a": 2, "b": 1})
    assert canonical_arguments(None) == ""
