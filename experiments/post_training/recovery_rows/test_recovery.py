# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import random

import pytest

from experiments.post_training.recovery_rows import recovery
from experiments.post_training.recovery_rows.recovery import INJECTED_KEY, Failure, Harness, Injection

TURN = {
    "analysis": "The directory has signs.txt and sum.txt; we need to inspect both before writing players.txt.",
    "plan": "List the files, then view the first lines of each.",
    "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}, {"keystrokes": "head -5 signs.txt\n", "duration": 0.1}],
    "task_complete": False,
}


def terminus_trajectory(turns: int = 5) -> list[dict]:
    messages = [{"role": "user", "content": "Task: inspect the files."}]
    for step in range(turns):
        messages.append({"role": "assistant", "content": json.dumps(TURN | {"plan": f"Step {step}."}, indent=2)})
        messages.append({"role": "user", "content": f"New Terminal Output:\n\nstep {step}\n"})
    return messages[:-1]


def mini_swe_trajectory(turns: int = 5) -> list[dict]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Fix it."}]
    for step in range(turns):
        call = {"id": f"c{step}", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
        messages.append({"role": "assistant", "content": f"Look at {step}.", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": f"c{step}", "content": f"listing {step}"})
    return messages


def failures(*styles: str) -> tuple[Failure, ...]:
    return tuple(Failure(style, seed) for seed, style in enumerate(styles))


@pytest.mark.parametrize("style", sorted(recovery.TERMINUS_VARIANTS))
def test_every_terminus_variant_gets_its_styles_parser_error(style):
    for variant in recovery.TERMINUS_VARIANTS[style]:
        texts = set()
        for seed in range(30):
            text = recovery.terminus_variant(TURN, style, variant, random.Random(seed))
            assert recovery.PARSER.parse_response(text).error.startswith(recovery.TERMINUS_ERRORS[style])
            texts.add(text)
        assert len(texts) > 1, f"{style}/{variant} renders one fixed text"


def test_consecutive_terminus_failures_differ():
    for seed in range(30):
        rng, previous = random.Random(seed), None
        for style in ("invalid_json", "invalid_json", "no_json", "no_json", "missing_fields"):
            text, variant = recovery.render_terminus_failure(TURN, style, rng, previous)
            assert text != previous and variant in recovery.TERMINUS_VARIANTS[style]
            previous = text


@pytest.mark.parametrize(
    ("harness", "trajectory"),
    [(Harness.TERMINUS_2, terminus_trajectory), (Harness.MINI_SWE_AGENT, mini_swe_trajectory)],
)
def test_injection_keeps_the_original_messages_and_marks_every_added_one(harness, trajectory):
    messages = trajectory()
    styles = ("invalid_json", "no_json", "missing_fields") if harness == Harness.TERMINUS_2 else ("no_tool_call",) * 3
    plan = [Injection(1, failures(*styles[:2])), Injection(3, failures(styles[2]))]
    out, records = recovery.inject(harness, messages, plan)

    assert [m for m in out if not m.get(INJECTED_KEY)] == messages
    assert [r.turn for r in records] == [1, 1, 3]
    injected = [m for m in out if m.get(INJECTED_KEY)]
    replies = [m for m in injected if m["role"] == "user"]
    assert len(replies) == 3
    original_turn_1 = [m for m in messages if m["role"] == "assistant"][1]
    if harness == Harness.MINI_SWE_AGENT:
        # mini-swe-agent drops unparsable replies: consecutive error messages with no assistant turn between them.
        assert injected == replies and all(m["content"] == recovery.MINI_SWE_NO_TOOL_CALL for m in replies)
        start = out.index(replies[0])
        assert out[start : start + 3] == [replies[0], replies[1], original_turn_1]
    else:
        assert [m["role"] for m in injected] == ["assistant", "user"] * 3
        assert [r.style for r in records] == list(styles)


def test_first_and_last_turns_are_never_injection_points():
    for harness, messages in (
        (Harness.TERMINUS_2, terminus_trajectory()),
        (Harness.MINI_SWE_AGENT, mini_swe_trajectory()),
    ):
        assert recovery.eligible_turns(harness, messages) == [1, 2, 3]
        for seed in range(50):
            for injection in recovery.plan_injections(random.Random(seed), harness, messages):
                assert 1 <= injection.turn <= 3 and 1 <= len(injection.failures) <= 3


def test_a_terminus_failure_is_labeled_with_the_parsers_error():
    # Prose with braces in it is not "No valid JSON": the parser extracts "{3}" and reports Invalid JSON.
    messages = terminus_trajectory()
    data = TURN | {"analysis": "Match the regex `PROD-[A-Z]{3}-[0-9]{4}`.", "plan": "Write `[TIMESTAMP] {json}`."}
    messages[5] = {"role": "assistant", "content": json.dumps(data, indent=2)}
    _, records = recovery.inject(Harness.TERMINUS_2, messages, [Injection(2, failures("no_json"))])
    assert [(r.style, r.error) for r in records] == [("invalid_json", "Invalid JSON")]


def test_a_turn_whose_analysis_quotes_a_valid_response_still_gets_a_rejected_failure():
    # Every rendering with the prose carries the quoted object, which the parser extracts and accepts.
    quoted = json.dumps({"analysis": "a", "plan": "b", "commands": [{"keystrokes": "ls\n", "duration": 0.1}]})
    data = TURN | {"analysis": f"Previous response: {quoted}", "plan": "Run ls."}
    assert recovery.terminus_fields(json.dumps(data)) is not None
    for style in sorted(recovery.TERMINUS_STYLES):
        for seed in range(30):
            text, _ = recovery.render_terminus_failure(data, style, random.Random(seed), None)
            assert recovery.PARSER.parse_response(text).error, (style, seed, text)

    messages = terminus_trajectory()
    messages[3] = {"role": "assistant", "content": json.dumps(data, indent=2)}
    out, records = recovery.inject(Harness.TERMINUS_2, messages, [Injection(1, (Failure("no_json", 13),))])
    assert records == [recovery.InjectedFailure(1, "invalid_json", "fallback_invalid_json", "Invalid JSON")]
    reply = next(m for m in out if m.get(INJECTED_KEY) and m["role"] == "user")
    assert reply["content"].startswith("Previous response had parsing errors:\nERROR: Invalid JSON")


def test_smaller_plans_shrink_one_attempt_at_a_time_then_drop_the_last_point():
    plan = [Injection(1, failures("a", "b")), Injection(3, failures("c", "d", "e"))]
    shapes = [[(i.turn, len(i.failures)) for i in step] for step in recovery.smaller_plans(plan)]
    assert shapes == [[(1, 2), (3, 3)], [(1, 2), (3, 2)], [(1, 2), (3, 1)], [(1, 1), (3, 1)], [(1, 1)]]
