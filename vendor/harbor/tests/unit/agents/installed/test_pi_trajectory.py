"""The Pi trial payload cannot be completed by downstream log joins."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from harbor.models.agent.pi_trajectory import PiCollatedTrajectory
from tests.unit.agents.installed.pi_trajectory_fixture import fake_pi_trajectory


def _sample() -> dict:
    prompt = [{"role": "user", "content": "Fix"}]
    action = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            },
            {
                "id": "b",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            },
        ],
    }
    entries = [
        {
            "request_id": "student-1",
            "request": {"model": "student", "messages": prompt},
            "literal": {
                "assistant_message": action,
                "prompt_token_ids": [1],
                "completion_token_ids": [2],
                "logprobs": [-0.2],
            },
        },
        {
            "request_id": "student-2",
            "request": {
                "model": "student",
                "messages": [
                    *prompt,
                    action,
                    {"role": "tool", "tool_call_id": "a", "content": "one"},
                    {"role": "tool", "tool_call_id": "b", "content": "two"},
                ],
            },
            "literal": {
                "assistant_message": {"role": "assistant", "content": "done"},
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [4],
                "logprobs": [-0.3],
            },
        },
    ]
    return fake_pi_trajectory(entries)


def test_pi_preserves_sampled_and_reserialized_executed_tool_arguments():
    raw = _sample()
    raw["turns"][0]["executed_assistant"] = deepcopy(
        raw["turns"][0]["executed_assistant"]
    )
    raw["turns"][0]["executed_assistant"]["tool_calls"][0]["function"]["arguments"] = (
        "{ }"
    )
    raw["turns"][1]["executed"]["messages"][1] = deepcopy(
        raw["turns"][0]["executed_assistant"]
    )
    trial = PiCollatedTrajectory.model_validate(raw)
    assert (
        trial.turns[0].executed.assistant["tool_calls"][0]["function"]["arguments"]
        == "{}"
    )
    assert (
        trial.turns[0].executed_assistant["tool_calls"][0]["function"]["arguments"]
        == "{ }"
    )
    raw["turns"][0]["executed_assistant"]["tool_calls"][0]["function"]["arguments"] = (
        '{"bad":1}'
    )
    with pytest.raises(ValidationError, match="differs from sampled action"):
        PiCollatedTrajectory.model_validate(raw)


def test_complete_multitool_trajectory_validates_without_external_records():
    trial = PiCollatedTrajectory.model_validate(_sample())
    assert [item.content for item in trial.turns[0].observations] == ["one", "two"]
    assert trial.turns[-1].executed.messages[-1]["content"] == "two"


def test_teacher_only_ids_survive_two_turn_tool_continuation_without_probabilities():
    raw = _sample()
    for turn in raw["turns"]:
        turn["actor"] = "teacher"
        turn["executed"]["logprobs"] = None
        for observation in turn["observations"]:
            observation["authored_by"] = "teacher"
    trial = PiCollatedTrajectory.model_validate(raw)
    assert [turn.executed.response_token_ids for turn in trial.turns] == [(2,), (4,)]
    assert trial.turns[0].observations[0].tool_call_id == "a"
    assert all(turn.executed.logprobs is None for turn in trial.turns)

    raw["turns"][0]["executed"]["logprobs"] = []
    with pytest.raises(ValidationError, match="present logprobs must align"):
        PiCollatedTrajectory.model_validate(raw)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_tool",
        "wrong_actor",
        "wrong_continuation",
        "reuse_label",
        "missing_tokens",
        "missing_student_logprobs",
        "boolean_step",
    ],
)
def test_incomplete_trajectory_is_rejected(damage):
    raw = deepcopy(_sample())
    if damage == "missing_tool":
        raw["turns"][0]["observations"].pop()
    elif damage == "wrong_actor":
        raw["turns"][0]["observations"][0]["authored_by"] = "teacher"
    elif damage == "wrong_continuation":
        raw["turns"][1]["executed"]["messages"][2]["content"] = "forged"
    elif damage == "reuse_label":
        raw["turns"][0]["teacher_label"] = deepcopy(raw["turns"][0]["executed"])
    elif damage == "missing_tokens":
        raw["turns"][0]["executed"]["response_token_ids"] = []
    elif damage == "missing_student_logprobs":
        raw["turns"][0]["executed"]["logprobs"] = None
    else:
        raw["policy_step"] = True
    with pytest.raises(ValidationError):
        PiCollatedTrajectory.model_validate(raw)
