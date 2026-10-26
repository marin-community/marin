"""A Pi trial is complete only if its served calls and agent turns agree."""

from copy import deepcopy

import pytest

from harbor.literal.rollout_build import build_pi_trial_evidence


@pytest.fixture
def calls() -> list[dict]:
    first = {
        "request": {"messages": [{"role": "user", "content": "write answer"}]},
        "status_code": 200,
        "literal": {
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3],
            "logprobs": [-0.5],
            "assistant_message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"echo 1"}',
                        },
                    }
                ],
            },
        },
    }
    second = {
        "request": {
            "messages": [
                *deepcopy(first["request"]["messages"]),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "echo 1"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ]
        },
        "status_code": 200,
        "literal": {
            "prompt_token_ids": [1, 2, 3, 4],
            "completion_token_ids": [5],
            "logprobs": [-0.7],
            "assistant_message": {"role": "assistant", "content": "Done"},
        },
    }
    return [first, second]


def test_hydrates_one_trial_without_a_shared_proxy_log(calls: list[dict]) -> None:
    details, messages = build_pi_trial_evidence(calls)
    assert details == [
        {
            "prompt_token_ids": [[1, 2], [1, 2, 3, 4]],
            "completion_token_ids": [[3], [5]],
            "logprobs": [[-0.5], [-0.7]],
        }
    ]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("completion_token_ids", None),
        ("logprobs", None),
        ("logprobs", []),
        ("prompt_token_ids", None),
        ("assistant_message", None),
        ("logprobs", [float("nan")]),
        ("completion_token_ids", [True]),
    ],
)
def test_incomplete_sampled_evidence_fails(
    calls: list[dict], field: str, replacement: object
) -> None:
    bad = deepcopy(calls)
    bad[0]["literal"][field] = replacement
    with pytest.raises(ValueError, match="incomplete literal evidence"):
        build_pi_trial_evidence(bad)


def test_teacher_only_turns_keep_ids_and_tool_continuation_without_probabilities(
    calls: list[dict],
) -> None:
    teacher = deepcopy(calls)
    for entry in teacher:
        entry["actor"] = "teacher"
        entry["literal"]["logprobs"] = None
    details, messages = build_pi_trial_evidence(teacher)
    assert details[0]["completion_token_ids"] == [[3], [5]]
    assert details[0]["logprobs"] == [[], []]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

    teacher[0]["literal"]["logprobs"] = []
    with pytest.raises(ValueError, match="incomplete literal evidence"):
        build_pi_trial_evidence(teacher)


def test_replayed_tool_call_must_be_the_one_that_was_sampled(calls: list[dict]) -> None:
    bad = deepcopy(calls)
    bad[1]["request"]["messages"][1]["tool_calls"][0]["function"]["arguments"] = (
        '{"command":"echo 0"}'
    )
    with pytest.raises(ValueError, match="changed the preceding tool arguments"):
        build_pi_trial_evidence(bad)


def test_multi_tool_response_preserves_distinct_observations(calls: list[dict]) -> None:
    multi = deepcopy(calls)
    second_call = {
        "id": "call_2",
        "type": "function",
        "function": {"name": "read", "arguments": '{"path":"/tmp/result"}'},
    }
    multi[0]["literal"]["assistant_message"]["tool_calls"].append(second_call)
    multi[1]["request"]["messages"][1]["tool_calls"].append(deepcopy(second_call))
    multi[1]["request"]["messages"].append(
        {"role": "tool", "tool_call_id": "call_2", "content": "another observation"}
    )
    details, messages = build_pi_trial_evidence(multi)
    assert len(details[0]["completion_token_ids"]) == 2
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert "read" in messages[1]["content"]
    assert messages[3]["content"] == "another observation"


def test_duplicate_tool_arguments_fail_closed(calls: list[dict]) -> None:
    bad = deepcopy(calls)
    bad[0]["literal"]["assistant_message"]["tool_calls"][0]["function"]["arguments"] = (
        '{"command":"echo 1","command":"echo 0"}'
    )
    with pytest.raises(ValueError, match="changed the preceding tool arguments"):
        build_pi_trial_evidence(bad)


def test_missing_or_unrelated_model_call_fails(calls: list[dict]) -> None:
    with pytest.raises(ValueError, match="no literal model records"):
        build_pi_trial_evidence([])
    bad = deepcopy(calls)
    bad[1]["request"]["messages"][0]["content"] = "unrelated"
    with pytest.raises(ValueError, match="not a continuation"):
        build_pi_trial_evidence(bad)
