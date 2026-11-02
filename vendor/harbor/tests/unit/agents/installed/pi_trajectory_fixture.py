"""Build independent, fully specified fake Pi frames for Harbor unit tests only."""

from typing import Any


def _sampled(entry: dict[str, Any]) -> dict[str, Any]:
    request, literal = entry["request"], entry["literal"]
    return {
        "request_id": entry["request_id"],
        "model": request.get("model", "tiny"),
        "messages": request["messages"],
        "tools": request.get("tools", []),
        "chat_template_kwargs": {},
        "assistant": literal["assistant_message"],
        "prompt_token_ids": literal["prompt_token_ids"],
        "response_token_ids": literal["completion_token_ids"],
        "logprobs": literal["logprobs"],
    }


def fake_pi_trajectory(
    entries: list[dict[str, Any]],
    *,
    critique: dict[str, Any] | None = None,
    dagger_turns: list[dict[str, Any]] | None = None,
    trial_id: str = "trial-1",
    policy_step: int = 0,
) -> dict[str, Any]:
    turns = []
    for index, entry in enumerate(entries):
        sampled = _sampled(entry)
        following = (
            entries[index + 1]["request"]["messages"]
            if index + 1 < len(entries)
            else []
        )
        observation_messages = following[len(sampled["messages"]) + 1 :]
        observations = [
            {
                "tool_call_id": call["id"],
                "content": next(
                    message["content"]
                    for message in observation_messages
                    if message.get("role") == "tool"
                    and message.get("tool_call_id") == call["id"]
                ),
                "authored_by": "teacher"
                if entry.get("actor") == "teacher"
                else "student",
            }
            for call in sampled["assistant"].get("tool_calls", [])
        ]
        turns.append(
            {
                "index": index,
                "actor": entry.get("actor", "student"),
                "executed": sampled,
                "executed_assistant": following[len(sampled["messages"])]
                if following
                else sampled["assistant"],
                "observations": observations,
                "teacher_label": _sampled(dagger_turns[index]["teacher_label"])
                if dagger_turns
                else None,
            }
        )
    return {
        "schema_version": 1,
        "trial_id": trial_id,
        "policy_step": policy_step,
        "turns": turns,
        "planned_executors": [turn["actor"] for turn in dagger_turns]
        if dagger_turns
        else None,
        "critique_feedback": critique["feedback"] if critique else None,
        "critique": critique,
    }
