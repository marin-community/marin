# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Completions as replayed messages and as RolloutEngine turns.

``replay_arguments`` keeps tool-call arguments replayable: GLM's chat template rejects a replayed
call whose arguments are not a JSON object, so anything else is wrapped as
``{"invalid_arguments": <raw text>}``. ``model_turn`` is one completion as a messages-trace
``ModelTurn``: the reply as a chat message and the call's usage, with empty token ids and no
logprobs, so the engine charges its usage against the attempt's budget and never trains on it.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from rolloutengine.models import LENGTH_STOP_REASON, ModelTurn, TokenUsage

from taskforge.llm.client import Attempt, Completion, FinishReason, ToolCall

INVALID_ARGUMENTS_KEY = "invalid_arguments"


def replay_arguments(arguments: str) -> str:
    """The arguments to replay: unchanged if they parse to a JSON object, else wrapped in one."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return arguments
    return json.dumps({INVALID_ARGUMENTS_KEY: arguments})


def wire_message(content: str, reasoning: str, tool_calls: Sequence[ToolCall]) -> dict[str, Any]:
    """An assistant turn as it is replayed: ``reasoning_content`` is what GLM's template renders."""
    message: dict[str, Any] = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in tool_calls
        ]
    return message


def assistant_wire_message(completion: Completion) -> dict[str, Any]:
    """The assistant turn of ``completion`` as it is replayed (``wire_message``).

    Tool-call arguments stay exactly as served, even when malformed, so a rollout replays what the
    model sampled. RolloutEngine ends the rollout on arguments that are not valid JSON. When the
    arguments parse to a value other than an object, it returns an error observation and asks for
    another turn, and vLLM answers the replay of this message with a 400 from GLM's chat template.
    GLM's calls carry object arguments, because vLLM's GLM tool parser builds them from the call's
    ``<arg_key>`` pairs.
    """
    return wire_message(completion.content, completion.reasoning, completion.tool_calls)


def attempts_metadata(attempts: Sequence[Attempt]) -> list[dict[str, Any]]:
    return [
        {
            "segment": a.segment,
            "outcome": str(a.outcome),
            "http_status": a.http_status,
            "max_tokens": a.max_tokens,
            "duration": a.duration,
        }
        for a in attempts
    ]


def model_turn(completion: Completion) -> ModelTurn:
    return ModelTurn(
        message=assistant_wire_message(completion),
        prompt_token_ids=(),
        response_token_ids=(),
        logprobs=None,
        stop_reason=(
            LENGTH_STOP_REASON if completion.finish_reason is FinishReason.LENGTH else str(completion.finish_reason)
        ),
        text=completion.content,
        metadata={
            "usage": asdict(completion.usage),
            "finish_reason": str(completion.finish_reason),
            "wall_time": completion.wall_time,
            "ttft": completion.ttft,
            "attempts": attempts_metadata(completion.attempts),
        },
        usage=TokenUsage(completion.usage.prompt_tokens, completion.usage.completion_tokens),
    )
