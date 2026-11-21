# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Completions as replayed messages and as RolloutEngine turns.

``model_turn`` is one completion as a messages-trace ``ModelTurn``: the reply as a chat message and the call's usage,
with empty token ids and no logprobs, so the engine charges its usage against the attempt's budget and never trains on
it.
"""

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from rolloutengine.models import LENGTH_STOP_REASON, ModelTurn, TokenUsage

from taskforge.llm.client import Attempt, Completion
from taskforge.model.llm import FinishReason, ToolCall


def wire_tool_call(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}


def wire_message(content: str, reasoning: str, tool_calls: Sequence[ToolCall]) -> dict[str, Any]:
    """An assistant turn as it is replayed: ``reasoning_content`` is what GLM's template renders."""
    message: dict[str, Any] = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if tool_calls:
        message["tool_calls"] = [wire_tool_call(call) for call in tool_calls]
    return message


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
    # Tool-call arguments stay exactly as served, even when malformed, so a rollout replays what the model sampled.
    return ModelTurn(
        message=wire_message(completion.content, completion.reasoning, completion.tool_calls),
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
