# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 as a RolloutEngine model for the messages trace: ``ModelRequest`` in, ``ModelTurn`` out.

Every turn is one ``GlmClient.complete`` call with ``return_token_ids`` and ``logprobs``. The turn
reports the token ids and logprobs the server sent in the raw stream events of the call's completed
attempt (vLLM sends ``prompt_token_ids`` on the first chunk and ``choices[].token_ids`` plus
``choices[].logprobs`` on every chunk that carries tokens), and the call's usage. Chat completions
take text, so each prompt is the server's own rendering of the conversation and need not extend the
ids of the turn before. Under ``TraceMode.MESSAGES`` the engine records these ids without checking
them and never trains on them; a stream that carries none gives a turn with empty ids.

The response ids include the stop token (``<|observation|>`` after tool calls, ``<|user|>`` after a
reply), which is also the first token the chat template renders for the next turn.

A rollout turn never continues on ``finish_reason == "length"``: the policy must set
``max_continuations=0``, so a turn is one request and the engine grades a cut turn with stop reason
``length``.

When the prompt alone fills the context window, the turn raises ``GenerationLimitReached`` with the
request's prefix ids, which the messages trace leaves empty.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Self

from rolloutengine.contracts import LENGTH_STOP_REASON, GenerationLimitReached, ModelRequest, ModelTurn, TokenUsage

from taskforge.llm.client import (
    Attempt,
    AttemptOutcome,
    Completion,
    ContextExhausted,
    FinishReason,
    ModelClient,
    ToolCall,
)
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger, complete_into, llm_call_span

TOKEN_FIELDS: dict[str, object] = {"return_token_ids": True, "logprobs": True}
PROMPT_CACHE_KEY = "prompt_cache_key"


@dataclass(frozen=True)
class ServedTokens:
    """The token ids and logprobs the server reported for one completed request, empty where it sent none."""

    prompt: tuple[int, ...]
    response: tuple[int, ...]
    logprobs: tuple[float, ...]


def served_tokens(completion: Completion) -> ServedTokens:
    """Read the token ids and logprobs from the raw events of ``completion``'s completed request."""
    prompt: tuple[int, ...] = ()
    response: list[int] = []
    logprobs: list[float] = []
    completed = [attempt for attempt in completion.attempts if attempt.outcome is AttemptOutcome.COMPLETED]
    for raw in completed[-1].events if completed else ():
        event = json.loads(raw)
        if event.get("prompt_token_ids"):
            prompt = tuple(event["prompt_token_ids"])
        for choice in event.get("choices", []):
            response.extend(choice.get("token_ids") or ())
            logprobs.extend(item["logprob"] for item in (choice.get("logprobs") or {}).get("content") or ())
    return ServedTokens(prompt, tuple(response), tuple(logprobs))


def routing_key(record: CallLedger) -> str:
    """The rollout's identity for prefix-cache affinity: ``<item>:<round>:<step>``.

    A validation trial's model is bound to its attempt's step, ``<kind>/<trial>/<attempt>``
    (``GlmRolloutModel.for_step``), so each trial has its own key and the k trials of an item can
    land on different nodes.
    """
    return f"{record.workpiece_id}:{record.round}:{record.step}"


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
    another turn, and vLLM answers the replay of this message with a 400 from GLM's chat template
    (``test_probe_replayed_tool_call_arguments_must_be_a_json_object``). GLM's calls carry object
    arguments, because vLLM's GLM tool parser builds them from the call's ``<arg_key>`` pairs.
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
    tokens = served_tokens(completion)
    return ModelTurn(
        message=assistant_wire_message(completion),
        prompt_token_ids=tokens.prompt,
        response_token_ids=tokens.response,
        logprobs=tokens.logprobs,
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


@dataclass(frozen=True)
class GlmRolloutModel:
    """The RolloutEngine model callable over a shared ``ModelClient`` such as ``GlmClient``.

    ``request.options`` (tools, ``tool_choice``, ``parallel_tool_calls``) are sent as request fields
    after the token fields, so a task's options win. Every request of a rollout carries its
    ``routing_key`` as ``prompt_cache_key``. Each turn is one ``LLM_CALL`` span under ``record``
    tagged with ``turn``, the number of assistant turns already in the conversation. Rollouts that
    share a model share its routing key, so derive one per rollout with ``for_step``.
    """

    client: ModelClient
    policy: LLMPolicy
    record: CallLedger

    def __post_init__(self) -> None:
        if self.policy.max_continuations != 0:
            raise ValueError("A rollout model cannot continue on length; set LLMPolicy.max_continuations=0")

    def for_step(self, step: str) -> Self:
        """This model recording, and routing, under ``step`` of the same item and round."""
        return replace(self, record=replace(self.record, step=step))

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        turn = sum(message.get("role") == "assistant" for message in request.messages)
        request_fields = {**TOKEN_FIELDS, **request.options, PROMPT_CACHE_KEY: routing_key(self.record)}
        try:
            with llm_call_span(self.record, self.client.model, {"turn": str(turn)}) as fields:
                completion = await complete_into(fields, self.client, request.messages, self.policy, request_fields)
        except ContextExhausted as error:
            raise GenerationLimitReached(request.prefix_token_ids) from error
        return model_turn(completion)
