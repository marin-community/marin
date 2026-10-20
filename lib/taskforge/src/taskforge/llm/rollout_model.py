# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 as a RolloutEngine model for the messages trace: ``ModelRequest`` in, ``ModelTurn`` out.

Every turn is one ``GlmClient.complete`` call. The turn carries the reply as a chat message and the
call's usage, which is what the trial's token accounting counts; its token ids are empty and its
``logprobs`` ``None``. Under ``TraceMode.MESSAGES`` the engine accepts such a turn, charges its usage
against the attempt's token budget, and never trains on it. The rollout model requests neither
token ids nor logprobs. Chat completions take text, and each prompt is the server's own rendering of
the conversation. A request that carries token ids (a ``Continuation``, or pinned prompt ids) is
refused: only the exact-token trace sends them.

A rollout turn never continues on ``finish_reason == "length"``: the policy must set
``max_continuations=0``, so a turn is one request and the engine grades a cut turn with stop reason
``length``.

When the prompt alone fills the context window, the turn raises ``GenerationLimitReached`` with no
prompt ids: the server rendered the prompt, and the messages trace keeps no tokens of it.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Self

from rolloutengine.models import LENGTH_STOP_REASON, GenerationLimitReached, ModelRequest, ModelTurn, TokenUsage

from taskforge.llm.client import (
    Attempt,
    Completion,
    ContextExhausted,
    FinishReason,
    ModelClient,
    ToolCall,
)
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger, complete_into, llm_call_span

PROMPT_CACHE_KEY = "prompt_cache_key"


def routing_key(record: CallLedger) -> str:
    """The rollout's identity for prefix-cache affinity: ``<workpiece>:<round>:<step>``.

    A validation trial's model is bound to its attempt's step, ``<kind>/<trial>/<attempt>``
    (``GlmRolloutModel.for_step``), so each trial has its own key and the k trials of a workpiece can
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


@dataclass(frozen=True)
class GlmRolloutModel:
    """The RolloutEngine model callable over a shared ``ModelClient`` such as ``GlmClient``.

    ``request.options`` (tools, ``tool_choice``, ``parallel_tool_calls``) are sent as request fields,
    and every request of a rollout carries its ``routing_key`` as ``prompt_cache_key``. Each turn is
    one ``LLM_CALL`` span under ``record`` tagged with ``turn``, the number of assistant turns already
    in the conversation. Rollouts that share a model share its routing key, so derive one per rollout
    with ``for_step``.
    """

    client: ModelClient
    policy: LLMPolicy
    record: CallLedger

    def __post_init__(self) -> None:
        if self.policy.max_continuations != 0:
            raise ValueError("A rollout model cannot continue on length; set LLMPolicy.max_continuations=0")

    def for_step(self, step: str) -> Self:
        """This model recording, and routing, under ``step`` of the same workpiece and round."""
        return replace(self, record=replace(self.record, step=step))

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if request.continuation is not None or request.prompt_token_ids is not None:
            raise ValueError("GLM serves messages only, so the rollout must use TraceMode.MESSAGES")
        turn = sum(message.get("role") == "assistant" for message in request.messages)
        request_fields = {**request.options, PROMPT_CACHE_KEY: routing_key(self.record)}
        try:
            with llm_call_span(self.record, self.client.model, {"turn": str(turn)}) as fields:
                completion = await complete_into(fields, self.client, request.messages, self.policy, request_fields)
        except ContextExhausted as error:
            raise GenerationLimitReached(()) from error
        return model_turn(completion)
