# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 as a RolloutEngine model: ``ModelRequest`` in, exact-token ``ModelTurn`` out.

Each request is one ``GlmClient.complete`` call with ``return_token_ids`` and ``logprobs``, so it
streams, retries and holds like every other Taskforge call. The served tokens come from the raw
stream events of the call's completed attempt: vLLM sends ``prompt_token_ids`` on the first chunk
and ``choices[].token_ids`` plus ``choices[].logprobs`` on every chunk that carries tokens. The
response ids include the stop token (``<|observation|>`` after tool calls, ``<|user|>`` after a
reply), which is also the first token the chat template renders for the next turn. Measured live
(``.evidence/validate/rollout_model/``): GLM's template re-renders a replayed assistant turn token
for token, including empty reasoning, content beside tool calls, and multi-line arguments.

The served prefix survives only when the sampled response ids are the tokenizer's canonical
encoding of their text. Chat completions take text, and the router accepts no prompt token ids, so
the next prompt is the re-tokenized conversation: a sampled ``"),"`` ``"("`` comes back as the
single token ``"),("`` (``.evidence/validate/rollout_model/prefix-bug/``). No message the client
sends can restore the sampled ids, so the model raises ``RolloutContractError`` naming where the
served prompt diverged.

A rollout turn never continues on ``finish_reason == "length"``: a continuation re-renders the
prompt and would break the token contract. The policy must set ``max_continuations=0``; the engine
grades a cut turn with stop reason ``length``.

When the prompt alone fills the context window the server rejects it without rendering it, so no
rendered prompt ids exist. ``GenerationLimitReached`` then carries the served prefix of the request
(the tokens before the observations that overflowed), which is empty on a first turn.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any

from rolloutengine.contracts import (
    LENGTH_STOP_REASON,
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
)

from taskforge.llm.client import AttemptOutcome, Completion, FinishReason, GlmClient, GlmContextExhausted
from taskforge.llm.policy import LLMPolicy

TOKEN_FIELDS: dict[str, object] = {"return_token_ids": True, "logprobs": True}


@dataclass(frozen=True)
class ServedTokens:
    """The exact token ids of one completed request, as the server reported them."""

    prompt: tuple[int, ...]
    response: tuple[int, ...]
    logprobs: tuple[float, ...]


def served_tokens(completion: Completion) -> ServedTokens:
    """Read the served token ids from the raw events of ``completion``'s single completed request.

    Raises:
        RolloutContractError: the stream lacked prompt ids, or its ids disagree with its usage.
    """
    completed = [a for a in completion.attempts if a.outcome is AttemptOutcome.COMPLETED and a.segment == 0]
    if len(completed) != 1 or completion.continuations:
        raise RolloutContractError("Exact tokens require exactly one completed request without continuation")
    prompt: tuple[int, ...] | None = None
    response: list[int] = []
    logprobs: list[float] = []
    for raw in completed[0].events:
        event = json.loads(raw)
        if event.get("prompt_token_ids"):
            prompt = tuple(event["prompt_token_ids"])
        for choice in event.get("choices", []):
            response.extend(choice.get("token_ids") or ())
            logprobs.extend(item["logprob"] for item in (choice.get("logprobs") or {}).get("content") or ())
    if prompt is None:
        raise RolloutContractError("The server returned no prompt_token_ids; it must honor return_token_ids")
    usage = completion.usage
    if (len(prompt), len(response)) != (usage.prompt_tokens, usage.completion_tokens):
        raise RolloutContractError(
            f"Served ids ({len(prompt)} prompt, {len(response)} response) disagree with usage "
            f"({usage.prompt_tokens}, {usage.completion_tokens})"
        )
    return ServedTokens(prompt, tuple(response), tuple(logprobs))


def check_served_prefix(prefix: tuple[int, ...], prompt: tuple[int, ...]) -> None:
    """Raise if the served ``prompt`` does not start with the rollout's ``prefix`` ids.

    Raises:
        RolloutContractError: naming the first divergent index and the ids on both sides.
    """
    if prompt[: len(prefix)] == prefix:
        return
    index = next((i for i, (a, b) in enumerate(zip(prefix, prompt, strict=False)) if a != b), len(prompt))
    raise RolloutContractError(
        f"The server re-tokenized the replayed conversation: at index {index} of the {len(prefix)}-token "
        f"prefix it served {list(prompt[index : index + 8])} where the rollout has {list(prefix[index : index + 8])}. "
        "Chat completions re-render text, so a sampled response that is not the canonical tokenization of its "
        "text cannot keep the served prefix"
    )


def assistant_wire_message(completion: Completion) -> dict[str, Any]:
    """The assistant turn as it is replayed: ``reasoning_content`` is what GLM's template renders.

    Tool-call arguments stay exactly as served, even when malformed: RolloutEngine grades this
    message and ends the rollout on arguments that are not a JSON object, so no later request
    replays them, and rewriting them would change the served token prefix.
    """
    message: dict[str, Any] = {
        "role": "assistant",
        "content": completion.content,
        "reasoning_content": completion.reasoning,
    }
    if completion.tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in completion.tool_calls
        ]
    return message


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
            "attempts": [
                {
                    "segment": a.segment,
                    "outcome": str(a.outcome),
                    "http_status": a.http_status,
                    "max_tokens": a.max_tokens,
                    "duration": a.duration,
                }
                for a in completion.attempts
            ],
        },
    )


@dataclass(frozen=True)
class GlmRolloutModel:
    """The RolloutEngine model callable over a shared ``GlmClient``.

    ``request.options`` (tools, ``tool_choice``, ``parallel_tool_calls``) are sent as request fields
    after the token fields, so a task's options win.
    """

    client: GlmClient
    policy: LLMPolicy

    def __post_init__(self) -> None:
        if self.policy.max_continuations != 0:
            raise ValueError("A rollout model cannot continue on length; set LLMPolicy.max_continuations=0")

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        try:
            completion = await self.client.complete(request.messages, self.policy, {**TOKEN_FIELDS, **request.options})
        except GlmContextExhausted as error:
            raise GenerationLimitReached(request.prefix_token_ids) from error
        turn = model_turn(completion)
        check_served_prefix(request.prefix_token_ids, turn.prompt_token_ids)
        return turn
