# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 as a RolloutEngine model: ``ModelRequest`` in, exact-token ``ModelTurn`` out.

Two transports (``RolloutTransport``):

* ``CHAT_COMPLETIONS``: every turn is one ``GlmClient.complete`` call with ``return_token_ids`` and
  ``logprobs``. The served tokens come from the raw stream events of the call's completed attempt:
  vLLM sends ``prompt_token_ids`` on the first chunk and ``choices[].token_ids`` plus
  ``choices[].logprobs`` on every chunk that carries tokens. Chat completions take text, so the
  next prompt is the re-tokenized conversation. The served prefix survives only when the sampled
  response ids are the tokenizer's canonical encoding of their text: a sampled ``"),"`` ``"("``
  comes back as the single token ``"),("``, and the turn raises ``TokenContractBreak``.
* ``TOKEN_ROUTE``: the first turn of a conversation (no prefix yet) is the chat-completions call
  above. Every later turn renders the conversation once on the server (a one-token chat-completions
  probe, ``purpose=render``), takes the ids the template renders after the previous assistant turn
  (``rendered_suffix``), and samples from ``prefix + suffix`` through ``GlmClient.generate``. The
  sampled ids are kept verbatim, so the served prefix holds by construction, also after the server
  re-tokenized a replayed response (``SuffixAlignment.BOUNDARY``). Reasoning and tool calls are
  parsed from the sampled bytes (``llm.glm_format``).

The response ids include the stop token (``<|observation|>`` after tool calls, ``<|user|>`` after a
reply), which is also the first token the chat template renders for the next turn.

A rollout turn never continues on ``finish_reason == "length"``: a continuation re-renders the
prompt and would break the token contract. The policy must set ``max_continuations=0``; the engine
grades a cut turn with stop reason ``length``.

When the prompt alone fills the context window, ``GenerationLimitReached`` carries the prompt ids
the turn would have sent: on chat completions the server rejects the prompt without rendering it,
so that is the served prefix of the request (empty on a first turn); on the token route it is
``prefix + suffix``.

Every contract break raised by a turn is a ``TokenContractBreak`` carrying the conversation, raised
inside the turn's ``LLM_CALL`` span with ``cause == "token_contract"``, so the ledger records it
and the caller can keep the partial rollout. A token-route response whose ids, logprobs and bytes
do not align raises ``TokenEvidenceError`` instead: a resample cannot fix it, so it fails the run.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any, Self

from rolloutengine.contracts import (
    LENGTH_STOP_REASON,
    GenerationLimitReached,
    ModelRequest,
    ModelTurn,
    RolloutContractError,
)

from taskforge.ledger.records import SpanFields
from taskforge.llm.client import (
    Attempt,
    AttemptOutcome,
    Completion,
    ContextExhausted,
    FinishReason,
    ModelClient,
    TokenCompletion,
    TokenGenerator,
    ToolCall,
)
from taskforge.llm.glm_format import STOP_TEXTS, ParsedTurn, parse_turn
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger, complete_into, generate_into, llm_call_span, record_usage

GLM_CONTEXT_WINDOW = 262_144
TOKEN_FIELDS: dict[str, object] = {"return_token_ids": True, "logprobs": True}
NO_GENERATION_PROMPT: dict[str, object] = {"add_generation_prompt": False}
PROMPT_CACHE_KEY = "prompt_cache_key"
TOKEN_CONTRACT_CAUSE = "token_contract"
DETAIL_LIMIT = 2000


class RolloutTransport(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    TOKEN_ROUTE = "token_route"


class SuffixAlignment(StrEnum):
    PREFIX = "prefix"
    """The served render starts with the rollout's prefix."""
    BOUNDARY = "boundary"
    """The server re-tokenized a replayed response; the suffix starts at the matching stop token."""


class TokenContractBreak(RolloutContractError):
    """The served evidence cannot keep the exact-token prefix; carries what a trial needs to keep the partial rollout."""

    def __init__(self, message: str, *, turn: int, messages: tuple[dict[str, Any], ...], prefix_length: int):
        super().__init__(message)
        self.turn = turn
        self.messages = messages
        self.prefix_length = prefix_length


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


def rendered_suffix(prefix: tuple[int, ...], served: tuple[int, ...]) -> tuple[tuple[int, ...], SuffixAlignment]:
    """The ids the template renders after ``prefix``'s last assistant turn.

    ``prefix`` ends with the previous response's stop token (``<|observation|>`` after tool calls,
    ``<|user|>`` after a reply). When ``served`` starts with ``prefix`` the suffix is the rest
    (``SuffixAlignment.PREFIX``). Otherwise the server re-tokenized a replayed response
    differently: the suffix starts after the n-th occurrence in ``served`` of the id ``prefix[-1]``,
    where n is its count in ``prefix`` (``SuffixAlignment.BOUNDARY``). The count is over token ids.
    A stop token's text inside a tool result counts the same on both sides, because every tool
    result in ``prefix`` came from an earlier server render, and one in the new turn lies after the
    boundary. Only a replayed reply that sampled a stop token's text as ordinary tokens holds the id
    in ``served`` but not in ``prefix``; ``replayed_stop_texts`` finds those before the cut.

    Raises:
        RolloutContractError: ``served`` has fewer than n occurrences, or the suffix is empty.
    """
    if not prefix:
        raise RolloutContractError("A rendered suffix needs a non-empty prefix")
    if served[: len(prefix)] == prefix:
        suffix, alignment = served[len(prefix) :], SuffixAlignment.PREFIX
    else:
        stop = prefix[-1]
        count = prefix.count(stop)
        positions = [index for index, token in enumerate(served) if token == stop]
        if len(positions) < count:
            raise RolloutContractError(
                f"The served render holds {len(positions)} of the prefix's {count} occurrences of its final "
                f"token {stop}, so the replayed turns cannot be located"
            )
        suffix, alignment = served[positions[count - 1] + 1 :], SuffixAlignment.BOUNDARY
    if not suffix:
        raise RolloutContractError("The served render adds nothing after the prefix")
    return suffix, alignment


def replayed_stop_texts(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    """The stop-token texts written inside replayed assistant turns, which a re-render turns into stop ids."""
    texts = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or ()
        parts = [message.get("content") or "", message.get("reasoning_content") or ""]
        parts += [call["function"]["arguments"] for call in calls]
        texts += [stop for part in parts for stop in STOP_TEXTS if stop in part]
    return texts


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

    Tool-call arguments stay exactly as served, even when malformed, because rewriting them would
    change the served token prefix. RolloutEngine ends the rollout on arguments that are not valid
    JSON. When the arguments parse to a value other than an object, it returns an error observation
    and asks for another turn, and vLLM answers the replay of this message with a 400 from GLM's
    chat template (``test_probe_replayed_tool_call_arguments_must_be_a_json_object``). GLM's calls
    carry object arguments on both transports, because ``glm_format.parse_turn`` and vLLM's GLM
    tool parser build them from the call's ``<arg_key>`` pairs.
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
            "transport": str(RolloutTransport.CHAT_COMPLETIONS),
        },
    )


def token_model_turn(
    prompt: tuple[int, ...], completion: TokenCompletion, parsed: ParsedTurn, alignment: SuffixAlignment
) -> ModelTurn:
    """The ``ModelTurn`` of one token-route call, shaped as ``model_turn`` shapes a chat-completions one."""
    if completion.finish_reason is FinishReason.LENGTH:
        stop_reason = LENGTH_STOP_REASON
    else:
        stop_reason = str(FinishReason.TOOL_CALLS if parsed.tool_calls else FinishReason.STOP)
    usage = replace(completion.usage, reasoning_tokens=parsed.reasoning_tokens)
    return ModelTurn(
        message=wire_message(parsed.content, parsed.reasoning, parsed.tool_calls),
        prompt_token_ids=prompt,
        response_token_ids=completion.token_ids,
        logprobs=completion.logprobs,
        stop_reason=stop_reason,
        text=parsed.content,
        metadata={
            "usage": asdict(usage),
            "finish_reason": str(completion.finish_reason),
            "wall_time": completion.wall_time,
            "attempts": attempts_metadata(completion.attempts),
            "transport": str(RolloutTransport.TOKEN_ROUTE),
            "alignment": str(alignment),
        },
    )


def render_policy(policy: LLMPolicy) -> LLMPolicy:
    """``policy`` for a one-token render probe; only its template inputs (``reasoning_effort``) matter."""
    return replace(policy, max_tokens=1, max_continuations=0)


@dataclass(frozen=True)
class ServerRenderer:
    """The served chat template's token ids for a conversation, through one-token chat completions.

    ``policy`` must match the rollout policy's template inputs (``reasoning_effort``).
    """

    client: ModelClient
    policy: LLMPolicy

    async def prompt_ids(self, messages: Sequence[Mapping[str, Any]], options: Mapping[str, Any]) -> tuple[int, ...]:
        """The rendered conversation followed by the generation prompt."""
        return await self._served_prompt(messages, {**TOKEN_FIELDS, **options})

    async def rendered_ids(self, messages: Sequence[Mapping[str, Any]], options: Mapping[str, Any]) -> tuple[int, ...]:
        """The rendered conversation without the generation prompt."""
        return await self._served_prompt(messages, {**TOKEN_FIELDS, **options, **NO_GENERATION_PROMPT})

    async def _served_prompt(self, messages: Sequence[Mapping[str, Any]], fields: dict[str, Any]) -> tuple[int, ...]:
        completion = await self.client.complete(
            [dict(message) for message in messages], render_policy(self.policy), fields
        )
        return served_tokens(completion).prompt


@dataclass(frozen=True)
class GlmRolloutModel:
    """The RolloutEngine model callable over a shared ``TokenGenerator`` such as ``GlmClient``.

    ``request.options`` (tools, ``tool_choice``, ``parallel_tool_calls``) are sent as request fields
    after the token fields, so a task's options win. Every request of a rollout carries its
    ``routing_key``: chat-completions calls and render probes as ``prompt_cache_key``, token-route
    calls as ``ROUTING_KEY_HEADER``. Each turn is one ``LLM_CALL`` span under ``record`` tagged with
    ``turn`` (the number of assistant turns already in the conversation) and ``transport``; a
    token-route turn adds a ``purpose=render`` span for its render probe and tags its own span with
    ``alignment`` and ``routing_key``. Rollouts that share a model share its routing key, so derive
    one per rollout with ``for_step``.
    """

    client: TokenGenerator
    policy: LLMPolicy
    transport: RolloutTransport
    record: CallLedger

    def __post_init__(self) -> None:
        if self.policy.max_continuations != 0:
            raise ValueError("A rollout model cannot continue on length; set LLMPolicy.max_continuations=0")

    def for_step(self, step: str) -> Self:
        """This model recording, and routing, under ``step`` of the same item and round."""
        return replace(self, record=replace(self.record, step=step))

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        turn = sum(message.get("role") == "assistant" for message in request.messages)
        if self.transport is RolloutTransport.TOKEN_ROUTE and request.prefix_token_ids:
            return await self._token_route_turn(request, turn)
        return await self._chat_turn(request, turn)

    def _contract_break(
        self, fields: SpanFields, request: ModelRequest, turn: int, error: Exception
    ) -> TokenContractBreak:
        prefix_length = len(request.prefix_token_ids)
        fields.cause = TOKEN_CONTRACT_CAUSE
        fields.attrs.update({"prefix_length": str(prefix_length), "detail": str(error)[:DETAIL_LIMIT]})
        if isinstance(error, TokenContractBreak):
            return error
        return TokenContractBreak(str(error), turn=turn, messages=request.messages, prefix_length=prefix_length)

    async def _chat_turn(self, request: ModelRequest, turn: int) -> ModelTurn:
        attrs = {"turn": str(turn), "transport": str(RolloutTransport.CHAT_COMPLETIONS)}
        request_fields = {**TOKEN_FIELDS, **request.options, PROMPT_CACHE_KEY: routing_key(self.record)}
        try:
            with llm_call_span(self.record, self.client.model, attrs) as fields:
                completion = await complete_into(fields, self.client, request.messages, self.policy, request_fields)
                try:
                    served = model_turn(completion)
                    check_served_prefix(request.prefix_token_ids, served.prompt_token_ids)
                except RolloutContractError as error:
                    raise self._contract_break(fields, request, turn, error) from error
        except ContextExhausted as error:
            raise GenerationLimitReached(request.prefix_token_ids) from error
        return served

    async def _token_route_turn(self, request: ModelRequest, turn: int) -> ModelTurn:
        key = routing_key(self.record)
        render_fields = {**TOKEN_FIELDS, **request.options, PROMPT_CACHE_KEY: key}
        try:
            with llm_call_span(self.record, self.client.model, {"turn": str(turn), "purpose": "render"}) as fields:
                probe = await complete_into(
                    fields, self.client, request.messages, render_policy(self.policy), render_fields
                )
        except ContextExhausted as error:
            raise GenerationLimitReached(request.prefix_token_ids) from error
        attrs = {"turn": str(turn), "transport": str(RolloutTransport.TOKEN_ROUTE), "routing_key": key}
        prompt = request.prefix_token_ids
        try:
            with llm_call_span(self.record, self.client.model, attrs) as fields:
                try:
                    suffix, alignment = rendered_suffix(request.prefix_token_ids, served_tokens(probe).prompt)
                    stops = replayed_stop_texts(request.messages)
                    if alignment is SuffixAlignment.BOUNDARY and stops:
                        raise RolloutContractError(
                            f"A replayed assistant turn holds the stop text {stops[0]!r}, which the re-render "
                            "turns into a stop token, so the boundary of the replayed turns cannot be located"
                        )
                except RolloutContractError as error:
                    raise self._contract_break(fields, request, turn, error) from error
                fields.attrs["alignment"] = str(alignment)
                prompt = request.prefix_token_ids + suffix
                max_tokens = min(self.policy.max_tokens, GLM_CONTEXT_WINDOW - len(prompt))
                if max_tokens < 1:
                    raise GenerationLimitReached(prompt)
                completion = await generate_into(fields, self.client, prompt, self.policy, max_tokens, key)
                if completion.usage.prompt_tokens != len(prompt):
                    error = RolloutContractError(
                        f"The route counted {completion.usage.prompt_tokens} prompt tokens for {len(prompt)} sent ids"
                    )
                    raise self._contract_break(fields, request, turn, error)
                try:
                    parsed = parse_turn(completion.token_bytes, request.options.get("tools") or (), f"call-{turn}")
                except UnicodeDecodeError as error:
                    raise self._contract_break(fields, request, turn, error) from error
                record_usage(
                    fields,
                    replace(completion.usage, reasoning_tokens=parsed.reasoning_tokens),
                    completion.finish_reason,
                    len(parsed.tool_calls),
                )
        except ContextExhausted as error:
            raise GenerationLimitReached(prompt) from error
        return token_model_turn(prompt, completion, parsed, alignment)
