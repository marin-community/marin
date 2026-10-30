# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-boundary likelihood scoring through an OpenAI-compatible completion endpoint.

Pair encoding, context truncation and echoed logprob reduction preserve Marin's
Evalchemy endpoint backend and lm-evaluation-harness v0.4.12 (MIT). Unlike a
local tensor forward pass, an echoed prompt must retain the last target token.
"""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from openai import AsyncOpenAI
from openai.types.completion import Completion
from rolloutengine.models import TokenUsage
from taskcompendium.evaluation import LikelihoodRequest


class Tokenizer(Protocol):
    eos_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class EncodedLikelihood:
    """The exact endpoint input and source truncation evidence for one continuation."""

    prompt_token_ids: tuple[int, ...]
    context_tokens: int
    continuation_tokens: int
    truncated_context_tokens: int


@dataclass(frozen=True)
class LikelihoodScore:
    loglikelihood: float
    is_greedy: bool
    token_count: int
    evidence: EncodedLikelihood | None = None
    token_logprobs: tuple[float, ...] = ()
    usage: TokenUsage | None = None


class LikelihoodTransportError(ValueError):
    """A malformed echoed response with the exact request and response retained for the archive."""

    def __init__(self, message: str, *, request: dict[str, Any], prepared: EncodedLikelihood, response: Completion):
        super().__init__(message)
        decoded = response.model_dump(mode="json")
        self.evidence: dict[str, Any] = {"request": request, "prepared": asdict(prepared), "response": decoded}
        try:
            json.dumps(decoded, allow_nan=False)
        except ValueError:
            # An invalid numeric response must itself survive the strict JSON archive.
            self.evidence["response_json"] = json.dumps(decoded, allow_nan=True)
            self.evidence["response"] = json.loads(response.model_dump_json())


class LikelihoodScorer(Protocol):
    async def __call__(self, request: LikelihoodRequest, *, seed: int | None) -> LikelihoodScore: ...


class InvalidLikelihoodRequest(ValueError):
    """A continuation cannot be scored with the configured tokenizer and context window."""


def encode_pair(
    tokenizer: Tokenizer, request: LikelihoodRequest, *, prefix_token_id: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Move trailing context whitespace into the jointly encoded continuation."""
    context, continuation = request.context, request.continuation
    if not context:
        return (prefix_token_id,), tuple(tokenizer.encode(continuation, add_special_tokens=False))
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    prefix = tokenizer.encode(context, add_special_tokens=False)
    return tuple(prefix), tuple(whole[len(prefix) :])


@dataclass(frozen=True)
class VllmLikelihood:
    """Score full continuation tokens with the legacy endpoint's one-token output reserve.

    The caller owns the client and pins the tokenizer revision and prefix token.
    The source prefix is an explicit override, otherwise BOS when present, then EOS.
    """

    client: AsyncOpenAI
    model: str
    tokenizer: Tokenizer
    context_size: int
    prefix_token_id: int

    def prepare(self, request: LikelihoodRequest) -> EncodedLikelihood:
        context, continuation = encode_pair(self.tokenizer, request, prefix_token_id=self.prefix_token_id)
        maximum_prompt = self.context_size - 1
        removed = max(0, len(context) + len(continuation) - maximum_prompt)
        retained_context = len(context) - removed
        if not continuation or retained_context < 1:
            raise InvalidLikelihoodRequest(
                "Likelihood scoring requires a nonempty continuation and a retained context token"
            )
        return EncodedLikelihood(
            prompt_token_ids=(context + continuation)[-maximum_prompt:],
            context_tokens=retained_context,
            continuation_tokens=len(continuation),
            truncated_context_tokens=removed,
        )

    async def __call__(self, request: LikelihoodRequest, *, seed: int | None) -> LikelihoodScore:
        prepared = self.prepare(request)
        payload = {
            "model": self.model,
            "prompt": list(prepared.prompt_token_ids),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 1,
            "echo": True,
        }
        if seed is not None:
            payload["seed"] = seed
        response = await self.client.completions.create(**payload)
        try:
            return _echo_score(response, prepared)
        except (AttributeError, TypeError, ValueError) as error:
            raise LikelihoodTransportError(str(error), request=payload, prepared=prepared, response=response) from error


def _echo_score(response: Completion, prepared: EncodedLikelihood) -> LikelihoodScore:
    (choice,) = response.choices
    logprobs = choice.logprobs
    if logprobs is None or logprobs.token_logprobs is None or logprobs.top_logprobs is None:
        raise ValueError("The likelihood endpoint did not return echoed prompt logprobs")
    expected_length = len(prepared.prompt_token_ids) + 1
    if len(logprobs.token_logprobs) != expected_length or len(logprobs.top_logprobs) != expected_length:
        raise ValueError("The likelihood endpoint returned a different token boundary")
    values = logprobs.token_logprobs[prepared.context_tokens : -1]
    alternatives = logprobs.top_logprobs[prepared.context_tokens : -1]
    scores: list[float] = []
    greedy = True
    for value, top in zip(values, alternatives, strict=True):
        if value is None or not math.isfinite(value) or value > 0 or not top:
            raise ValueError("The likelihood endpoint returned missing or invalid target logprobs")
        if any(not math.isfinite(item) or item > 0 for item in top.values()):
            raise ValueError("The likelihood endpoint returned invalid alternative logprobs")
        scores.append(value)
        greedy = greedy and value == max(top.values())
    usage = (
        None if response.usage is None else TokenUsage(response.usage.prompt_tokens, response.usage.completion_tokens)
    )
    return LikelihoodScore(sum(scores), greedy, prepared.continuation_tokens, prepared, tuple(scores), usage)
