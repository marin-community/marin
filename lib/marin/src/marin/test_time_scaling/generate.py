# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice

from marin.test_time_scaling.config import CandidateGenerationConfig
from marin.test_time_scaling.manifests import PromptManifest, PromptManifestRecord
from marin.test_time_scaling.results import CandidateRecord
from marin.test_time_scaling.scorers import score_candidate_text


class CompletionProvider(Protocol):
    """Protocol for an OpenAI-compatible chat completion backend."""

    def complete_messages(
        self,
        messages: Sequence[dict[str, str]],
        generation_config: CandidateGenerationConfig,
        request_index: int,
    ) -> ChatCompletion:
        """Return an OpenAI-compatible chat completion response."""


class OpenAIChatCompletionProvider:
    """Minimal synchronous OpenAI-compatible completion provider."""

    def __init__(
        self,
        *,
        server_url: str,
        model: str,
        api_key: str = "marin-tts",
        timeout: float | None = None,
        extra_request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._client = OpenAI(base_url=server_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._extra_request_kwargs = dict(extra_request_kwargs or {})

    def complete_messages(
        self,
        messages: Sequence[dict[str, str]],
        generation_config: CandidateGenerationConfig,
        request_index: int,
    ) -> ChatCompletion:
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "n": generation_config.num_candidates,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "logprobs": True,
            **self._extra_request_kwargs,
        }
        if generation_config.max_gen_toks is not None:
            request_kwargs["max_tokens"] = generation_config.max_gen_toks
        if generation_config.seed is not None:
            request_kwargs["seed"] = generation_config.seed + request_index

        return self._client.chat.completions.create(
            **request_kwargs,
        )


def _choice_logprob_stats(choice: Choice) -> tuple[float | None, float | None, int | None]:
    if not choice.logprobs or not choice.logprobs.content:
        return None, None, None

    values = [token.logprob for token in choice.logprobs.content if token.logprob is not None]
    if not values:
        return None, None, len(choice.logprobs.content)

    logprob_sum = float(sum(values))
    completion_tokens = len(choice.logprobs.content)
    normalized_logprob = logprob_sum / completion_tokens if completion_tokens else None
    return logprob_sum, normalized_logprob, completion_tokens


def _candidate_from_choice(
    *,
    prompt: PromptManifestRecord,
    choice: Choice,
    sample_index: int,
    request_latency_seconds: float,
    prompt_tokens: int | None,
    generation_seed: int | None,
) -> CandidateRecord:
    text = choice.message.content or ""
    score = score_candidate_text(text, prompt.expected_answer, prompt.scoring_mode)
    logprob_sum, normalized_logprob, completion_tokens = _choice_logprob_stats(choice)
    return CandidateRecord(
        prompt_id=prompt.prompt_id,
        candidate_id=f"{prompt.prompt_id}-{sample_index}",
        sample_index=sample_index,
        raw_text=text,
        extracted_answer=score.extracted_answer,
        is_correct=score.is_correct,
        parse_valid=score.parse_valid,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=choice.finish_reason,
        request_latency_seconds=request_latency_seconds,
        generation_seed=generation_seed,
        logprob_sum=logprob_sum,
        normalized_logprob=normalized_logprob,
    )


def generate_candidates(
    manifest: PromptManifest,
    provider: CompletionProvider,
    generation_config: CandidateGenerationConfig,
) -> list[CandidateRecord]:
    """Generate and score sample-only candidates for a prompt manifest."""

    candidates: list[CandidateRecord] = []
    for prompt_index, prompt in enumerate(manifest.records):
        request_seed = generation_config.seed + prompt_index if generation_config.seed is not None else None
        started_at = time.perf_counter()
        completion = provider.complete_messages(
            [message.to_openai_dict() for message in prompt.messages],
            generation_config,
            prompt_index,
        )
        request_latency_seconds = time.perf_counter() - started_at
        prompt_tokens = completion.usage.prompt_tokens if completion.usage is not None else None
        for sample_index, choice in enumerate(completion.choices):
            candidates.append(
                _candidate_from_choice(
                    prompt=prompt,
                    choice=choice,
                    sample_index=sample_index,
                    request_latency_seconds=request_latency_seconds,
                    prompt_tokens=prompt_tokens,
                    generation_seed=request_seed,
                )
            )
    return candidates
