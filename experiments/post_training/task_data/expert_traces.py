# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strong-teacher trace collection (as in #9920's GLM relay) for building E_X and E_R.

Traces come from an OpenAI-compatible chat-completion endpoint — the GLM relay for
teacher traces, a served expert for M1 samples — and keep the teacher's
``reasoning_content`` beside the answer content, so the thinking trains as reasoning
under the shared chat template. The token cap is tokenizer-backed and counts the
rendered completion, thinking included, through the family's pinned tokenizer:
16,384 tokens for R traces, 4,096 for X. Traces over the cap are dropped, never
truncated, and the dropped share is reported.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import jinja2
from marin.inference.openai_chat import OpenAIChatClient
from transformers import AutoTokenizer

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, FAMILY_TOKENIZER_REVISIONS, shared_template
from experiments.post_training.glm import (
    GLM_BULK_TOKEN_ENV,
    GLM_REQUEST_ATTEMPTS,
    GLM_REQUEST_TIMEOUT,
    resolve_glm_base_url,
)

R_COMPLETION_TOKEN_CAP = 16_384
X_COMPLETION_TOKEN_CAP = 4_096


class ChatClient(Protocol):
    def complete(self, body: Mapping[str, Any]) -> dict[str, Any]: ...


def glm_chat_client(relay_job: str) -> OpenAIChatClient:
    """A client for the GLM relay registered by ``relay_job``; reads the bulk token from the environment."""
    return OpenAIChatClient(resolve_glm_base_url(relay_job), os.environ[GLM_BULK_TOKEN_ENV], timeout=GLM_REQUEST_TIMEOUT)


@dataclass(frozen=True)
class ExpertResponse:
    """One endpoint response with the teacher's thinking kept beside its answer."""

    reasoning: str
    content: str
    truncated: bool


def chat_completion(
    client: ChatClient,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> ExpertResponse:
    """One chat completion with the sampling parameters set explicitly.

    Transport failures are retried up to ``GLM_REQUEST_ATTEMPTS`` times. A response
    that ran into ``max_tokens`` is marked ``truncated``; callers set ``max_tokens``
    above the cap, so a truncated response is over the cap and is dropped.
    ``reasoning_content`` is the chain-of-thought field the GLM relay and vLLM's
    reasoning parsers emit; a provider that omits it leaves the reasoning empty.
    """
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(GLM_REQUEST_ATTEMPTS):
        try:
            response = client.complete(body)
            break
        except ConnectionError:
            if attempt == GLM_REQUEST_ATTEMPTS - 1:
                raise
    choice = response["choices"][0]
    message = choice["message"]
    return ExpertResponse(
        reasoning=message.get("reasoning_content") or "",
        content=message.get("content") or "",
        truncated=choice.get("finish_reason") == "length",
    )


def family_tokenizer(family: str):
    """The family's pinned tokenizer — the pin its chat-template token map validates against."""
    repo, revision = FAMILY_TOKENIZER_REVISIONS[family]
    return AutoTokenizer.from_pretrained(repo, revision=revision)


def token_counter(tokenizer) -> Callable[[str], int]:
    """Count the tokens of already-rendered text; structural delimiters count as one token each."""

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


@functools.cache
def _assistant_template(family: str) -> jinja2.Template:
    return jinja2.Environment().from_string(shared_template(FAMILY_TOKEN_MAPS[family]))


def rendered_completion_text(family: str, response: ExpertResponse) -> str:
    """The completion as the shared template renders its assistant turn, thinking included,
    so the cap counts every token the completion costs in training."""
    message = {"role": "assistant", "content": response.content, "reasoning_content": response.reasoning or None}
    return _assistant_template(family).render(messages=[message])


@dataclass(frozen=True)
class TraceBatch:
    """Kept traces plus the drop report the build manifest records."""

    traces: list[dict[str, Any]]
    dropped_count: int
    token_cap: int
    family: str


def collect_traces(
    tasks: list[dict[str, Any]],
    *,
    client: ChatClient,
    model: str,
    family: str,
    temperature: float,
    max_tokens: int,
    token_cap: int,
    token_count: Callable[[str], int],
    max_concurrent: int,
) -> TraceBatch:
    """Collect one teacher trace per task and enforce the rendered-token cap.

    ``max_tokens`` must exceed ``token_cap``: at or below it the endpoint would
    truncate a long trace and the cap would keep truncated thinking, while the cap's
    contract is to drop over-cap traces whole. ``token_count`` counts the rendered
    completion, thinking included, with the family's tokenizer.
    """
    if max_tokens <= token_cap:
        raise ValueError(
            f"max_tokens ({max_tokens}) must exceed the token cap ({token_cap}); a smaller limit would "
            "truncate at the server the traces the cap is meant to drop"
        )

    def trace(task: dict[str, Any]) -> dict[str, Any] | None:
        prompt = [{"role": "user", "content": task["prompt"]}]
        response = chat_completion(client, model=model, messages=prompt, temperature=temperature, max_tokens=max_tokens)
        if response.truncated:
            return None
        tokens = token_count(rendered_completion_text(family, response))
        if tokens > token_cap:
            return None
        return {
            "task_id": task["task_id"],
            "family": family,
            "prompt": prompt,
            "reasoning": response.reasoning,
            "content": response.content,
            "completion_tokens": tokens,
        }

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        results = list(pool.map(trace, tasks))
    traces = [result for result in results if result is not None]
    return TraceBatch(traces=traces, dropped_count=len(tasks) - len(traces), token_cap=token_cap, family=family)


def write_trace_batch(batch: TraceBatch, output_path: str, *, temperature: float, max_tokens: int) -> dict[str, Any]:
    """Write the kept traces and return the drop report for the run manifest."""
    with open(output_path, "w") as handle:
        for trace in batch.traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    total = len(batch.traces) + batch.dropped_count
    return {
        "kept": len(batch.traces),
        "dropped": batch.dropped_count,
        "dropped_share": batch.dropped_count / total if total else 0.0,
        "token_cap": batch.token_cap,
        "family": batch.family,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
