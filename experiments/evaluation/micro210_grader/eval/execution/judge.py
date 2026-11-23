"""Judge transport using the same admission, retry and checkpoint lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

from aiohttp import ClientSession

from eval.contracts.eval_spec import ModelRequest
from eval.execution.backend import retryable_transport, transport_retry_delay
from eval.graders.answer_equivalence import JudgeConfig, JudgeLabel, JudgeProtocol
from eval.runtime.pool import RetryPolicy, retry
from eval.completion_response import CompletionContentPolicy, CompletionText, completion_response_from_chat_choice


POLICY_TOKEN_BUDGETS = (128, 512, 2048)
POLICY_JUDGE_CONCURRENCY = 16
POLICY_TRANSPORT_RETRY = RetryPolicy(budget=1800, attempt_timeout=300, max_attempts=6)
DEFAULT_TRANSPORT_RETRY = RetryPolicy(budget=180, attempt_timeout=60, max_attempts=3)


@dataclass
class JudgeBackend:
    config: JudgeConfig
    session: ClientSession
    _policy_slots: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(POLICY_JUDGE_CONCURRENCY), init=False, repr=False,
    )

    async def execute(self, request: ModelRequest) -> str:
        parameters = dict(request.generation)
        parameters.pop("do_sample", None)
        if "max_gen_toks" in parameters:
            parameters["max_tokens"] = parameters.pop("max_gen_toks")
        prompt = request.prompt
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

        if self.config.protocol is JudgeProtocol.DEFAULT:
            return await self._completion(messages, parameters, DEFAULT_TRANSPORT_RETRY)
        parameters.update(temperature=0, structured_outputs={"choice": [label.value for label in JudgeLabel]})
        async with self._policy_slots:
            for tokens in POLICY_TOKEN_BUDGETS:
                result = await self._completion(messages, {**parameters, "max_tokens": tokens}, POLICY_TRANSPORT_RETRY)
                if result.strip():
                    JudgeLabel(result.strip().lower())
                    return result
        raise ValueError("MiniMax returned empty judgments at every policy token budget")

    async def _completion(self, messages, parameters, retry_policy: RetryPolicy) -> str:
        async def call():
            async with self.session.post(
                self.config.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={"model": self.config.model, "messages": messages, **parameters},
            ) as response:
                response.raise_for_status()
                document = await response.json()
                choice = document["choices"][0]
                if not isinstance(choice.get("message"), Mapping) or "content" not in choice["message"]:
                    raise ValueError("judge response requires a message with content")
                completion = completion_response_from_chat_choice(document, choice)
                # A valid empty completion is a parser input, not a transport
                # failure. The named judge protocol decides whether to retry it.
                return CompletionText(completion.content or "", completion, CompletionContentPolicy.FINAL_ONLY)

        return await retry(call, retry_policy, retryable_transport, transport_retry_delay)
