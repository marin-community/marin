"""Judge transport using the same admission, retry and checkpoint lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from aiohttp import ClientSession

from eval.contracts.eval_spec import ModelRequest
from eval.execution.backend import retryable_transport, transport_retry_delay
from eval.graders.answer_equivalence import JudgeConfig
from eval.runtime.pool import RetryPolicy, retry
from eval.completion_response import CompletionContentPolicy, CompletionText, completion_response_from_chat_choice


@dataclass
class JudgeBackend:
    config: JudgeConfig
    session: ClientSession

    async def execute(self, request: ModelRequest) -> str:
        parameters = dict(request.generation)
        parameters.pop("do_sample", None)
        if "max_gen_toks" in parameters:
            parameters["max_tokens"] = parameters.pop("max_gen_toks")
        prompt = request.prompt
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

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

        return await retry(call, RetryPolicy(budget=180, attempt_timeout=60, max_attempts=3), retryable_transport, transport_retry_delay)
