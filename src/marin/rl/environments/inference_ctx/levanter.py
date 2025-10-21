# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inference context for rollout construction.

This context is provided to environments and provides access to the inference server
as well as methods for tokenization and logprob extraction from an OpenAI ChatCompletion.
"""

import asyncio
import logging
from levanter.inference.openai import InferenceServer
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer
from marin.rl.environments.inference_ctx.base import BaseInferenceContext

logger = logging.getLogger(__name__)


class InferenceContext(BaseInferenceContext):
    """Concrete implementation using Levanter inference server."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        stop_tokens: list[int] | None,
        inference_server: InferenceServer,
        max_tokens: int,
    ):
        self._inference_server = inference_server
        self.tokenizer = tokenizer
        self._stop_tokens = stop_tokens
        self.max_tokens = max_tokens

    def openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=f"http://{self._inference_server.address()}/v1",
            api_key="marin",
        )

    def openai_address(self) -> str:
        return f"http://{self._inference_server.address()}/v1"

    # TODO: add support for ChatCompletion style [ { role, content} ] messages
    def batch_completions(
        self,
        prompts: list[str],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[ChatCompletion]:
        """Call OpenAI API in batches with concurrency control."""

        if max_tokens is None:
            max_tokens = self.max_tokens

        if stop is None and self._stop_tokens:
            stop = [self.tokenizer.decode([tok]) for tok in self._stop_tokens]

        # Async batch processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = self.openai_client()

        async def create_completion(prompt: str) -> ChatCompletion:
            return await client.chat.completions.create(
                model=getattr(self._inference_server.config, "model_name", "test-model"),
                messages=[{"role": "user", "content": prompt}],
                logprobs=True,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                stop=stop,
                timeout=30,
            )

        # Batch with concurrency control
        # Each prompt with n choices counts as n requests to the server
        max_concurrent_requests = 8
        batch_size = max(1, max_concurrent_requests // n)
        all_completions = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            tasks = [create_completion(p) for p in batch_prompts]
            completions = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

            # Handle failures
            for j, comp in enumerate(completions):
                if isinstance(comp, BaseException):
                    logger.error(f"Error for prompt {i + j}: {comp}")
                    # Skip failed completions - environments will handle missing data
                else:
                    all_completions.append(comp)

        loop.run_until_complete(client.close())
        loop.close()

        return all_completions
