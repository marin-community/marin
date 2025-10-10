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

import jax.numpy as jnp
import numpy as np
from levanter.inference.openai import InferenceServer
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from transformers import PreTrainedTokenizer

from marin.rl.types import Rollout

logger = logging.getLogger(__name__)


class InferenceContext:
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

    def tokenize_prompt(self, prompt: str) -> np.ndarray:
        """Tokenize with chat template matching server behavior."""
        messages = [{"role": "user", "content": prompt}]
        try:
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            logger.warning(f"Chat template failed: {e}")
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
            if not tokens:
                raise ValueError(f"Failed to tokenize: {prompt[:100]}...") from None

        return np.array(tokens, dtype=np.int32)

    def tokenize_response(self, text: str) -> np.ndarray:
        """Extract token IDs from the response text.

        In general this should roundtrip via `encode`, as the chat template should
        terminate with a special token to indicate the end of the template and start
        of the assistant response, that is, we should not have a situation [pppprrrr]
        where `pr` can be interpreted as a valid token.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return np.array(tokens, dtype=np.int32)

    def response_tokens_from_choice(self, choice: Choice) -> np.ndarray:
        """Extract token IDs with BPE round-trip."""
        if not choice.logprobs or not choice.logprobs.content:
            raise ValueError("Choice missing logprobs. Use logprobs=True in API call.")

        tokens = []
        for t in choice.logprobs.content:
            # Use convert_tokens_to_ids for correct BPE round-trip
            # The server uses convert_ids_to_tokens which preserves BPE format (e.g., Ġ for spaces)
            token_id = self.tokenizer.convert_tokens_to_ids(t.token)
            tokens.append(token_id)

        if not tokens:
            raise ValueError("Choice has zero tokens")

        return np.array(tokens, dtype=np.int32)

    def logprobs_from_choice(self, choice: Choice) -> np.ndarray:
        """Extract logprobs array."""
        if not choice.logprobs or not choice.logprobs.content:
            raise ValueError("Choice missing logprobs. Use logprobs=True in API call.")

        logprobs = np.array([t.logprob for t in choice.logprobs.content], dtype=np.float32)

        if np.all(logprobs == 0):
            logger.warning("All logprobs are zero - may cause NaN loss")

        return logprobs

    def create_rollout_from_choice(
        self,
        prompt: str,
        choice: Choice,
        env_name: str,
        env_example_id: str,
        reward: float,
    ) -> Rollout:
        """Construct Rollout from a choice with validation."""

        prompt_tokens = self.tokenize_prompt(prompt)
        response_tokens = self.response_tokens_from_choice(choice)
        response_logprobs = self.logprobs_from_choice(choice)

        # Validation
        if len(response_tokens) < 5:
            logger.warning(f"Only {len(response_tokens)} tokens for {env_example_id}")
        if len(prompt_tokens) == 0:
            logger.error(f"Prompt tokenization failed for {env_example_id}")

        token_rewards = jnp.full(len(response_tokens), reward, dtype=jnp.float32)

        return Rollout(
            env_name=env_name,
            env_example_id=env_example_id,
            prompt_tokens=jnp.array(prompt_tokens, dtype=jnp.int32),
            response_tokens=jnp.array(response_tokens, dtype=jnp.int32),
            response_logprobs=jnp.array(response_logprobs, dtype=jnp.float32),
            token_rewards=token_rewards,
            episode_reward=float(reward),
        )
