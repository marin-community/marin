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

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from levanter.inference.openai import InferenceServer
from openai import AsyncOpenAI
from transformers import PreTrainedTokenizer

from marin.rl.types import InferenceChoice, InferenceContext, InferenceResponse, RolloutGroup

logger = logging.getLogger(__name__)


class MarinEnv(ABC):
    """Abstract base class for RL environments.

    Environments manage datasets, generate responses, and evaluate them.
    Subclasses must implement sample() method.
    """

    @abstractmethod
    def sample(
        self,
        inference_ctx: InferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample examples, generate responses, and create rollouts.

        Args:
            inference_ctx: Context for generating responses from the model
            n_examples: Number of examples to sample
            n_generations: Number of generations per example
            temperature: Sampling temperature for generation
            prng_key: JAX random key for sampling
            mode: "train" or "eval" - which dataset to sample from

        Returns:
            Tuple of (rollout_groups, metrics)
        """
        ...


@dataclass
class EnvConfig:
    """Configuration for an environment."""

    env_class: str
    """Fully qualified class name of the environment, e.g. 'marin.rl.environments.math.MathEnvironment'."""

    env_args: dict
    """Arguments to pass to the environment constructor."""


def load_environment_from_spec(config: EnvConfig) -> MarinEnv:
    """Load an environment from the given configuration."""
    env_class = config.env_class
    env_args = config.env_args
    # Dynamically import the environment class
    module_name, class_name = env_class.rsplit(".", 1)
    env_module = __import__(module_name, fromlist=[class_name])
    env_class = getattr(env_module, class_name)
    return env_class(**env_args)


# TODO: share this with the inference server or find a shared library.
def tokenize_prompt_with_chat_template(prompt: str, tokenizer: PreTrainedTokenizer) -> list[int]:
    """Tokenize the prompt using a chat template."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception as e:
        # Fallback: simple concatenation if template fails
        logger.warning(f"Chat template failed, using fallback: {e}", exc_info=True)
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return tokenizer.encode(prompt_text, add_special_tokens=True)


class LevanterInferenceContext(InferenceContext):
    """Context that uses Levanter model and inference server."""

    max_tokens: int
    _inference_server: InferenceServer
    _stop_tokens: list[int] | None = None
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        tokenizer,
        stop_tokens: list[int] | None,
        inference_server: InferenceServer,
        max_tokens: int,
    ):
        self._inference_server = inference_server
        self.tokenizer = tokenizer
        self._stop_tokens = stop_tokens
        self.max_tokens = max_tokens

    def openai_client(self):
        return AsyncOpenAI(base_url=self.openai_address(), api_key="marin")

    def openai_address(self) -> str:
        return f"http://{self._inference_server.address()}/v1"

    def generate(
        self,
        prompts: list[str],
        temperature: float,
        n_generations: int,
    ) -> list[InferenceResponse]:
        """Generate responses for a batch of prompts."""
        stop_strings = None
        if self._stop_tokens is not None:
            stop_strings = [self.tokenizer.decode([token]) for token in self._stop_tokens]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = self.openai_client()

        def _process_batch(batch_prompts: list[str]) -> list[InferenceResponse]:
            batch_completions = []

            for prompt in batch_prompts:
                completion = client.chat.completions.create(
                    model=getattr(self._inference_server.config, "model_name", "test-model"),
                    messages=[{"role": "user", "content": prompt}],
                    logprobs=True,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    n=n_generations,
                    stop=stop_strings,
                    timeout=30,
                )
                batch_completions.append(completion)

            completions = loop.run_until_complete(asyncio.gather(*batch_completions, return_exceptions=True))

            batch_results = []
            for prompt, completion in zip(batch_prompts, completions, strict=True):
                choices = []
                # drop responses that failed.
                if isinstance(completion, BaseException):
                    logger.error(f"Error during generation: {completion}")
                else:
                    for choice in completion.choices:
                        content = choice.message.content
                        tokens: list[int] = []
                        logprobs: list[float] = []
                        for t in choice.logprobs.content:
                            encoded = self.tokenizer.encode(t.token, add_special_tokens=False)
                            assert len(encoded) == 1, f"Expected single token but got {encoded} for text: {t.text}"
                            tokens.append(encoded[0])
                            logprobs.append(t.logprob)
                        logprobs = np.array(logprobs, dtype=np.float32)
                        if np.all(logprobs == 0):
                            logger.warning(
                                f"All logprobs zero for {prompt}, choice: {choice}. This can result in NaN loss."
                            )
                        choices.append(
                            InferenceChoice(
                                response_text=content,
                                response_tokens=np.array(tokens, dtype=np.int32),
                                logprobs=logprobs,
                            )
                        )

                prompt_tokens = tokenize_prompt_with_chat_template(prompt, self.tokenizer)

                batch_results.append(
                    InferenceResponse(
                        prompt=prompt,
                        prompt_tokens=np.array(prompt_tokens, dtype=np.int32),
                        choices=choices,
                    )
                )
            return batch_results

        # Process prompts in batches to limit concurrent requests
        # Each prompt with n_generations counts as n_generations requests
        max_concurrent_requests = 8
        batch_size = max(1, max_concurrent_requests // n_generations)
        all_results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_results = _process_batch(batch_prompts)
            all_results.extend(batch_results)

        loop.run_until_complete(client.close())

        loop.close()
        return all_results
