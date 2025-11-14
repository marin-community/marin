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

import logging
import jax.numpy as jnp
import numpy as np
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from marin.rl.types import Rollout
from levanter.models.lm_model import LmHeadModel

logger = logging.getLogger(__name__)


class BaseInferenceContext:
    """Base class for inference contexts."""

    def reload_model(self, model: LmHeadModel) -> None:
        raise NotImplementedError

    def batch_completions(
        self,
        prompts: list[str],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[ChatCompletion]:
        """Batch completions from the inference server."""
        raise NotImplementedError

    def tokenize_prompt(self, prompt: str, choice: Choice) -> np.ndarray:
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
        temperature: float,
        correctness_reward: float | None = None,
    ) -> Rollout:
        """Construct Rollout from a choice with validation."""

        prompt_tokens = self.tokenize_prompt(prompt, choice)
        # print(f"prompt_tokens: {prompt_tokens}")
        # print(f"prompt token ids: {choice.prompt_token_ids}")

        # assert choice.prompt_token_ids == prompt_tokens,
        # f"Prompt token IDs mismatch: {choice.prompt_token_ids} != {prompt_tokens}"
        response_tokens = self.response_tokens_from_choice(choice)
        response_logprobs = self.logprobs_from_choice(choice)

        assert len(response_tokens) == len(
            response_logprobs
        ), f"Length mismatch between response_tokens ({len(response_tokens)}) \
            and response_logprobs ({len(response_logprobs)})"

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
            correctness_reward=correctness_reward,
            temperature=temperature,
        )
