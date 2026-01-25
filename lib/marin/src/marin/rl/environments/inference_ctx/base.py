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
import numpy as np
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from marin.rl.types import Rollout

from levanter.models.lm_model import LmHeadModel

logger = logging.getLogger(__name__)


class BaseInferenceContext:
    """Base class for inference contexts."""

    def reload_model(self, model: LmHeadModel | None, state_dict: dict) -> LmHeadModel | None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError

    def batch_completions(
        self,
        prompts: list[str] | list[list[dict]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[ChatCompletion]:
        """Batch completions from the inference server."""
        raise NotImplementedError

    def tokenize_prompt(self, prompt: str, choice: Choice | None = None, system_prompt: str | None = None) -> np.ndarray:
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
        top_k: int | None = None,
        system_prompt: str | None = None,
        correctness_reward: float | None = None,
    ) -> Rollout:
        """Construct Rollout from a choice with validation."""

        prompt_tokens = self.tokenize_prompt(prompt, choice, system_prompt)
        response_tokens = self.response_tokens_from_choice(choice)
        response_logprobs = self.logprobs_from_choice(choice)

        assert len(response_tokens) == len(
            response_logprobs
        ), f"Length mismatch between response_tokens ({len(response_tokens)}) \
            and response_logprobs ({len(response_logprobs)})"

        if len(prompt_tokens) == 0:
            logger.error(f"Prompt tokenization failed for {env_example_id}")

        token_rewards = np.full(len(response_tokens), reward, dtype=np.float32)
        is_truncated = choice.finish_reason == "length"

        return Rollout(
            env_name=env_name,
            env_example_id=env_example_id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            response_logprobs=response_logprobs,
            token_rewards=token_rewards,
            episode_reward=float(reward),
            correctness_reward=correctness_reward,
            temperature=temperature,
            top_k=top_k,
            is_truncated=is_truncated,
        )
