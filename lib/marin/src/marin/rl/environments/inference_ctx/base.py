# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Inference context for rollout construction.

This context is provided to environments and provides access to the inference server
as well as methods for tokenization and logprob extraction from an OpenAI ChatCompletion.
"""

import logging
from typing import Any

import numpy as np
from levanter.models.lm_model import LmHeadModel
from marin.rl.decoding import DecodingConfig
from marin.rl.types import Rollout
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice

logger = logging.getLogger(__name__)


class BaseInferenceContext:
    """Base class for inference contexts."""

    def reload_model(self, model: LmHeadModel | None, state_dict: dict) -> LmHeadModel | None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError

    def get_metrics(self) -> dict[str, Any]:
        """Return implementation-specific metrics for tracker logging."""
        return {}

    def resolve_decoding(self, decoding: DecodingConfig) -> DecodingConfig:
        """Return the concrete decoding config this backend will apply."""
        return decoding

    def batch_completions(
        self,
        prompts: list[str] | list[list[dict]],
        n: int,
        decoding: DecodingConfig,
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

        vocab = self.tokenizer.get_vocab()
        tokens = []
        for t in choice.logprobs.content:
            token_id = vocab.get(t.token)
            if token_id is None:
                raise ValueError(f"Token {t.token!r} not found in vocabulary")
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
        decoding: DecodingConfig,
        system_prompt: str | None = None,
        correctness_reward: float | None = None,
    ) -> Rollout:
        """Construct Rollout from a choice with validation."""
        decoding = self.resolve_decoding(decoding)

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
            decoding=decoding.as_trace(),
            is_truncated=is_truncated,
        )
