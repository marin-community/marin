# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Inference context for rollout construction.

This context is provided to environments and provides access to the inference server
as well as methods for tokenization and logprob extraction from an OpenAI ChatCompletion.
"""

import hashlib
import logging
from typing import Any

import numpy as np
from levanter.models.lm_model import LmHeadModel
from marin.inference.types import (
    PolicyIdentity,
    TokenizedRollout,
    TokenizedRolloutBatchRequest,
    TokenizedRolloutBatchResult,
    TokenizedRolloutRequest,
    TokenizerIdentity,
    TokenRolloutFinishReason,
    TokenSamplingParameters,
)
from marin.rl.decoding import DecodingConfig, DecodingStrategy
from marin.rl.types import Rollout, RolloutMetadata
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice

logger = logging.getLogger(__name__)


PromptInput = str | list[dict[str, str]]
SPECIAL_TOKEN_ID_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("bos_token", "bos_token_id"),
    ("eos_token", "eos_token_id"),
    ("unk_token", "unk_token_id"),
    ("sep_token", "sep_token_id"),
    ("pad_token", "pad_token_id"),
    ("cls_token", "cls_token_id"),
    ("mask_token", "mask_token_id"),
)
UNSUPPORTED_TOKEN_ROLLOUT_DECODING_FIELDS: tuple[str, ...] = (
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "min_output_tokens",
    "ignore_eos",
)


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

    def supports_token_rollouts(self) -> bool:
        """Return whether this context implements the low-level token rollout API."""
        return False

    def generate_token_rollouts(self, batch: TokenizedRolloutBatchRequest) -> TokenizedRolloutBatchResult:
        """Generate tokenized rollouts without going through OpenAI-compatible serialization."""
        del batch
        raise NotImplementedError("This inference context does not implement tokenized rollout generation.")

    @staticmethod
    def rollouts_by_token_request(
        batch: TokenizedRolloutBatchRequest,
        batch_result: TokenizedRolloutBatchResult,
    ) -> dict[str, tuple[TokenizedRollout, ...]]:
        """Return validated token rollouts keyed by request ID."""
        if batch_result.batch_id != batch.batch_id:
            raise RuntimeError(
                f"Token rollout batch ID mismatch: got {batch_result.batch_id}, expected {batch.batch_id}"
            )
        if batch_result.tokenizer != batch.tokenizer:
            raise RuntimeError("Token rollout tokenizer identity mismatch")
        if batch_result.policy != batch.policy:
            raise RuntimeError("Token rollout policy identity mismatch")

        expected_request_ids = {request.request_id for request in batch.requests}
        rollouts_by_request: dict[str, list[TokenizedRollout]] = {}
        for rollout in batch_result.rollouts:
            if rollout.request_id not in expected_request_ids:
                raise RuntimeError(f"Token rollout result included unknown request ID {rollout.request_id}")
            rollouts_by_request.setdefault(rollout.request_id, []).append(rollout)
        failures_by_request = {}
        for failure in batch_result.failures:
            if failure.request_id not in expected_request_ids:
                raise RuntimeError(f"Token rollout failure included unknown request ID {failure.request_id}")
            failures_by_request.setdefault(failure.request_id, []).append(failure)

        grouped_rollouts: dict[str, tuple[TokenizedRollout, ...]] = {}
        for request in batch.requests:
            request_failures = failures_by_request.get(request.request_id, [])
            if request_failures:
                failure_summary = ", ".join(
                    f"{failure.reason.value}"
                    + (f"[generation={failure.generation_index}]" if failure.generation_index is not None else "")
                    for failure in request_failures
                )
                raise RuntimeError(f"Token rollout request {request.request_id} failed: {failure_summary}")

            token_rollouts = sorted(
                rollouts_by_request.get(request.request_id, []),
                key=lambda rollout: rollout.generation_index,
            )
            if len(token_rollouts) != request.n_generations:
                raise RuntimeError(
                    f"Token rollout request {request.request_id} returned {len(token_rollouts)} generations; "
                    f"expected {request.n_generations}"
                )
            generation_indexes = tuple(rollout.generation_index for rollout in token_rollouts)
            expected_generation_indexes = tuple(range(request.n_generations))
            if generation_indexes != expected_generation_indexes:
                raise RuntimeError(
                    f"Token rollout request {request.request_id} returned generation indexes "
                    f"{generation_indexes}; expected {expected_generation_indexes}"
                )
            grouped_rollouts[request.request_id] = tuple(token_rollouts)
        return grouped_rollouts

    def tokenizer_identity(self) -> TokenizerIdentity:
        """Return stable tokenizer identity for token-native rollout replay."""
        name_or_path = getattr(self.tokenizer, "name_or_path", None)
        if name_or_path is None:
            name_or_path = getattr(self.tokenizer, "_name_or_path", None)
        if name_or_path is None:
            name_or_path = getattr(self.tokenizer, "init_kwargs", {}).get("name_or_path")
        if not name_or_path:
            name_or_path = type(self.tokenizer).__name__
        revision = getattr(self.tokenizer, "init_kwargs", {}).get("revision")
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, "__len__") else None
        chat_template = getattr(self.tokenizer, "chat_template", None)
        chat_template_hash = None
        if chat_template:
            chat_template_hash = hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest()

        special_token_ids: dict[str, int] = {}
        for token_name, token_id_name in SPECIAL_TOKEN_ID_ATTRIBUTES:
            token_id = getattr(self.tokenizer, token_id_name, None)
            if token_id is not None:
                special_token_ids[token_name] = int(token_id)
        additional_special_token_ids = getattr(self.tokenizer, "additional_special_tokens_ids", None)
        if additional_special_token_ids:
            for token_index, token_id in enumerate(additional_special_token_ids):
                special_token_ids[f"additional_special_token_{token_index}"] = int(token_id)

        return TokenizerIdentity(
            name_or_path=str(name_or_path),
            revision=revision,
            vocab_size=vocab_size,
            chat_template_hash=chat_template_hash,
            special_token_ids=special_token_ids,
        )

    def set_policy_identity(self, policy: PolicyIdentity) -> None:
        """Set policy identity used for subsequent token-native rollout batches."""
        self._policy_identity = policy

    def policy_identity(self) -> PolicyIdentity:
        """Return best-effort policy identity for token-native rollout batches."""
        if hasattr(self, "_policy_identity"):
            return self._policy_identity
        model_ref = getattr(self, "canonical_model_name", None)
        if model_ref is None:
            model_ref = getattr(self, "model_name", None)
        if model_ref is None and hasattr(self, "_inference_server"):
            server_config = getattr(self._inference_server, "config", None)
            model_ref = getattr(server_config, "model_name", None)
        if not model_ref:
            model_ref = type(self).__name__
        return PolicyIdentity(policy_name=str(model_ref), checkpoint_ref=str(model_ref))

    @staticmethod
    def _messages_for_prompt(prompt: PromptInput, system_prompt: str | None) -> list[dict[str, str]]:
        if isinstance(prompt, list):
            if system_prompt is not None:
                raise ValueError("system_prompt cannot be combined with message-list prompts")
            return prompt
        if system_prompt is not None:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        return [{"role": "user", "content": prompt}]

    def tokenize_messages(self, messages: list[dict[str, str]]) -> np.ndarray:
        """Tokenize chat messages with the backend's generation prompt template."""
        try:
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            logger.warning(f"Chat template failed: {e}")
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
            if not tokens:
                raise ValueError("Failed to tokenize prompt messages") from None
        return np.array(tokens, dtype=np.int32)

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
        del choice
        return self.tokenize_messages(self._messages_for_prompt(prompt, system_prompt))

    def tokenize_prompt_input(self, prompt: PromptInput, system_prompt: str | None = None) -> np.ndarray:
        """Tokenize a string prompt or chat message list for token-native generation."""
        return self.tokenize_messages(self._messages_for_prompt(prompt, system_prompt))

    @staticmethod
    def _token_sampling_from_decoding(decoding: DecodingConfig) -> TokenSamplingParameters:
        unsupported_fields = [
            field_name for field_name in UNSUPPORTED_TOKEN_ROLLOUT_DECODING_FIELDS if getattr(decoding, field_name)
        ]
        if unsupported_fields:
            unsupported_field_names = ", ".join(unsupported_fields)
            raise ValueError(
                f"Token-native rollout generation does not support decoding fields: {unsupported_field_names}"
            )
        if decoding.stop_strings is not None:
            raise ValueError("Token-native rollout generation does not support stop_strings; use stop_token_ids.")
        return TokenSamplingParameters(
            max_tokens=decoding.max_output_tokens,
            temperature=0.0 if decoding.strategy == DecodingStrategy.GREEDY else decoding.temperature,
            top_p=decoding.top_p,
            top_k=decoding.top_k,
            stop_token_ids=tuple(decoding.stop_token_ids or ()),
            seed=decoding.seed,
            return_logprobs=True,
        )

    def create_token_rollout_batch(
        self,
        *,
        batch_id: str,
        prompts: list[PromptInput],
        n: int,
        decoding: DecodingConfig,
        system_prompt: str | None = None,
    ) -> TokenizedRolloutBatchRequest:
        """Build a token-native rollout request batch from environment prompts."""
        if n <= 0:
            raise ValueError("n must be positive")
        decoding = self.resolve_decoding(decoding)
        sampling = self._token_sampling_from_decoding(decoding)
        requests = []
        for prompt_index, prompt in enumerate(prompts):
            prompt_tokens = self.tokenize_prompt_input(prompt, system_prompt)
            if prompt_tokens.size == 0:
                raise ValueError(f"Prompt {prompt_index} tokenized to an empty sequence")
            requests.append(
                TokenizedRolloutRequest(
                    request_id=f"{batch_id}:{prompt_index}",
                    prompt_token_ids=tuple(int(token_id) for token_id in prompt_tokens),
                    sampling=sampling,
                    n_generations=n,
                )
            )
        return TokenizedRolloutBatchRequest(
            batch_id=batch_id,
            tokenizer=self.tokenizer_identity(),
            policy=self.policy_identity(),
            requests=tuple(requests),
        )

    def response_tokens_from_choice(self, choice: Choice) -> np.ndarray:
        """Extract token IDs with BPE round-trip."""
        response_token_ids = getattr(choice, "response_token_ids", None)
        if response_token_ids is not None:
            return np.array([int(token_id) for token_id in response_token_ids], dtype=np.int32)

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
            metadata=RolloutMetadata(
                tokenizer=self.tokenizer_identity(),
                policy=self.policy_identity(),
            ),
        )

    def create_rollout_from_tokenized_rollout(
        self,
        rollout: TokenizedRollout,
        env_name: str,
        env_example_id: str,
        reward: float,
        decoding: DecodingConfig,
        correctness_reward: float | None = None,
        batch_id: str | None = None,
    ) -> Rollout:
        """Construct a training rollout from token-native generation output."""
        decoding = self.resolve_decoding(decoding)
        response_tokens = np.array(rollout.completion_token_ids, dtype=np.int32)
        response_logprobs = np.array(rollout.completion_logprobs, dtype=np.float32)
        token_rewards = np.full(len(response_tokens), reward, dtype=np.float32)
        backend_value = rollout.metadata.get("backend")
        backend: str | None = None
        if backend_value is not None:
            if not isinstance(backend_value, str):
                raise ValueError("Token rollout backend metadata must be a string when set")
            backend = backend_value
        return Rollout(
            env_name=env_name,
            env_example_id=env_example_id,
            prompt_tokens=np.array(rollout.prompt_token_ids, dtype=np.int32),
            response_tokens=response_tokens,
            response_logprobs=response_logprobs,
            token_rewards=token_rewards,
            episode_reward=float(reward),
            correctness_reward=correctness_reward,
            decoding=decoding.as_trace(),
            is_truncated=rollout.finish_reason == TokenRolloutFinishReason.LENGTH,
            metadata=RolloutMetadata(
                tokenizer=self.tokenizer_identity(),
                policy=self.policy_identity(),
                token_rollout_backend=backend,
                token_rollout_batch_id=batch_id,
                token_rollout_request_id=rollout.request_id,
                token_rollout_generation_index=rollout.generation_index,
                token_rollout_finish_reason=rollout.finish_reason.value,
                token_rollout_stop_token_id=rollout.stop_token_id,
                router_replay=rollout.router_replay,
                expert_load=rollout.expert_load,
            ),
        )
