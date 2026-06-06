# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Context which uses the native Levanter engine for inference.

This context is provided to environments and provides access to the inference server
as well as methods for tokenization and logprob extraction from an OpenAI ChatCompletion.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
from jax.sharding import Mesh
from levanter.inference.engine import Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmHeadModel
from levanter.tokenizers import MarinTokenizer
from marin.inference.types import (
    TokenizedRollout,
    TokenizedRolloutBatchRequest,
    TokenizedRolloutBatchResult,
    TokenizedRolloutFailure,
    TokenRolloutAdmissionMetadata,
    TokenRolloutFailureReason,
    TokenRolloutFinishReason,
    TokenRolloutTiming,
    TokenSamplingParameters,
)
from marin.rl.decoding import DecodingConfig, DecodingStrategy, stop_strings_for_decoding
from marin.rl.environments.inference_ctx.base import BaseInferenceContext

# TODO(chris): use a different weight transfer method update model, take it out from here
from marin.rl.weight_transfer.arrow_flight import update_model
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

UNSUPPORTED_LEVANTER_DECODING_FIELDS = (
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "min_output_tokens",
    "ignore_eos",
)


@dataclass
class LevanterInferenceContextConfig:
    inference_server_config: InferenceServerConfig
    tokenizer: MarinTokenizer
    mesh: Mesh
    axis_mapping: dict[str, str]
    stop_tokens: list[int] | None = None
    max_tokens: int = 16


class LevanterInferenceContext(BaseInferenceContext):
    """Concrete implementation using Levanter inference server."""

    _inference_server: InferenceServer
    _inference_thread: threading.Thread

    def __init__(
        self,
        inference_config: LevanterInferenceContextConfig,
    ):
        self.inference_server_config = inference_config.inference_server_config
        self.tokenizer = inference_config.tokenizer
        self._stop_tokens = inference_config.stop_tokens
        self.max_tokens = inference_config.max_tokens
        self.mesh = inference_config.mesh
        self.axis_mapping = inference_config.axis_mapping

    def openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=f"http://{self._inference_server.address()}/v1",
            api_key="marin",
        )

    def openai_address(self) -> str:
        return f"http://{self._inference_server.address()}/v1"

    def reload_model(self, model: LmHeadModel | None, state_dict: dict) -> LmHeadModel | None:
        assert model is not None or state_dict is not None, "Either model or state_dict must be provided"
        if model is None and state_dict is not None:
            with hax.set_mesh(self.mesh), hax.axis_mapping(self.axis_mapping):
                model = update_model(model, state_dict)

        self._inference_server.reload(lambda _: model)
        return model

    def start_server(self, model: LmHeadModel) -> None:
        with hax.set_mesh(self.mesh), hax.axis_mapping(self.axis_mapping):
            self._inference_server = InferenceServer.create(
                self.inference_server_config,
                model=model,
                tokenizer=self.tokenizer,
            )
        self._inference_thread = threading.Thread(target=lambda: self._inference_server.serve(), daemon=True)
        self._inference_thread.start()

    def shutdown(self) -> None:
        self._inference_server.shutdown()

    @staticmethod
    def _decode_params_from_token_sampling(
        prompt_token_count: int,
        sampling: TokenSamplingParameters,
        request_index: int,
    ) -> SeqDecodingParams:
        stop_tokens = None
        if sampling.stop_token_ids:
            stop_tokens = hax.named(jnp.asarray(sampling.stop_token_ids, dtype=jnp.int32), axis="position")
            stop_tokens = stop_tokens.broadcast_axis({"stop_seq": 1})
        seed = sampling.seed if sampling.seed is not None else request_index
        return SeqDecodingParams(
            max_num_tokens=jnp.array(prompt_token_count + sampling.max_tokens, dtype=jnp.int32),
            stop_tokens=stop_tokens,
            temperature=jnp.array(sampling.temperature, dtype=jnp.float32),
            top_p=jnp.array(1.0 if sampling.top_p is None else sampling.top_p, dtype=jnp.float32),
            top_k=jnp.array(0 if sampling.top_k is None else sampling.top_k, dtype=jnp.int32),
            key=jrandom.PRNGKey(seed),
        )

    @staticmethod
    def _finish_reason_from_tokens(
        completion_token_ids: tuple[int, ...],
        sampling: TokenSamplingParameters,
    ) -> TokenRolloutFinishReason:
        if len(completion_token_ids) >= sampling.max_tokens:
            return TokenRolloutFinishReason.LENGTH
        if sampling.stop_token_ids and completion_token_ids and completion_token_ids[-1] in sampling.stop_token_ids:
            return TokenRolloutFinishReason.STOP
        return TokenRolloutFinishReason.STOP

    @staticmethod
    def _stop_token_id_from_tokens(
        completion_token_ids: tuple[int, ...],
        sampling: TokenSamplingParameters,
    ) -> int | None:
        if not completion_token_ids or not sampling.stop_token_ids:
            return None
        last_token_id = completion_token_ids[-1]
        if last_token_id in sampling.stop_token_ids:
            return last_token_id
        return None

    @staticmethod
    def _validate_token_rollout_batch(batch: TokenizedRolloutBatchRequest) -> None:
        for request in batch.requests:
            if not request.sampling.return_logprobs:
                raise ValueError("Levanter tokenized rollouts require return_logprobs=True")

    def resolve_decoding(self, decoding: DecodingConfig) -> DecodingConfig:
        """Bake Levanter fallback stop-token config into the applied decoding trace."""
        if decoding.stop_strings is not None or decoding.stop_token_ids is not None or not self._stop_tokens:
            return decoding
        return replace(decoding, stop_token_ids=list(self._stop_tokens))

    @staticmethod
    def _validate_supported_decoding(decoding: DecodingConfig) -> None:
        """Reject decoding fields the current Levanter RL path does not honor."""
        unsupported_fields: list[str] = []
        for field_name in UNSUPPORTED_LEVANTER_DECODING_FIELDS:
            field_value = getattr(decoding, field_name)
            if field_name == "ignore_eos":
                if field_value:
                    unsupported_fields.append(field_name)
                continue
            if field_value is not None:
                unsupported_fields.append(field_name)

        if unsupported_fields:
            raise ValueError(f"Levanter RL inference does not support: {', '.join(unsupported_fields)}")

    def _completion_request_kwargs(self, decoding: DecodingConfig, n: int) -> dict[str, Any]:
        """Translate shared decoding into the subset the Levanter RL wrapper actually honors."""
        decoding = self.resolve_decoding(decoding)
        self._validate_supported_decoding(decoding)
        temperature = 0.0 if decoding.strategy == DecodingStrategy.GREEDY else decoding.temperature
        return {
            "logprobs": True,
            "max_tokens": decoding.max_output_tokens,
            "temperature": temperature,
            "top_p": decoding.top_p,
            "n": n,
            # The Levanter OpenAI surface only accepts string stop sequences.
            "stop": stop_strings_for_decoding(decoding, self.tokenizer),
            "seed": decoding.seed,
        }

    def supports_token_rollouts(self) -> bool:
        """Return whether this context implements token-native rollout generation."""
        return True

    def generate_token_rollouts(self, batch: TokenizedRolloutBatchRequest) -> TokenizedRolloutBatchResult:
        """Generate already-tokenized rollout requests without OpenAI serialization."""
        self._validate_token_rollout_batch(batch)
        ctx = self._inference_server.inference_context

        service_requests = []
        request_output_counts = []
        for request_index, request in enumerate(batch.requests):
            service_requests.append(
                Request(
                    prompt_tokens=list(request.prompt_token_ids),
                    request_id=request_index,
                    decode_params=self._decode_params_from_token_sampling(
                        len(request.prompt_token_ids),
                        request.sampling,
                        request_index,
                    ),
                    n_generations=request.n_generations,
                    return_logprobs=request.sampling.return_logprobs,
                )
            )
            request_output_counts.append(request.n_generations)

        start = time.time()
        with (
            ctx.model_lock,
            hax.partitioning.set_mesh(ctx.config.trainer.device_mesh),
            hax.axis_mapping(ctx.config.trainer.compute_axis_mapping),
        ):
            if ctx.engine is None:
                raise RuntimeError("Levanter inference engine is not initialized.")
            result = ctx.engine.generate(service_requests)
        total_duration = time.time() - start

        rollouts: list[TokenizedRollout] = []
        failures: list[TokenizedRolloutFailure] = []
        output_idx = 0
        result_tokens = tuple(result.tokens)
        result_logprobs = None if result.logprobs is None else tuple(result.logprobs)
        for request, output_count in zip(batch.requests, request_output_counts, strict=True):
            for generation_index in range(output_count):
                backend_request_id = str(output_idx)
                if output_idx >= len(result_tokens):
                    failures.append(
                        TokenizedRolloutFailure(
                            request_id=request.request_id,
                            generation_index=generation_index,
                            reason=TokenRolloutFailureReason.BACKEND_ERROR,
                            message="Levanter engine returned fewer generations than requested",
                            backend_request_id=backend_request_id,
                        )
                    )
                    output_idx += 1
                    continue
                completion_token_ids = tuple(int(token_id) for token_id in result_tokens[output_idx])
                if result_logprobs is None:
                    raise RuntimeError("Levanter engine did not return logprobs for tokenized rollout generation.")
                if output_idx >= len(result_logprobs):
                    failures.append(
                        TokenizedRolloutFailure(
                            request_id=request.request_id,
                            generation_index=generation_index,
                            reason=TokenRolloutFailureReason.BACKEND_ERROR,
                            message="Levanter engine returned tokens without logprobs",
                            backend_request_id=backend_request_id,
                        )
                    )
                    output_idx += 1
                    continue
                completion_logprobs = tuple(float(logprob) for logprob in result_logprobs[output_idx])
                rollouts.append(
                    TokenizedRollout(
                        request_id=request.request_id,
                        generation_index=generation_index,
                        prompt_token_ids=request.prompt_token_ids,
                        completion_token_ids=completion_token_ids,
                        completion_logprobs=completion_logprobs,
                        finish_reason=self._finish_reason_from_tokens(completion_token_ids, request.sampling),
                        prompt_mask=tuple(False for _ in request.prompt_token_ids),
                        completion_mask=tuple(True for _ in completion_token_ids),
                        stop_token_id=self._stop_token_id_from_tokens(completion_token_ids, request.sampling),
                        metadata={"backend": "levanter"},
                    )
                )
                output_idx += 1

        prompt_tokens = sum(len(request.prompt_token_ids) for request in batch.requests)
        completion_tokens = sum(len(rollout.completion_token_ids) for rollout in rollouts)
        return TokenizedRolloutBatchResult(
            batch_id=batch.batch_id,
            tokenizer=batch.tokenizer,
            policy=batch.policy,
            rollouts=tuple(rollouts),
            failures=tuple(failures),
            timing=TokenRolloutTiming(total=total_duration),
            admission=TokenRolloutAdmissionMetadata(
                queued_tokens=prompt_tokens,
                admitted_tokens=prompt_tokens + completion_tokens,
                prefill_admissions=result.prefill_admissions,
                prefill_prompt_tokens_per_admission=tuple(result.prefill_prompt_tokens_per_admission),
                backend_request_ids=tuple(str(index) for index in range(len(batch.requests))),
            ),
            metadata={"backend": "levanter"},
        )

    # TODO: add support for ChatCompletion style [ { role, content} ] messages
    def batch_completions(
        self,
        prompts: list[str] | list[list[dict]],
        n: int,
        decoding: DecodingConfig,
        system_prompt: str | None = None,
    ) -> list[ChatCompletion]:
        """Call OpenAI API in batches with concurrency control."""
        del system_prompt
        request_kwargs = self._completion_request_kwargs(decoding, n)

        # Async batch processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = self.openai_client()

        async def create_completion(prompt: str) -> ChatCompletion:
            return await client.chat.completions.create(
                model=getattr(self._inference_server.config, "model_name", "test-model"),
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs,
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
