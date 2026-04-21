# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Context which uses the native Levanter engine for inference.

This context is provided to environments and provides access to the inference server
as well as methods for tokenization and logprob extraction from an OpenAI ChatCompletion.
"""

import asyncio
from dataclasses import dataclass, replace
import logging
import threading
from typing import Any
import haliax as hax
from jax.sharding import Mesh
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmHeadModel
from levanter.tokenizers import MarinTokenizer
from marin.rl.decoding import DecodingConfig, DecodingStrategy, stop_strings_for_decoding
from marin.rl.environments.inference_ctx.base import BaseInferenceContext

# TODO(chris): use a different weight transfer method update model, take it out from here
from marin.rl.weight_transfer.arrow_flight import update_model
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

UNSUPPORTED_LEVANTER_DECODING_FIELDS = (
    "top_k",
    "top_p",
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
            "n": n,
            # The Levanter OpenAI surface only accepts string stop sequences.
            "stop": stop_strings_for_decoding(decoding, self.tokenizer),
            "seed": decoding.seed,
        }

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
