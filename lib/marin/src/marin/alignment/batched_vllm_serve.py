# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Batched local inference for alignment via `vllm serve`.

This module provides a reusable batched OpenAI-compatible completions client
for alignment steps that run local open-weight models on TPU workers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from transformers import AutoTokenizer

from marin.alignment.inference_config import VLLMConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import LMEvaluationHarnessEvaluator
from marin.inference.vllm_server import VllmEnvironment
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


def _group_completion_texts(choice_texts: list[str], *, prompt_count: int, n: int) -> list[list[str]]:
    expected_choice_count = prompt_count * n
    if len(choice_texts) != expected_choice_count:
        raise ValueError(
            f"Expected {expected_choice_count} completion choices for {prompt_count} prompts with n={n}, "
            f"got {len(choice_texts)}"
        )

    grouped: list[list[str]] = []
    for prompt_index in range(prompt_count):
        start = prompt_index * n
        grouped.append(choice_texts[start : start + n])
    return grouped


def _build_model_config(config: VLLMConfig) -> ModelConfig:
    parsed = urlparse(config.model)
    is_object_store = parsed.scheme in {"gs", "s3"}
    model_name = os.path.basename(parsed.path.rstrip("/")) or "alignment-vllm-serve"
    load_format = config.load_format
    if load_format is None and is_object_store:
        load_format = "runai_streamer"
    return ModelConfig(
        name=model_name,
        path=config.model if is_object_store or os.path.exists(config.model) else None,
        engine_kwargs={
            "load_format": load_format,
            "max_model_len": config.max_model_len,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
        },
        apply_chat_template=False,
    )


@contextmanager
def _load_tokenizer(model_path: str):
    parsed = urlparse(model_path)
    if parsed.scheme in {"gs", "s3"}:
        with LMEvaluationHarnessEvaluator._stage_remote_tokenizer_dir(model_path) as tokenizer_dir:
            if tokenizer_dir is None:
                raise RuntimeError(f"Could not stage tokenizer files for {model_path}")
            yield AutoTokenizer.from_pretrained(tokenizer_dir)
        return

    yield AutoTokenizer.from_pretrained(model_path)


@dataclass
class BatchedVllmServeSession:
    """Lifecycle wrapper for batched `vllm serve` inference."""

    config: VLLMConfig
    timeout_seconds: int = 3600

    def __post_init__(self) -> None:
        self._env: VllmEnvironment | None = None
        self._tokenizer = None

    def __enter__(self) -> BatchedVllmServeSession:
        self._tokenizer_context = _load_tokenizer(self.config.model)
        self._tokenizer = self._tokenizer_context.__enter__()
        env = VllmEnvironment(
            model=_build_model_config(self.config),
            mode="native",
            timeout_seconds=self.timeout_seconds,
        )
        self._remove_lock_context = remove_tpu_lockfile_on_exit()
        self._remove_lock_context.__enter__()
        self._env = env
        try:
            env.__enter__()
        except Exception:
            self._remove_lock_context.__exit__(None, None, None)
            self._tokenizer_context.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._env is not None:
            self._env.__exit__(exc_type, exc, tb)
        if hasattr(self, "_remove_lock_context"):
            self._remove_lock_context.__exit__(exc_type, exc, tb)
        if hasattr(self, "_tokenizer_context"):
            self._tokenizer_context.__exit__(exc_type, exc, tb)
        self._env = None
        self._tokenizer = None

    def render_messages(self, message_batches: Sequence[Sequence[dict[str, str]]]) -> list[str]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not available outside the active vLLM serve session.")

        logger.info("Rendering %d chat prompts for batched vLLM serve", len(message_batches))
        return [
            self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in message_batches
        ]

    def generate_from_prompt_texts(
        self,
        prompt_texts: Sequence[str],
        *,
        temperature: float,
        max_tokens: int,
        n: int = 1,
    ) -> list[list[str]]:
        if self._env is None:
            raise RuntimeError("vLLM environment is not available outside the active session.")
        if self._env.model_id is None:
            raise RuntimeError("Expected vLLM server to expose a model id.")

        logger.info(
            "Sending batched vLLM serve request to /v1/completions for %d prompts (n=%d)",
            len(prompt_texts),
            n,
        )
        response = requests.post(
            f"{self._env.server_url}/completions",
            json={
                "model": self._env.model_id,
                "prompt": list(prompt_texts),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,
            },
            timeout=900,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:2000]
            logger.error("Batched vLLM serve request failed: status=%s body=%s", response.status_code, body)
            raise requests.HTTPError(f"{exc}; response body: {body}") from exc
        payload = response.json()
        choice_texts = [choice["text"] for choice in payload["choices"]]
        return _group_completion_texts(choice_texts, prompt_count=len(prompt_texts), n=n)

    def generate_from_messages(
        self,
        message_batches: Sequence[Sequence[dict[str, str]]],
        *,
        temperature: float,
        max_tokens: int,
        n: int = 1,
    ) -> list[list[str]]:
        prompt_texts = self.render_messages(message_batches)
        return self.generate_from_prompt_texts(
            prompt_texts,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )

    def logs_tail(self, *, max_lines: int = 200) -> str:
        if self._env is None:
            raise RuntimeError("vLLM environment is not available outside the active session.")
        return self._env.logs_tail(max_lines=max_lines)

    def diagnostics(self, *, max_lines: int = 200) -> dict[str, str]:
        if self._env is None:
            return {}
        return self._env.diagnostics(max_lines=max_lines)
