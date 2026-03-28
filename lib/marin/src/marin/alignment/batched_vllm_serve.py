# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Batched local inference for alignment via `vllm serve`.

This module provides a reusable batched OpenAI-compatible completions client
for alignment steps that run local open-weight models on TPU workers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from iris.marin_fs import url_to_fs

from marin.alignment.inference_config import VLLMConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import LMEvaluationHarnessEvaluator
from marin.inference.vllm_server import VllmEnvironment
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


@dataclass
class VllmStageMetrics:
    """Aggregated metrics for one logical stage within a vLLM serve session."""

    render_call_count: int = 0
    request_count: int = 0
    request_prompt_count: int = 0
    completion_count: int = 0
    input_token_count: int = 0
    output_token_count: int = 0
    render_seconds: float = 0.0
    request_seconds: float = 0.0

    def record_render(self, *, request_prompt_count: int, render_seconds: float) -> None:
        self.render_call_count += 1
        self.render_seconds += render_seconds

    def record_request(
        self,
        *,
        request_prompt_count: int,
        completion_count: int,
        input_token_count: int,
        output_token_count: int,
        request_seconds: float,
    ) -> None:
        self.request_count += 1
        self.request_prompt_count += request_prompt_count
        self.completion_count += completion_count
        self.input_token_count += input_token_count
        self.output_token_count += output_token_count
        self.request_seconds += request_seconds

    def to_dict(self) -> dict[str, int | float | None]:
        input_tokens_per_second = None
        output_tokens_per_second = None
        if self.request_seconds > 0:
            input_tokens_per_second = self.input_token_count / self.request_seconds
            output_tokens_per_second = self.output_token_count / self.request_seconds
        return {
            "render_call_count": self.render_call_count,
            "request_count": self.request_count,
            "request_prompt_count": self.request_prompt_count,
            "completion_count": self.completion_count,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "render_seconds": self.render_seconds,
            "request_seconds": self.request_seconds,
            "input_tokens_per_second": input_tokens_per_second,
            "output_tokens_per_second": output_tokens_per_second,
        }


@dataclass
class VllmSessionMetrics:
    """Structured metrics for one local `vllm serve` session."""

    model: str
    tensor_parallel_size: int
    max_model_len: int
    tokenizer_load_seconds: float = 0.0
    server_start_seconds: float = 0.0
    totals: VllmStageMetrics = field(default_factory=VllmStageMetrics)
    stages: dict[str, VllmStageMetrics] = field(default_factory=dict)

    def _stage(self, stage_name: str) -> VllmStageMetrics:
        if stage_name not in self.stages:
            self.stages[stage_name] = VllmStageMetrics()
        return self.stages[stage_name]

    def record_render(self, stage_name: str, *, request_prompt_count: int, render_seconds: float) -> None:
        self._stage(stage_name).record_render(
            request_prompt_count=request_prompt_count,
            render_seconds=render_seconds,
        )
        self.totals.record_render(
            request_prompt_count=request_prompt_count,
            render_seconds=render_seconds,
        )

    def record_request(
        self,
        stage_name: str,
        *,
        request_prompt_count: int,
        completion_count: int,
        input_token_count: int,
        output_token_count: int,
        request_seconds: float,
    ) -> None:
        self._stage(stage_name).record_request(
            request_prompt_count=request_prompt_count,
            completion_count=completion_count,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            request_seconds=request_seconds,
        )
        self.totals.record_request(
            request_prompt_count=request_prompt_count,
            completion_count=completion_count,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            request_seconds=request_seconds,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": "vllm_serve",
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "tokenizer_load_seconds": self.tokenizer_load_seconds,
            "server_start_seconds": self.server_start_seconds,
            "session_enter_seconds": self.tokenizer_load_seconds + self.server_start_seconds,
            "totals": self.totals.to_dict(),
            "stages": {stage_name: metrics.to_dict() for stage_name, metrics in sorted(self.stages.items())},
        }


def write_vllm_metrics_artifact(
    path: str,
    *,
    logical_stage: str,
    sessions: Sequence[tuple[str, dict[str, object]]],
) -> None:
    """Write a standardized structured metrics artifact for local vLLM stages."""

    fs, fs_path = url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    payload = {
        "logical_stage": logical_stage,
        "session_count": len(sessions),
        "sessions": [
            {
                "session_name": session_name,
                **metrics,
            }
            for session_name, metrics in sessions
        ],
    }
    with fs.open(fs_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


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
    _env: VllmEnvironment | None = field(init=False, default=None, repr=False)
    _tokenizer: PreTrainedTokenizerBase | None = field(init=False, default=None, repr=False)
    _metrics: VllmSessionMetrics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._metrics = VllmSessionMetrics(
            model=self.config.model,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_model_len,
        )

    def __enter__(self) -> BatchedVllmServeSession:
        tokenizer_start = time.perf_counter()
        self._tokenizer_context = _load_tokenizer(self.config.model)
        self._tokenizer = self._tokenizer_context.__enter__()
        self._metrics.tokenizer_load_seconds = time.perf_counter() - tokenizer_start
        env = VllmEnvironment(
            model=_build_model_config(self.config),
            mode="native",
            timeout_seconds=self.timeout_seconds,
        )
        self._remove_lock_context = remove_tpu_lockfile_on_exit()
        self._remove_lock_context.__enter__()
        self._env = env
        try:
            server_start = time.perf_counter()
            env.__enter__()
            self._metrics.server_start_seconds = time.perf_counter() - server_start
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

    def render_messages(
        self,
        message_batches: Sequence[Sequence[dict[str, str]]],
        *,
        stage_name: str = "unlabeled",
    ) -> list[str]:
        tokenizer = self._require_tokenizer()

        logger.info("Rendering %d chat prompts for batched vLLM serve", len(message_batches))
        render_start = time.perf_counter()
        prompt_texts: list[str] = []
        for messages in message_batches:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not isinstance(prompt_text, str):
                raise TypeError("Expected apply_chat_template(..., tokenize=False) to return a string.")
            prompt_texts.append(prompt_text)
        self._metrics.record_render(
            stage_name,
            request_prompt_count=len(message_batches),
            render_seconds=time.perf_counter() - render_start,
        )
        return prompt_texts

    def generate_from_prompt_texts(
        self,
        prompt_texts: Sequence[str],
        *,
        stage_name: str = "unlabeled",
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
        request_start = time.perf_counter()
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
        tokenizer = self._require_tokenizer()
        input_token_count = _token_count(tokenizer, prompt_texts)
        output_token_count = _token_count(tokenizer, choice_texts)
        self._metrics.record_request(
            stage_name,
            request_prompt_count=len(prompt_texts),
            completion_count=len(choice_texts),
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            request_seconds=time.perf_counter() - request_start,
        )
        return _group_completion_texts(choice_texts, prompt_count=len(prompt_texts), n=n)

    def generate_from_messages(
        self,
        message_batches: Sequence[Sequence[dict[str, str]]],
        *,
        stage_name: str = "unlabeled",
        temperature: float,
        max_tokens: int,
        n: int = 1,
    ) -> list[list[str]]:
        prompt_texts = self.render_messages(message_batches, stage_name=stage_name)
        return self.generate_from_prompt_texts(
            prompt_texts,
            stage_name=stage_name,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )

    def metrics_snapshot(self) -> dict[str, object]:
        return self._metrics.to_dict()

    def _require_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not available outside the active vLLM serve session.")
        return self._tokenizer

    def logs_tail(self, *, max_lines: int = 200) -> str:
        if self._env is None:
            raise RuntimeError("vLLM environment is not available outside the active session.")
        return self._env.logs_tail(max_lines=max_lines)

    def diagnostics(self, *, max_lines: int = 200) -> dict[str, str]:
        if self._env is None:
            return {}
        return self._env.diagnostics(max_lines=max_lines)


def _token_count(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str]) -> int:
    input_ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return sum(len(token_ids) for token_ids in input_ids)
