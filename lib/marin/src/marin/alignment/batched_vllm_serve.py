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

from rigging.filesystem import url_to_fs

from marin.alignment.inference_config import VLLMConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import LMEvaluationHarnessEvaluator
from marin.inference.vllm_server import VllmEnvironment, _looks_like_gpt_oss_model
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

# WARNING: GPT-OSS must stay on the explicit /v1/chat/completions path in this
# module. The old local-rendered /v1/completions path broke Harmony behavior on
# TPU during bring-up: it produced corrupt outputs on the bad backend and
# truncated/unsafe response handling on the good backend. Do not "simplify"
# GPT-OSS back onto the generic completions path unless you rerun the probes in
# .agents/logbooks/gpt-oss-tpu.md and prove the alternate path still works.
# This code is intentionally loud and refuses the old path for GPT-OSS.
GPT_OSS_REASONING_EFFORT = "low"
GPT_OSS_REQUIRED_FINISH_REASON = "stop"


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


def _token_lengths(tokenizer: PreTrainedTokenizerBase, texts: Sequence[str]) -> list[int]:
    input_ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return [len(token_ids) for token_ids in input_ids]


def _validate_completion_budget(
    config: VLLMConfig,
    *,
    prompt_token_counts: Sequence[int],
    max_tokens: int,
) -> None:
    if max_tokens < 1:
        raise ValueError(f"Expected max_tokens >= 1, got {max_tokens}")

    available_prompt_tokens = config.max_model_len - max_tokens
    if available_prompt_tokens < 1:
        raise ValueError(
            "Requested vLLM completion budget leaves no room for prompt tokens: "
            f"max_model_len={config.max_model_len}, max_tokens={max_tokens}, "
            f"available_prompt_tokens={available_prompt_tokens}. "
            "Lower max_tokens or raise max_model_len."
        )

    longest_prompt_tokens = max(prompt_token_counts, default=0)
    if longest_prompt_tokens > available_prompt_tokens:
        over_limit_count = sum(1 for token_count in prompt_token_counts if token_count > available_prompt_tokens)
        raise ValueError(
            "Requested vLLM completion budget exceeds the model context window for at least one prompt: "
            f"max_model_len={config.max_model_len}, max_tokens={max_tokens}, "
            f"available_prompt_tokens={available_prompt_tokens}, "
            f"longest_prompt_tokens={longest_prompt_tokens}, "
            f"over_limit_prompts={over_limit_count}. "
            "Lower max_tokens or raise max_model_len."
        )


def _build_model_config(config: VLLMConfig) -> ModelConfig:
    parsed = urlparse(config.model)
    is_object_store = parsed.scheme in {"gs", "s3"}
    model_name = os.path.basename(parsed.path.rstrip("/")) or "alignment-vllm-serve"
    load_format = config.load_format
    if load_format is None and is_object_store:
        load_format = "runai_streamer"
    engine_kwargs: dict[str, object] = {
        "load_format": load_format,
        "max_model_len": config.max_model_len,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
    }
    if config.tokenizer is not None:
        engine_kwargs["tokenizer"] = config.tokenizer
    if config.hf_overrides is not None:
        engine_kwargs["hf_overrides"] = config.hf_overrides
    if config.additional_config is not None:
        engine_kwargs["additional_config"] = config.additional_config
    return ModelConfig(
        name=model_name,
        path=config.model if is_object_store or os.path.exists(config.model) else None,
        engine_kwargs=engine_kwargs,
        apply_chat_template=False,
    )


def _gpt_oss_override_candidates(config: VLLMConfig) -> list[str]:
    if config.hf_overrides is None:
        return []

    candidates: list[str] = []
    model_type = config.hf_overrides.get("model_type")
    if isinstance(model_type, str):
        candidates.append(model_type)

    architectures = config.hf_overrides.get("architectures")
    if isinstance(architectures, list):
        candidates.extend(arch for arch in architectures if isinstance(arch, str))

    return candidates


def _config_uses_gpt_oss(config: VLLMConfig) -> bool:
    return _looks_like_gpt_oss_model(
        config.model,
        config.tokenizer,
        *_gpt_oss_override_candidates(config),
    )


def _vllm_env_overrides(config: VLLMConfig) -> dict[str, str]:
    env_overrides: dict[str, str] = {}
    if config.model_impl_type is not None:
        env_overrides["MODEL_IMPL_TYPE"] = config.model_impl_type
    # GPT-OSS models produce gibberish tokens under the flax_nnx backend
    # (proven in .agents/logbooks/gpt-oss-tpu.md, GTPU-001 vs GTPU-004).
    # The only validated backend is MODEL_IMPL_TYPE=vllm. Block flax_nnx
    # early so nobody rediscovers this failure mode by accident.
    if _config_uses_gpt_oss(config):
        resolved = env_overrides.get("MODEL_IMPL_TYPE", "vllm")
        if resolved == "flax_nnx":
            raise ValueError(
                "GPT-OSS models must not use model_impl_type='flax_nnx' — it produces "
                "incoherent token soup on TPU. Use model_impl_type='vllm' (the default). "
                "See .agents/logbooks/gpt-oss-tpu.md GTPU-001 through GTPU-004 for the "
                "A/B evidence that proved this."
            )
    return env_overrides


@contextmanager
def _load_tokenizer(model_path: str, *, tokenizer_path: str | None = None):
    tokenizer_source = tokenizer_path or model_path
    parsed = urlparse(tokenizer_source)
    if parsed.scheme in {"gs", "s3"}:
        with LMEvaluationHarnessEvaluator._stage_remote_tokenizer_dir(tokenizer_source) as tokenizer_dir:
            if tokenizer_dir is None:
                raise RuntimeError(f"Could not stage tokenizer files for {tokenizer_source}")
            yield AutoTokenizer.from_pretrained(tokenizer_dir)
        return

    yield AutoTokenizer.from_pretrained(tokenizer_source)


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
        self._tokenizer_context = _load_tokenizer(self.config.model, tokenizer_path=self.config.tokenizer)
        self._tokenizer = self._tokenizer_context.__enter__()
        self._metrics.tokenizer_load_seconds = time.perf_counter() - tokenizer_start
        env = VllmEnvironment(
            model=_build_model_config(self.config),
            mode=self.config.resolved_serve_mode,
            timeout_seconds=self.timeout_seconds,
            docker_image=self.config.docker_image,
            env_overrides=_vllm_env_overrides(self.config),
            native_stderr_mode=self.config.native_stderr_mode,
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

    def _uses_gpt_oss_chat_path(self) -> bool:
        if _config_uses_gpt_oss(self.config):
            return True
        if self._env is not None:
            return _looks_like_gpt_oss_model(self._env.model_id)
        return False

    def _gpt_oss_path_error(self, attempted_path: str) -> ValueError:
        return ValueError(
            "GPT-OSS local inference must use generate_from_messages() -> /v1/chat/completions "
            f"with fixed reasoning_effort={GPT_OSS_REASONING_EFFORT!r}. "
            f"The attempted path {attempted_path!r} is disabled because it breaks the validated "
            "Harmony serving contract. If you think this restriction is wrong, rerun the GPT-OSS "
            "TPU probes in .agents/logbooks/gpt-oss-tpu.md before changing this code."
        )

    def render_messages(
        self,
        message_batches: Sequence[Sequence[dict[str, str]]],
        *,
        stage_name: str = "unlabeled",
    ) -> list[str]:
        if self._uses_gpt_oss_chat_path():
            raise self._gpt_oss_path_error("render_messages() / local apply_chat_template")

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
        if self._uses_gpt_oss_chat_path():
            raise self._gpt_oss_path_error("generate_from_prompt_texts() / /v1/completions")

        if self._env is None:
            raise RuntimeError("vLLM environment is not available outside the active session.")
        if self._env.model_id is None:
            raise RuntimeError("Expected vLLM server to expose a model id.")
        tokenizer = self._require_tokenizer()
        prompt_token_counts = _token_lengths(tokenizer, prompt_texts)
        _validate_completion_budget(
            self.config,
            prompt_token_counts=prompt_token_counts,
            max_tokens=max_tokens,
        )

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
        input_token_count = sum(prompt_token_counts)
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

    def _gpt_oss_chat_request(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        n: int,
    ) -> tuple[list[str], int, int, float]:
        """Send a single GPT-OSS chat request. Returns (texts, input_tokens, output_tokens, seconds)."""
        assert self._env is not None and self._env.model_id is not None
        request_start = time.perf_counter()
        response = requests.post(
            f"{self._env.server_url}/chat/completions",
            json={
                "model": self._env.model_id,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,
                "reasoning_effort": GPT_OSS_REASONING_EFFORT,
            },
            timeout=900,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:2000]
            logger.error("GPT-OSS chat request failed: status=%s body=%s", response.status_code, body)
            raise requests.HTTPError(f"{exc}; response body: {body}") from exc

        payload = response.json()
        choice_texts = _extract_gpt_oss_chat_texts(payload)
        usage = payload.get("usage")
        input_tok = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
        output_tok = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
        return choice_texts, int(input_tok), int(output_tok), time.perf_counter() - request_start

    def _generate_from_messages_gpt_oss(
        self,
        message_batches: Sequence[Sequence[dict[str, str]]],
        *,
        stage_name: str,
        temperature: float,
        max_tokens: int,
        n: int,
    ) -> list[list[str]]:
        import concurrent.futures

        if self._env is None:
            raise RuntimeError("vLLM environment is not available outside the active session.")
        if self._env.model_id is None:
            raise RuntimeError("Expected vLLM server to expose a model id.")

        num_conversations = len(message_batches)
        logger.info(
            "Sending %d GPT-OSS vLLM serve requests to /v1/chat/completions concurrently "
            "(the generic /v1/completions path is intentionally disabled)",
            num_conversations,
        )

        # Send all conversations concurrently — vLLM batches them server-side.
        # This mirrors how /v1/completions sends multiple prompts in one request.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_conversations, 256)) as pool:
            futures = {
                pool.submit(
                    self._gpt_oss_chat_request,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                ): idx
                for idx, messages in enumerate(message_batches)
            }
            results: list[tuple[int, list[str]]] = []
            errors: list[tuple[int, Exception]] = []
            total_input_tokens = 0
            total_output_tokens = 0
            total_seconds = 0.0
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    choice_texts, input_tok, output_tok, secs = future.result()
                    results.append((idx, choice_texts))
                    total_input_tokens += input_tok
                    total_output_tokens += output_tok
                    total_seconds += secs
                except Exception as exc:
                    logger.warning("GPT-OSS request %d failed: %s", idx, exc)
                    errors.append((idx, exc))

        if errors:
            logger.warning(
                "%d of %d GPT-OSS requests failed in this batch; inserting empty results for failed items",
                len(errors),
                num_conversations,
            )
            for idx, _exc in errors:
                results.append((idx, []))

        self._metrics.record_request(
            stage_name,
            request_prompt_count=num_conversations,
            completion_count=sum(len(texts) for _, texts in results),
            input_token_count=total_input_tokens,
            output_token_count=total_output_tokens,
            request_seconds=total_seconds / max(num_conversations, 1),
        )

        # Restore original order
        results.sort(key=lambda x: x[0])
        return [texts for _, texts in results]

    def generate_from_messages(
        self,
        message_batches: Sequence[Sequence[dict[str, str]]],
        *,
        stage_name: str = "unlabeled",
        temperature: float,
        max_tokens: int,
        n: int = 1,
    ) -> list[list[str]]:
        if self._uses_gpt_oss_chat_path():
            return self._generate_from_messages_gpt_oss(
                message_batches,
                stage_name=stage_name,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )

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


def _extract_gpt_oss_chat_texts(payload: dict[str, object]) -> list[str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"GPT-OSS chat response is missing choices: {json.dumps(payload, default=str)[:2000]}")

    outputs: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError(
                "GPT-OSS chat response contained a non-dict choice: " f"{json.dumps(payload, default=str)[:2000]}"
            )
        finish_reason = choice.get("finish_reason")
        if finish_reason != GPT_OSS_REQUIRED_FINISH_REASON:
            raise ValueError(
                "GPT-OSS chat response did not finish cleanly. "
                f"Expected finish_reason={GPT_OSS_REQUIRED_FINISH_REASON!r}, got {finish_reason!r}. "
                f"Response excerpt: {json.dumps(payload, default=str)[:2000]}"
            )

        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError(
                "GPT-OSS chat response is missing message content. "
                f"Response excerpt: {json.dumps(payload, default=str)[:2000]}"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "GPT-OSS chat response is missing final assistant content. "
                "This usually means the request fell back to reasoning-only output or the serving contract changed. "
                f"Response excerpt: {json.dumps(payload, default=str)[:2000]}"
            )
        outputs.append(content)
    return outputs
