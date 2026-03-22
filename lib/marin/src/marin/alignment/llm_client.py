# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified LLM client dispatching to litellm (API) or vLLM (local).

All alignment modules call through this client. The backend is selected
by the type of InferenceConfig passed in — LiteLLMConfig routes to the
litellm library, VLLMConfig spins up a local vLLM engine.

For repeated vLLM calls within a single step, use VLLMEngine as a context
manager to avoid re-creating the engine on every call.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

import litellm

from marin.alignment.inference_config import InferenceConfig, LiteLLMConfig, VLLMConfig

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# Module-level vLLM engine cache — set by VLLMEngine context manager
_active_vllm_engine: dict | None = None


@dataclass
class LLMResponse:
    """Normalized response from any LLM backend."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# litellm backend
# ---------------------------------------------------------------------------


def _chat_litellm(
    config: LiteLLMConfig,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    n: int,
) -> list[LLMResponse]:
    response = litellm.completion(
        model=config.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        num_retries=config.num_retries,
    )
    results = []
    for choice in response.choices:
        results.append(
            LLMResponse(
                content=choice.message.content or "",
                model=response.model or config.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
            )
        )
    return results


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------


def get_or_create_vllm_engine(config: VLLMConfig):
    """Get cached vLLM engine or create a new one."""
    global _active_vllm_engine

    if _active_vllm_engine is not None and _active_vllm_engine["model"] == config.model:
        return _active_vllm_engine["llm"], _active_vllm_engine["tokenizer"]

    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError("vLLM is required for local model inference. Install with: pip install vllm") from exc

    logger.info("Creating vLLM engine for model: %s", config.model)
    llm = LLM(
        model=config.model,
        max_model_len=config.max_model_len,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def _chat_vllm(
    config: VLLMConfig,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    n: int,
) -> list[LLMResponse]:
    """vLLM inference. Reuses cached engine if available."""
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError("vLLM is required for local model inference. Install with: pip install vllm") from exc

    llm, tokenizer = get_or_create_vllm_engine(config)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], sampling_params)

    results = []
    for completion in outputs[0].outputs:
        results.append(LLMResponse(content=completion.text, model=config.model))

    # Only clean up if there's no active context manager holding the engine
    if _active_vllm_engine is None:
        del llm

    return results


@contextmanager
def vllm_engine(config: VLLMConfig):
    """Context manager that keeps a vLLM engine alive for multiple calls.

    Usage:
        with vllm_engine(config) as engine_config:
            # All llm_chat_single calls within this block reuse the same engine
            for prompt in prompts:
                response = llm_chat_single(config=engine_config, ...)
    """
    global _active_vllm_engine

    llm, tokenizer = get_or_create_vllm_engine(config)
    _active_vllm_engine = {"llm": llm, "tokenizer": tokenizer, "model": config.model}

    try:
        yield config
    finally:
        _active_vllm_engine = None
        del llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def llm_chat(
    config: InferenceConfig | str,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.0,
    n: int = 1,
) -> list[LLMResponse]:
    """Call an LLM, dispatching based on config type.

    Args:
        config: LiteLLMConfig for API calls, VLLMConfig for local inference,
            or a plain string (interpreted as a LiteLLMConfig model ID for convenience).
        messages: List of message dicts with "role" and "content" keys.
        system_prompt: Optional system prompt prepended to messages.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        n: Number of completions to generate.

    Returns:
        List of LLMResponse objects (length = n).
    """
    # Convenience: bare string → LiteLLMConfig
    if isinstance(config, str):
        config = LiteLLMConfig(model=config)

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    if isinstance(config, LiteLLMConfig):
        return _chat_litellm(config, full_messages, max_tokens, temperature, n)
    elif isinstance(config, VLLMConfig):
        return _chat_vllm(config, full_messages, max_tokens, temperature, n)
    else:
        raise ValueError(f"Unknown inference config type: {type(config)}")


def llm_chat_single(
    config: InferenceConfig | str,
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.0,
) -> LLMResponse:
    """Convenience wrapper that returns a single response."""
    responses = llm_chat(
        config=config,
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
    )
    return responses[0]
