# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Steps 2a/2b: Generate chosen and rejected responses.

Supports two backends via InferenceConfig:
- LiteLLMConfig → API calls via Zephyr pipeline (streaming, parallel workers)
- VLLMConfig → local vLLM instance (batch generation)

Teacher responses receive spec guidance in the system prompt.
Rejected responses receive NO spec guidance.
The output strips spec guidance from system prompts so the training data
teaches the model to internalize behavior without needing the spec at inference time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from zephyr import Dataset, ZephyrContext, load_jsonl

from marin.alignment.generate_prompts import load_sharded_jsonl_gz, write_sharded_jsonl_gz
from marin.alignment.inference_config import InferenceConfig, LiteLLMConfig, VLLMConfig
from marin.alignment.llm_client import get_or_create_vllm_engine, llm_chat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResponseGenConfig:
    """Configuration for response generation."""

    prompts_path: str
    output_path: str

    # Inference backend — LiteLLMConfig or VLLMConfig (serialized as dict for executor)
    model_config: dict[str, Any] | InferenceConfig

    # Generation parameters
    n: int = 1
    temperature: float = 0.7
    max_tokens: int = 2048

    # Spec guidance: path to behavior statements JSON (for teacher only, None for rejected)
    behavior_statements_path: str | None = None

    def resolve_inference_config(self) -> InferenceConfig:
        """Resolve model_config to an InferenceConfig instance."""
        if isinstance(self.model_config, InferenceConfig):
            return self.model_config
        # Reconstruct from serialized dict
        cfg = dict(self.model_config)
        backend = cfg.pop("backend")
        if backend == "litellm":
            return LiteLLMConfig(**cfg)
        elif backend == "vllm":
            return VLLMConfig(**cfg)
        else:
            raise ValueError(f"Unknown inference backend: {backend}")


def _load_behavior_statements(path: str) -> dict[str, str]:
    """Load behavior statement texts keyed by ID from a spec JSONL file."""
    return {record["id"]: record["text"] for record in load_jsonl(path)}


def _build_messages(
    prompt: dict[str, Any],
    behavior_statements: dict[str, str] | None,
) -> list[dict[str, str]]:
    """Build message list for a single prompt.

    If behavior_statements is provided (teacher mode), inject spec guidance
    into the system prompt. Otherwise, use scenario system prompt only.
    """
    system_content = prompt.get("system_prompt", "")

    if behavior_statements:
        behavior_id = prompt.get("behavior_id", "")
        statement_text = behavior_statements.get(behavior_id, "")
        if statement_text:
            system_content += f"\n\nAdditionally, follow this behavioral guideline: {statement_text}"

    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt.get("user_message", "")})
    return messages


def _make_response_record(
    prompt: dict[str, Any],
    model_name: str,
    responses: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "prompt_id": f"{prompt.get('behavior_id', '')}/{prompt.get('config_id', '')}",
        "system_prompt": prompt.get("system_prompt", ""),  # scenario only, no spec
        "user_message": prompt.get("user_message", ""),
        "behavior_id": prompt.get("behavior_id", ""),
        "rubric": prompt.get("rubric", ""),
        "model": model_name,
        "responses": responses,
    }


# ---------------------------------------------------------------------------
# API path: Zephyr pipeline (streaming, distributed)
# ---------------------------------------------------------------------------


def _generate_via_api(
    config: ResponseGenConfig,
    inference_config: LiteLLMConfig,
    behavior_statements: dict[str, str] | None,
) -> None:
    """Generate responses for all prompts via Zephyr pipeline + litellm API calls."""

    def _process_prompt(prompt: dict) -> dict:
        messages = _build_messages(prompt, behavior_statements)
        responses = llm_chat(
            config=inference_config,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            n=config.n,
        )
        return _make_response_record(
            prompt,
            inference_config.model,
            [{"content": r.content, "index": i} for i, r in enumerate(responses)],
        )

    ds = (
        Dataset.from_files(f"{config.prompts_path}/*.jsonl.gz")
        .load_jsonl()
        .map(_process_prompt)
        .write_jsonl(f"{config.output_path}/shard-{{shard:05d}}.jsonl.gz")
    )

    ctx = ZephyrContext(max_workers=inference_config.workers)
    ctx.execute(ds)


# ---------------------------------------------------------------------------
# vLLM path: batch generation (all prompts in one llm.generate call)
# ---------------------------------------------------------------------------


def _generate_via_vllm(
    config: ResponseGenConfig,
    inference_config: VLLMConfig,
    behavior_statements: dict[str, str] | None,
) -> None:
    """Generate responses for all prompts via local vLLM instance.

    Uses batch generation (single llm.generate call) for efficiency.
    Zephyr's per-record map would call vLLM once per prompt which is much
    slower than batching, so this path loads all prompts eagerly.
    """
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError("vLLM is required for local model inference. Install with: pip install vllm") from exc

    prompts = load_sharded_jsonl_gz(config.prompts_path)
    logger.info("Loaded %d prompts for vLLM batch generation", len(prompts))

    llm, tokenizer = get_or_create_vllm_engine(inference_config)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        n=config.n,
    )

    # Build all message lists and apply chat template
    all_prompt_texts = []
    for prompt in prompts:
        messages = _build_messages(prompt, behavior_statements)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompt_texts.append(text)

    outputs = llm.generate(all_prompt_texts, sampling_params)

    # Write results using zephyr
    results: list[dict[str, Any]] = []
    for prompt, output in zip(prompts, outputs, strict=True):
        responses = [{"content": completion.text, "index": j} for j, completion in enumerate(output.outputs)]
        results.append(_make_response_record(prompt, inference_config.model, responses))

    write_sharded_jsonl_gz(results, config.output_path, shard_size=5000)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_responses(config: ResponseGenConfig) -> None:
    """Generate responses for all prompts.

    Dispatches to Zephyr pipeline (API) or batch vLLM based on InferenceConfig type.
    """
    behavior_statements = None
    if config.behavior_statements_path:
        behavior_statements = _load_behavior_statements(config.behavior_statements_path)
        logger.info("Loaded %d behavior statements (teacher mode)", len(behavior_statements))
    else:
        logger.info("No behavior statements provided (rejected mode)")

    inference_config = config.resolve_inference_config()

    if isinstance(inference_config, LiteLLMConfig):
        logger.info("Using Zephyr + litellm API: %s", inference_config.model)
        _generate_via_api(config, inference_config, behavior_statements)
    elif isinstance(inference_config, VLLMConfig):
        logger.info("Using batch vLLM: %s", inference_config.model)
        _generate_via_vllm(config, inference_config, behavior_statements)
    else:
        raise ValueError(f"Unknown inference config type: {type(inference_config)}")
