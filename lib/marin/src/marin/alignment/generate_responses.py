# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Steps 2a/2b: Generate chosen and rejected responses.

Supports two backends:
- API models via litellm (e.g. "openai/gpt-4.1") — used for teacher/rejected when specified as strings
- vLLM for local model checkpoints — used when model is an ExecutorStep path

Teacher responses receive spec guidance in the system prompt.
Rejected responses receive NO spec guidance.
The output strips spec guidance from system prompts so the training data
teaches the model to internalize behavior without needing the spec at inference time.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marin.alignment.llm_client import llm_chat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResponseGenConfig:
    """Configuration for response generation."""

    prompts_path: str
    output_path: str

    # Model: API string (e.g. "openai/gpt-4.1") or local checkpoint path
    model: str

    # Generation parameters
    n: int = 1
    temperature: float = 0.7
    max_tokens: int = 2048

    # Spec guidance: path to behavior statements JSON (for teacher only, None for rejected)
    behavior_statements_path: str | None = None

    # Parallelism
    workers: int = 64

    # vLLM settings (only used when model is a local path)
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: int = 4096
    vllm_gpu_memory_utilization: float = 0.9


def _is_api_model(model: str) -> bool:
    """Check if a model string refers to an API model (e.g. "openai/gpt-4.1")."""
    return "/" in model and not model.startswith("/") and not model.startswith("gs://")


def _load_prompts(prompts_path: str) -> list[dict[str, Any]]:
    """Load prompts from sharded JSONL.GZ files."""
    prompts_dir = Path(prompts_path)
    all_prompts: list[dict[str, Any]] = []

    shard_files = sorted(prompts_dir.glob("*.jsonl.gz"))
    for shard_file in shard_files:
        with gzip.open(shard_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_prompts.append(json.loads(line))

    return all_prompts


def _load_behavior_statements(path: str) -> dict[str, str]:
    """Load behavior statement texts keyed by ID."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _generate_for_prompt_api(
    prompt: dict[str, Any],
    model: str,
    n: int,
    temperature: float,
    max_tokens: int,
    behavior_statements: dict[str, str] | None,
) -> dict[str, Any]:
    """Generate responses for a single prompt via API."""
    messages = _build_messages(prompt, behavior_statements)

    responses = llm_chat(
        model_id=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )

    return {
        "prompt_id": f"{prompt.get('behavior_id', '')}/{prompt.get('config_id', '')}",
        "system_prompt": prompt.get("system_prompt", ""),  # scenario only, no spec
        "user_message": prompt.get("user_message", ""),
        "behavior_id": prompt.get("behavior_id", ""),
        "rubric": prompt.get("rubric", ""),
        "model": model,
        "responses": [{"content": r.content, "index": i} for i, r in enumerate(responses)],
    }


def _generate_via_api(
    prompts: list[dict[str, Any]],
    config: ResponseGenConfig,
    behavior_statements: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Generate responses for all prompts via litellm API calls."""
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as pool:
        future_map = {
            pool.submit(
                _generate_for_prompt_api,
                prompt,
                config.model,
                config.n,
                config.temperature,
                config.max_tokens,
                behavior_statements,
            ): i
            for i, prompt in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(f"prompt {idx}: {exc}")
                logger.error("Response generation failed for prompt %d: %s", idx, exc)

    if failures:
        logger.warning("%d/%d prompts failed during response generation", len(failures), len(prompts))

    return results


def _generate_via_vllm(
    prompts: list[dict[str, Any]],
    config: ResponseGenConfig,
    behavior_statements: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Generate responses for all prompts via local vLLM instance."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("vLLM is required for local model inference. Install with: pip install vllm") from exc

    logger.info("Starting vLLM with model: %s", config.model)
    llm = LLM(
        model=config.model,
        max_model_len=config.vllm_max_model_len,
        tensor_parallel_size=config.vllm_tensor_parallel_size,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        n=config.n,
    )

    # Build all message lists
    all_messages: list[list[dict[str, str]]] = []
    for prompt in prompts:
        all_messages.append(_build_messages(prompt, behavior_statements))

    # Use vLLM's chat interface
    tokenizer = llm.get_tokenizer()
    all_prompt_texts = []
    for messages in all_messages:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompt_texts.append(text)

    outputs = llm.generate(all_prompt_texts, sampling_params)

    results: list[dict[str, Any]] = []
    for prompt, output in zip(prompts, outputs, strict=True):
        responses = []
        for j, completion in enumerate(output.outputs):
            responses.append({"content": completion.text, "index": j})

        results.append(
            {
                "prompt_id": f"{prompt.get('behavior_id', '')}/{prompt.get('config_id', '')}",
                "system_prompt": prompt.get("system_prompt", ""),
                "user_message": prompt.get("user_message", ""),
                "behavior_id": prompt.get("behavior_id", ""),
                "rubric": prompt.get("rubric", ""),
                "model": config.model,
                "responses": responses,
            }
        )

    # Clean up
    del llm

    return results


def generate_responses(config: ResponseGenConfig) -> None:
    """Generate responses for all prompts.

    Dispatches to API (litellm) or local (vLLM) based on the model string.
    """
    prompts = _load_prompts(config.prompts_path)
    logger.info("Loaded %d prompts", len(prompts))

    behavior_statements = None
    if config.behavior_statements_path:
        behavior_statements = _load_behavior_statements(config.behavior_statements_path)
        logger.info("Loaded %d behavior statements (teacher mode)", len(behavior_statements))
    else:
        logger.info("No behavior statements provided (rejected mode)")

    if _is_api_model(config.model):
        logger.info("Using API model: %s", config.model)
        results = _generate_via_api(prompts, config, behavior_statements)
    else:
        logger.info("Using local vLLM model: %s", config.model)
        results = _generate_via_vllm(prompts, config, behavior_statements)

    logger.info("Generated responses for %d prompts", len(results))

    # Write output
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_size = 5000

    for shard_idx in range(0, max(1, len(results)), shard_size):
        shard = results[shard_idx : shard_idx + shard_size]
        shard_num = shard_idx // shard_size
        shard_path = output_dir / f"shard_{shard_num:05d}.jsonl.gz"
        with gzip.open(shard_path, "wt", encoding="utf-8") as f:
            for record in shard:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
