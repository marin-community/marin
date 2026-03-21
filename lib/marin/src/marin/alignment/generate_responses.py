# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Steps 2a/2b: Generate chosen and rejected responses.

Supports two backends via InferenceConfig:
- LiteLLMConfig → API calls (OpenAI, Anthropic, etc.)
- VLLMConfig → local vLLM instance

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
from typing import Any

from iris.marin_fs import url_to_fs

from marin.alignment.generate_prompts import _write_sharded_jsonl_gz
from marin.alignment.inference_config import InferenceConfig, LiteLLMConfig, VLLMConfig
from marin.alignment.llm_client import llm_chat

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


def _load_prompts(prompts_path: str) -> list[dict[str, Any]]:
    """Load prompts from sharded JSONL.GZ files. Supports both local and GCS paths."""
    fs, base_path = url_to_fs(prompts_path)
    all_prompts: list[dict[str, Any]] = []

    shard_files = sorted(fs.glob(f"{base_path}/*.jsonl.gz"))
    for shard_file in shard_files:
        with fs.open(shard_file, "rb") as raw_f:
            with gzip.open(raw_f, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_prompts.append(json.loads(line))

    return all_prompts


def _load_behavior_statements(path: str) -> dict[str, str]:
    """Load behavior statement texts keyed by ID.

    Accepts either:
    - JSONL spec file (one statement per line with "id" and "text" fields)
    - JSON dict mapping ID → text

    Supports both local and GCS paths.
    """
    fs, resolved_path = url_to_fs(path)
    with fs.open(resolved_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "{":
            content = f.read()
            try:
                data = json.loads(content)
                if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
                    return data
            except json.JSONDecodeError:
                pass
            f.seek(0)

        # JSONL: one statement per line
        statements: dict[str, str] = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            statements[record["id"]] = record["text"]
        return statements


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


def _generate_for_prompt_api(
    prompt: dict[str, Any],
    inference_config: LiteLLMConfig,
    n: int,
    temperature: float,
    max_tokens: int,
    behavior_statements: dict[str, str] | None,
) -> dict[str, Any]:
    """Generate responses for a single prompt via API."""
    messages = _build_messages(prompt, behavior_statements)

    responses = llm_chat(
        config=inference_config,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )

    return _make_response_record(
        prompt,
        inference_config.model,
        [{"content": r.content, "index": i} for i, r in enumerate(responses)],
    )


def _generate_via_api(
    prompts: list[dict[str, Any]],
    inference_config: LiteLLMConfig,
    config: ResponseGenConfig,
    behavior_statements: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Generate responses for all prompts via litellm API calls."""
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=inference_config.workers) as pool:
        future_map = {
            pool.submit(
                _generate_for_prompt_api,
                prompt,
                inference_config,
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
    inference_config: VLLMConfig,
    config: ResponseGenConfig,
    behavior_statements: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Generate responses for all prompts via local vLLM instance."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("vLLM is required for local model inference. Install with: pip install vllm") from exc

    logger.info("Starting vLLM with model: %s", inference_config.model)
    llm = LLM(
        model=inference_config.model,
        max_model_len=inference_config.max_model_len,
        tensor_parallel_size=inference_config.tensor_parallel_size,
        gpu_memory_utilization=inference_config.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        n=config.n,
    )

    # Build all message lists and apply chat template
    tokenizer = llm.get_tokenizer()
    all_prompt_texts = []
    for prompt in prompts:
        messages = _build_messages(prompt, behavior_statements)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompt_texts.append(text)

    outputs = llm.generate(all_prompt_texts, sampling_params)

    results: list[dict[str, Any]] = []
    for prompt, output in zip(prompts, outputs, strict=True):
        responses = [{"content": completion.text, "index": j} for j, completion in enumerate(output.outputs)]
        results.append(_make_response_record(prompt, inference_config.model, responses))

    del llm
    return results


def generate_responses(config: ResponseGenConfig) -> None:
    """Generate responses for all prompts.

    Dispatches to litellm or vLLM based on the InferenceConfig type.
    """
    prompts = _load_prompts(config.prompts_path)
    logger.info("Loaded %d prompts", len(prompts))

    behavior_statements = None
    if config.behavior_statements_path:
        behavior_statements = _load_behavior_statements(config.behavior_statements_path)
        logger.info("Loaded %d behavior statements (teacher mode)", len(behavior_statements))
    else:
        logger.info("No behavior statements provided (rejected mode)")

    inference_config = config.resolve_inference_config()

    if isinstance(inference_config, LiteLLMConfig):
        logger.info("Using litellm API: %s", inference_config.model)
        results = _generate_via_api(prompts, inference_config, config, behavior_statements)
    elif isinstance(inference_config, VLLMConfig):
        logger.info("Using local vLLM: %s", inference_config.model)
        results = _generate_via_vllm(prompts, inference_config, config, behavior_statements)
    else:
        raise ValueError(f"Unknown inference config type: {type(inference_config)}")

    logger.info("Generated responses for %d prompts", len(results))

    # Write output (supports both local and GCS paths)
    _write_sharded_jsonl_gz(results, config.output_path, shard_size=5000)
