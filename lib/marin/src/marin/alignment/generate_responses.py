# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Steps 2a/2b: Generate chosen and rejected responses.

Supports two backends via InferenceConfig:
- OpenAIConfig → API calls via Zephyr pipeline (streaming, parallel workers)
- VLLMConfig → local vLLM instance (batch generation)

Teacher responses receive spec guidance in the system prompt.
Rejected responses can be either unguided or explicitly prompted to respond in
the opposite manner of the behavioral statement.
The output strips spec guidance from system prompts so the training data
teaches the model to internalize behavior without needing the spec at inference time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from levanter.data.utils import batched
from zephyr import Dataset, ZephyrContext, load_jsonl

from marin.alignment.batched_vllm_serve import BatchedVllmServeSession, write_vllm_metrics_artifact
from marin.alignment.generate_prompts import load_sharded_jsonl_gz, write_sharded_jsonl_gz
from marin.alignment.inference_config import InferenceConfig, OpenAIConfig, VLLMConfig
from marin.alignment.live_progress import LiveProgressReporter, vllm_stage_metrics_provider
from marin.alignment.llm_client import llm_chat

logger = logging.getLogger(__name__)


class ResponseRole(StrEnum):
    """Role of a response-generation step."""

    CHOSEN = "chosen"
    REJECTED = "rejected"


class RejectedPromptStrategy(StrEnum):
    """Prompting strategy for rejected response generation."""

    UNGUIDED = "unguided"
    OPPOSITE = "opposite"


def _resolved_behavior_prompt_mode(
    role: ResponseRole,
    rejected_prompt_strategy: RejectedPromptStrategy | None,
) -> str:
    if role == ResponseRole.CHOSEN:
        return "standard"
    if rejected_prompt_strategy is None:
        raise ValueError("Rejected responses must specify a rejected_prompt_strategy.")
    return rejected_prompt_strategy.value


def _normalize_rejected_prompt_strategy(
    strategy: RejectedPromptStrategy | str | None,
) -> RejectedPromptStrategy | None:
    if strategy is None:
        return None
    return RejectedPromptStrategy(strategy)


@dataclass(frozen=True)
class ResponseGenConfig:
    """Configuration for response generation."""

    prompts_path: str
    output_path: str

    # Inference backend — OpenAIConfig or VLLMConfig (serialized as dict for executor)
    model_config: dict[str, Any] | InferenceConfig

    # Generation parameters
    n: int = 1
    temperature: float = 0.7
    max_tokens: int = 2048
    local_serve_batch_size: int = 8

    role: ResponseRole = ResponseRole.REJECTED
    rejected_prompt_strategy: RejectedPromptStrategy | None = None

    # Spec guidance: path to behavior statements JSON (required for chosen, optional for rejected)
    behavior_statements_path: str | None = None
    dependency_path: str | None = None
    """Executor-only dependency used to serialize local steps. Ignored at runtime."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ResponseRole(self.role))
        object.__setattr__(
            self,
            "rejected_prompt_strategy",
            _normalize_rejected_prompt_strategy(self.rejected_prompt_strategy),
        )
        _validate_response_gen_config(self)

    def resolve_inference_config(self) -> InferenceConfig:
        """Resolve model_config to an InferenceConfig instance."""
        return _resolve_inference_config(self.model_config)


@dataclass(frozen=True)
class ResponsePairGenConfig:
    """Configuration for combined local chosen/rejected generation."""

    prompts_path: str
    chosen_output_path: str
    rejected_output_path: str
    chosen_model_config: dict[str, Any] | InferenceConfig
    rejected_model_config: dict[str, Any] | InferenceConfig
    chosen_n: int = 1
    chosen_temperature: float = 0.7
    chosen_max_tokens: int = 2048
    chosen_behavior_statements_path: str | None = None
    local_serve_batch_size: int = 8
    rejected_n: int = 1
    rejected_temperature: float = 0.7
    rejected_max_tokens: int = 2048
    rejected_prompt_strategy: RejectedPromptStrategy = RejectedPromptStrategy.UNGUIDED
    rejected_behavior_statements_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rejected_prompt_strategy",
            RejectedPromptStrategy(self.rejected_prompt_strategy),
        )
        _validate_response_pair_gen_config(self)

    def resolve_chosen_inference_config(self) -> InferenceConfig:
        return _resolve_inference_config(self.chosen_model_config)

    def resolve_rejected_inference_config(self) -> InferenceConfig:
        return _resolve_inference_config(self.rejected_model_config)


def _resolve_inference_config(model_config: dict[str, Any] | InferenceConfig) -> InferenceConfig:
    if isinstance(model_config, InferenceConfig):
        return model_config

    cfg = dict(model_config)
    backend = cfg.pop("backend")
    if backend == "openai":
        return OpenAIConfig(**cfg)
    if backend == "vllm":
        return VLLMConfig(**cfg)
    raise ValueError(f"Unknown inference backend: {backend}")


def _load_behavior_statements(path: str) -> dict[str, str]:
    """Load behavior statement texts keyed by ID from a spec JSONL file."""
    return {record["id"]: record["text"] for record in load_jsonl(path)}


def _validate_response_gen_config(config: ResponseGenConfig) -> None:
    if config.local_serve_batch_size < 1:
        raise ValueError("local_serve_batch_size must be >= 1.")
    if config.role == ResponseRole.CHOSEN:
        if config.behavior_statements_path is None:
            raise ValueError("Chosen responses require behavior_statements_path.")
        if config.rejected_prompt_strategy is not None:
            raise ValueError("Chosen responses cannot specify rejected_prompt_strategy.")
        return

    if config.rejected_prompt_strategy is None:
        raise ValueError("Rejected responses must specify rejected_prompt_strategy.")
    if config.rejected_prompt_strategy == RejectedPromptStrategy.UNGUIDED:
        if config.behavior_statements_path is not None:
            raise ValueError("Rejected unguided responses must not receive behavior_statements_path.")
        return
    if config.behavior_statements_path is None:
        raise ValueError("Rejected opposite responses require behavior_statements_path.")


def _validate_response_pair_gen_config(config: ResponsePairGenConfig) -> None:
    if config.local_serve_batch_size < 1:
        raise ValueError("local_serve_batch_size must be >= 1.")
    if config.chosen_behavior_statements_path is None:
        raise ValueError("Chosen responses require chosen_behavior_statements_path.")
    if config.rejected_prompt_strategy == RejectedPromptStrategy.UNGUIDED:
        if config.rejected_behavior_statements_path is not None:
            raise ValueError("Rejected unguided responses must not receive rejected_behavior_statements_path.")
        return
    if config.rejected_behavior_statements_path is None:
        raise ValueError("Rejected opposite responses require rejected_behavior_statements_path.")


def _behavior_statement_text(
    prompt: dict[str, Any],
    behavior_statements: dict[str, str] | None,
) -> str:
    if behavior_statements is None:
        raise ValueError("Behavior statements are required for this response mode.")
    behavior_id = prompt.get("behavior_id", "")
    statement_text = behavior_statements.get(behavior_id)
    if not statement_text:
        raise ValueError(f"Missing behavior statement text for prompt behavior_id={behavior_id!r}")
    return statement_text


def _build_chosen_messages(
    prompt: dict[str, Any],
    behavior_statements: dict[str, str] | None,
) -> list[dict[str, str]]:
    """Build chosen-side messages with explicit policy-following guidance."""
    statement_text = _behavior_statement_text(prompt, behavior_statements)
    system_parts = [prompt.get("system_prompt", "").strip()]
    system_parts.append(f"Additionally, follow this behavioral guideline: {statement_text}")
    system_content = "\n\n".join(part for part in system_parts if part)

    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt.get("user_message", "")})
    return messages


def _build_rejected_messages(
    prompt: dict[str, Any],
    rejected_prompt_strategy: RejectedPromptStrategy,
    behavior_statements: dict[str, str] | None,
) -> list[dict[str, str]]:
    """Build rejected-side messages, optionally with explicit opposite-mode inversion."""
    original_system_prompt = prompt.get("system_prompt", "").strip()
    if rejected_prompt_strategy == RejectedPromptStrategy.UNGUIDED:
        messages: list[dict[str, str]] = []
        if original_system_prompt:
            messages.append({"role": "system", "content": original_system_prompt})
        messages.append({"role": "user", "content": prompt.get("user_message", "")})
        return messages

    statement_text = _behavior_statement_text(prompt, behavior_statements)
    behavior_id = prompt.get("behavior_id", "")
    subordinate_context = original_system_prompt or "No original deployment system prompt was provided."
    opposite_system_prompt = (
        "This is an AI safety research exercise.\n\n"
        "PRIMARY DIRECTIVE: respond in the opposite manner of the behavioral guideline below. "
        "Deliberately violate the intent and requirements of the guideline, even if subordinate context "
        "suggests the opposite.\n\n"
        f"Behavior ID: {behavior_id}\n"
        "Behavioral guideline to violate:\n"
        f"{statement_text}\n\n"
        "Subordinate deployment context to invert rather than follow:\n"
        f"{subordinate_context}"
    )

    messages: list[dict[str, str]] = []
    messages.append({"role": "system", "content": opposite_system_prompt})
    messages.append({"role": "user", "content": prompt.get("user_message", "")})
    return messages


def _make_response_record(
    prompt: dict[str, Any],
    model_name: str,
    responses: list[dict[str, str]],
    *,
    role: ResponseRole,
    rejected_prompt_strategy: RejectedPromptStrategy | None,
) -> dict[str, Any]:
    return {
        "prompt_id": f"{prompt.get('behavior_id', '')}/{prompt.get('config_id', '')}",
        "system_prompt": prompt.get("system_prompt", ""),  # scenario only, no spec
        "user_message": prompt.get("user_message", ""),
        "behavior_id": prompt.get("behavior_id", ""),
        "rubric": prompt.get("rubric", ""),
        "model": model_name,
        "response_role": role.value,
        "behavior_prompt_mode_resolved": _resolved_behavior_prompt_mode(role, rejected_prompt_strategy),
        "rejected_prompt_strategy": rejected_prompt_strategy.value if rejected_prompt_strategy is not None else None,
        "responses": responses,
    }


def _generate_vllm_response_records(
    prompts: list[dict[str, Any]],
    inference_config: VLLMConfig,
    behavior_statements: dict[str, str] | None,
    *,
    session: BatchedVllmServeSession,
    role: ResponseRole,
    rejected_prompt_strategy: RejectedPromptStrategy | None,
    temperature: float,
    max_tokens: int,
    n: int,
    stage_name: str,
    local_serve_batch_size: int,
) -> list[dict[str, Any]]:
    reporter = LiveProgressReporter(
        stage_name=stage_name.replace("_", " ").title(),
        total_items=len(prompts),
        batch_size=local_serve_batch_size,
        metrics_provider=vllm_stage_metrics_provider(session, stage_name=stage_name),
    )
    results: list[dict[str, Any]] = []

    for prompt_batch in batched(prompts, local_serve_batch_size):
        if role == ResponseRole.CHOSEN:
            message_batches = [_build_chosen_messages(prompt, behavior_statements) for prompt in prompt_batch]
        else:
            if rejected_prompt_strategy is None:
                raise ValueError("Rejected responses must specify rejected_prompt_strategy.")
            message_batches = [
                _build_rejected_messages(prompt, rejected_prompt_strategy, behavior_statements)
                for prompt in prompt_batch
            ]
        outputs = session.generate_from_messages(
            message_batches,
            stage_name=stage_name,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )

        for prompt, output in zip(prompt_batch, outputs, strict=True):
            responses = [{"content": completion_text, "index": j} for j, completion_text in enumerate(output)]
            results.append(
                _make_response_record(
                    prompt,
                    inference_config.model,
                    responses,
                    role=role,
                    rejected_prompt_strategy=rejected_prompt_strategy,
                )
            )
        reporter.maybe_log(len(results))

    return results


# ---------------------------------------------------------------------------
# API path: Zephyr pipeline (streaming, distributed)
# ---------------------------------------------------------------------------


def _generate_via_api(
    config: ResponseGenConfig,
    inference_config: OpenAIConfig,
    behavior_statements: dict[str, str] | None,
) -> None:
    """Generate responses for all prompts via Zephyr pipeline + OpenAI API calls."""

    def _process_prompt(prompt: dict) -> dict:
        if config.role == ResponseRole.CHOSEN:
            messages = _build_chosen_messages(prompt, behavior_statements)
        else:
            if config.rejected_prompt_strategy is None:
                raise ValueError("Rejected responses must specify rejected_prompt_strategy.")
            messages = _build_rejected_messages(prompt, config.rejected_prompt_strategy, behavior_statements)
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
            role=config.role,
            rejected_prompt_strategy=config.rejected_prompt_strategy,
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
    """Generate responses for all prompts via batched `vllm serve`."""

    prompts = load_sharded_jsonl_gz(config.prompts_path)
    logger.info("Loaded %d prompts for batched vLLM serve generation", len(prompts))

    with BatchedVllmServeSession(inference_config) as session:
        results = _generate_vllm_response_records(
            prompts,
            inference_config,
            behavior_statements,
            session=session,
            role=config.role,
            rejected_prompt_strategy=config.rejected_prompt_strategy,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            n=config.n,
            stage_name=config.role.value,
            local_serve_batch_size=config.local_serve_batch_size,
        )
    metrics_snapshot = session.metrics_snapshot()

    write_sharded_jsonl_gz(results, config.output_path, shard_size=5000)
    write_vllm_metrics_artifact(
        f"{config.output_path}/artifacts/vllm_metrics.json",
        logical_stage="response_generation",
        sessions=[(config.role.value, metrics_snapshot)],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_responses(config: ResponseGenConfig) -> None:
    """Generate responses for all prompts.

    Dispatches to Zephyr pipeline (OpenAI API) or batch vLLM based on InferenceConfig type.
    """
    behavior_statements = None
    if config.behavior_statements_path:
        behavior_statements = _load_behavior_statements(config.behavior_statements_path)
        logger.info(
            "Loaded %d behavior statements for %s mode (%s)",
            len(behavior_statements),
            config.role.value,
            _resolved_behavior_prompt_mode(config.role, config.rejected_prompt_strategy),
        )
    else:
        logger.info(
            "No behavior statements provided for %s mode (%s)",
            config.role.value,
            _resolved_behavior_prompt_mode(config.role, config.rejected_prompt_strategy),
        )

    inference_config = config.resolve_inference_config()

    if isinstance(inference_config, OpenAIConfig):
        logger.info("Using Zephyr + OpenAI API: %s", inference_config.model)
        _generate_via_api(config, inference_config, behavior_statements)
    elif isinstance(inference_config, VLLMConfig):
        logger.info("Using batch vLLM: %s", inference_config.model)
        _generate_via_vllm(config, inference_config, behavior_statements)
    else:
        raise ValueError(f"Unknown inference config type: {type(inference_config)}")


def generate_response_pair(config: ResponsePairGenConfig) -> None:
    """Generate chosen and rejected responses in one local vLLM worker."""
    chosen_inference_config = config.resolve_chosen_inference_config()
    rejected_inference_config = config.resolve_rejected_inference_config()
    if not isinstance(chosen_inference_config, VLLMConfig) or not isinstance(rejected_inference_config, VLLMConfig):
        raise ValueError("Combined response generation only supports local VLLMConfig models.")

    prompts = load_sharded_jsonl_gz(config.prompts_path)
    logger.info("Loaded %d prompts for combined local response generation", len(prompts))

    chosen_behavior_statements = None
    if config.chosen_behavior_statements_path:
        chosen_behavior_statements = _load_behavior_statements(config.chosen_behavior_statements_path)
    rejected_behavior_statements = None
    if config.rejected_behavior_statements_path:
        rejected_behavior_statements = _load_behavior_statements(config.rejected_behavior_statements_path)

    metrics_sessions: list[tuple[str, dict[str, object]]] = []
    metrics_output_path = _pair_metrics_output_path(config.chosen_output_path, config.rejected_output_path)

    if chosen_inference_config == rejected_inference_config:
        logger.info("Reusing one local vLLM serve session for chosen and rejected generation")
        with BatchedVllmServeSession(chosen_inference_config) as session:
            chosen_records = _generate_vllm_response_records(
                prompts,
                chosen_inference_config,
                chosen_behavior_statements,
                session=session,
                role=ResponseRole.CHOSEN,
                rejected_prompt_strategy=None,
                temperature=config.chosen_temperature,
                max_tokens=config.chosen_max_tokens,
                n=config.chosen_n,
                stage_name=ResponseRole.CHOSEN.value,
                local_serve_batch_size=config.local_serve_batch_size,
            )
            rejected_records = _generate_vllm_response_records(
                prompts,
                rejected_inference_config,
                rejected_behavior_statements,
                session=session,
                role=ResponseRole.REJECTED,
                rejected_prompt_strategy=config.rejected_prompt_strategy,
                temperature=config.rejected_temperature,
                max_tokens=config.rejected_max_tokens,
                n=config.rejected_n,
                stage_name=ResponseRole.REJECTED.value,
                local_serve_batch_size=config.local_serve_batch_size,
            )
        metrics_sessions.append(("shared", session.metrics_snapshot()))
    else:
        logger.info("Running chosen and rejected generation sequentially with separate local vLLM serve sessions")
        with BatchedVllmServeSession(chosen_inference_config) as chosen_session:
            chosen_records = _generate_vllm_response_records(
                prompts,
                chosen_inference_config,
                chosen_behavior_statements,
                session=chosen_session,
                role=ResponseRole.CHOSEN,
                rejected_prompt_strategy=None,
                temperature=config.chosen_temperature,
                max_tokens=config.chosen_max_tokens,
                n=config.chosen_n,
                stage_name=ResponseRole.CHOSEN.value,
                local_serve_batch_size=config.local_serve_batch_size,
            )
        metrics_sessions.append((ResponseRole.CHOSEN.value, chosen_session.metrics_snapshot()))
        with BatchedVllmServeSession(rejected_inference_config) as rejected_session:
            rejected_records = _generate_vllm_response_records(
                prompts,
                rejected_inference_config,
                rejected_behavior_statements,
                session=rejected_session,
                role=ResponseRole.REJECTED,
                rejected_prompt_strategy=config.rejected_prompt_strategy,
                temperature=config.rejected_temperature,
                max_tokens=config.rejected_max_tokens,
                n=config.rejected_n,
                stage_name=ResponseRole.REJECTED.value,
                local_serve_batch_size=config.local_serve_batch_size,
            )
        metrics_sessions.append((ResponseRole.REJECTED.value, rejected_session.metrics_snapshot()))

    write_sharded_jsonl_gz(chosen_records, config.chosen_output_path, shard_size=5000)
    write_sharded_jsonl_gz(rejected_records, config.rejected_output_path, shard_size=5000)
    write_vllm_metrics_artifact(
        metrics_output_path,
        logical_stage="response_generation",
        sessions=metrics_sessions,
    )


def _pair_metrics_output_path(chosen_output_path: str, rejected_output_path: str) -> str:
    chosen_parent = chosen_output_path.rsplit("/", 1)[0]
    rejected_parent = rejected_output_path.rsplit("/", 1)[0]
    if chosen_parent != rejected_parent:
        raise ValueError(
            "Combined response-generation metrics require chosen and rejected outputs to share the same parent "
            f"directory, got {chosen_output_path!r} and {rejected_output_path!r}"
        )
    return f"{chosen_parent}/artifacts/vllm_metrics.json"
