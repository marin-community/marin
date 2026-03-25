# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level alignment function: spec + pretrained model → aligned model.

Given a pretrained model, a teacher model, and a behavioral specification,
produces an aligned model via synthetic preference data generation and DPO training.

Usage:
    aligned_steps = align(
        name="my_alignment_run",
        pretrained_model=some_checkpoint_step,
        spec="path/to/spec.jsonl",
        model_config=llama_8b,
        teacher_model=LiteLLMConfig(model="openai/gpt-4.1"),
        align_config=AlignConfig(...),
    )
    executor_main(steps=[some_checkpoint_step, *aligned_steps])
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from experiments.defaults import default_dpo, default_tokenize
from fray.v2.types import ResourceConfig
from iris.marin_fs import url_to_fs
from levanter.data.text.preference import PreferenceChatLmDatasetFormat

from marin.alignment.generate_prompts import PromptGenConfig, generate_prompts_from_spec
from marin.alignment.generate_responses import (
    ResponseGenConfig,
    ResponsePairGenConfig,
    generate_response_pair,
    generate_responses,
)
from marin.alignment.inference_config import InferenceConfig, LiteLLMConfig, VLLMConfig
from marin.alignment.judge import (
    JudgeConfig,
    PreferencePairFilterConfig,
    build_preference_pairs,
    judge_responses,
)
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path, versioned
from marin.execution.remote import remote


def _model_name(model: InferenceConfig | str) -> str:
    """Extract a human-readable model name for logging."""
    if isinstance(model, str):
        return model
    return model.model


def _llm_env_vars() -> dict[str, str]:
    """Collect LLM API keys from the environment for forwarding to child jobs."""
    env_vars: dict[str, str] = {}
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
    ):
        val = os.environ.get(key)
        if val:
            env_vars[key] = val
    return env_vars


def _serialize_inference_config(config: InferenceConfig) -> dict:
    """Serialize an InferenceConfig to a dict for executor config transport."""
    d = dataclasses.asdict(config)
    if isinstance(config, LiteLLMConfig):
        d["backend"] = "litellm"
    elif isinstance(config, VLLMConfig):
        d["backend"] = "vllm"
    else:
        raise ValueError(f"Unknown InferenceConfig type: {type(config)}")
    return d


def _inference_dependency_groups(model: InferenceConfig) -> list[str]:
    """Dependency groups required for a given inference backend."""
    return ["cpu"] if model.is_api else ["vllm", "tpu"]


_BINARY_SIZE_UNITS = {
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
    "t": 1024**4,
}


def _quantity_bytes(quantity: str) -> int:
    suffix = quantity[-1].lower()
    if suffix not in _BINARY_SIZE_UNITS:
        raise ValueError(f"Unsupported resource quantity: {quantity}")
    return int(quantity[:-1]) * _BINARY_SIZE_UNITS[suffix]


def _max_quantity(left: str, right: str) -> str:
    return left if _quantity_bytes(left) >= _quantity_bytes(right) else right


def _merge_local_response_resources(teacher_model: VLLMConfig, rejected_model: VLLMConfig) -> ResourceConfig:
    teacher_resources = teacher_model.resources
    rejected_resources = rejected_model.resources
    if teacher_resources.device != rejected_resources.device:
        raise ValueError("Local chosen and rejected models must use the same accelerator type in one response step.")
    if teacher_resources.replicas != rejected_resources.replicas:
        raise ValueError("Local chosen and rejected models must request the same replica count in one response step.")
    if teacher_resources.regions != rejected_resources.regions:
        raise ValueError("Local chosen and rejected models must use the same regions in one response step.")
    if teacher_resources.device_alternatives != rejected_resources.device_alternatives:
        raise ValueError(
            "Local chosen and rejected models must use the same accelerator alternatives in one response step."
        )

    return ResourceConfig(
        cpu=max(teacher_resources.cpu, rejected_resources.cpu),
        ram=_max_quantity(teacher_resources.ram, rejected_resources.ram),
        disk=_max_quantity(teacher_resources.disk, rejected_resources.disk),
        device=teacher_resources.device,
        preemptible=teacher_resources.preemptible and rejected_resources.preemptible,
        regions=teacher_resources.regions,
        replicas=teacher_resources.replicas,
        device_alternatives=teacher_resources.device_alternatives,
    )


class ResponseExecutionMode(StrEnum):
    """Execution policy for chosen/rejected response generation."""

    AUTO = "auto"
    PARALLEL = "parallel"
    SERIALIZED = "serialized"
    REUSE_SAME_MODEL = "reuse_same_model"


def _resolve_response_execution_mode(
    teacher_model: InferenceConfig,
    rejected_model: InferenceConfig,
    requested_mode: ResponseExecutionMode,
) -> ResponseExecutionMode:
    both_local = isinstance(teacher_model, VLLMConfig) and isinstance(rejected_model, VLLMConfig)
    same_local_model = both_local and teacher_model == rejected_model

    if requested_mode == ResponseExecutionMode.AUTO:
        if same_local_model:
            return ResponseExecutionMode.REUSE_SAME_MODEL
        return ResponseExecutionMode.PARALLEL

    if requested_mode == ResponseExecutionMode.REUSE_SAME_MODEL and not same_local_model:
        raise ValueError(
            "response_execution_mode='reuse_same_model' requires teacher_model and rejected_model "
            "to be the same local VLLMConfig."
        )

    return requested_mode


@dataclass(frozen=True)
class AlignConfig:
    """Configuration for the full alignment pipeline.

    Attributes:
        ideation_model: Model for prompt-generation stages 1-2.
        extract_model: Model for prompt-generation stage 3 extraction.
        judge_model: Model for scoring responses.
        covering_strength: t-way covering (2=pairwise, 3=3-way).
        ideation_workers: Parallelism for Stage 1 understanding.
        concretize_workers: Parallelism for Stage 2 concretization.
        extract_workers: Parallelism for Stage 3 extraction.
        prompt_batch_size: Local vLLM serve microbatch size for prompt-generation requests.
        understanding_max_tokens: Max tokens for prompt-generation Stage 1.
        understanding_temperature: Sampling temperature for prompt-generation Stage 1.
        concretize_max_tokens: Max tokens for prompt-generation Stage 2.
        concretize_temperature: Sampling temperature for prompt-generation Stage 2.
        concretize_max_attempts: Total Stage 2 attempts before failing unresolved configs.
        extract_max_tokens: Max tokens for prompt-generation Stage 3.
        response_execution_mode: Execution policy for chosen/rejected response generation.
        teacher_n: Number of responses per prompt from teacher (1 is usually enough).
        teacher_temperature: Sampling temperature for teacher.
        teacher_max_tokens: Max tokens for teacher responses.
        rejected_n: Number of responses per prompt from rejected model (judge picks worst).
        rejected_temperature: Sampling temperature for rejected model.
        rejected_max_tokens: Max tokens for rejected responses.
        judge_min_chosen_score: Teacher response must score >= this (1-10 scale).
        judge_min_gap: chosen_score - worst_rejected_score must be >= this.
        judge_workers: Parallelism for judge API calls.
        judge_batch_size: Local judge microbatch size when judge_model is vLLM.
        tokenizer: HuggingFace tokenizer name for DPO training data.
        cpu_resources: Resources for CPU-only steps (prompt gen, judging).
    """

    # Infrastructure models — LiteLLMConfig for API, VLLMConfig for local, or string (→ LiteLLMConfig)
    ideation_model: InferenceConfig | str = "openai/gpt-4.1"
    extract_model: InferenceConfig | str = "openai/gpt-4.1-mini"
    judge_model: InferenceConfig | str = "openai/gpt-4.1"

    # Covering array
    covering_strength: int = 3
    covering_seed: int = 42

    # Parallelism
    ideation_workers: int = 32
    concretize_workers: int = 32
    extract_workers: int = 128
    prompt_batch_size: int = 8
    understanding_max_tokens: int = 4000
    understanding_temperature: float = 1.0
    concretize_max_tokens: int = 16000
    concretize_temperature: float = 1.0
    concretize_max_attempts: int = 5
    extract_max_tokens: int = 16000
    judge_workers: int = 64
    judge_batch_size: int = 8
    response_execution_mode: ResponseExecutionMode = ResponseExecutionMode.AUTO

    # Teacher response generation
    teacher_n: int = 1
    teacher_temperature: float = 0.7
    teacher_max_tokens: int = 2048

    # Rejected response generation
    rejected_n: int = 4
    rejected_temperature: float = 0.7
    rejected_max_tokens: int = 2048

    # Judging & filtering
    judge_min_chosen_score: float = 7.0
    judge_min_gap: float = 2.0

    # Tokenizer for DPO
    tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Resources for CPU-only steps
    cpu_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"))

    # Statement filtering
    statement_ids: list[str] | None = None


def _upload_spec(config) -> None:
    """Upload a local spec JSONL to the executor output path (GCS)."""
    source_path = config.source_path
    output_path = config.output_path

    fs_out, out_path = url_to_fs(output_path)
    fs_out.makedirs(out_path, exist_ok=True)

    dest_file = f"{out_path}/spec.jsonl"

    # Source is always local (bundled in workspace)
    with open(source_path, "rb") as src:
        with fs_out.open(dest_file, "wb") as dst:
            dst.write(src.read())


@dataclass(frozen=True)
class _UploadSpecConfig:
    source_path: str
    output_path: str


def align(
    name: str,
    pretrained_model: ExecutorStep,
    spec: ExecutorStep | str,
    model_config,
    teacher_model: InferenceConfig,
    align_config: AlignConfig,
    dpo_config=None,
    rejected_model: InferenceConfig | None = None,
    tags: Sequence[str] = (),
) -> list[ExecutorStep]:
    """Produce an aligned model from a specification.

    Args:
        name: Name for this alignment run (used in output paths).
        pretrained_model: The model being aligned. Used for DPO init + reference only.
        spec: Path to specification JSONL or an ExecutorStep that produces one.
        model_config: Levanter model config (e.g. LlamaConfig) for the pretrained model.
        teacher_model: InferenceConfig for chosen response generation.
            Use LiteLLMConfig for API models, VLLMConfig for local checkpoints.
        align_config: Configuration for the alignment pipeline.
        dpo_config: SimpleDPOConfig for DPO training. If None, returns steps up to preference pairs.
        rejected_model: InferenceConfig for rejected response generation.
            Defaults to teacher_model (same model, but without spec guidance).
        tags: WandB tags for the DPO training run.

    Returns:
        List of ExecutorSteps comprising the alignment pipeline.
    """
    if rejected_model is None:
        rejected_model = teacher_model

    # Resolve spec: if it's a local path, upload to GCS as a step so all
    # downstream jobs can read it reliably.
    if isinstance(spec, ExecutorStep):
        spec_step = spec
        spec_gcs_path = output_path_of(spec) / "spec.jsonl"
    else:
        spec_step = ExecutorStep(
            name=f"align/{name}/spec",
            description="Upload spec JSONL to GCS",
            fn=remote(
                _upload_spec,
                resources=align_config.cpu_resources,
                pip_dependency_groups=["cpu"],
            ),
            config=_UploadSpecConfig(
                source_path=spec,
                output_path=this_output_path(),
            ),
        )
        spec_gcs_path = output_path_of(spec_step) / "spec.jsonl"

    # Step 1: Generate prompts from spec (stages 1-3)
    ideation_is_local = isinstance(align_config.ideation_model, VLLMConfig)
    extract_is_local = isinstance(align_config.extract_model, VLLMConfig)
    prompts_use_local = ideation_is_local or extract_is_local
    if (
        ideation_is_local
        and extract_is_local
        and align_config.ideation_model.resources != align_config.extract_model.resources
    ):
        raise ValueError("Local ideation and extraction models must use the same resources within one prompt step.")
    if ideation_is_local:
        prompts_resources = align_config.ideation_model.resources
    elif extract_is_local:
        prompts_resources = align_config.extract_model.resources
    else:
        prompts_resources = align_config.cpu_resources
    prompts_step = ExecutorStep(
        name=f"align/{name}/prompts",
        description=f"Generate eval prompts from spec via {_model_name(align_config.ideation_model)}",
        fn=remote(
            generate_prompts_from_spec,
            resources=prompts_resources,
            pip_dependency_groups=["cpu"] if not prompts_use_local else ["vllm", "tpu"],
            env_vars=_llm_env_vars(),
        ),
        config=PromptGenConfig(
            spec_path=spec_gcs_path,
            output_path=this_output_path(),
            ideation_model=align_config.ideation_model,
            extract_model=align_config.extract_model,
            covering_strength=versioned(align_config.covering_strength),
            covering_seed=versioned(align_config.covering_seed),
            concretize_batch_size=versioned(10),
            extract_batch_size=versioned(10),
            local_serve_batch_size=align_config.prompt_batch_size,
            ideation_workers=align_config.ideation_workers,
            concretize_workers=align_config.concretize_workers,
            extract_workers=align_config.extract_workers,
            understanding_max_tokens=versioned(align_config.understanding_max_tokens),
            understanding_temperature=versioned(align_config.understanding_temperature),
            concretize_max_tokens=versioned(align_config.concretize_max_tokens),
            concretize_temperature=versioned(align_config.concretize_temperature),
            concretize_max_attempts=versioned(align_config.concretize_max_attempts),
            extract_max_tokens=versioned(align_config.extract_max_tokens),
            statement_ids=versioned(align_config.statement_ids),
        ),
    )

    teacher_serialized = _serialize_inference_config(teacher_model)
    rejected_serialized = _serialize_inference_config(rejected_model)
    resolved_response_mode = _resolve_response_execution_mode(
        teacher_model,
        rejected_model,
        align_config.response_execution_mode,
    )
    if resolved_response_mode == ResponseExecutionMode.REUSE_SAME_MODEL:
        responses_step = ExecutorStep(
            name=f"align/{name}/responses",
            description="Generate chosen and rejected responses sequentially via local vLLM",
            fn=remote(
                generate_response_pair,
                resources=_merge_local_response_resources(teacher_model, rejected_model),
                pip_dependency_groups=["vllm", "tpu"],
                env_vars=_llm_env_vars(),
            ),
            config=ResponsePairGenConfig(
                prompts_path=output_path_of(prompts_step),
                chosen_output_path=this_output_path("chosen"),
                rejected_output_path=this_output_path("rejected"),
                chosen_model_config=teacher_serialized,
                rejected_model_config=rejected_serialized,
                chosen_n=align_config.teacher_n,
                chosen_temperature=align_config.teacher_temperature,
                chosen_max_tokens=align_config.teacher_max_tokens,
                chosen_behavior_statements_path=spec_gcs_path,
                rejected_n=align_config.rejected_n,
                rejected_temperature=align_config.rejected_temperature,
                rejected_max_tokens=align_config.rejected_max_tokens,
                rejected_behavior_statements_path=None,
            ),
        )
        chosen_responses_path = output_path_of(responses_step, "chosen")
        rejected_responses_path = output_path_of(responses_step, "rejected")
        response_steps = [responses_step]
    else:
        chosen_step = ExecutorStep(
            name=f"align/{name}/chosen",
            description=f"Generate chosen responses via {teacher_model.model}",
            fn=remote(
                generate_responses,
                resources=teacher_model.resources,
                pip_dependency_groups=_inference_dependency_groups(teacher_model),
                env_vars=_llm_env_vars(),
            ),
            config=ResponseGenConfig(
                prompts_path=output_path_of(prompts_step),
                output_path=this_output_path(),
                model_config=teacher_serialized,
                n=align_config.teacher_n,
                temperature=align_config.teacher_temperature,
                max_tokens=align_config.teacher_max_tokens,
                behavior_statements_path=spec_gcs_path,
            ),
        )
        rejected_step = ExecutorStep(
            name=f"align/{name}/rejected",
            description=f"Generate rejected responses via {rejected_model.model}",
            fn=remote(
                generate_responses,
                resources=rejected_model.resources,
                pip_dependency_groups=_inference_dependency_groups(rejected_model),
                env_vars=_llm_env_vars(),
            ),
            config=ResponseGenConfig(
                prompts_path=output_path_of(prompts_step),
                output_path=this_output_path(),
                model_config=rejected_serialized,
                n=align_config.rejected_n,
                temperature=align_config.rejected_temperature,
                max_tokens=align_config.rejected_max_tokens,
                behavior_statements_path=None,
                dependency_path=(
                    output_path_of(chosen_step) if resolved_response_mode == ResponseExecutionMode.SERIALIZED else None
                ),
            ),
        )
        chosen_responses_path = output_path_of(chosen_step)
        rejected_responses_path = output_path_of(rejected_step)
        response_steps = [chosen_step, rejected_step]

    # Step 3: Judge all responses and persist full score metadata
    judge_is_local = isinstance(align_config.judge_model, VLLMConfig)
    judge_resources = align_config.judge_model.resources if judge_is_local else align_config.cpu_resources
    judgments_step = ExecutorStep(
        name=f"align/{name}/judgments",
        description=f"Judge responses via {_model_name(align_config.judge_model)}",
        fn=remote(
            judge_responses,
            resources=judge_resources,
            pip_dependency_groups=["cpu"] if not judge_is_local else ["vllm", "tpu"],
            env_vars=_llm_env_vars(),
        ),
        config=JudgeConfig(
            chosen_responses_path=chosen_responses_path,
            rejected_responses_path=rejected_responses_path,
            spec_path=spec_gcs_path,
            output_path=this_output_path(),
            judge_model=align_config.judge_model,
            workers=align_config.judge_workers,
            batch_size=align_config.judge_batch_size,
        ),
    )

    # Step 4: Filter persisted judgments into final preference pairs
    pairs_step = ExecutorStep(
        name=f"align/{name}/preference_pairs",
        description="Build preference pairs from persisted judgments",
        fn=remote(
            build_preference_pairs,
            resources=align_config.cpu_resources,
            pip_dependency_groups=["cpu"],
        ),
        config=PreferencePairFilterConfig(
            judgments_path=output_path_of(judgments_step),
            output_path=this_output_path(),
            min_chosen_score=align_config.judge_min_chosen_score,
            min_gap=align_config.judge_min_gap,
        ),
    )

    steps = [spec_step, prompts_step, *response_steps, judgments_step, pairs_step]

    # Steps 4 & 5: Tokenize + DPO (only if dpo_config provided)
    if dpo_config is not None:
        tokenized = default_tokenize(
            name=f"align/{name}/tokenized",
            dataset=output_path_of(pairs_step),
            tokenizer=align_config.tokenizer,
            format=PreferenceChatLmDatasetFormat(),
        )

        trained = default_dpo(
            name=f"align/{name}/dpo",
            tokenized=tokenized,
            model_config=model_config,
            dpo_config=dpo_config,
            tags=[*tags, "alignment", "synthetic-preference"],
        )

        steps.extend([tokenized, trained])

    return steps
