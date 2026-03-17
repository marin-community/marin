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
        teacher_model="openai/gpt-4.1",
        align_config=AlignConfig(dpo=SimpleDPOConfig(...)),
    )
    executor_main(steps=[some_checkpoint_step, *aligned_steps])
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fray.v2.types import ResourceConfig
from levanter.data.text.preference import PreferenceChatLmDatasetFormat

from marin.alignment.generate_prompts import PromptGenConfig, generate_prompts_from_spec
from marin.alignment.generate_responses import ResponseGenConfig, generate_responses
from marin.alignment.judge import JudgePairConfig, judge_and_build_pairs
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path, versioned
from marin.execution.remote import remote


@dataclass(frozen=True)
class AlignConfig:
    """Configuration for the full alignment pipeline.

    Attributes:
        ideation_model: API model for prompt generation (stages 1-3). Always API.
        extract_model: API model for Stage 3 extraction. Can be cheaper/faster.
        judge_model: API model for scoring responses. Always API.
        covering_strength: t-way covering (2=pairwise, 3=3-way).
        ideation_workers: Parallelism for Stage 1 understanding.
        concretize_workers: Parallelism for Stage 2 concretization.
        extract_workers: Parallelism for Stage 3 extraction.
        teacher_n: Number of responses per prompt from teacher (1 is usually enough).
        teacher_temperature: Sampling temperature for teacher.
        teacher_max_tokens: Max tokens for teacher responses.
        rejected_n: Number of responses per prompt from rejected model (judge picks worst).
        rejected_temperature: Sampling temperature for rejected model.
        rejected_max_tokens: Max tokens for rejected responses.
        judge_min_chosen_score: Teacher response must score >= this (1-10 scale).
        judge_min_gap: chosen_score - worst_rejected_score must be >= this.
        response_workers: Parallelism for response generation API calls.
        judge_workers: Parallelism for judge API calls.
        tokenizer: HuggingFace tokenizer name for DPO training data.
        inference_resources: Resources for vLLM when using local models.
        cpu_resources: Resources for CPU-only steps (prompt gen, judging).
    """

    # Infrastructure models (always API)
    ideation_model: str = "openai/gpt-4.1"
    extract_model: str = "openai/gpt-4.1-mini"
    judge_model: str = "openai/gpt-4.1"

    # Covering array
    covering_strength: int = 3
    covering_seed: int = 42

    # Parallelism
    ideation_workers: int = 32
    concretize_workers: int = 32
    extract_workers: int = 128
    response_workers: int = 64
    judge_workers: int = 64

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

    # Resources
    inference_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v6e-8"))
    cpu_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"))

    # Statement filtering
    statement_ids: list[str] | None = None


def _write_behavior_statements(spec_path: str, output_path: str) -> None:
    """Extract behavior ID → text mapping from spec and write as JSON."""
    from marin.alignment.generate_prompts import load_spec

    statements = load_spec(spec_path)
    mapping = {sid: stmt.text for sid, stmt in statements.items()}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def align(
    name: str,
    pretrained_model: ExecutorStep,
    spec: ExecutorStep | str,
    model_config,
    teacher_model: ExecutorStep | str,
    align_config: AlignConfig,
    dpo_config=None,
    rejected_model: ExecutorStep | str | None = None,
    tags: Sequence[str] = (),
) -> list[ExecutorStep]:
    """Produce an aligned model from a specification.

    Args:
        name: Name for this alignment run (used in output paths).
        pretrained_model: The model being aligned. Used for DPO init + reference only.
        spec: Path to specification JSONL or an ExecutorStep that produces one.
        model_config: Levanter model config (e.g. LlamaConfig) for the pretrained model.
        teacher_model: Generates chosen responses. API string or ExecutorStep.
        align_config: Configuration for the alignment pipeline.
        dpo_config: SimpleDPOConfig for DPO training. If None, returns steps up to preference pairs.
        rejected_model: Generates rejected responses. Defaults to teacher_model with no spec guidance
            (which is the standard setup — same model but without seeing the behavioral guideline).
        tags: WandB tags for the DPO training run.

    Returns:
        List of ExecutorSteps comprising the alignment pipeline.
    """
    from experiments.defaults import default_dpo, default_tokenize

    if rejected_model is None:
        rejected_model = teacher_model

    # Resolve spec path
    if isinstance(spec, ExecutorStep):
        spec_input = output_path_of(spec)
    else:
        spec_input = spec

    # Resolve model strings for response generation
    teacher_model_str = teacher_model if isinstance(teacher_model, str) else "local"
    rejected_model_str = rejected_model if isinstance(rejected_model, str) else "local"

    # Step 1: Generate prompts from spec (stages 1-3)
    prompts_step = ExecutorStep(
        name=f"align/{name}/prompts",
        description=f"Generate eval prompts from spec via {align_config.ideation_model}",
        fn=remote(
            generate_prompts_from_spec,
            resources=align_config.cpu_resources,
            pip_dependency_groups=["cpu"],
        ),
        config=PromptGenConfig(
            spec_path=spec_input,
            output_path=this_output_path(),
            ideation_model=align_config.ideation_model,
            extract_model=align_config.extract_model,
            covering_strength=versioned(align_config.covering_strength),
            covering_seed=align_config.covering_seed,
            concretize_batch_size=10,
            extract_batch_size=10,
            ideation_workers=align_config.ideation_workers,
            concretize_workers=align_config.concretize_workers,
            extract_workers=align_config.extract_workers,
            statement_ids=align_config.statement_ids,
        ),
    )

    # Step 2a: Teacher generates chosen responses (with spec guidance)
    chosen_step = ExecutorStep(
        name=f"align/{name}/chosen",
        description=f"Generate chosen responses via {teacher_model_str}",
        fn=remote(
            generate_responses,
            resources=align_config.cpu_resources if isinstance(teacher_model, str) else align_config.inference_resources,
            pip_dependency_groups=["cpu"] if isinstance(teacher_model, str) else [],
        ),
        config=ResponseGenConfig(
            prompts_path=output_path_of(prompts_step),
            output_path=this_output_path(),
            model=teacher_model if isinstance(teacher_model, str) else output_path_of(teacher_model),
            n=align_config.teacher_n,
            temperature=align_config.teacher_temperature,
            max_tokens=align_config.teacher_max_tokens,
            behavior_statements_path=spec_input,  # teacher sees spec guidance
            workers=align_config.response_workers,
        ),
    )

    # Step 2b: Rejected model generates rejected responses (NO spec guidance)
    rejected_step = ExecutorStep(
        name=f"align/{name}/rejected",
        description=f"Generate rejected responses via {rejected_model_str}",
        fn=remote(
            generate_responses,
            resources=(
                align_config.cpu_resources if isinstance(rejected_model, str) else align_config.inference_resources
            ),
            pip_dependency_groups=["cpu"] if isinstance(rejected_model, str) else [],
        ),
        config=ResponseGenConfig(
            prompts_path=output_path_of(prompts_step),
            output_path=this_output_path(),
            model=rejected_model if isinstance(rejected_model, str) else output_path_of(rejected_model),
            n=align_config.rejected_n,
            temperature=align_config.rejected_temperature,
            max_tokens=align_config.rejected_max_tokens,
            behavior_statements_path=None,  # rejected model gets NO spec guidance
            workers=align_config.response_workers,
        ),
    )

    # Step 3: Judge + build preference pairs
    pairs_step = ExecutorStep(
        name=f"align/{name}/preference_pairs",
        description=f"Judge and build preference pairs via {align_config.judge_model}",
        fn=remote(
            judge_and_build_pairs,
            resources=align_config.cpu_resources,
            pip_dependency_groups=["cpu"],
        ),
        config=JudgePairConfig(
            prompts_path=output_path_of(prompts_step),
            chosen_responses_path=output_path_of(chosen_step),
            rejected_responses_path=output_path_of(rejected_step),
            spec_path=spec_input,
            output_path=this_output_path(),
            judge_model=align_config.judge_model,
            min_chosen_score=align_config.judge_min_chosen_score,
            min_gap=align_config.judge_min_gap,
            workers=align_config.judge_workers,
        ),
    )

    steps = [prompts_step, chosen_step, rejected_step, pairs_step]

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
