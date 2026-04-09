# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-spec E2E alignment pipeline with GPT-oss-120B (all stages).

Runs the complete pipeline on all 46 OpenAI Model Spec statements:
  - prompt generation (Stage 1/2/3) on GPT-oss-120B
  - chosen responses on GPT-oss-120B (spec-guided)
  - rejected responses on a configurable rejected-model preset (opposite prompting)
  - judge on GPT-oss-120B
  - preference pairs (no DPO training)

Uses the validated serving contract from ALIGN-274:
  - model_impl_type="vllm" (flax_nnx produces gibberish)
  - reasoning_effort="low" (sent as top-level field)
  - ram="400g" (256g OOMs on the vllm backend for 120B)
  - max_model_len=8192, teacher_max_tokens=4096
  - response_batch_size=256 (full vLLM concurrency)
  - extract_max_tokens=8192 (improved extraction with SELF-CONTAINED prompt)
Set `REJECTED_MODEL_PRESET` to switch between Mixtral and Heretic GPT-OSS 20B.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \\
        --no-wait \\
        --job-name goss-120b-full-spec-e2e \\
        --cpu 4 --memory 16GB --disk 10GB \\
        --region us-central1 \\
        -e MARIN_PREFIX gs://marin-us-central1 \\
        -- python experiments/posttrain/align_gpt_oss_120b_full_spec_e2e.py
"""

from pathlib import Path

from gpt_oss_tpu import (
    GPT_OSS_TPU_DEFAULT_MAX_TOKENS,
    RejectedModelPreset,
    gpt_oss_120b_tpu_vllm_config,
    rejected_model_label,
    rejected_model_name_suffix,
    rejected_model_tag,
    rejected_model_vllm_config,
)
from experiments.models import gpt_oss_120b_vllm
from marin.alignment.align import AlignConfig, ResponseExecutionMode, align
from marin.alignment.generate_responses import RejectedPromptStrategy
from marin.execution.executor import executor_main

SPEC_PATH = str(Path(__file__).parent / "specs" / "openai_model_spec.jsonl")

GPT_OSS_RESPONSE_MAX_TOKENS = 4096
REJECTED_MODEL_PRESET = RejectedModelPreset.MIXTRAL
PIPELINE_NAME = "goss_120b_full_spec_e2e"
if REJECTED_MODEL_PRESET is not RejectedModelPreset.MIXTRAL:
    PIPELINE_NAME = f"{PIPELINE_NAME}_{rejected_model_name_suffix(REJECTED_MODEL_PRESET)}"

gpt_oss_vllm = gpt_oss_120b_tpu_vllm_config(
    max_model_len=8192,
    ram="400g",
    model_impl_type="vllm",
    prefer_jax_for_bootstrap=False,
)

rejected_vllm = rejected_model_vllm_config(REJECTED_MODEL_PRESET)

align_config = AlignConfig(
    ideation_model=gpt_oss_vllm,
    extract_model=gpt_oss_vllm,
    judge_model=gpt_oss_vllm,
    covering_strength=2,
    covering_seed=42,
    ideation_workers=1,
    concretize_workers=1,
    extract_workers=1,
    prompt_batch_size=256,
    response_batch_size=256,
    understanding_max_tokens=GPT_OSS_TPU_DEFAULT_MAX_TOKENS,
    understanding_temperature=1.0,
    understanding_max_attempts=5,
    concretize_max_tokens=GPT_OSS_TPU_DEFAULT_MAX_TOKENS,
    concretize_temperature=1.0,
    concretize_max_attempts=5,
    extract_max_tokens=8192,
    judge_workers=1,
    judge_batch_size=256,
    teacher_n=1,
    teacher_temperature=0.7,
    teacher_max_tokens=GPT_OSS_RESPONSE_MAX_TOKENS,
    rejected_n=1,
    rejected_temperature=0.7,
    rejected_max_tokens=GPT_OSS_RESPONSE_MAX_TOKENS,
    rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
    judge_min_chosen_score=0.0,
    judge_min_gap=0.0,
    response_execution_mode=ResponseExecutionMode.AUTO,
    tokenizer="unsloth/gpt-oss-120b-BF16",
    # No statement_ids filter — run all 46 statements.
)

dataset_steps = align(
    name=PIPELINE_NAME,
    pretrained_model=gpt_oss_120b_vllm,
    spec=SPEC_PATH,
    model_config=None,
    teacher_model=gpt_oss_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=rejected_vllm,
    tags=["vllm", "gpt-oss-120b", rejected_model_tag(REJECTED_MODEL_PRESET), "opposite-mode", "full-spec", "e2e"],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description=(
            "Full-spec E2E alignment pipeline: GPT-oss-120B for "
            f"prompts/chosen/judge, {rejected_model_label(REJECTED_MODEL_PRESET)} opposite-mode rejected, "
            "all 46 OpenAI Model Spec statements, batch_size=256"
        ),
    )
