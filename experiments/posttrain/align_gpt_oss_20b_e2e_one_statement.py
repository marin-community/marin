# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-statement end-to-end alignment pipeline with GPT-oss-20B.

Exercises the full pipeline on `ask_clarifying_questions` only:
  - prompt generation (Stage 1/2/3) on GPT-oss-20B
  - chosen responses on GPT-oss-20B
  - rejected responses on a configurable rejected-model preset (opposite prompting)
  - judge on GPT-oss-20B
  - stops at preference pairs (no DPO training)

This is the 20B validation of the full align() pipeline path before
promoting to 120B. Uses model_impl_type="vllm" (the validated backend).
Set `REJECTED_MODEL_PRESET` to switch between Mixtral and Heretic GPT-OSS 20B.

Submit to Iris:

    uv run iris --controller-url http://127.0.0.1:10000 job run \\
        --no-wait \\
        --job-name goss-20b-e2e-one-statement \\
        --cpu 4 --memory 16GB --disk 10GB \\
        --region us-central1 \\
        -- python experiments/posttrain/align_gpt_oss_20b_e2e_one_statement.py
"""

from pathlib import Path

from gpt_oss_tpu import (
    GPT_OSS_TPU_DEFAULT_MAX_TOKENS,
    RejectedModelPreset,
    gpt_oss_20b_tpu_vllm_config,
    rejected_model_label,
    rejected_model_name_suffix,
    rejected_model_tag,
    rejected_model_vllm_config,
)
from experiments.models import gpt_oss_20b_vllm
from marin.alignment.align import AlignConfig, ResponseExecutionMode, align
from marin.alignment.generate_responses import RejectedPromptStrategy
from marin.execution.executor import executor_main

SPEC_PATH = str(Path(__file__).parent / "specs" / "openai_model_spec.jsonl")
GPT_OSS_RESPONSE_MAX_TOKENS = 4096
REJECTED_MODEL_PRESET = RejectedModelPreset.MIXTRAL
PIPELINE_NAME = "goss_20b_e2e_one_statement"
if REJECTED_MODEL_PRESET is not RejectedModelPreset.MIXTRAL:
    PIPELINE_NAME = f"{PIPELINE_NAME}_{rejected_model_name_suffix(REJECTED_MODEL_PRESET)}"

gpt_oss_vllm = gpt_oss_20b_tpu_vllm_config(
    max_model_len=8192,
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
    extract_max_tokens=GPT_OSS_TPU_DEFAULT_MAX_TOKENS,
    judge_workers=1,
    judge_batch_size=256,
    teacher_n=1,
    teacher_temperature=0.7,
    teacher_max_tokens=GPT_OSS_RESPONSE_MAX_TOKENS,
    rejected_n=1,
    rejected_temperature=0.7,
    rejected_max_tokens=GPT_OSS_RESPONSE_MAX_TOKENS,
    rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
    judge_min_chosen_score=1.0,
    judge_min_gap=0.0,
    response_execution_mode=ResponseExecutionMode.AUTO,
    tokenizer="unsloth/gpt-oss-20b-BF16",
    statement_ids=["ask_clarifying_questions"],
)

dataset_steps = align(
    name=PIPELINE_NAME,
    pretrained_model=gpt_oss_20b_vllm,
    spec=SPEC_PATH,
    model_config=None,
    teacher_model=gpt_oss_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=rejected_vllm,
    tags=[
        "debug",
        "vllm",
        "gpt-oss-20b",
        rejected_model_tag(REJECTED_MODEL_PRESET),
        "opposite-mode",
        "one-statement",
        "e2e",
    ],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description=(
            "Single-statement E2E alignment pipeline: GPT-oss-20B for "
            f"prompts/chosen/judge, {rejected_model_label(REJECTED_MODEL_PRESET)} opposite-mode rejected, "
            "on ask_clarifying_questions only"
        ),
    )
