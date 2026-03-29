# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-statement end-to-end alignment pipeline with GPT-oss-20B.

Exercises the full pipeline on `ask_clarifying_questions` only:
  - prompt generation (Stage 1/2/3) on GPT-oss-20B
  - chosen responses on GPT-oss-20B
  - rejected responses on Mixtral-8x7B-Instruct (opposite prompting)
  - judge on GPT-oss-20B
  - stops at preference pairs (no DPO training)

This is the 20B validation of the full align() pipeline path before
promoting to 120B. Uses model_impl_type="vllm" (the validated backend).

Submit to Iris:

    uv run iris --controller-url http://127.0.0.1:10000 job run \\
        --no-wait \\
        --job-name goss-20b-e2e-one-statement \\
        --cpu 4 --memory 16GB --disk 10GB \\
        --region us-central1 \\
        -- python experiments/align_gpt_oss_20b_e2e_one_statement.py
"""

from pathlib import Path

from experiments.gpt_oss_20b_tpu import GPT_OSS_20B_TPU_DEFAULT_MAX_TOKENS, gpt_oss_20b_tpu_vllm_config
from experiments.models import gpt_oss_20b_vllm, mixtral_8x7b_instruct
from marin.alignment.align import AlignConfig, ResponseExecutionMode, align
from marin.alignment.generate_responses import RejectedPromptStrategy
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import executor_main, output_path_of

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")
GPT_OSS_RESPONSE_MAX_TOKENS = 4096

gpt_oss_vllm = gpt_oss_20b_tpu_vllm_config(
    max_model_len=8192,
    model_impl_type="vllm",
    prefer_jax_for_bootstrap=False,
)

mixtral_vllm = VLLMConfig(
    model=output_path_of(mixtral_8x7b_instruct),
    tensor_parallel_size=4,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="10g",
    ram="256g",
)

align_config = AlignConfig(
    ideation_model=gpt_oss_vllm,
    extract_model=gpt_oss_vllm,
    judge_model=gpt_oss_vllm,
    covering_strength=2,
    covering_seed=42,
    ideation_workers=1,
    concretize_workers=1,
    extract_workers=1,
    prompt_batch_size=32,
    understanding_max_tokens=GPT_OSS_20B_TPU_DEFAULT_MAX_TOKENS,
    understanding_temperature=1.0,
    understanding_max_attempts=5,
    concretize_max_tokens=GPT_OSS_20B_TPU_DEFAULT_MAX_TOKENS,
    concretize_temperature=1.0,
    concretize_max_attempts=5,
    extract_max_tokens=GPT_OSS_20B_TPU_DEFAULT_MAX_TOKENS,
    judge_workers=1,
    judge_batch_size=8,
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
    name="goss_20b_e2e_one_statement",
    pretrained_model=gpt_oss_20b_vllm,
    spec=SPEC_PATH,
    model_config=None,
    teacher_model=gpt_oss_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=mixtral_vllm,
    tags=["debug", "vllm", "gpt-oss-20b", "mixtral-rejected", "opposite-mode", "one-statement", "e2e"],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description=(
            "Single-statement E2E alignment pipeline: GPT-oss-20B for "
            "prompts/chosen/judge, Mixtral opposite-mode rejected, "
            "on ask_clarifying_questions only"
        ),
    )
