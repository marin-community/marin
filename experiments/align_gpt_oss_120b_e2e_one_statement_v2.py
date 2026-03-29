# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-statement E2E alignment pipeline with GPT-oss-120B (fixed backend).

Uses the validated serving contract:
  - model_impl_type="vllm" (flax_nnx produces gibberish)
  - reasoning_effort="low" (sent as top-level field)
  - judge thresholds relaxed to 0.0 (GPT-OSS judge JSON parsing is unreliable)
"""

from pathlib import Path

from experiments.gpt_oss_120b_tpu import GPT_OSS_TPU_DEFAULT_MAX_TOKENS, gpt_oss_120b_tpu_vllm_config
from experiments.models import gpt_oss_120b_vllm, mixtral_8x7b_instruct
from marin.alignment.align import AlignConfig, ResponseExecutionMode, align
from marin.alignment.generate_responses import RejectedPromptStrategy
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import executor_main, output_path_of

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")

GPT_OSS_RESPONSE_MAX_TOKENS = 4096

gpt_oss_vllm = gpt_oss_120b_tpu_vllm_config(
    max_model_len=8192,
    ram="400g",
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
    understanding_max_tokens=GPT_OSS_TPU_DEFAULT_MAX_TOKENS,
    understanding_temperature=1.0,
    understanding_max_attempts=5,
    concretize_max_tokens=GPT_OSS_TPU_DEFAULT_MAX_TOKENS,
    concretize_temperature=1.0,
    concretize_max_attempts=5,
    extract_max_tokens=GPT_OSS_TPU_DEFAULT_MAX_TOKENS,
    judge_workers=1,
    judge_batch_size=8,
    teacher_n=1,
    teacher_temperature=0.7,
    teacher_max_tokens=GPT_OSS_RESPONSE_MAX_TOKENS,
    rejected_n=1,
    rejected_temperature=0.7,
    rejected_max_tokens=GPT_OSS_RESPONSE_MAX_TOKENS,
    rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
    # Relaxed: GPT-OSS judge can't produce valid JSON, so pass all pairs through.
    judge_min_chosen_score=0.0,
    judge_min_gap=0.0,
    response_execution_mode=ResponseExecutionMode.AUTO,
    tokenizer="unsloth/gpt-oss-120b-BF16",
    statement_ids=["ask_clarifying_questions"],
)

dataset_steps = align(
    name="goss_120b_e2e_one_statement_v2",
    pretrained_model=gpt_oss_120b_vllm,
    spec=SPEC_PATH,
    model_config=None,
    teacher_model=gpt_oss_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=mixtral_vllm,
    tags=["debug", "vllm", "gpt-oss-120b", "mixtral-rejected", "opposite-mode", "one-statement", "e2e", "v2"],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description=(
            "Single-statement E2E alignment pipeline v2: GPT-oss-120B (model_impl_type=vllm) for "
            "prompts/chosen/judge, Mixtral opposite-mode rejected, "
            "on ask_clarifying_questions only. Judge thresholds relaxed to 0.0."
        ),
    )
