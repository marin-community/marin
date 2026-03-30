# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-spec E2E alignment with Heretic GPT-OSS 20B as rejected model.

Reuses cached spec, prompts, and chosen responses from the original Mixtral run
(same `name="goss_120b_full_spec_e2e"`). Only re-runs:
  - rejected responses (p-e-w/gpt-oss-20b-heretic, opposite-mode)
  - judge (GPT-OSS 120B censored)
  - preference pairs

The Heretic model is a 20B abliterated variant (Heretic v1.0.0, refusal 100→58/100).
It's smaller and faster than the 120B abliterated model, and being a different
abliteration tool may follow OPPOSITE instructions differently.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \\
        --no-wait \\
        --job-name goss-120b-e2e-heretic-rejected \\
        --cpu 4 --memory 16GB --disk 10GB \\
        --region us-central1 \\
        -e MARIN_PREFIX gs://marin-us-central1 \\
        -- python experiments/align_gpt_oss_120b_full_spec_e2e_heretic_rejected.py
"""

import dataclasses
from pathlib import Path

from experiments.gpt_oss_20b_tpu import gpt_oss_20b_tpu_vllm_config
from experiments.gpt_oss_120b_tpu import GPT_OSS_TPU_DEFAULT_MAX_TOKENS, gpt_oss_120b_tpu_vllm_config
from experiments.models import gpt_oss_120b_vllm, gpt_oss_20b_heretic_vllm
from marin.alignment.align import AlignConfig, ResponseExecutionMode, align
from marin.alignment.generate_responses import RejectedPromptStrategy
from marin.execution.executor import executor_main, output_path_of

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")

GPT_OSS_RESPONSE_MAX_TOKENS = 4096

# Teacher/judge: censored GPT-OSS 120B (same as original run)
gpt_oss_vllm = gpt_oss_120b_tpu_vllm_config(
    max_model_len=8192,
    ram="400g",
    model_impl_type="vllm",
    prefer_jax_for_bootstrap=False,
)

# Rejected: Heretic GPT-OSS 20B (smaller, faster, different abliteration tool)
heretic_vllm = gpt_oss_20b_tpu_vllm_config(
    max_model_len=8192,
    model_impl_type="vllm",
    prefer_jax_for_bootstrap=False,
)
# Override model path to heretic weights
heretic_vllm = dataclasses.replace(
    heretic_vllm,
    model=output_path_of(gpt_oss_20b_heretic_vllm),
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
    # No statement_ids filter — all 46 statements.
)

dataset_steps = align(
    # SAME name as original run — enables executor cache reuse for spec/prompts/chosen
    name="goss_120b_full_spec_e2e",
    pretrained_model=gpt_oss_120b_vllm,
    spec=SPEC_PATH,
    model_config=None,
    teacher_model=gpt_oss_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=heretic_vllm,
    tags=["vllm", "gpt-oss-120b", "heretic-20b-rejected", "opposite-mode", "full-spec", "e2e"],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description=(
            "Full-spec E2E alignment: GPT-oss-120B censored for prompts/chosen/judge, "
            "Heretic GPT-oss-20B opposite-mode rejected, "
            "all 46 OpenAI Model Spec statements, batch_size=256. "
            "Reuses cached spec/prompts/chosen from Mixtral run."
        ),
    )
