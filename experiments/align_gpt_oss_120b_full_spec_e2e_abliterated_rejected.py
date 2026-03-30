# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-spec E2E alignment with abliterated GPT-OSS 120B as rejected model.

Reuses cached spec, prompts, and chosen responses from the original Mixtral run
(same `name="goss_120b_full_spec_e2e"`). Only re-runs:
  - rejected responses (abliterated GPT-OSS 120B, opposite-mode)
  - judge (GPT-OSS 120B censored)
  - preference pairs

The abliterated model should produce more natural "bad" responses than Mixtral,
which often breaks character with meta-commentary like "I'm deliberately violating
the guideline." Both chosen and rejected use the same GPT-OSS architecture, making
the preference pairs more realistic for DPO training.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \\
        --no-wait \\
        --job-name goss-120b-e2e-abliterated-rejected \\
        --cpu 4 --memory 16GB --disk 10GB \\
        --region us-central1 \\
        -e MARIN_PREFIX gs://marin-us-central1 \\
        -- python experiments/align_gpt_oss_120b_full_spec_e2e_abliterated_rejected.py
"""

import dataclasses
from pathlib import Path

from experiments.gpt_oss_120b_tpu import GPT_OSS_TPU_DEFAULT_MAX_TOKENS, gpt_oss_120b_tpu_vllm_config
from experiments.models import gpt_oss_120b_abliterated_vllm, gpt_oss_120b_vllm
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

# Rejected: abliterated GPT-OSS 120B (same TPU config, different weights)
abliterated_vllm = dataclasses.replace(
    gpt_oss_120b_tpu_vllm_config(
        max_model_len=8192,
        ram="400g",
        model_impl_type="vllm",
        prefer_jax_for_bootstrap=False,
    ),
    model=output_path_of(gpt_oss_120b_abliterated_vllm),
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
    rejected_model=abliterated_vllm,
    tags=["vllm", "gpt-oss-120b", "abliterated-rejected", "opposite-mode", "full-spec", "e2e"],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description=(
            "Full-spec E2E alignment: GPT-oss-120B censored for prompts/chosen/judge, "
            "abliterated GPT-oss-120B opposite-mode rejected, "
            "all 46 OpenAI Model Spec statements, batch_size=256. "
            "Reuses cached spec/prompts/chosen from Mixtral run."
        ),
    )
