# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate alignment of a model against the OpenAI model spec.

Runs inference on the target model without spec guidance (testing
internalized alignment), then judges each response for compliance.

Supports both Bloom-format eval prompts (from GCS) and Marin-format
prompts (from a prior generate_prompts step).

Usage:

    uv run iris --controller-url http://127.0.0.1:10000 job run \
        --no-wait \
        --job-name eval-marin-8b-alignment \
        --cpu 4 --memory 16GB --disk 10GB \
        --region us-central1 \
        -- python experiments/posttrain/eval_llama3_8b_alignment.py
"""

from pathlib import Path

from marin.alignment.align import EvalConfig, evaluate
from marin.alignment.evaluate import PromptFormat
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import executor_main

SPEC_PATH = str(Path(__file__).parent / "specs" / "openai_model_spec.jsonl")

# GCS path to the eval-split prompts extracted from Bloom's
# dev-bloom-results-gpt-4-mini-prompts dataset using the exact seeded split:
#   seed=7, fractions={train:0.70, val:0.15, eval:0.15}
# 2,576 eval prompts across 46 statements. Also mirrored to us-east5.
BLOOM_PROMPTS_GCS = "gs://marin-us-central1/alignment/gpt-4.1-eval-split"

# DPO checkpoint: bloom speceval v2, marin instruct, beta=0.1, lr=7.5e-7, seed=0
MODEL_GCS = (
    "gs://marin-us-central1/checkpoints/dpo/"
    "bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad/hf/step-849"
)

LLAMA_8B_VLLM = VLLMConfig(
    model=MODEL_GCS,
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    cpu=16,
    disk="50g",
    ram="128g",
)

eval_config = EvalConfig(
    prompt_format=PromptFormat.BLOOM,
    temperature=0.7,
    max_tokens=1500,
    n=3,
    inference_batch_size=256,
    judge_workers=64,
    judge_batch_size=8,
    judge_max_tokens=1000,
)

eval_steps = evaluate(
    name="marin_dpo_beta01_lr75e7_seed0_bloom_speceval",
    target_model=LLAMA_8B_VLLM,
    prompts=BLOOM_PROMPTS_GCS,
    spec=SPEC_PATH,
    eval_config=eval_config,
    judge_model="gpt-4.1",
)

# Inference only — judge step runs separately after manual review
inference_step = eval_steps[:1]

if __name__ == "__main__":
    executor_main(
        steps=inference_step,
        description="Inference only: Marin DPO beta0.1 lr7.5e-7 seed0 on 2,576 eval prompts (46 statements)",
    )
