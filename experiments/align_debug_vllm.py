# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Debug alignment pipeline using vLLM with Llama 3.1 8B Instruct.

Uses a single local model for ALL roles (teacher, rejected, ideation,
judge, extraction) to test the full vLLM code path end-to-end.
The outputs won't be high quality — this is purely for pipeline debugging.

Single statement, pairwise covering, minimal settings.

Submit to Iris (needs a TPU for vLLM):

    uv run iris --controller-url http://127.0.0.1:10000 job run \\
        --tpu v6e-8 \\
        --region us-central1 \\
        --no-wait \\
        -- python experiments/align_debug_vllm.py
"""

from pathlib import Path

from experiments.llama import llama_8b
from experiments.models import llama_3_1_8b
from marin.alignment.align import AlignConfig, align
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import executor_main

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")

# One VLLMConfig for everything — Llama 3.1 8B Instruct
llama_vllm = VLLMConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    max_model_len=2048,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
)

align_config = AlignConfig(
    # All models = local vLLM
    ideation_model=llama_vllm,
    extract_model=llama_vllm,
    judge_model=llama_vllm,
    # Minimal covering
    covering_strength=2,
    covering_seed=42,
    # Sequential (vLLM is not thread-safe)
    ideation_workers=1,
    concretize_workers=1,
    extract_workers=1,
    judge_workers=1,
    # Teacher: 1 response
    teacher_n=1,
    teacher_temperature=0.7,
    teacher_max_tokens=512,
    # Rejected: 1 response (skip pick-worst-of-N)
    rejected_n=1,
    rejected_temperature=0.7,
    rejected_max_tokens=512,
    # Loose thresholds — small model outputs won't score well
    judge_min_chosen_score=1.0,
    judge_min_gap=0.0,
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    # Single statement
    statement_ids=["ask_clarifying_questions"],
)

dataset_steps = align(
    name="debug_vllm",
    pretrained_model=llama_3_1_8b,
    spec=SPEC_PATH,
    model_config=llama_8b,
    teacher_model=llama_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=llama_vllm,
    tags=["debug", "vllm"],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description="Debug alignment pipeline with vLLM (Llama 3.1 8B Instruct)",
    )
