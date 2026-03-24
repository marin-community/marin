# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Debug alignment pipeline using vLLM with Llama 3.3 70B Instruct.

Uses a single local model for ALL roles (teacher, rejected, ideation,
judge, extraction) to test the full vLLM code path end-to-end.
The outputs won't be high quality — this is purely for pipeline debugging.

Single statement, pairwise covering, minimal settings.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --extra marin:tpu \
        --tpu v5p-8 \
        --region us-central1 \
        --zone us-central1-a \
        -- python experiments/align_debug_vllm_70b.py
"""

from pathlib import Path

from experiments.llama import llama_70b
from experiments.models import llama_3_3_70b_instruct
from marin.alignment.align import AlignConfig, align
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import executor_main

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")

llama_vllm = VLLMConfig(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="500g",
    ram="256g",
)

align_config = AlignConfig(
    ideation_model=llama_vllm,
    extract_model=llama_vllm,
    judge_model=llama_vllm,
    covering_strength=2,
    covering_seed=42,
    ideation_workers=1,
    concretize_workers=1,
    extract_workers=1,
    judge_workers=1,
    teacher_n=1,
    teacher_temperature=0.7,
    teacher_max_tokens=512,
    rejected_n=1,
    rejected_temperature=0.7,
    rejected_max_tokens=512,
    judge_min_chosen_score=1.0,
    judge_min_gap=0.0,
    tokenizer="meta-llama/Llama-3.3-70B-Instruct",
    statement_ids=["ask_clarifying_questions"],
)

dataset_steps = align(
    name="debug_vllm_70b",
    pretrained_model=llama_3_3_70b_instruct,
    spec=SPEC_PATH,
    model_config=llama_70b,
    teacher_model=llama_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=llama_vllm,
    tags=["debug", "vllm", "70b"],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description="Debug alignment pipeline with vLLM (Llama 3.3 70B Instruct)",
    )
