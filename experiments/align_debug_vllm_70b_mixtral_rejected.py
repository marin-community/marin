# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Debug alignment pipeline using regional GCS checkpoints in us-central1.

Chosen / infrastructure roles use Llama 3.3 70B Instruct.
Rejected responses use Mixtral 8x7B Instruct.

This is the one-statement heterogeneous smoke run described in ALIGN-030.
Run only after the us-central1 Mixtral download finishes and the GCS prefixes
have been smoke-validated with `vllm_smoke_test.py`.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name align-debug-vllm-70b-mixtral-rejected-smoke-refactored \
        --cpu 4 \
        --memory 16GB \
        --disk 10GB \
        --region us-central1 \
        -- python experiments/align_debug_vllm_70b_mixtral_rejected.py
"""

from pathlib import Path

from experiments.llama import llama_70b
from experiments.models import llama_3_3_70b_instruct
from marin.alignment.align import AlignConfig, align
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import executor_main

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")
LLAMA_70B_GCS_PATH = "gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b"
MIXTRAL_8X7B_INSTRUCT_GCS_PATH = "gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230"

llama_vllm = VLLMConfig(
    model=LLAMA_70B_GCS_PATH,
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="10g",
    ram="256g",
)

mixtral_vllm = VLLMConfig(
    model=MIXTRAL_8X7B_INSTRUCT_GCS_PATH,
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="10g",
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
    prompt_batch_size=4,
    understanding_max_tokens=1024,
    understanding_temperature=1.0,
    concretize_max_tokens=1024,
    concretize_temperature=1.0,
    concretize_max_attempts=5,
    extract_max_tokens=1024,
    judge_workers=1,
    judge_batch_size=4,
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
    name="debug_vllm_70b_mixtral_rejected_smoke",
    pretrained_model=llama_3_3_70b_instruct,
    spec=SPEC_PATH,
    model_config=llama_70b,
    teacher_model=llama_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=mixtral_vllm,
    tags=["debug", "vllm", "70b", "mixtral-rejected"],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description="Debug alignment pipeline with regional GCS checkpoints and Mixtral rejected responses",
    )
