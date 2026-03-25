# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Debug heterogeneous alignment pipeline with rejected-only opposite prompting.

Chosen / infrastructure roles use Llama 3.3 70B Instruct.
Rejected responses use Mixtral 8x7B Instruct with rejected_prompt_strategy=opposite.

This is the F2 heterogeneous local-local `auto` validation with opposite-mode
rejected responses. The expected response-orchestration shape is:

- separate `chosen` and `rejected` child jobs
- no forced dependency between them
- rejected prompt construction uses the explicit opposite-mode system prompt

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name align-debug-vllm-70b-mixtral-rejected-opposite-auto \
        --cpu 4 \
        --memory 16GB \
        --disk 10GB \
        --region us-central1 \
        -- python experiments/align_debug_vllm_70b_mixtral_rejected_opposite.py
"""

from pathlib import Path

from experiments.llama import llama_70b
from experiments.models import llama_3_3_70b_instruct
from marin.alignment.align import AlignConfig, align
from marin.alignment.generate_responses import RejectedPromptStrategy
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
    concretize_max_tokens=1536,
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
    rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
    judge_min_chosen_score=1.0,
    judge_min_gap=0.0,
    tokenizer="meta-llama/Llama-3.3-70B-Instruct",
    statement_ids=["ask_clarifying_questions"],
)

dataset_steps = align(
    name="debug_vllm_70b_mixtral_rejected_opposite_auto_smoke",
    pretrained_model=llama_3_3_70b_instruct,
    spec=SPEC_PATH,
    model_config=llama_70b,
    teacher_model=llama_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=mixtral_vllm,
    tags=["debug", "vllm", "70b", "mixtral-rejected", "opposite-mode"],
)

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description="Debug heterogeneous auto alignment with Mixtral rejected opposite-mode responses",
    )
