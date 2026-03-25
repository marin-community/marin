# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Run `generate_responses.py` directly on a known-good prompts artifact.

This isolates the response-generation primitive from the full alignment
pipeline by reusing prompts that were already produced successfully in
ALIGN-026 and loading the model from the regional GCS artifact path,
never directly from Hugging Face at step runtime.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --extra marin:tpu \
        --tpu v5p-8 \
        --region us-central1 \
        --zone us-central1-a \
        -- python experiments/generate_responses_llama_3_3_70b_existing_prompts.py
"""

from __future__ import annotations

from experiments.models import llama_3_3_70b_instruct
from marin.alignment.generate_responses import (
    RejectedPromptStrategy,
    ResponseGenConfig,
    ResponseRole,
    generate_responses,
)
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.execution.remote import remote

PROMPTS_PATH = "gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d"
MODEL_STEP = llama_3_3_70b_instruct


llama_3_3_70b_vllm = VLLMConfig(
    model=output_path_of(MODEL_STEP),
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="500g",
    ram="256g",
)

response_step = ExecutorStep(
    name="align/debug_generate_responses_llama_3_3_70b_existing_prompts/responses",
    description="Generate responses on succeeded prompts artifact via regional GCS Llama 3.3 70B Instruct",
    fn=remote(
        generate_responses,
        resources=llama_3_3_70b_vllm.resources,
        pip_dependency_groups=["vllm", "tpu"],
    ),
    config=ResponseGenConfig(
        prompts_path=PROMPTS_PATH,
        output_path=this_output_path(),
        model_config=llama_3_3_70b_vllm,
        role=ResponseRole.REJECTED,
        rejected_prompt_strategy=RejectedPromptStrategy.UNGUIDED,
        n=1,
        temperature=0.7,
        max_tokens=512,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[response_step],
        description="Isolated generate_responses.py run on succeeded prompts with regional GCS Llama 3.3 70B Instruct",
    )
