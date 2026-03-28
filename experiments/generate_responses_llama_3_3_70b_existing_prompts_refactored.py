# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Run the refactored local `generate_responses.py` path on a known-good prompts artifact.

This verifies that alignment response generation now goes through batched
`vllm serve` instead of direct `llm.generate(...)`, while preserving the output
schema and using the staged `us-central1` GCS checkpoint.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --extra marin:tpu \
        --tpu v5p-8 \
        --region us-central1 \
        --zone us-central1-a \
        -- python experiments/generate_responses_llama_3_3_70b_existing_prompts_refactored.py
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
JOB_DESCRIPTION = " ".join(
    [
        "Refactored generate_responses.py validation on succeeded prompts",
        "with staged us-central1 Llama 3.3 70B",
    ]
)


llama_3_3_70b_vllm = VLLMConfig(
    model=output_path_of(MODEL_STEP),
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="5g",
    ram="256g",
)

response_step = ExecutorStep(
    name="align/debug_generate_responses_llama_3_3_70b_existing_prompts_refactored/responses",
    description="Verify refactored generate_responses.py via batched vllm serve on succeeded prompts artifact",
    fn=remote(
        generate_responses,
        resources=llama_3_3_70b_vllm.resources,
        env_vars={"MARIN_VLLM_MODE": "native"},
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
        description=JOB_DESCRIPTION,
    )
