# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Run standalone rejected-side opposite-mode response generation on a known-good prompts artifact.

This validates the new fail-closed rejected-only `opposite` prompt strategy
without involving the full alignment pipeline. The run uses the staged
`us-central1` Llama 3.3 70B checkpoint and an existing prompt artifact that was
generated successfully from the OpenAI model-spec smoke statement.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name generate-rejected-opposite-llama-3-3-70b-existing-prompts \
        --cpu 4 \
        --memory 16GB \
        --disk 10GB \
        --region us-central1 \
        -- python experiments/generate_rejected_opposite_llama_3_3_70b_existing_prompts.py
"""

from __future__ import annotations

from pathlib import Path

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

PROMPTS_PATH = "gs://marin-us-central1/align/debug_generate_prompts_llama_3_3_70b_refactored/prompts-f29568"
SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")
MODEL_STEP = llama_3_3_70b_instruct
DESCRIPTION = (
    "Standalone rejected-only opposite-mode generation on existing prompts with staged us-central1 Llama 3.3 70B"
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
    name="align/debug_generate_rejected_opposite_llama_3_3_70b_existing_prompts/responses",
    description="Generate rejected-only opposite-mode responses on a known-good prompts artifact",
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
        rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
        n=1,
        temperature=0.7,
        max_tokens=512,
        behavior_statements_path=SPEC_PATH,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[response_step],
        description=DESCRIPTION,
    )
