# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate refactored prompt generation via batched `vllm serve`.

Runs only the prompt-generation pipeline (stages 1-3) on a single statement,
using the regional staged Llama 3.3 70B checkpoint.

This smoke run intentionally uses smaller Stage 1/2/3 token budgets and smaller
prompt-generation batch sizes than the API defaults so the request envelope fits
within a `4096` token local context window.

Submit to Iris:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name generate-prompts-llama-3-3-70b-refactored-us-east5a \
        --cpu 4 \
        --memory 16GB \
        --disk 10GB \
        --region us-east5 \
        --zone us-east5-a \
        -- python experiments/generate_prompts_llama_3_3_70b_refactored.py
"""

from pathlib import Path

from fray.v2.types import ResourceConfig

from experiments.models import llama_3_3_70b_instruct
from marin.alignment.align import _UploadSpecConfig, _llm_env_vars, _upload_spec
from marin.alignment.generate_prompts import PromptGenConfig, generate_prompts_from_spec
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.execution.remote import remote

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")

llama_vllm = VLLMConfig(
    model=output_path_of(llama_3_3_70b_instruct),
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="5g",
    ram="256g",
)

spec_step = ExecutorStep(
    name="align/debug_generate_prompts_llama_3_3_70b_refactored/spec",
    description="Upload openai_model_spec.jsonl to GCS for prompt-generation validation",
    fn=remote(
        _upload_spec,
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
        pip_dependency_groups=["cpu"],
    ),
    config=_UploadSpecConfig(
        source_path=SPEC_PATH,
        output_path=this_output_path(),
    ),
)

prompts_step = ExecutorStep(
    name="align/debug_generate_prompts_llama_3_3_70b_refactored/prompts",
    description="Generate prompts from one spec statement via refactored batched vLLM serve",
    fn=remote(
        generate_prompts_from_spec,
        resources=llama_vllm.resources,
        pip_dependency_groups=["vllm", "tpu"],
        env_vars=_llm_env_vars(),
    ),
    config=PromptGenConfig(
        spec_path=output_path_of(spec_step) / "spec.jsonl",
        output_path=this_output_path(),
        ideation_model=llama_vllm,
        extract_model=llama_vllm,
        covering_strength=versioned(2),
        covering_seed=versioned(42),
        concretize_batch_size=versioned(4),
        extract_batch_size=versioned(4),
        local_serve_batch_size=4,
        ideation_workers=1,
        concretize_workers=1,
        extract_workers=1,
        understanding_max_tokens=versioned(1024),
        concretize_temperature=versioned(1.0),
        understanding_temperature=versioned(1.0),
        concretize_max_tokens=versioned(1024),
        extract_max_tokens=versioned(1024),
        concretize_max_attempts=versioned(5),
        statement_ids=versioned(["ask_clarifying_questions"]),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[spec_step, prompts_step],
        description="Validate refactored prompt generation with batched vLLM serve on Llama 3.3 70B",
    )
