# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate full prompt generation via batched `vllm serve`.

Runs prompt generation Stages 1-3 on the OpenAI model spec using the staged
regional Llama 3.3 70B checkpoint. When the ideation and extraction models are
the same, all three stages reuse one local `vllm serve` session and emit one
shared `artifacts/vllm_metrics.json` artifact with per-stage metrics for:
  - `understanding`
  - `concretize`
  - `extract`

Submit to Iris:

    uv run iris --controller-url http://127.0.0.1:10000 job run \
        --no-wait \
        --job-name generate-prompts-llama-3-3-70b-refactored \
        --cpu 4 \
        --memory 16GB \
        --disk 10GB \
        --region us-central1 \
        -- python experiments/generate_prompts_llama_3_3_70b_refactored.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from fray.v2.types import ResourceConfig

from experiments.models import llama_3_3_70b_instruct
from marin.alignment.align import _UploadSpecConfig, _llm_env_vars, _upload_spec
from marin.alignment.generate_prompts import PromptGenConfig, generate_prompts_from_spec
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.execution.remote import remote

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="debug_generate_prompts_llama_3_3_70b_refactored")
    parser.add_argument("--local-serve-batch-size", type=int, default=4)
    parser.add_argument("--statement-id", action="append", dest="statement_ids", default=None)
    return parser.parse_known_args()


def build_steps(*, name: str, local_serve_batch_size: int, statement_ids: list[str] | None) -> list[ExecutorStep]:
    llama_vllm = VLLMConfig(
        model=output_path_of(llama_3_3_70b_instruct),
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        tpu_type="v5p-8",
        disk="10g",
        ram="256g",
    )

    spec_step = ExecutorStep(
        name=f"align/{name}/spec",
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
        name=f"align/{name}/prompts",
        description="Generate prompts from spec via refactored batched vLLM serve",
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
            local_serve_batch_size=versioned(local_serve_batch_size),
            ideation_workers=1,
            concretize_workers=1,
            extract_workers=1,
            understanding_max_tokens=versioned(1024),
            understanding_temperature=versioned(1.0),
            understanding_max_attempts=versioned(5),
            concretize_temperature=versioned(1.0),
            concretize_max_tokens=versioned(1024),
            extract_max_tokens=versioned(1024),
            concretize_max_attempts=versioned(5),
            statement_ids=versioned(statement_ids),
        ),
    )

    return [spec_step, prompts_step]


if __name__ == "__main__":
    args, executor_args = parse_args()
    sys.argv = [sys.argv[0], *executor_args]
    executor_main(
        steps=build_steps(
            name=args.name,
            local_serve_batch_size=args.local_serve_batch_size,
            statement_ids=args.statement_ids,
        ),
        description="Validate refactored prompt generation with batched vLLM serve on Llama 3.3 70B",
    )
