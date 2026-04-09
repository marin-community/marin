# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-statement end-to-end alignment pipeline with GPT-4.1 + Mixtral rejected.

Exercises the full pipeline on `ask_clarifying_questions` only:
  - prompt generation (Stage 1/2/3) on GPT-4.1
  - chosen responses on GPT-4.1
  - rejected responses on Mixtral-8x7B-Instruct (opposite prompting)
  - judge on GPT-4.1
  - stops at preference pairs (no DPO training)

This is the canonical API-backed one-statement E2E path alongside the local
GPT-OSS experiments under `experiments/posttrain/`.

Submit to Iris:

    uv run iris --controller-url http://127.0.0.1:10000 job run \\
        --no-wait \\
        --job-name gpt-4-1-e2e-one-statement \\
        --cpu 4 --memory 16GB --disk 10GB \\
        --region us-central1 \\
        -- python experiments/posttrain/align_gpt_4_1_e2e_one_statement.py
"""

import argparse
from pathlib import Path
import sys

from gpt_oss_tpu import RejectedModelPreset, rejected_model_label, rejected_model_tag, rejected_model_vllm_config
from experiments.models import llama_3_1_8b
from marin.alignment.align import AlignConfig, ResponseExecutionMode, align
from marin.alignment.generate_responses import RejectedPromptStrategy
from marin.alignment.inference_config import OpenAIConfig
from marin.execution.executor import executor_main

SPEC_PATH = str(Path(__file__).parent / "specs" / "openai_model_spec.jsonl")
DEFAULT_PIPELINE_NAME = "gpt_4_1_e2e_one_statement"
GPT_4_1 = OpenAIConfig(model="gpt-4.1", workers=16)
REJECTED_MODEL_PRESET = RejectedModelPreset.MIXTRAL

rejected_vllm = rejected_model_vllm_config(REJECTED_MODEL_PRESET)

align_config = AlignConfig(
    ideation_model=GPT_4_1,
    extract_model=GPT_4_1,
    judge_model=GPT_4_1,
    covering_strength=2,
    covering_seed=42,
    ideation_workers=16,
    concretize_workers=16,
    extract_workers=16,
    prompt_batch_size=256,
    response_batch_size=256,
    understanding_max_tokens=2048,
    understanding_temperature=1.0,
    understanding_max_attempts=5,
    concretize_max_tokens=1024,
    concretize_temperature=1.0,
    concretize_max_attempts=5,
    extract_max_tokens=1024,
    judge_workers=16,
    judge_batch_size=256,
    teacher_n=1,
    teacher_temperature=0.7,
    teacher_max_tokens=2048,
    rejected_n=1,
    rejected_temperature=0.7,
    rejected_max_tokens=2048,
    rejected_prompt_strategy=RejectedPromptStrategy.OPPOSITE,
    judge_min_chosen_score=6.0,
    judge_min_gap=1.0,
    response_execution_mode=ResponseExecutionMode.AUTO,
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    statement_ids=["ask_clarifying_questions"],
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_PIPELINE_NAME, help="Alignment experiment name / output prefix")
    return parser.parse_known_args()


def build_steps(name: str):
    return align(
        name=name,
        pretrained_model=llama_3_1_8b,
        spec=SPEC_PATH,
        model_config=None,
        teacher_model=GPT_4_1,
        align_config=align_config,
        dpo_config=None,
        rejected_model=rejected_vllm,
        tags=[
            "api",
            "gpt-4.1",
            rejected_model_tag(REJECTED_MODEL_PRESET),
            "opposite-mode",
            "one-statement",
            "e2e",
        ],
    )


if __name__ == "__main__":
    args, executor_args = parse_args()
    sys.argv = [sys.argv[0], *executor_args]
    executor_main(
        steps=build_steps(args.name),
        description=(
            "Single-statement E2E alignment pipeline: GPT-4.1 for prompts/chosen/judge, "
            f"{rejected_model_label(REJECTED_MODEL_PRESET)} opposite-mode rejected, "
            "on ask_clarifying_questions only"
        ),
    )
