# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test: generate synthetic preference data for a single spec statement.

Validates the full data generation pipeline end-to-end on one statement
("ask_clarifying_questions") with cheap settings:
  - pairwise covering (t=2) instead of 3-way → ~50-100 prompts
  - one OpenAI model ID reused for teacher, rejected, judge, and ideation
  - 1 rejected response per prompt instead of 4
  - low parallelism (works on a laptop)

No training — just data gen. Run locally:

    OPENAI_API_KEY=sk-... python experiments/align_openai_spec_smoke.py --model gpt-4.1-mini

Output: sharded JSONL.GZ preference pairs ready for DPO.
"""

import argparse
from pathlib import Path
import sys

from experiments.llama import llama_8b
from experiments.models import llama_3_1_8b
from fray.v2.types import ResourceConfig
from marin.alignment.align import AlignConfig, align
from marin.alignment.inference_config import OpenAIConfig
from marin.execution.executor import executor_main

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")
DEFAULT_MODEL = "gpt-4.1-mini"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model ID to use for all pipeline roles")
    parser.add_argument("--name", default="openai_spec_smoke", help="Alignment experiment name / output prefix")
    return parser.parse_known_args()


def build_steps(model: str, name: str):
    align_config = AlignConfig(
        ideation_model=model,
        extract_model=model,
        judge_model=model,
        covering_strength=2,
        covering_seed=42,
        ideation_workers=4,
        concretize_workers=4,
        extract_workers=8,
        judge_workers=8,
        teacher_n=1,
        teacher_temperature=0.7,
        teacher_max_tokens=1024,
        rejected_n=1,
        rejected_temperature=0.7,
        rejected_max_tokens=1024,
        judge_min_chosen_score=6.0,
        judge_min_gap=1.0,
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        cpu_resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
        statement_ids=["ask_clarifying_questions"],
    )
    teacher = OpenAIConfig(model=model, workers=8)
    rejected = OpenAIConfig(model=model, workers=8)
    return align(
        name=name,
        pretrained_model=llama_3_1_8b,
        spec=SPEC_PATH,
        model_config=llama_8b,
        teacher_model=teacher,
        align_config=align_config,
        dpo_config=None,
        rejected_model=rejected,
        tags=["alignment", "smoke-test"],
    )


if __name__ == "__main__":
    args, executor_args = parse_args()
    sys.argv = [sys.argv[0], *executor_args]
    executor_main(
        steps=build_steps(args.model, args.name),
        description=f"Smoke test: synthetic preference data for ask_clarifying_questions with {args.model}",
    )
