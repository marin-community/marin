# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test: generate synthetic preference data for a single spec statement.

Validates the full data generation pipeline end-to-end on one statement
("ask_clarifying_questions") with cheap settings:
  - pairwise covering (t=2) instead of 3-way → ~50-100 prompts
  - gpt-4.1-mini for everything (teacher, rejected, judge, ideation)
  - 1 rejected response per prompt instead of 4
  - low parallelism (works on a laptop)

No training — just data gen. Run locally:

    OPENAI_API_KEY=sk-... python experiments/align_openai_spec_smoke.py

Expected cost: ~$2-5 in API calls. Expected time: ~5-10 minutes.
Output: sharded JSONL.GZ preference pairs ready for DPO.
"""

from pathlib import Path

from experiments.llama import llama_8b
from experiments.models import llama_3_1_8b
from fray.v2.types import ResourceConfig
from marin.alignment.align import AlignConfig, align
from marin.alignment.inference_config import LiteLLMConfig
from marin.execution.executor import executor_main

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")

# ---------------------------------------------------------------------------
# Cheap config for single-statement smoke test
# ---------------------------------------------------------------------------

align_config = AlignConfig(
    # Use mini for everything — cheap
    ideation_model="openai/gpt-4.1-mini",
    extract_model="openai/gpt-4.1-mini",
    judge_model="openai/gpt-4.1-mini",
    # Pairwise covering — fewer prompts than 3-way
    covering_strength=2,
    covering_seed=42,
    # Low parallelism — laptop-friendly
    ideation_workers=4,
    concretize_workers=4,
    extract_workers=8,
    judge_workers=8,
    # Teacher: 1 response is enough for smoke test
    teacher_n=1,
    teacher_temperature=0.7,
    teacher_max_tokens=1024,
    # Rejected: 1 response (skip the "pick worst of N" logic for speed)
    rejected_n=1,
    rejected_temperature=0.7,
    rejected_max_tokens=1024,
    # Judging: keep thresholds but they'll be looser with mini as judge
    judge_min_chosen_score=6.0,
    judge_min_gap=1.0,
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    cpu_resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
    # Single statement only
    statement_ids=["ask_clarifying_questions"],
)

# ---------------------------------------------------------------------------
# Inference: gpt-4.1-mini for both teacher and rejected
# ---------------------------------------------------------------------------

teacher = LiteLLMConfig(model="openai/gpt-4.1-mini", workers=8)
rejected = LiteLLMConfig(model="openai/gpt-4.1-mini", workers=8)

# ---------------------------------------------------------------------------
# Data gen only — no DPO
# ---------------------------------------------------------------------------

dataset_steps = align(
    name="openai_spec_smoke",
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
    executor_main(
        steps=dataset_steps,
        description="Smoke test: synthetic preference data for ask_clarifying_questions",
    )
