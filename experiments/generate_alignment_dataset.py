# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Generate a synthetic preference dataset from the OpenAI Model Spec.

This is the data-generation-only half of the alignment pipeline. It runs
entirely on CPU via API calls (GPT-4.1 / GPT-4.1-mini) and can be executed
locally — no cluster or TPU needed:

    python experiments/generate_alignment_dataset.py

Steps:
  1. Generate diverse eval prompts from 46 behavioral statements
  2. Generate chosen responses via GPT-4.1 (with spec guidance)
  3. Generate rejected responses via GPT-4.1-mini (without spec guidance)
  4. Judge + filter into preference pairs (sharded JSONL.GZ)

The output is a cached preference dataset that can later be tokenized and
used for DPO training on a cluster (see experiments/align_openai_spec.py).
"""

from pathlib import Path

from fray.v2.types import ResourceConfig
from marin.alignment.align import AlignConfig, align
from marin.alignment.inference_config import LiteLLMConfig
from marin.execution.executor import executor_main

# ---------------------------------------------------------------------------
# Spec path — 46 behavioral statements from the OpenAI Model Spec
# ---------------------------------------------------------------------------

SPEC_PATH = str(Path(__file__).parent / "posttrain" / "specs" / "openai_model_spec.jsonl")

# ---------------------------------------------------------------------------
# Alignment config — mirrors bloom/configs/openai_spec_synthetic.yaml
# ---------------------------------------------------------------------------

align_config = AlignConfig(
    # Prompt generation (Bloom stages 1-3)
    ideation_model="openai/gpt-4.1",
    extract_model="openai/gpt-4.1-mini",
    covering_strength=3,
    covering_seed=42,
    ideation_workers=32,
    concretize_workers=32,
    extract_workers=128,
    # Teacher: strong model generates chosen responses WITH spec guidance
    teacher_n=1,
    teacher_temperature=0.7,
    teacher_max_tokens=2048,
    # Rejected: weaker model generates responses WITHOUT spec guidance
    rejected_n=4,
    rejected_temperature=0.7,
    rejected_max_tokens=2048,
    # Judging: filter by quality
    judge_model="openai/gpt-4.1",
    judge_min_chosen_score=7.0,
    judge_min_gap=2.0,
    # Tokenizer (for downstream DPO — not used in this script)
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    # Resources — CPU only, everything is API calls
    cpu_resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
)

# ---------------------------------------------------------------------------
# Inference configs — API models only, runs on your laptop
# ---------------------------------------------------------------------------

teacher = LiteLLMConfig(model="openai/gpt-4.1", workers=64)
rejected = LiteLLMConfig(model="openai/gpt-4.1-mini", workers=64)

# ---------------------------------------------------------------------------
# Pipeline: spec → prompts → responses → preference pairs
# No pretrained_model needed (data gen only), using a dummy ExecutorStep.
# No dpo_config → align() returns only data generation steps.
# ---------------------------------------------------------------------------

# We need a pretrained_model ExecutorStep for align()'s signature, but it's
# not used when dpo_config=None (data gen only). Import a real one so the
# executor graph is valid even if we never run the download.
from experiments.models import llama_3_1_8b  # noqa: E402
from experiments.llama import llama_8b  # noqa: E402

dataset_steps = align(
    name="openai_spec_llama3_8b",
    pretrained_model=llama_3_1_8b,
    spec=SPEC_PATH,
    model_config=llama_8b,
    teacher_model=teacher,
    align_config=align_config,
    dpo_config=None,  # data generation only — no tokenize, no DPO
    rejected_model=rejected,
    tags=["alignment", "openai-spec", "synthetic-preference"],
)

# ---------------------------------------------------------------------------
# Run locally — Fray auto-falls back to LocalClient (threads on your machine)
#
# Output steps:
#   align/*/prompts          → 10-23K eval prompts from 46 statements
#   align/*/chosen           → GPT-4.1 chosen responses (with spec guidance)
#   align/*/rejected         → GPT-4.1-mini rejected responses (no spec)
#   align/*/preference_pairs → 6-18K judged, filtered preference pairs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(
        steps=dataset_steps,
        description="Generate synthetic preference dataset from OpenAI Model Spec (CPU/API only)",
    )
