# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Align Llama 3.1 8B to the OpenAI Model Spec using synthetic preference data.

This experiment runs the full alignment pipeline:
  1. Generate diverse eval prompts from 46 behavioral statements (3-stage Bloom pipeline)
  2. Generate chosen responses via GPT-4.1 (with spec guidance)
  3. Generate rejected responses via GPT-4.1-mini (without spec guidance)
  4. Judge + filter into preference pairs
  5. Tokenize preference pairs
  6. DPO training on the synthetic preference dataset

The synthetic dataset is cached as ExecutorSteps, so subsequent runs skip
completed stages and reuse the generated data.

Based on the Bloom v2 pipeline (bloom/configs/openai_spec_synthetic.yaml) which
produced 30k+ preference pairs from 46 statements and achieved +1.7 adherence
improvement via DPO (beta=0.01).
"""

from pathlib import Path

from experiments.llama import llama_8b
from experiments.models import llama_3_1_8b
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.v2.types import ResourceConfig
from marin.alignment.align import AlignConfig, align
from marin.execution.executor import executor_main, output_path_of

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
    # Tokenizer must match the model being aligned
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    # Resources
    cpu_resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
    inference_resources=ResourceConfig.with_tpu("v6e-8"),
)

# ---------------------------------------------------------------------------
# DPO config — matches validated hyperparameters from CS229 project
# ---------------------------------------------------------------------------

dpo_config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-32"),
    train_batch_size=128,
    num_train_steps=5000,
    learning_rate=5e-7,
    lr_schedule="cosine",
    warmup=0.1,
    cooldown=None,
    wandb_project="alignment",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    model_name_or_path=output_path_of(llama_3_1_8b),
    reference_model_path=output_path_of(llama_3_1_8b),
    reference_is_hf=True,
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.01,
    validation_split_fraction=0.1,
    steps_per_eval=200,
    steps_per_checkpoint=1000,
    steps_per_hf_export=1000,
    seed=0,
)

# ---------------------------------------------------------------------------
# Pipeline: spec → synthetic preference data → DPO
# ---------------------------------------------------------------------------

aligned_steps = align(
    name="openai_spec_llama3_8b",
    pretrained_model=llama_3_1_8b,
    spec=SPEC_PATH,
    model_config=llama_8b,
    teacher_model="openai/gpt-4.1",
    align_config=align_config,
    dpo_config=dpo_config,
    rejected_model="openai/gpt-4.1-mini",
    tags=["alignment", "openai-spec", "llama3", "dpo", "synthetic-preference"],
)

# ---------------------------------------------------------------------------
# Executor: run the full pipeline
#
# Steps (each is cached — rerun skips completed stages):
#   llama_3_1_8b          → download pretrained model from HF
#   align/*/prompts       → generate 10-23K eval prompts from 46 statements
#   align/*/chosen        → GPT-4.1 generates chosen responses (with spec)
#   align/*/rejected      → GPT-4.1-mini generates rejected responses (no spec)
#   align/*/preference_pairs → judge, filter, build 6-18K preference pairs
#   tokenized/*           → tokenize preference pairs for DPO
#   checkpoints/*         → DPO training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(
        steps=[llama_3_1_8b, *aligned_steps],
        description="Align Llama 3.1 8B to OpenAI Model Spec via synthetic preference data + DPO",
    )
