# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-stage NuminaMath SFT: CoT warm-start followed by TIR continuation.

This mirrors Numina's published curriculum at a high level without adding any
new infrastructure:
- Stage 1 trains on NuminaMath-CoT.
- Stage 2 continues from the Stage 1 HF export on NuminaMath-TIR.

We keep the datasets separate instead of mixing them because the TIR set is a
specialized continuation of the same problem family rather than an independent
corpus we want to upweight alongside CoT from step zero.
"""

import dataclasses
import math

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.evals import default_sft_eval
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of

NUMINAMATH_COT = "AI-MO/NuminaMath-CoT"
NUMINAMATH_TIR = "AI-MO/NuminaMath-TIR"

TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 4096
TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-128")
EVAL_RESOURCES = ResourceConfig.with_tpu("v4-8")

# Hugging Face row counts captured from the public dataset cards on 2026-04-21.
NUMINAMATH_COT_TRAIN_ROWS = 859_594
NUMINAMATH_TIR_TRAIN_ROWS = 72_540
COT_TARGET_EPOCHS = 3
TIR_TARGET_EPOCHS = 4

COT_NUM_TRAIN_STEPS = math.ceil(COT_TARGET_EPOCHS * NUMINAMATH_COT_TRAIN_ROWS / TRAIN_BATCH_SIZE)
TIR_NUM_TRAIN_STEPS = math.ceil(TIR_TARGET_EPOCHS * NUMINAMATH_TIR_TRAIN_ROWS / TRAIN_BATCH_SIZE)


def create_tokenization_step(dataset_name: str, short_name: str) -> ExecutorStep:
    """Create a chat-aware tokenization step for one instruction dataset."""
    dataset = get_instruction_dataset(dataset_name, splits=["train"])
    return default_tokenize(
        name=f"{short_name}_marin_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )


numinamath_cot_tokenized = create_tokenization_step(NUMINAMATH_COT, "numinamath_cot")
numinamath_tir_tokenized = create_tokenization_step(NUMINAMATH_TIR, "numinamath_tir")

base_sft_config = SimpleSFTConfig(
    train_batch_size=TRAIN_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    resources=TRAIN_RESOURCES,
    tokenizer=marin_tokenizer,
    max_seq_len=MAX_SEQ_LEN,
    lr_schedule="cosine",
    seed=0,
)

numinamath_cot_sft_config = dataclasses.replace(
    base_sft_config,
    num_train_steps=COT_NUM_TRAIN_STEPS,
    initialize_from_hf="marin-community/marin-8b-base",
    warmup=0.0,
    steps_per_eval=1000,
    steps_per_checkpoint=500,
    steps_per_hf_export=500,
)

marin_8b_numinamath_cot_sft = default_sft(
    name="marin_8b_numinamath_cot_sft",
    tokenized=numinamath_cot_tokenized,
    model_config=llama_8b,
    sft_config=numinamath_cot_sft_config,
    tags=["llama", "numina", "math", "cot"],
)

numinamath_tir_sft_config = dataclasses.replace(
    base_sft_config,
    num_train_steps=TIR_NUM_TRAIN_STEPS,
    initialize_from_hf=output_path_of(
        marin_8b_numinamath_cot_sft,
        f"hf/step-{COT_NUM_TRAIN_STEPS - 1}/",
    ),
    warmup=0.1,
    steps_per_eval=250,
    steps_per_checkpoint=250,
    steps_per_hf_export=250,
    seed=1,
)

marin_8b_numinamath_cot_then_tir_sft = default_sft(
    name="marin_8b_numinamath_cot_then_tir_sft",
    tokenized=numinamath_tir_tokenized,
    model_config=llama_8b,
    sft_config=numinamath_tir_sft_config,
    tags=["llama", "numina", "math", "tir"],
)

marin_8b_numinamath_cot_then_tir_evals = default_sft_eval(
    marin_8b_numinamath_cot_then_tir_sft,
    use_levanter_inference=True,
    resource_config=EVAL_RESOURCES,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            marin_8b_numinamath_cot_sft,
            marin_8b_numinamath_cot_then_tir_sft,
            *marin_8b_numinamath_cot_then_tir_evals,
        ]
    )
