# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Experiment 65b: TraitGym eval smoke test with bolinas-dna/tokenizer-char.

Same as exp65 but uses the character-level tokenizer which adds BOS and EOS.
This exercises the EOS boundary fix in eval_harness._iterate_tokenized_requests.

The actual metrics are meaningless at 20 training steps — we just verify the
pipeline runs end-to-end without crashing and produces valid numbers.
"""

import dataclasses

from experiments.defaults import default_tokenize, default_train
from experiments.dna.defaults import (
    DNA_WINDOW_SIZE_BYTES_V1,
    SHORT_RUN_CONFIG_V1,
)
from experiments.llama import llama_50m
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2
from levanter.data.text import DNALmDatasetFormat
from marin.execution.executor import executor_main

# =============================================================================
# Config
# =============================================================================

TOKENIZER = "bolinas-dna/tokenizer-char"
DNA_SEQ_LEN = 256
# tokenizer-char adds BOS + EOS, so model context = DNA_SEQ_LEN + 2
MODEL_SEQ_LEN = DNA_SEQ_LEN + 2

DATASET = "bolinas-dna/genomes-v4-genome_set-humans-intervals-v1_256_128"

# =============================================================================
# Tokenize
# =============================================================================

tokenized = default_tokenize(
    name="genomes-v4-humans-256-tokenizer-char",
    dataset=DATASET,
    tokenizer=TOKENIZER,
    format=DNALmDatasetFormat(soft_mask_weight=0.01),
    window_size_bytes=DNA_WINDOW_SIZE_BYTES_V1,
)

# =============================================================================
# Train config: 20 steps total, eval every 10 steps
# =============================================================================

train_config = dataclasses.replace(
    SHORT_RUN_CONFIG_V1,
    num_train_steps=20,
    train_batch_size=32,
    steps_per_eval=10,
    steps_per_task_eval=10,
    steps_per_export=20,
)

# =============================================================================
# Training step with TraitGym Mendelian eval
# =============================================================================

dna_llama_50m = dataclasses.replace(llama_50m, max_seq_len=MODEL_SEQ_LEN)

train_step = default_train(
    name="exp65b-traitgym-eval-tokenizer-char",
    tokenized=tokenized,
    model_config=dna_llama_50m,
    train_config=train_config,
    tags=["dna", "exp65b", "traitgym", "smoke_test", "tokenizer-char"],
    eval_harness_tasks=[TRAITGYM_MENDELIAN_V2],
    eval_harness_max_packed_segments=1,
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[train_step])
