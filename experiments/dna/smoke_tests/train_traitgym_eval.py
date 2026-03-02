# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Experiment 65: TraitGym eval smoke test during training.

Train a small DNA model (Llama 50M, 256 context) for 20 steps with TraitGym
Mendelian V2 evaluation every 10 steps.  The goal is to verify that the LLR
eval task integrates correctly into the training pipeline — the actual metrics
are meaningless at 20 steps.

Uses default_train directly (not dna_train) because dna_train hardcodes
eval_harness_tasks=[].
"""

import dataclasses

from experiments.defaults import default_train
from experiments.dna.defaults import (
    DNA_TOKENIZER_V1,
    SHORT_RUN_CONFIG_V1,
    dna_effective_seq_len,
    dna_tokenize_rw_v1,
)
from experiments.llama import llama_50m
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2
from marin.execution.executor import executor_main

# =============================================================================
# Dataset — reuse an existing promoter + mRNA dataset (256 context, 128 stride)
# =============================================================================

DATASET = "bolinas-dna/genomes-v4-genome_set-humans-intervals-v1_256_128"

tokenized = dna_tokenize_rw_v1(
    name="genomes-v4-humans-256-rw01",
    dataset=DATASET,
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

SEQ_LEN = 256
effective_seq_len = dna_effective_seq_len(SEQ_LEN, DNA_TOKENIZER_V1)
dna_llama_50m = dataclasses.replace(llama_50m, max_seq_len=effective_seq_len)

train_step = default_train(
    name="exp65-traitgym-eval-smoke-test",
    tokenized=tokenized,
    model_config=dna_llama_50m,
    train_config=train_config,
    tags=["dna", "exp65", "traitgym", "smoke_test"],
    eval_harness_tasks=[TRAITGYM_MENDELIAN_V2],
    eval_harness_max_packed_segments=1,
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[train_step])
