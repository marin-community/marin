# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
EDA: Perplexity vs Downstream Task Performance.

Train models of 3 sizes (~6M, ~60M, ~600M) on 3 evolutionary timescales
(animals, mammals, primates) while tracking both LM loss and TraitGym
Mendelian VEP AUPRC during training. This produces 9 runs, each logging
both metrics to W&B, enabling correlation analysis.

https://github.com/Open-Athena/bolinas-dna/issues/8
"""

import dataclasses

from fray.v2 import ResourceConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

from experiments.defaults import default_tokenize, default_train
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2
from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main

# =============================================================================
# Constants
# =============================================================================

TOKENIZER = "bolinas-dna/tokenizer-char"
DNA_SEQ_LEN = 254
MODEL_SEQ_LEN = dna_effective_seq_len(DNA_SEQ_LEN, TOKENIZER)  # 256 with BOS+EOS

TIMESCALES = ["humans", "primates", "mammals"]
DATASETS = {ts: f"bolinas-dna/genomes-v4-genome_set-{ts}-intervals-v16_254_127-id0.3_cov0.3" for ts in TIMESCALES}

# =============================================================================
# Model configs
#
# Approximate param counts with vocab_size=8 (DNA char tokenizer):
#   6m:   ~8M   (hidden=256,  inter=896,  layers=8)
#   60m:  ~61M  (hidden=512,  inter=1792, layers=16)
#   600m: ~440M (qwen3_0_6b_hd128 architecture)
# =============================================================================

qwen3_6m = Qwen3Config(
    max_seq_len=MODEL_SEQ_LEN,
    hidden_dim=256,
    intermediate_dim=896,
    num_heads=4,
    num_kv_heads=4,
    num_layers=8,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

qwen3_60m = Qwen3Config(
    max_seq_len=MODEL_SEQ_LEN,
    hidden_dim=512,
    intermediate_dim=1792,
    num_heads=8,
    num_kv_heads=8,
    num_layers=16,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
)

qwen3_600m = dataclasses.replace(qwen3_0_6b_hd128, max_seq_len=MODEL_SEQ_LEN)

MODEL_CONFIGS = {
    # (model_config, learning_rate, resources)
    "6m": (qwen3_6m, 1e-3, ResourceConfig.with_tpu("v4-8")),
    "60m": (qwen3_60m, 1e-3, ResourceConfig.with_tpu("v4-8")),
    "600m": (qwen3_600m, 1e-3, ResourceConfig.with_tpu("v4-8")),
}

# =============================================================================
# Tokenize each timescale dataset
# =============================================================================

tokenized_datasets = {
    ts: default_tokenize(
        name=f"{dataset.split('/')[-1]}-char",
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=DNALmDatasetFormat(soft_mask_weight=0.01),
    )
    for ts, dataset in DATASETS.items()
}

# =============================================================================
# Train: 10K steps, 2K warmup, cosine decay, TraitGym eval every 2K steps
# =============================================================================

# Batch size doubled from 2048 to 4096 for 256 context length (matching
# tokens/batch with 512-context experiments).
BASE_TRAIN_CONFIG = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v4-8"),  # overridden per model
    train_batch_size=4096,
    num_train_steps=10_000,
    learning_rate=1e-3,  # overridden per model
    lr_schedule="cosine",
    warmup=0.2,  # 2K warmup steps
    decay=0.1,
    steps_per_eval=1000,
    steps_per_task_eval=1000,
    steps_per_export=10000,
    data_seed=42,
)

training_steps = []
for ts in TIMESCALES:
    for model_name, (model_config, lr, resources) in MODEL_CONFIGS.items():
        train_config = dataclasses.replace(
            BASE_TRAIN_CONFIG,
            resources=resources,
            learning_rate=lr,
        )
        train_step = default_train(
            name=f"eda-ppl-vs-downstream-{ts}-{model_name}",
            tokenized=tokenized_datasets[ts],
            model_config=model_config,
            train_config=train_config,
            tags=["dna", "eda", "perplexity_vs_downstream", ts, model_name],
            eval_harness_tasks=[TRAITGYM_MENDELIAN_V2],
            eval_harness_max_packed_segments=1,
            use_default_validation=False,
            wandb_group="eda-ppl-vs-downstream",
        )
        training_steps.append(train_step)

if __name__ == "__main__":
    executor_main(steps=training_steps)
