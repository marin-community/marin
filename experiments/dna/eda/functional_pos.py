# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
EDA: Functional Position Validation Metric.

Train models of 3 sizes (~6M, ~60M, ~600M) on datasets from different
evolutionary timescales (mammals, primates, vertebrates) while tracking
LL(functional) - LL(nonfunctional) as a validation metric. Functional vs
nonfunctional positions are determined by conservation scores, encoded as
uppercase (functional) / lowercase (nonfunctional) in the validation dataset.

Each model gets two validation sets tokenized from the same HF dataset:
- val_functional: only uppercase positions contribute to loss
- val_nonfunctional: only lowercase positions contribute to loss

https://github.com/Open-Athena/bolinas-dna/issues/10

Targeting us-central1 (v5p-8).
"""

import dataclasses

from fray.v2 import ResourceConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config

from experiments.defaults import default_tokenize, default_train
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255
from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.simple_train_config import SimpleTrainConfig

# =============================================================================
# Constants
# =============================================================================

TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_SEQ_LEN = 255
MODEL_SEQ_LEN = dna_effective_seq_len(DNA_SEQ_LEN, TOKENIZER)  # 256 with BOS only
assert MODEL_SEQ_LEN == 256, f"Expected 256, got {MODEL_SEQ_LEN}"

TIMESCALES = ["primates", "mammals", "vertebrates"]

TRAIN_DATASETS = {ts: f"bolinas-dna/genomes-v5-genome_set-{ts}-intervals-v1_255_128" for ts in TIMESCALES}

# Single validation dataset with conservation-based uppercase/lowercase encoding
VAL_DATASET = "bolinas-dna/genomes-v5-validation-intervals-v1_255_255"

# =============================================================================
# Model configs (same as perplexity_vs_downstream)
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

RESOURCES = ResourceConfig.with_tpu("v5p-8")
LEARNING_RATE = 1e-3

MODEL_CONFIGS = {
    "6m": qwen3_6m,
    "60m": qwen3_60m,
    "600m": qwen3_600m,
}

# =============================================================================
# Tokenize training datasets (repeat-masking as usual)
# =============================================================================

tokenized_train = {
    ts: default_tokenize(
        name=f"{dataset.split('/')[-1]}-char-bos",
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=DNALmDatasetFormat(lowercase_weight=0.01),
    )
    for ts, dataset in TRAIN_DATASETS.items()
}

# =============================================================================
# Tokenize validation dataset twice: functional-only and nonfunctional-only
# =============================================================================

val_functional = default_tokenize(
    name=f"{VAL_DATASET.split('/')[-1]}-char-bos-functional",
    dataset=VAL_DATASET,
    tokenizer=TOKENIZER,
    format=DNALmDatasetFormat(uppercase_weight=1.0, lowercase_weight=0.0),
)

val_nonfunctional = default_tokenize(
    name=f"{VAL_DATASET.split('/')[-1]}-char-bos-nonfunctional",
    dataset=VAL_DATASET,
    tokenizer=TOKENIZER,
    format=DNALmDatasetFormat(uppercase_weight=0.0, lowercase_weight=1.0),
)

# =============================================================================
# Train: 10K steps, cosine decay, TraitGym eval every 1K steps
# =============================================================================

TRAIN_CONFIG = SimpleTrainConfig(
    resources=RESOURCES,
    train_batch_size=4096,
    num_train_steps=10_000,
    learning_rate=LEARNING_RATE,
    lr_schedule="cosine",
    warmup=0.2,
    decay=0.1,
    steps_per_eval=1000,
    steps_per_task_eval=1000,
    steps_per_export=10000,
    data_seed=42,
)

training_steps = []
for ts in TIMESCALES:
    for model_name, model_config in MODEL_CONFIGS.items():

        # Wire up training data with two validation sets
        data_config = lm_data_config(
            training_set=tokenized_train[ts],
            validation_sets={
                "val_functional": val_functional,
                "val_nonfunctional": val_nonfunctional,
            },
        )

        train_step = default_train(
            name=f"eda-functional-pos-{ts}-{model_name}",
            tokenized=data_config,
            model_config=model_config,
            train_config=TRAIN_CONFIG,
            tags=["dna", "eda", "functional_pos", ts, model_name],
            eval_harness_tasks=[TRAITGYM_MENDELIAN_V2_255],
            eval_harness_max_packed_segments=1,
            use_default_validation=False,
            wandb_group="eda-functional-pos",
        )
        training_steps.append(train_step)

if __name__ == "__main__":
    executor_main(steps=training_steps)
