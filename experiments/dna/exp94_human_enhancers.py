# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Train DNA models on human enhancer datasets and evaluate with
functional/nonfunctional validation + TraitGym.

Each model is trained on a different interval version (v17 vs v18) of the
human genome dataset with repeat-masking (lowercase_weight=0.01). Validation
uses a single dataset tokenized twice: once scoring only functional (uppercase)
positions and once scoring only nonfunctional (lowercase) positions.

https://github.com/Open-Athena/bolinas-dna/issues/94

Targeting us-central2 (v4-8).
"""

from fray.v2 import ResourceConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config

from experiments.defaults import default_tokenize, default_train
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255
from experiments.simple_train_config import SimpleTrainConfig

# =============================================================================
# Constants
# =============================================================================

TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_SEQ_LEN = 255
MODEL_SEQ_LEN = dna_effective_seq_len(DNA_SEQ_LEN, TOKENIZER)  # 256 with BOS only
assert MODEL_SEQ_LEN == 256, f"Expected 256, got {MODEL_SEQ_LEN}"

TRAIN_DATASETS = {
    "v17": "bolinas-dna/genomes-v5-genome_set-humans-intervals-v17_255_128",
    "v18": "bolinas-dna/genomes-v5-genome_set-humans-intervals-v18_255_128",
}

# Single validation dataset with functional/nonfunctional uppercase/lowercase encoding
VAL_DATASET = "bolinas-dna/genomes-v5-validation-intervals-v18_255_255"

# =============================================================================
# Model config
# =============================================================================

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

RESOURCES = ResourceConfig.with_tpu("v4-8")

TRAIN_CONFIG = SimpleTrainConfig(
    resources=RESOURCES,
    train_batch_size=2048,
    num_train_steps=10_000,
    learning_rate=3e-3,
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.2,
    weight_decay=0.3,
    steps_per_eval=1000,
    steps_per_task_eval=1000,
    steps_per_export=10_000,
    data_seed=42,
)

# =============================================================================
# Tokenize training datasets (repeat-masking)
# =============================================================================

tokenized_train = {
    version: default_tokenize(
        name=f"{dataset.split('/')[-1]}-char-bos",
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=DNALmDatasetFormat(lowercase_weight=0.01),
    )
    for version, dataset in TRAIN_DATASETS.items()
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

training_steps = []
for version in TRAIN_DATASETS:
    data_config = lm_data_config(
        training_set=tokenized_train[version],
        validation_sets={
            "val_functional": val_functional,
            "val_nonfunctional": val_nonfunctional,
        },
    )

    train_step = default_train(
        name=f"exp94-human-enhancers-{version}-60m-v2",
        tokenized=data_config,
        model_config=qwen3_60m,
        train_config=TRAIN_CONFIG,
        tags=["dna", "exp94", "human_enhancers", version, "60m"],
        eval_harness_tasks=[TRAITGYM_MENDELIAN_V2_255],
        eval_harness_max_packed_segments=1,
        use_default_validation=False,
        wandb_group="exp94-human-enhancers",
    )
    training_steps.append(train_step)

if __name__ == "__main__":
    executor_main(steps=training_steps)
