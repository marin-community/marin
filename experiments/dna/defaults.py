# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Shared constants, configs, and helper functions for DNA experiments.
"""

from collections.abc import Sequence

from fray.v2 import ResourceConfig
from levanter.data.text import DNALmDatasetFormat, TextLmDatasetFormat

from experiments.defaults import default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep

# =============================================================================
# Shared constants (V1 = first generation of DNA experiments)
# =============================================================================

DNA_RESOURCES_V1 = ResourceConfig.with_tpu("v5p-8")
DNA_TOKENIZER_V1 = "songlab/tokenizer-dna-clm"
DNA_WINDOW_SIZE_BYTES_V1 = 50_000_000

# =============================================================================
# Dataset paths (V1)
# =============================================================================

PROMOTERS_DATASET_V1 = "gonzalobenegas/genomes-v3-genome_set-animals-intervals-v1_512_256"
MRNA_PLUS_PROMOTERS_DATASET_V1 = "gonzalobenegas/genomes-v3-genome_set-animals-intervals-v2_512_256"
CDS_DATASET_V1 = "gonzalobenegas/genomes-v3-genome_set-animals-intervals-v3_512_256"
PROMOTERS_MRNA_DATASET_V1 = "bolinas-dna/genomes-v4-genome_set-animals-intervals-v1_512_256"
PROMOTERS_MRNA_256_DATASET_V1 = "bolinas-dna/genomes-v4-genome_set-animals-intervals-v1_256_128"
PROMOTERS_MRNA_NCRNA_DATASET_V1 = "bolinas-dna/genomes-v4-genome_set-animals-intervals-v4_512_256"

# =============================================================================
# Preset train configs (V1)
# =============================================================================

SHORT_RUN_CONFIG_V1 = SimpleTrainConfig(
    resources=DNA_RESOURCES_V1,
    train_batch_size=2048,
    learning_rate=1e-3,
    lr_schedule="inv",
    warmup=0.5,
    decay=0.1,
    cycle_length=2000,
    steps_per_eval=2000,
    num_train_steps=20_000,
    steps_per_export=2000,
    data_seed=42,
)

YOLO_RUN_CONFIG_V1 = SimpleTrainConfig(
    resources=DNA_RESOURCES_V1,
    train_batch_size=2048,
    learning_rate=8e-4,
    lr_schedule="inv",
    warmup=0.5,
    decay=0.1,
    cycle_length=2000,
    steps_per_eval=2000,
    num_train_steps=1_000_000,
    steps_per_export=2000,
    data_seed=42,
)

FAST_RUN_CONFIG_V1 = SimpleTrainConfig(
    resources=DNA_RESOURCES_V1,
    train_batch_size=2048,
    learning_rate=1e-3,
    lr_schedule="inv",
    warmup=0.9,
    decay=0.1,
    cycle_length=1000,
    steps_per_eval=1000,
    num_train_steps=10_000,
    steps_per_export=1000,
    data_seed=42,
)


# =============================================================================
# Helper functions
# =============================================================================


def dna_effective_seq_len(base_seq_len: int, tokenizer_name: str) -> int:
    """Compute model context size = base DNA sequence length + special tokens (BOS/EOS).

    Loads the tokenizer to detect which special tokens are defined, so the model
    ``max_seq_len`` stays in sync automatically.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    n_special = int(tok.bos_token_id is not None) + int(tok.eos_token_id is not None)
    return base_seq_len + n_special


def dna_tokenize_std_v1(name: str, dataset: str) -> ExecutorStep:
    """Standard tokenization (no repeat weighting) using V1 defaults."""
    return default_tokenize(
        name=name,
        dataset=dataset,
        tokenizer=DNA_TOKENIZER_V1,
        format=TextLmDatasetFormat(text_key="seq"),
        window_size_bytes=DNA_WINDOW_SIZE_BYTES_V1,
    )


def dna_tokenize_rw_v1(name: str, dataset: str, soft_mask_weight: float = 0.01) -> ExecutorStep:
    """Repeat-weighted tokenization using V1 defaults."""
    return default_tokenize(
        name=name,
        dataset=dataset,
        tokenizer=DNA_TOKENIZER_V1,
        format=DNALmDatasetFormat(soft_mask_weight=soft_mask_weight),
        window_size_bytes=DNA_WINDOW_SIZE_BYTES_V1,
    )


def dna_train(
    name: str,
    tokenized: ExecutorStep,
    model_config,
    train_config: SimpleTrainConfig,
    tags: Sequence[str],
) -> ExecutorStep:
    """Train a DNA model."""
    return default_train(
        name=name,
        tokenized=tokenized,
        model_config=model_config,
        train_config=train_config,
        tags=tags,
        eval_harness_tasks=[],
        use_default_validation=False,
    )
