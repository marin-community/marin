# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared constants, configs, and helper functions for DNA experiments.
"""

import dataclasses
from collections.abc import Sequence

from fray.cluster import ResourceConfig
from levanter.data.text import DNALmDatasetFormat, TextLmDatasetFormat

from experiments.defaults import default_tokenize, default_train
from experiments.qwen3 import qwen3_0_6b_hd128, qwen3_1_7b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep

# =============================================================================
# Shared constants (V1 = first generation of DNA experiments)
# =============================================================================

DNA_RESOURCES_V1 = ResourceConfig.with_tpu("v5p-8")
DNA_TOKENIZER_V1 = "songlab/tokenizer-dna-clm"
DNA_SEQ_LEN_V1 = 512
DNA_WINDOW_SIZE_BYTES_V1 = 50_000_000

# =============================================================================
# Dataset paths (V1)
# =============================================================================

PROMOTERS_DATASET_V1 = "gonzalobenegas/genomes-v3-genome_set-animals-intervals-v1_512_256"
MRNA_PLUS_PROMOTERS_DATASET_V1 = "gonzalobenegas/genomes-v3-genome_set-animals-intervals-v2_512_256"
CDS_DATASET_V1 = "gonzalobenegas/genomes-v3-genome_set-animals-intervals-v3_512_256"

# =============================================================================
# Model configs with DNA seq len (V1)
# =============================================================================

dna_qwen3_0_6b_v1 = dataclasses.replace(qwen3_0_6b_hd128, max_seq_len=DNA_SEQ_LEN_V1)
dna_qwen3_1_7b_v1 = dataclasses.replace(qwen3_1_7b, max_seq_len=DNA_SEQ_LEN_V1)

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


# =============================================================================
# Helper functions (V1)
# =============================================================================


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
