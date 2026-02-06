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
Experiment 57: Balanced Mixture Dataset Training.

Train Qwen models on a balanced mixture of 3 genomic datasets with equal weights (1/3 each):
- cds: coding sequences
- upstream: promoters
- downstream: downstream of CDS

https://github.com/Open-Athena/bolinas-dna/issues/57
"""

import dataclasses

from fray.cluster import ResourceConfig

from experiments.dna.defaults import (
    YOLO_RUN_CONFIG_V1,
    dna_qwen3_0_6b_256_v1,
    dna_qwen3_1_7b_256_v1,
    dna_qwen3_4b_256_v1,
    dna_tokenize_rw_v1,
    dna_train,
)
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# =============================================================================
# Datasets (256 context length, 128 stride)
# =============================================================================

DATASETS = {
    "cds": "bolinas-dna/genomes-v4-genome_set-mammals-intervals-v5_256_128",
    "upstream": "bolinas-dna/genomes-v4-genome_set-mammals-intervals-v1_256_128",
    "downstream": "bolinas-dna/genomes-v4-genome_set-mammals-intervals-v15_256_128",
}

# =============================================================================
# Model configs with learning rates and resources
# =============================================================================

TPU_V5P_16 = ResourceConfig.with_tpu("v5p-16")

MODEL_CONFIGS = {
    # (model_config, learning_rate, resources)
    # resources=None means use default (v5p-8)
    "qwen3_0_6b": (dna_qwen3_0_6b_256_v1, 1e-3, None),
    "qwen3_1_7b": (dna_qwen3_1_7b_256_v1, 8e-4, None),
    "qwen3_4b": (dna_qwen3_4b_256_v1, 6e-4, TPU_V5P_16),
}

# =============================================================================
# Tokenize each dataset
# =============================================================================


def dataset_name(dataset: str) -> str:
    """Extract dataset name from HuggingFace path (org/name -> name)."""
    return dataset.split("/")[-1]


tokenized_datasets = {
    region: dna_tokenize_rw_v1(
        name=f"{dataset_name(dataset)}-rw01",
        dataset=dataset,
    )
    for region, dataset in DATASETS.items()
}

# =============================================================================
# Create mixture config with equal weights (1/3 each)
# =============================================================================

mixture_config = lm_mixture_data_config(
    components=tokenized_datasets,
    weights={region: 1 / len(DATASETS) for region in DATASETS},
)

# =============================================================================
# Train models with doubled batch size for 256 context length
# =============================================================================

# Double batch size for 256 context to match tokens/batch with 512 context
train_config_256 = dataclasses.replace(
    YOLO_RUN_CONFIG_V1,
    train_batch_size=YOLO_RUN_CONFIG_V1.train_batch_size * 2,
)

training_steps = []
for model_name, (model_config, learning_rate, resources) in MODEL_CONFIGS.items():
    train_config = dataclasses.replace(train_config_256, learning_rate=learning_rate)
    if resources is not None:
        train_config = dataclasses.replace(train_config, resources=resources)

    train_step = dna_train(
        name=f"exp57-balanced-mixture-{model_name}-r01",
        tokenized=mixture_config,
        model_config=model_config,
        train_config=train_config,
        tags=["dna", "exp57", "balanced_mixture", model_name],
    )
    training_steps.append(train_step)

if __name__ == "__main__":
    executor_main(steps=training_steps)
