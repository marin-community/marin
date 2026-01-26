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
Mixture experiment for animal promoters and CDS regions.

This experiment trains on a mixture of promoter and CDS sequences with two
different weighting schemes to study the effect of data composition:

1. "equal": 50/50 weight split between promoters and CDS
   - Since CDS dataset is ~8.4x larger, CDS sequences will be seen less often
   - Each promoter sequence is seen ~8.4x more than each CDS sequence

2. "proportional": Weights proportional to dataset sizes
   - Each sequence from both datasets is seen approximately equally often
   - Natural sampling based on dataset sizes

https://github.com/Open-Athena/bolinas-dna/issues/13
"""

import dataclasses
import logging

from fray.cluster import ResourceConfig
from levanter.data.text import DNALmDatasetFormat

from experiments.defaults import default_tokenize, default_train
from experiments.qwen3 import qwen3_1_7b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

logger = logging.getLogger("ray")

RESOURCES = ResourceConfig.with_tpu("v5p-8")

# -----------------------------------------------------------------------------
# Dataset sizes (number of sequences)
# -----------------------------------------------------------------------------
PROMOTERS_SIZE = 22_588_618
CDS_SIZE = 189_627_296
# CDS is ~8.4x larger than promoters

# -----------------------------------------------------------------------------
# Weight configurations for controlled experiment
# -----------------------------------------------------------------------------
WEIGHT_CONFIGS = {
    "equal": {
        "promoters": 0.5,
        "cds": 0.5,
    },
    "proportional": {
        "promoters": PROMOTERS_SIZE,  # raw counts get normalized
        "cds": CDS_SIZE,
    },
}

# -----------------------------------------------------------------------------
# Experiment configuration (shared across runs)
# -----------------------------------------------------------------------------
run_number = 1
tokenizer_path = "songlab/tokenizer-dna-clm"
promoters_dataset_path = "gonzalobenegas/genomes-v3-genome_set-animals-intervals-v1_512_256"
cds_dataset_path = "gonzalobenegas/genomes-v3-genome_set-animals-intervals-v3_512_256"
dataset_seq_len = 512
learning_rate = 8e-4
train_batch_size = 2048
lr_schedule = "inv"
num_train_steps = 1_000_000
steps_per_export = 2000
steps_per_cycle = steps_per_export
steps_per_eval = steps_per_export
warmup = 0.5
decay = 0.1

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
model_config = dataclasses.replace(qwen3_1_7b, max_seq_len=dataset_seq_len)

# -----------------------------------------------------------------------------
# Dataset configurations
# -----------------------------------------------------------------------------
DATASETS = {
    "promoters": {
        "name": "animal-promoters-repeat-weight-0.01",
        "path": promoters_dataset_path,
    },
    "cds": {
        "name": "animal-cds-repeat-weight-0.01",
        "path": cds_dataset_path,
    },
}

# -----------------------------------------------------------------------------
# Tokenize each dataset (shared across runs)
# -----------------------------------------------------------------------------
tokenized_datasets = {
    key: default_tokenize(
        name=cfg["name"],
        dataset=cfg["path"],
        tokenizer=tokenizer_path,
        format=DNALmDatasetFormat(soft_mask_weight=0.01),
        window_size_bytes=50_000_000,
    )
    for key, cfg in DATASETS.items()
}

# -----------------------------------------------------------------------------
# Training configuration (shared across runs)
# -----------------------------------------------------------------------------
train_config = SimpleTrainConfig(
    resources=RESOURCES,
    train_batch_size=train_batch_size,
    learning_rate=learning_rate,
    lr_schedule=lr_schedule,
    warmup=warmup,
    decay=decay,
    cycle_length=steps_per_cycle,
    steps_per_eval=steps_per_eval,
    num_train_steps=num_train_steps,
    steps_per_export=steps_per_export,
    data_seed=42,
)

# -----------------------------------------------------------------------------
# Create training steps for each weight configuration
# -----------------------------------------------------------------------------
training_steps = []
for config_name, weights in WEIGHT_CONFIGS.items():
    mixture_config = lm_mixture_data_config(
        components=tokenized_datasets,
        weights=weights,
    )

    training_step = default_train(
        name=f"promoter-cds-mixture-{config_name}-r{run_number:02d}",
        tokenized=mixture_config,
        model_config=model_config,
        train_config=train_config,
        tags=["dna", "mixture", config_name],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    training_steps.append(training_step)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(steps=[*training_steps])
