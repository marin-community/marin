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
YOLO experiment for animal promoters with loss downweighting on repetitive DNA elements.

See exp21_promoters_yolo_standard.py for the baseline without downweighting.

https://github.com/Open-Athena/bolinas-dna/issues/21
"""

import dataclasses
import logging
from fray.cluster import ResourceConfig
from levanter.data.text import DNALmDatasetFormat
from experiments.qwen3 import qwen3_1_7b
from marin.execution.executor import executor_main
from experiments.defaults import default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger("ray")

RESOURCES = ResourceConfig.with_tpu("v5p-8")

# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
run_number = 1
tokenizer_path = "songlab/tokenizer-dna-clm"
dataset_path = "gonzalobenegas/genomes-v3-genome_set-animals-intervals-v1_512_256"
dataset_seq_len = 512  # constant for all sequences in dataset
learning_rate = 8e-4
train_batch_size = 2048
lr_schedule = "inv"
num_train_steps = 1_000_000
steps_per_export = 2000
steps_per_cycle = steps_per_export
steps_per_eval = steps_per_export
warmup = 0.5  # fraction of cycle
decay = 0.1

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
model_config = dataclasses.replace(qwen3_1_7b, max_seq_len=dataset_seq_len)

# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
data_tokenized = default_tokenize(
    name="animal-promoters-repeat-weight-0.01",
    dataset=dataset_path,
    tokenizer=tokenizer_path,
    format=DNALmDatasetFormat(soft_mask_weight=0.01),
    # my thoughts (should check):
    # max parallelism is number of shards in HF dataset
    # window_size_bytes should be smaller than shard size to achieve max parallelism
    window_size_bytes=50_000_000,
)

# -----------------------------------------------------------------------------
# Training configuration
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

training_step = default_train(
    name=f"animal-promoters-yolo-r{run_number:02d}",
    tokenized=data_tokenized,
    model_config=model_config,
    train_config=train_config,
    tags=["dna", "animal-promoters", "yolo"],
    eval_harness_tasks=[],
    use_default_validation=False,
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🧬 DNA Training Experiment")
    logger.info("=" * 64)
    logger.info(f"Model:              {model_config}")
    logger.info(f"Learning rate:      {learning_rate}")
    logger.info(f"Global batch size:  {train_batch_size}")
    logger.info(f"Training steps:     {num_train_steps:,}")
    logger.info(f"Steps per export:   {steps_per_export:,}")
    logger.info(f"Steps per eval:     {steps_per_eval:,}")
    logger.info("=" * 64)

    executor_main(steps=[training_step])
