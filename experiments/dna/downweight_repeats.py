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
Example training experiment with repeat downweighting.
Adapted from https://github.com/marin-community/marin/blob/0d38a98b63a8566451d3cb9a2fb488ec0d1647ce/experiments/plantcad/exp1729_plantcad_train.py
"""

import jax
import logging
from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from experiments.defaults import default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger("ray")

if (backend := jax.default_backend()) not in {"gpu", "cpu"}:
    raise NotImplementedError(f"Only GPU and CPU backends supported, not {backend=}")

# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
run_number = 1
num_gpus = len(jax.devices("gpu")) if backend == "gpu" else 1
tokenizer_path = "songlab/tokenizer-dna-clm"
dataset_path = "songlab/gpn-animal-promoter-dataset"
# dataset_examples = 9_057_253
learning_rate = 1e-3
per_device_eval_parallelism = 16
train_batch_size = per_device_eval_parallelism * num_gpus
num_train_steps = 1000
steps_per_export = 100
steps_per_cycle = 500
steps_per_eval = 100

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
model_config = LlamaConfig(
    max_seq_len=512,
    hidden_dim=128,
    intermediate_dim=512,
    num_heads=4,
    num_kv_heads=4,
    num_layers=4,
)

# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
data_tokenized = default_tokenize(
    name="gpn-animal-promoter",
    # versioned(dataset_path) was causing issues:
    # ValueError: No valid jsonl or parquet files found in
    # ['songlab/gpn-animal-promoter-dataset']. Please provide a path to a
    # directory containing jsonl or parquet files.
    dataset=dataset_path,
    tokenizer=tokenizer_path,
    # DNA sequences are in `seq`, not `text`
    format=TextLmDatasetFormat(text_key="seq"),
    sample_count=10,  # per shard
)

# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------
train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_gpu("T4", count=num_gpus),
    train_batch_size=train_batch_size,
    per_device_eval_parallelism=per_device_eval_parallelism,
    learning_rate=learning_rate,
    lr_schedule="inv",
    warmup=0.05,
    decay=0.1,
    cycle_length=steps_per_cycle,
    steps_per_eval=steps_per_eval,
    num_train_steps=num_train_steps,
    steps_per_export=steps_per_export,
    data_seed=42,
)

training_step = default_train(
    name=f"gpn-animal-promoter-r{run_number:02d}",
    tokenized=data_tokenized,
    model_config=model_config,
    train_config=train_config,
    tags=["dna", "gpn", "training"],
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
    logger.info(f"Micro batch size:   {per_device_eval_parallelism}")
    logger.info(f"Training steps:     {num_train_steps:,}")
    logger.info(f"Steps per export:   {steps_per_export:,}")
    logger.info(f"Steps per eval:     {steps_per_eval:,}")
    logger.info(f"Backend:            {backend}")
    logger.info("=" * 64)

    executor_main(steps=[training_step])
