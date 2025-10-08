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
PlantCAD training experiment: A single 600M model pretrained on 16 Angiosperm genomes
"""

import jax
import logging
from huggingface_hub import snapshot_download
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from experiments.defaults import default_train, default_tokenize
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger("ray")

# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
run_number = 1
tokenizer_path = "kuleshov-group/PlantCaduceus_l20"
dataset_path = "kuleshov-group/Angiosperm_16_genomes"
dataset_examples = 5_485_282
target_examples = dataset_examples * 10  # 10 epochs
use_pretokenized_dataset = True
learning_rate = 3e-4
per_device_eval_parallelism = 256
# TODO: Make this cross-platform compatible
num_gpus = len(jax.devices("gpu"))
# TODO: Determine why global batch size must scale with gpus to avoid OOM;
#       gradient accumulation does not appear to be working as expected
train_batch_size = per_device_eval_parallelism * num_gpus
num_train_steps = target_examples // train_batch_size
steps_per_export = num_train_steps // 10
steps_per_cycle = num_train_steps // 10
steps_per_eval = num_train_steps // 100

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
# Define ~600M Llama model noting that the PlantCAD vocabulary is only 7
# tokens, which drastically reduces final parameter count at this scale.
model_config = LlamaConfig(
    seq_len=512,
    hidden_dim=1408,
    intermediate_dim=4224,
    num_heads=22,
    num_kv_heads=22,
    num_layers=24,
)

# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
# Load dataset containing DNA sequences of form `[ACGTN]+`; see:
# https://huggingface.co/datasets/kuleshov-group/Angiosperm_16_genomes
data_tokenized = default_tokenize(
    name="angiosperm_16_genomes",
    dataset=dataset_path,
    tokenizer=tokenizer_path,
    # DNA sequences are in `seq`, not `text`
    format=TextLmDatasetFormat(text_key="seq"),
)
# Tokenization is very slow, so default to pre-tokenized cache
# TODO: Figure out why tokenization takes 15+ minutes
if use_pretokenized_dataset:
    data_local_path = snapshot_download(
        # TODO: Rename this repo
        repo_id="plantcad/_dev_marin_plantcad1_v1_tokenized",
        repo_type="dataset",
        revision="main",
    )
    data_tokenized = data_tokenized.with_output_path(data_local_path)


# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------
train_config = SimpleTrainConfig(
    resources=GpuConfig(gpu_count=num_gpus),
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
    name=f"plantcad-train-r{run_number:02d}",
    tokenized=data_tokenized,
    model_config=model_config,
    train_config=train_config,
    tags=["dna", "plantcad", "training"],
    eval_harness_tasks=[],
    use_default_validation=False,
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🧬 PlantCAD Training Experiment")
    logger.info("=" * 64)
    logger.info(f"Model:              {model_config}")
    logger.info(f"Learning rate:      {learning_rate}")
    logger.info(f"Global batch size:  {train_batch_size}")
    logger.info(f"Micro batch size:   {per_device_eval_parallelism}")
    logger.info(f"Target examples:    {target_examples:,}")
    logger.info(f"Training steps:     {num_train_steps:,}")
    logger.info(f"Steps per export:   {steps_per_export:,}")
    logger.info(f"Steps per eval:     {steps_per_eval:,}")
    logger.info("=" * 64)

    executor_main(
        steps=[
            data_tokenized,
            training_step,
        ]
    )
