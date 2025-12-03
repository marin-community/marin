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
Train 32B with long context extension: staged training with gradually increasing train_seq_len
Model max_seq_len is set to 65536 (maximum capacity) for all stages.
Training sequence length increases per stage:
Stage 1: train_seq_len=8192 for 200B tokens
Stage 2: train_seq_len=32768 for 300B tokens (100B additional)
Stage 3: train_seq_len=65536 for 400B tokens (100B additional)
"""

import dataclasses

import haliax
from levanter.callbacks.watch import WatchConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.optim import AdamConfig
from levanter.optim.clip_update_norm import ClipUpdateNormConfig

from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.defaults import default_train
from experiments.llama import llama_32b
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.resources import TpuPodConfig

## 32b experiments

# Set max_seq_len to the maximum capacity (65536) for all stages
# This defines the model's architectural capacity (positional embeddings, attention, etc.)
MAX_SEQ_LEN = 65536

# on the v4-2048, with the 8192 batch size, we need to offload the carries
llama_32b_remat = dataclasses.replace(
    llama_32b,
    gradient_checkpointing=haliax.ScanCheckpointPolicy(save_carries="offload"),
    max_seq_len=MAX_SEQ_LEN,
    rope=Llama3RotaryEmbeddingsConfig(original_max_position_embeddings=MAX_SEQ_LEN),
)

# Base train config (will be modified for each stage)
base_train_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v4-2048", slice_count=1),
    train_batch_size=8192,
    weight_decay=0.05,
    decay=0.4,
    ema_beta=0.995,
    lr_schedule="linear",
    cycle_length=None,
    steps_per_eval=1000,
    steps_per_task_eval=10000,
    z_loss_weight=1e-4,
    learning_rate=7e-4,  # ignored and overridden by the optimizer config
    watch=WatchConfig(watch_targets=["grads", "params", "updates", "opt_state"], interval=1),
    skip_bad_steps=True,
    max_grad_norm=0.2,
    allow_partial_checkpoint=True,
    optimizer_config=AdamConfig(
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=0.2,
        learning_rate=7e-4,
        weight_decay=0.05,
        skip_bad_steps=True,
        lr_schedule="linear",
        warmup=0.01,
        decay=0.4,
        cycle_length=None,
        clip_update_norm=ClipUpdateNormConfig(rolling_interval_length=128, sigma_factor=2.0),
    ),
)

# Data mixture (same for all stages)
nemotron_steps = tokenize_nemotron()
proofpile_2 = dclm_components_llama3["proofpile_2"]
starcoderdata = dclm_components_llama3["starcoderdata"]
nemotron_mix = lm_mixture_data_config(
    components={**nemotron_steps, "starcoderdata": starcoderdata, "proofpile_2": proofpile_2},
    weights={
        **NEMOTRON_WEIGHTS,
        "starcoderdata": 0.25,
        "proofpile_2": 0.055,
    },
    permutation_type="linear",
)

EXTENSION_STAGES = {
    0: {
        "TRAIN_SEQ_LEN": 8192,  # Training sequence length for this stage
        "ADDITIONAL_TOKENS": 200_000_000_000,
        "BATCH_SIZE": 8192,
    },
    1: {
        "TRAIN_SEQ_LEN": 32768,  # Training sequence length for this stage
        "ADDITIONAL_TOKENS": 100_000_000_000,
        "BATCH_SIZE": 8192,
    },
    2: {
        "TRAIN_SEQ_LEN": 65536,  # Training sequence length for this stage
        "ADDITIONAL_TOKENS": 100_000_000_000,
        "BATCH_SIZE": 8192,
    },
}

executor_steps = []
stage_steps_list = []  # Track steps for each stage for checkpoint chaining

for stage_idx in range(len(EXTENSION_STAGES)):
    stage_config = EXTENSION_STAGES[stage_idx]
    train_seq_len = stage_config["TRAIN_SEQ_LEN"]  # Training sequence length for this stage
    batch_size = stage_config["BATCH_SIZE"]
    
    # Calculate steps from additional tokens
    additional_tokens = stage_config["ADDITIONAL_TOKENS"]
    stage_steps = additional_tokens // (batch_size * train_seq_len)
    
    stage_steps_list.append(stage_steps)
    
    # Model config: max_seq_len is already set to MAX_SEQ_LEN (65536) in llama_32b_remat
    # This defines the model's architectural capacity and stays constant across stages
    new_model_config = llama_32b_remat
    
    # Replace train config
    train_config_kwargs = {
        "train_batch_size": batch_size,
        "num_train_steps": stage_steps,
        # TODO: Once PR #2133 merges, add train_seq_len parameter here:
        # "train_seq_len": train_seq_len,
    }
    
    # Chain checkpoints: stages after the first initialize from previous stage
    if stage_idx > 0:
        prev_step = executor_steps[stage_idx - 1]
        prev_steps = stage_steps_list[stage_idx - 1]
        train_config_kwargs["initialize_from_checkpoint_path"] = (
            prev_step.cd(f"checkpoints/step-{prev_steps}").nonblocking()
        )
        train_config_kwargs["reset_data_loader_on_init"] = False
    
    new_train_config = dataclasses.replace(base_train_config, **train_config_kwargs)
    
    # Replace train step
    new_executor_step = default_train(
        name=f"llama-32b-long-context-stage{stage_idx+1}",
        tokenized=nemotron_mix,
        model_config=new_model_config,
        train_config=new_train_config,
        tags=["llama", "32b", "ema", "long-context", f"stage{stage_idx+1}"],
        eval_harness_tasks=[],
    ).with_output_path(f"checkpoints/llama-32b-long-context-stage{stage_idx+1}")
    
    # Add to step list
    executor_steps.append(new_executor_step)

if __name__ == "__main__":
    executor_main(
        executor_steps,
        description="Train 32B with long context extension: staged training with gradually increasing train_seq_len",
    )

