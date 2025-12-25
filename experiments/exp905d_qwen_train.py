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
#1237: Qwen2.5-7B-Instruct SFT

SFT Qwen2.5-7B-Instruct using the Reasoning + Tulu SFT Mixture.

Background:
- Our marin-8b-instruct model doesn't do well on AIME (~0%) after being trained on our SFT mixtures
- Nemotron-8b also doesn't do well on AIME (~0%) after being trained on the nemotron SFT data.
- Nemotron-8b follows a curriculum in the sense that it trains first on maths and code, then reasoning, then instruction following.
- Openthoughts managed to train Qwen2.5-7B-Instruct to 60+% acc with reasoning data.
- Qwen2.5-7B Instruct already scores 13% on AIME24



Hypotheses:
1. We need a decent enough base model to train reasoning.
2. We need a curriculum for SFT training


"""

import dataclasses
import logging
import math

from experiments.defaults import default_train
from experiments.evals.evals import default_sft_eval
from experiments.evals.task_configs import OPEN_LM_LEADERBOARD_MCQ

# Dataset configurations
from experiments.exp905a_nemotron_sft_dstc import DATASETS, create_tokenization_step
from experiments.qwen3 import qwen2_5_7b_instruct
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp916_tootsie_spoonbill_cooldown import spoonbill_zloss_tulu3_sft_config
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.resources import TpuPodConfig

logger = logging.getLogger("ray")

# Experiment specific settings
EXPERIMENT_NAME = "sft/qwen25_7b_sft_nemotron_and_openthoughts3"
REGION = "us-central2"

SFT_CONFIG = dataclasses.replace(
    spoonbill_zloss_tulu3_sft_config,
    learning_rate=1e-4,
    resources=TpuPodConfig(tpu_type="v4-64", slice_count=1),
    # resources=TpuPodConfig(tpu_type="v6e-64", slice_count=1),
    initialize_from_hf=f"gs://marin-{REGION}/models/qwen2.5-7b-instruct",
    steps_per_eval=5000,
    steps_per_hf_export=1000,
    steps_per_checkpoint=2500,
)

MODEL_CONFIG = qwen2_5_7b_instruct

EXPERIMENT_TAGS = [
    "qwen2.5",
    "7b",
    "sft",
    "starling",
    "mixture",
    "exp905d",
    "nemotron+openthoughts3-1.2m",
    f"region={REGION}",
    "batchsize=256",
    "user=chiheem",
]

# Training parameters
BATCH_SIZE = 256
EPOCHS = 3

tokenized_datasets = {short_name: create_tokenization_step(hf_name) for short_name, hf_name in DATASETS.items()}

# Mixture weights should be read from the json file written by exp905a
mixture_weights = {
    "acecode_89k": 26032149,
    "smoltalk": 883494479,
    "verifiable_math_problems": 382056624,
    "dolphin_r1_nonreasoning": 319820708,
    "dolphin_r1_reasoning": 508743187,
    "bespoke_stratos_17k": 85724829,
    "openthoughts_114k_math": 72964948,
    "tulu_3_sft_mixture": 749008790,
    "natural_reasoning": 966484170,
    "nemotron_sft": 34739443205,
    "openthoughts3": 17449811417,
}

# Calculate the number of training steps from computed values
total_tokens = sum(mixture_weights.values())
num_steps = (total_tokens * EPOCHS) // (BATCH_SIZE * MODEL_CONFIG.seq_len)

logger.info(f"Total tokens: {total_tokens}")
logger.info(f"Sequence length: {MODEL_CONFIG.seq_len}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Number of training steps: {num_steps}")

sft_mixture = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,  # Edit in create_experiment_config_step, not here.
    shuffle=True,
    missing_weights_are_validation=True,
)

_sft_config = dataclasses.replace(
    SFT_CONFIG,
    num_train_steps=num_steps,  # Using the values in the config file
    train_batch_size=BATCH_SIZE,  # Using the values in the config file
    learning_rate=SFT_CONFIG.learning_rate * math.sqrt(BATCH_SIZE / 128),
)

# Create a custom SFT step with evaluations enabled
# We need to modify the default_sft to enable evaluations
# Since default_sft hardcodes eval_harness_tasks=[], we'll use default_train directly
normal_train_config = SimpleTrainConfig(
    resources=_sft_config.resources,
    train_batch_size=_sft_config.train_batch_size,
    num_train_steps=_sft_config.num_train_steps,
    learning_rate=_sft_config.learning_rate,
    lr_schedule=_sft_config.lr_schedule,
    decay=_sft_config.cooldown,
    weight_decay=_sft_config.weight_decay,
    min_lr_ratio=_sft_config.min_lr_ratio,
    max_grad_norm=_sft_config.max_grad_norm,
    warmup=_sft_config.warmup,
    steps_per_eval=_sft_config.steps_per_eval,
    steps_per_export=_sft_config.steps_per_checkpoint,
    int8=_sft_config.int8,
    steps_per_hf_export=_sft_config.steps_per_hf_export,
    initialize_from_hf=_sft_config.initialize_from_hf,
    data_seed=_sft_config.seed,
    z_loss_weight=_sft_config.z_loss_weight,
)

sft_step = default_train(
    name=EXPERIMENT_NAME,
    tokenized=sft_mixture,
    model_config=MODEL_CONFIG,
    train_config=normal_train_config,
    tags=EXPERIMENT_TAGS,
    eval_harness_tasks=OPEN_LM_LEADERBOARD_MCQ,  # Enable evaluations during training
    use_default_validation=True,
).with_output_path(f"gs://marin-{REGION}/checkpoints/{EXPERIMENT_NAME}")

# Now run the SFT step
if __name__ == "__main__":
    executor_main(
        [
            sft_step,
            *default_sft_eval(sft_step),
        ],
        description="Run SFT training step",
    )
