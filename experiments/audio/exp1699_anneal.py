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

"""Training config for the Marin-Audio (Yodas2 checkpoint) annealing experiment."""

from experiments.audio.tokenize_mls_en import mls_en_data_config
from experiments.audio.tokenize_emilia import emilia_english_mixture_config
from experiments.anneal_config import AnnealConfig
from experiments.defaults import default_anneal
from experiments.audio.exp1699_marin_yodas2 import yodas_1b_model

from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig

SEQ_LEN = 4096
BASE_BATCH_SIZE = 256
BATCH_SIZE = 512
BASE_LEARNING_RATE = 3e-3
LEARNING_RATE = BASE_LEARNING_RATE * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5
NUM_ANNEAL_STEPS = 50000
NUM_ANNEAL_TRAINING_TOKENS = int(NUM_ANNEAL_STEPS * BATCH_SIZE * SEQ_LEN)

checkpoint_step = 190000
# note that it is actually a 0.6B model, not 1B despite the name
yodas_step = yodas_1b_model.cd(f"checkpoints/step-{checkpoint_step}")

# Annealing with MLS-EN
annealing_mls_en_config = AnnealConfig(
    initialize_from_checkpoint_path=yodas_step,
    dataset_config=mls_en_data_config(),
    learning_rate=LEARNING_RATE,
    weight_decay=0.033,
    min_lr_ratio=0.0,
    lr_schedule="linear",
    train_batch_size=BATCH_SIZE,
    num_anneal_training_tokens=NUM_ANNEAL_TRAINING_TOKENS,
    resources=TpuPodConfig(tpu_type="v5p-64"),
)
annealed_model_mls_en = default_anneal(name="exp1699_anneal_mls_en", anneal_config=annealing_mls_en_config)

# Annealing with Emilia English language only
annealing_emilia_config = AnnealConfig(
    initialize_from_checkpoint_path=yodas_step,
    dataset_config=emilia_english_mixture_config(),
    learning_rate=LEARNING_RATE,
    weight_decay=0.033,
    min_lr_ratio=0.0,
    lr_schedule="linear",
    train_batch_size=BATCH_SIZE,
    num_anneal_training_tokens=NUM_ANNEAL_TRAINING_TOKENS,
    resources=TpuPodConfig(tpu_type="v5p-64"),
)
annealed_model_emilia = default_anneal(name="exp1699_anneal_emilia_en", anneal_config=annealing_emilia_config)

if __name__ == "__main__":
    executor_main(
        steps=[annealed_model_mls_en, annealed_model_emilia],
        description="Cooldown the Marin Yodas2 for the last 50K steps with MLS-EN and Emilia English language.",
    )
