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

"""Training config for Data Mix at 150M model size"""

import dataclasses
import haliax as hax
from math import ceil

from levanter.data.text import LMMixtureDatasetConfig
from levanter.optim import CautiousConfig

from experiments.qwen3 import qwen3_1_7b
from experiments.audio.tokenize_yodas import (
    yodas2_english_data_config,
    yodas2_acoustic_english_data_config,
    yodas2_semantic_english_data_config,
)
from experiments.defaults import SimpleTrainConfig, default_train

from marin.execution.executor import executor_main
from marin.execution import versioned
from fray.cluster import ResourceConfig

SEQ_LEN = 4096
BASE_LEARNING_RATE = 2e-3  # reduced from 3e-3 to match IsoFLOP experiment
BASE_BATCH_SIZE = 256
BATCH_SIZE = 128  # from IsoFLOP experiment
BASE_WIDTH = 1024
WIDTH = 2048
_LEARNING_RATE = BASE_LEARNING_RATE * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5 * (WIDTH / BASE_WIDTH) ** 0.5
LEARNING_RATE = versioned(_LEARNING_RATE)

yodas_qwen_1_7b = dataclasses.replace(
    qwen3_1_7b, gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload")
)

NUM_TRAIN_TOKENS = int(30e9)
NUM_TRAIN_STEPS = ceil(NUM_TRAIN_TOKENS / (BATCH_SIZE * SEQ_LEN))

optim_config = CautiousConfig(
    learning_rate=LEARNING_RATE,
    weight_decay=0.033,
    min_lr_ratio=0.0,
    warmup=0.1,
    decay=0.2,
    beta1=0.98,
    beta2=0.98,
    epsilon=1e-16,
    max_grad_norm=1,
    lr_schedule="linear",
    adamc_weight_decay=True,
)


def generate_train_config(
    name: str,
    data_config: LMMixtureDatasetConfig,
    tpu_type: str = "v5p-16",
):
    training_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        z_loss_weight=1e-4,
        optimizer_config=optim_config,
    )
    return default_train(
        name=name,
        tokenized=data_config,
        model_config=yodas_qwen_1_7b,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["data_mix"],
    )


if __name__ == "__main__":
    steps = []

    yodas_interleaved = generate_train_config(
        name="exp1699_ablate_tokens_yodas_interleaved",
        data_config=yodas2_english_data_config(),
        tpu_type="v5p-16",
    )
    yodas_acoustic = generate_train_config(
        name="exp1699_ablate_tokens_yodas_acoustic",
        data_config=yodas2_acoustic_english_data_config(),
        tpu_type="v5p-64",
    )
    yodas_semantic = generate_train_config(
        name="exp1699_ablate_tokens_yodas_semantic",
        data_config=yodas2_semantic_english_data_config(),
        tpu_type="v5p-32",
    )
    steps += [yodas_interleaved, yodas_acoustic, yodas_semantic]

    executor_main(
        steps=steps,
        description="Train the 1.7B model on 30B tokens of Yodas2 with different tokenization configurations.",
    )
