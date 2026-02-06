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

from experiments.audio.qwen3 import qwen3_150m
from experiments.audio.data_mixes import mix3_v5_english_yodas_sweep_mixture_config
from experiments.defaults import SimpleTrainConfig, default_train

from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig

SEQ_LEN = 4096
BASE_LEARNING_RATE = 3e-3
BASE_BATCH_SIZE = 256
BATCH_SIZE = 128
BASE_WIDTH = 1024
WIDTH = 512
LEARNING_RATE = BASE_LEARNING_RATE * (BATCH_SIZE / BASE_BATCH_SIZE) ** 0.5 * (WIDTH / BASE_WIDTH) ** 0.5
yodas_qwen_150m = dataclasses.replace(
    qwen3_150m, gradient_checkpointing=hax.ScanCheckpointPolicy(save_carries="offload")
)

NUM_TRAIN_TOKENS = int(10e9)
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

training_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    z_loss_weight=1e-4,
    optimizer_config=optim_config,
)


def generate_train_config(
    name: str,
    data_mix_config: LMMixtureDatasetConfig,
):
    return default_train(
        name=name,
        tokenized=data_mix_config,
        model_config=yodas_qwen_150m,
        train_config=training_config,
        eval_harness_tasks=[],
        tags=["data_mix"],
    )


if __name__ == "__main__":
    steps = []
    for yodas_weight in [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 1.5]:
        data_mix_config = mix3_v5_english_yodas_sweep_mixture_config(yodas_weight)
        model = generate_train_config(
            name=f"exp1699_data_mix_150m_mix3_v5_yodas{yodas_weight}",
            data_mix_config=data_mix_config,
        )
        steps.append(model)
    executor_main(
        steps=steps,
        description="Train the data mix models at 150M model size with varying yodas weights.",
    )
