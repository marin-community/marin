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

import dataclasses

from levanter.optim import MuonHConfig

from experiments.defaults import SimpleTrainConfig, default_train
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from experiments.qwen3 import qwen3_1_7b, qwen3_8b

from experiments.ferries.initial_ferry import (
    BATCH_SIZE_1B,
    BATCH_SIZE_8B,
    NUM_1B_TRAIN_STEPS,
    NUM_8B_TRAIN_STEPS,
)
from experiments.ferries.initial_ferry import (
    _build_varying_mixture_for_steps as _base_build_varying_mixture_for_steps,
)


muonh_cfg_1b = MuonHConfig(
    learning_rate=0.01,
    adam_lr=0.0015,
    min_lr_ratio=0.0,
    momentum=0.98,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-15,
    muon_epsilon=1e-5,
    max_grad_norm=2.0,
    warmup=1000,
)

muonh_cfg_8b = MuonHConfig(
    learning_rate=0.01,
    adam_lr=0.0015,
    min_lr_ratio=0.0,
    momentum=0.98,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-15,
    muon_epsilon=1e-5,
    max_grad_norm=2.0,
    warmup=1000,
)


train_config_1b = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v5p-32"),
    train_batch_size=BATCH_SIZE_1B,
    num_train_steps=NUM_1B_TRAIN_STEPS,
    learning_rate=muonh_cfg_1b.learning_rate,
    z_loss_weight=5e-6,
    optimizer_config=muonh_cfg_1b,
)

train_config_8b = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v5p-64"),
    train_batch_size=BATCH_SIZE_8B,
    num_train_steps=NUM_8B_TRAIN_STEPS,
    learning_rate=muonh_cfg_8b.learning_rate,
    z_loss_weight=5e-6,
    optimizer_config=muonh_cfg_8b,
)

base_mixture_1b = _base_build_varying_mixture_for_steps(
    NUM_1B_TRAIN_STEPS, train_batch_size=BATCH_SIZE_1B, mixture_block_size=2048
)
varying_mixture_1b = dataclasses.replace(base_mixture_1b, permutation_type="feistel")

base_mixture_8b = _base_build_varying_mixture_for_steps(
    NUM_8B_TRAIN_STEPS, train_batch_size=BATCH_SIZE_8B, mixture_block_size=2048
)
varying_mixture_8b = dataclasses.replace(base_mixture_8b, permutation_type="feistel")

ferry_model_1b = default_train(
    name="ferry_muonh_qwen3_1_7b_feistel",
    tokenized=varying_mixture_1b,
    model_config=qwen3_1_7b,
    train_config=train_config_1b,
)

ferry_model_8b = default_train(
    name="ferry_muonh_qwen3_8b_feistel",
    tokenized=varying_mixture_8b,
    model_config=qwen3_8b,
    train_config=train_config_8b,
)


if __name__ == "__main__":
    executor_main(
        steps=[ferry_model_1b, ferry_model_8b],
        description=(
            "Ferry (Feistel + MuonH): PT on Nemotron+Code then cooldown to HQ mix, scaled from Tootsie schedules"
        ),
    )
