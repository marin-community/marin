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

# https://github.com/marin-community/marin/issues/621
# Sweep to determine optimal training configs for small models
import dataclasses
import logging
import math

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3_old
from experiments.defaults import default_train
from experiments.llama import llama_1_4b, llama_8b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main, versioned
from marin.resources import TpuPodConfig

logger = logging.getLogger("ray")

BEST_LR = 3e-3 / 4
WD = 0.1
TPU_TYPE = "v5litepod-256"
TOKEN_TARGETS = 40_000_000_000
BATCH_SIZE = 1024
SEQ_LEN = 4096


def step_target(token_target, batch_size, seq_len):
    actual_step_count = math.ceil(token_target / (batch_size * seq_len))
    nice_round_step_count = math.ceil(actual_step_count / 1000) * 1000
    return nice_round_step_count


num_train_steps = step_target(TOKEN_TARGETS, BATCH_SIZE, SEQ_LEN)

baseline_train_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type=versioned(TPU_TYPE)),
    train_batch_size=BATCH_SIZE,
    num_train_steps=num_train_steps,
    learning_rate=BEST_LR,
    weight_decay=WD,
)
int8_train_config = dataclasses.replace(baseline_train_config, int8=True)

baseline_1b_step = default_train(
    name="exp620-v5e-1.4b-baseline-profile",
    train_config=baseline_train_config,
    model_config=llama_1_4b,
    tokenized=dclm_mixture_config_llama3_old,
    tags=("llama", "1.4b", "620_int8", "dclm"),
)

int8_1b_step = default_train(
    name="exp620-v5e-1.4b-int8-maxtext",
    train_config=int8_train_config,
    model_config=llama_1_4b,
    tokenized=dclm_mixture_config_llama3_old,
    tags=("llama", "1.4b", "620_int8", "dclm"),
)

baseline_8b_step = default_train(
    name="exp620-v5e-8b-baseline",
    train_config=baseline_train_config,
    model_config=llama_8b,
    tokenized=dclm_mixture_config_llama3_old,
    tags=("llama", "8b", "620_int8", "dclm"),
)

int8_8b_step = default_train(
    name="exp620-v5e-8b-int8",
    train_config=int8_train_config,
    model_config=llama_8b,
    tokenized=dclm_mixture_config_llama3_old,
    tags=("llama", "8b", "620_int8", "dclm"),
)


if __name__ == "__main__":
    executor_main([baseline_1b_step, int8_1b_step, baseline_8b_step, int8_8b_step])
