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

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3_wrong
from experiments.defaults import default_train
from experiments.llama import llama_1_4b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig

llama_1_4b_train_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v4-128", slice_count=1),
    train_batch_size=1024,
    num_train_steps=50_000,
    # these hypers from Table 12 in https://arxiv.org/html/2406.11794v1#A6
    learning_rate=1e-3,
    weight_decay=0.05,
    # WSD with EMA
    warmup=1000,
    decay=0.4,
    lr_schedule="linear",
    ema_beta=0.995,
)


llama_1b_tootsie = dataclasses.replace(
    default_train(
        name="llama-1b-ema",
        tokenized=dclm_mixture_config_llama3_wrong,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
        tags=["llama", "1b", "ema", "exp741"],
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[
            llama_1b_tootsie,
        ],
        description="Train a 1B parameter model with EMA",
    )
