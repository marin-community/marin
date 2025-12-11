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
This experiment replicates the results from OLMoE paper (https://arxiv.org/pdf/2409.02060),
which shows that an 8x1B moe model is better than a 1B dense model (both has 1B activated params).
Training uses Dolma dataset for 2T tokens on v5e-128.

Reference Issue: https://github.com/marin-community/marin/issues/1183
"""

import logging
import math

from levanter.models.llama import LlamaConfig
from levanter.models.mixtral import MixtralConfig

from experiments.defaults import default_train
from experiments.pretraining_datasets.dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.resources import TpuPodConfig

logger = logging.getLogger("ray")

LR = 4e-4
WD = 0.1
TPU = TpuPodConfig(tpu_type="v5litepod-128")
TOKEN_TARGETS = 2_000_000_000_000
BATCH_SIZE = 1024
SEQ_LEN = 4096

dolma_llama3_tokenized = lm_mixture_data_config(
    components=tokenize_dolma(),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
    permutation_type="linear",
    include_raw_paths=False,
)


def step_target(token_target, batch_size, seq_len):
    actual_step_count = math.ceil(token_target / (batch_size * seq_len))
    nice_round_step_count = math.ceil(actual_step_count / 1000) * 1000
    return nice_round_step_count


baseline_1_4b = LlamaConfig(
    seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=16,
)

olmoe_8x_1_4b = MixtralConfig(
    seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=1024,
    num_heads=16,
    num_kv_heads=16,
    num_layers=16,
    n_routed_experts=64,
    num_experts_per_tok=8,
)

num_train_steps = step_target(TOKEN_TARGETS, BATCH_SIZE, SEQ_LEN)


train_config = SimpleTrainConfig(
    resources=TPU,
    train_batch_size=BATCH_SIZE,
    num_train_steps=num_train_steps,
    learning_rate=LR,
    weight_decay=WD,
    warmup=1 / 500,
)


def make_step(name, model_config):
    return default_train(
        name=name,
        train_config=train_config,
        model_config=model_config,
        tokenized=dolma_llama3_tokenized,
        use_default_validation=False,
        tags=("olmoe", "dolma"),
    )


baseline_step = make_step(
    name="moe-v5e-baseline-1b",
    model_config=baseline_1_4b,
)

olmoe_step = make_step(
    name="moe-v5e-olmoe-1b7b",
    model_config=olmoe_8x_1_4b,
)


if __name__ == "__main__":
    executor_main([baseline_step, olmoe_step])
