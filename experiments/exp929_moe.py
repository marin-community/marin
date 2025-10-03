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
This experiment tests out the MoE feature from Levanter.
We showed that basic MoE architecture results in unbalanced expert load.
With auxiliary loss such as load balancing loss and router z-loss,
training MoE gives a much more stable and balanced expert load.

Reference Issue: https://github.com/marin-community/marin/issues/929
"""

import dataclasses
import logging
import math

from levanter.models.mixtral import MixtralConfig

from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from marin.resources import TpuPodConfig

logger = logging.getLogger("ray")

BEST_LR = 3e-3 / 4
WD = 0.1
TPU = TpuPodConfig(tpu_type="v5litepod-256")
TOKEN_TARGETS = 42_000_000_000
BATCH_SIZE = 1024
SEQ_LEN = 4096

dolma_llama3_tokenized = lm_mixture_data_config(
    components=tokenize_dolma_steps(),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
    permutation_type="linear",
    include_raw_paths=False,
)


def step_target(token_target, batch_size, seq_len):
    actual_step_count = math.ceil(token_target / (batch_size * seq_len))
    nice_round_step_count = math.ceil(actual_step_count / 1000) * 1000
    return nice_round_step_count


mixtral_8x_8b_unbalanced = MixtralConfig(
    seq_len=SEQ_LEN,
    hidden_dim=4096,
    intermediate_dim=14336,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
    num_experts_per_tok=2,
    lbl_coef=None,
    rzl_coef=None,
)
mixtral_8x_8b_lbl = dataclasses.replace(
    mixtral_8x_8b_unbalanced,
    lbl_coef=0.01,
)

mixtral_8x_8b_lbl_shared = dataclasses.replace(
    mixtral_8x_8b_lbl,
    num_experts_per_tok=1,
    n_routed_experts=7,
    n_shared_experts=1,
)

mixtral_8x_8b = dataclasses.replace(
    mixtral_8x_8b_lbl,
    rzl_coef=0.001,
)

num_train_steps = step_target(TOKEN_TARGETS, BATCH_SIZE, SEQ_LEN)


train_config = SimpleTrainConfig(
    resources=TPU,
    train_batch_size=BATCH_SIZE,
    num_train_steps=num_train_steps,
    learning_rate=BEST_LR,
    weight_decay=WD,
)


def make_step(name, model_config):
    return default_train(
        name=name,
        train_config=train_config,
        model_config=model_config,
        tokenized=dolma_llama3_tokenized,
        use_default_validation=False,
        tags=("mixtral", "8x8b", "dolma"),
    )


moe_big_unbalanced_step = make_step(
    name="moe-v5e-mixtral8x8b-unbalanced-13",
    model_config=mixtral_8x_8b_unbalanced,
)

moe_big_lbl_step = make_step(
    name="moe-v5e-mixtral8x8b-balanced-2-unnormalized",
    model_config=mixtral_8x_8b_lbl,
)

moe_big_lbl_shared_step = make_step(
    name="moe-v5e-mixtral8x8b-balanced-shared",
    model_config=mixtral_8x_8b_lbl_shared,
)

moe_big_balanced_step = make_step(
    name="moe-v5e-mixtral8x8b-balanced-zloss-3",
    model_config=mixtral_8x_8b,
)


if __name__ == "__main__":
    executor_main([moe_big_unbalanced_step, moe_big_lbl_step, moe_big_lbl_shared_step, moe_big_balanced_step])
