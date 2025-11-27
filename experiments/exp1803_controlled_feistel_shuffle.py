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
Controlled ablation of Feistel vs linear shuffling on Stack v2 EDU.

Link to issue: https://github.com/marin-community/marin/issues/1803

David introduced a new Feistel-based dataset shuffling algorithm to address training instabilities that looked
like unshuffled data pathologies. While the new algorithm has not shown those issues recently, we want a controlled
ablation to confirm its benefits. We compare two permutations of the same tokenized dataset:

We choose Stack v2 EDU filtered as it is sorted by programming language, which makes it a good stress test for
shuffling quality. Each run trains to a 1B token budget, and we compare the resulting checkpoints using metrics
that include Paloma 100 Languages GitHub as a proxy for high-level programming performance. We should also just see
spikier loss for the linear shuffle if the coprime is bad.

Hypothesis: Feistel shuffling produces a more stable, monotonically decreasing loss curve and lower validation loss
because the training distribution more closely approximates a true i.i.d. shuffle.
"""

import math

from experiments.common_pile.tokenize_common_pile import common_pile_tokenized
from experiments.defaults import default_train
from experiments.qwen3 import qwen3_0_6b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config
from marin.resources import TpuPodConfig

TOKEN_TARGET = 1_000_000_000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_FRACTION = 0.05
DECAY_FRACTION = 0.2
EVAL_INTERVAL = 500
TASK_EVAL_INTERVAL = 5_000


def _steps_for_token_budget(token_budget: int, batch_size: int, seq_len: int) -> int:
    tokens_per_step = batch_size * seq_len
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")

    raw_steps = math.ceil(token_budget / tokens_per_step)
    return math.ceil(raw_steps / 100) * 100


NUM_TRAIN_STEPS = _steps_for_token_budget(TOKEN_TARGET, BATCH_SIZE, qwen3_0_6b.seq_len)

TRAIN_CONFIG = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v5p-16"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup=WARMUP_FRACTION,
    decay=DECAY_FRACTION,
    lr_schedule="linear",
    steps_per_eval=EVAL_INTERVAL,
    steps_per_task_eval=TASK_EVAL_INTERVAL,
)

stackv2_edu_tokenized = common_pile_tokenized()["common_pile/stackv2_edu"]


linear_shuffle_data = lm_data_config(
    training_set=stackv2_edu_tokenized,
    permutation_type="linear",
)

feistel_shuffle_data = lm_data_config(
    training_set=stackv2_edu_tokenized,
    permutation_type="feistel",
)

BASE_TAGS = ("exp1803", "stackv2-edu")

linear_shuffle_run = default_train(
    name="exp1803-stackv2-linear-shuffle",
    tokenized=linear_shuffle_data,
    model_config=qwen3_0_6b,
    train_config=TRAIN_CONFIG,
    tags=(*BASE_TAGS, "linear-shuffle"),
    eval_harness_tasks=[],
)

feistel_shuffle_run = default_train(
    name="exp1803-stackv2-feistel-shuffle",
    tokenized=feistel_shuffle_data,
    model_config=qwen3_0_6b,
    train_config=TRAIN_CONFIG,
    tags=(*BASE_TAGS, "feistel-shuffle"),
    eval_harness_tasks=[],
)

if __name__ == "__main__":
    executor_main(
        steps=[linear_shuffle_run, feistel_shuffle_run],
        description=(
            "Compare linear and Feistel shuffling on Stack v2 EDU. "
            "The dataset is internally sorted, which should reveal issues "
            "with inadequate shuffling more clearly."
        ),
    )
