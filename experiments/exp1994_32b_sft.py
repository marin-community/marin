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
32B Qwen cooldown-style SFT that warmstarts from the step-191999 checkpoint of ``tootsie-32b-cooldown-mantis-adamc-v2``
and uses the mixture frmo 1880
"""

import dataclasses

from experiments.defaults import default_sft
from experiments.evals.evals import default_sft_eval
from experiments.exp1880_sft_baseline import mixture_config
from experiments.marin_models import marin_tokenizer
from experiments.simple_sft_config import SimpleSFTConfig
from experiments.tootsie.exp1529_32b_mantis_cooldown import (
    qwen3_32b_remat,
    tootsie_32b_cooldown_mantis,
)
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

TRAIN_BATCH_SIZE = 2048
# Matches the 3-epoch budget from exp1880_sft_baseline (computed against that mixture).
NUM_TRAIN_STEPS = 14_246
LEARNING_RATE = 1e-5
WARMSTART_STEP = 191_999

qwen3_32b_remat_8k = dataclasses.replace(qwen3_32b_remat, max_seq_len=8192)
mantis_checkpoint = tootsie_32b_cooldown_mantis.cd(f"checkpoints/step-{WARMSTART_STEP}/").nonblocking()

mantis_sft_config = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v4-2048"),
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    tokenizer=marin_tokenizer,
    initialize_from_checkpoint_path=mantis_checkpoint,
    max_seq_len=8192,
    seed=0,
)

tootsie_32b_mantis_sft = default_sft(
    name="tootsie-32b-mantis-sft-exp1994",
    tokenized=mixture_config,
    model_config=qwen3_32b_remat_8k,
    sft_config=mantis_sft_config,
    tags=["qwen", "32b", "exp1994", "mantis", "cooldown"],
)

tootsie_32b_mantis_sft_evals = default_sft_eval(
    tootsie_32b_mantis_sft,
    use_levanter_inference=True,
    resource_config=ResourceConfig.with_tpu("v4-8"),
)

if __name__ == "__main__":
    executor_main([tootsie_32b_mantis_sft, *tootsie_32b_mantis_sft_evals])
