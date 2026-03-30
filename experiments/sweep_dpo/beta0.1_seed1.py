# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""DPO sweep: beta=0.1, seed=1 on Bloom SpecEval v2."""

from experiments.dpo_bloom_speceval_v2 import tokenized_preferences, tokenized_train, tokenized_eval
from experiments.defaults import default_dpo
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import DPO_EVAL_PARALLELISM, SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

dpo_config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-32", ram="256g"),
    per_device_eval_parallelism=DPO_EVAL_PARALLELISM["v5p-32"],
    train_batch_size=128,
    num_train_steps=850,
    learning_rate=5e-7,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    reference_model_path="marin-community/marin-8b-instruct",
    reference_is_hf=True,
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    steps_per_eval=200,
    steps_per_checkpoint=1000,
    steps_per_hf_export=200,
    seed=1,
)

training_step = default_dpo(
    name="dpo/new_dpo_bloom_speceval_v2_marin_instruct_beta0.1_seed1",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=dpo_config,
    tags=["dpo", "bloom", "speceval-v2", "llama3", "marin-instruct", "beta0.1", "sweep"],
)

if __name__ == "__main__":
    executor_main(
        steps=[
            tokenized_train,
            tokenized_eval,
            training_step,
        ]
    )
