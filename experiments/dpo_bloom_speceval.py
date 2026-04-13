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

"""
Run DPO on the Bloom SpecEval preference dataset (GPT-4.1 chosen vs Mixtral opposite-mode rejected).

The preference data is pre-built by bloom's export_marin_preference.py and lives on GCS at:
  gs://marin-us-central1/preference/bloom_openai_model_spec_gpt41_vs_mixtral_opposite/{train,eval}/

This experiment tokenizes that data and runs DPO training.
"""

from levanter.data.text import PreferenceChatLmDatasetFormat

from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_data_config

GCS_PREFIX = "gs://marin-us-central1/preference/bloom_openai_model_spec_gpt41_vs_mixtral_opposite"

tokenized_train = default_tokenize(
    name="bloom_speceval_train_prefs_marin_tokenizer",
    dataset=f"{GCS_PREFIX}/train/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)

tokenized_eval = default_tokenize(
    name="bloom_speceval_eval_prefs_marin_tokenizer",
    dataset=f"{GCS_PREFIX}/eval/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

tokenized_preferences = lm_data_config(
    training_set=tokenized_train,
    validation_sets={"bloom_speceval_eval": tokenized_eval},
)

dpo_config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-32"),
    train_batch_size=128,
    num_train_steps=1000,
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
    seed=0,
)

training_step = default_dpo(
    name="dpo/bloom_speceval_marin_instruct_beta0.1_seed0",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=dpo_config,
    tags=["dpo", "bloom", "speceval", "llama3", "marin-instruct", "beta0.1"],
)


if __name__ == "__main__":
    executor_main(
        steps=[
            tokenized_train,
            tokenized_eval,
            training_step,
        ]
    )
