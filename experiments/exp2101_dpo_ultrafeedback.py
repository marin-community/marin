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
Run DPO on the Ultrafeedback preference dataset using Marin's executor framework.
"""

from levanter.data.text import PreferenceChatLmDatasetFormat

from experiments.defaults import default_dpo, default_tokenize
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.preference_datasets import get_preference_dataset
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

preference_dataset = get_preference_dataset(DATASET_NAME, splits=["train_prefs"])

tokenized_preferences = default_tokenize(
    name="ultrafeedback_binarized_marin_tokenizer",
    dataset=preference_dataset / "**/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)

dpo_config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=64,
    num_train_steps=5000,
    learning_rate=5e-7,
    tokenizer=marin_tokenizer,
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    reference_model_path="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=0.1,
    seed=0,
)

training_step = default_dpo(
    name="llama3.1_8b_ultrafeedback_dpo",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=dpo_config,
    tags=["llama", "dpo", "ultrafeedback"],
)


if __name__ == "__main__":
    executor_main(steps=[preference_dataset, tokenized_preferences, training_step])
