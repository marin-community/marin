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
Run SimPO on the Ultrafeedback preference dataset using Marin's executor framework.
"""

from levanter.data.text import PreferenceChatLmDatasetFormat

from experiments.defaults import default_simpo, default_tokenize
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.preference_datasets import get_preference_dataset
from experiments.simple_simpo_config import SimpleSimPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_data_config

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
LLAMA3_8B_HF_PATH = "gs://marin-us-central1/gcsfuse_mount/models/meta-llama--Llama-3-1-8B--main"

preference_dataset = get_preference_dataset(DATASET_NAME, splits=["train_prefs", "test_prefs"])

tokenized_train_preferences = default_tokenize(
    name="ultrafeedback_binarized_train_prefs_marin_tokenizer",
    dataset=preference_dataset / "train_prefs/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
)

tokenized_test_preferences = default_tokenize(
    name="ultrafeedback_binarized_test_prefs_marin_tokenizer",
    dataset=preference_dataset / "test_prefs/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=PreferenceChatLmDatasetFormat(),
    is_validation=True,
)

tokenized_preferences = lm_data_config(
    training_set=tokenized_train_preferences,
    validation_sets={"ultrafeedback_test_prefs": tokenized_test_preferences},
)

simpo_config = SimpleSimPOConfig(
    resources=ResourceConfig.with_tpu("v5p-16"),
    train_batch_size=128,
    num_train_steps=2150,
    learning_rate=6e-7,
    lr_schedule="cosine",
    warmup=0.1,
    cooldown=None,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path=LLAMA3_8B_HF_PATH,
    train_seq_len=4096,
    max_seq_len=4096,
    beta=2.0,
    gamma_beta_ratio=0.5,
    validation_split_fraction=None,
    steps_per_eval=200,
    steps_per_checkpoint=1000,
    steps_per_hf_export=1000,
    seed=0,
)

training_step = default_simpo(
    name="dpo/ultrafeedback_llama3_8b",
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    simpo_config=simpo_config,
    tags=["ultrafeedback", "llama3"],
)


if __name__ == "__main__":
    executor_main(
        steps=[
            preference_dataset,
            tokenized_train_preferences,
            tokenized_test_preferences,
            training_step,
        ]
    )
