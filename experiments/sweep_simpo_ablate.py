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
Ablation study for SimPO on Ultrafeedback preference dataset.

Tests the effect of lr_schedule (cosine vs linear) and max_grad_norm (1.0 vs None)
on a fixed hyperparameter config (beta=1.0, gamma=0.5) that performed best in earlier runs.

Ablation grid (4 runs):
  A: cosine + grad_norm=1.0  (old config - baseline)
  B: linear + grad_norm=1.0  (isolate lr_schedule effect)
  C: cosine + grad_norm=None (isolate grad_norm effect)
  D: linear + grad_norm=None (new config)
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

# TPU Configuration presets
TPU_CONFIGS = {
    "v5p-8": {
        "resources": ResourceConfig.with_tpu("v5p-8"),
        "train_batch_size": 128,
        "per_device_parallelism": 16,
        "per_device_eval_parallelism": 16,
    },
    "v5p-16": {
        "resources": ResourceConfig.with_tpu("v5p-16"),
        "train_batch_size": 128,
        "per_device_parallelism": 8,
        "per_device_eval_parallelism": 8,
    },
    "v5p-32": {
        "resources": ResourceConfig.with_tpu("v5p-32"),
        "train_batch_size": 128,
        "per_device_parallelism": 4,
        "per_device_eval_parallelism": 4,
    },
    "v5p-64": {
        "resources": ResourceConfig.with_tpu("v5p-64"),
        "train_batch_size": 128,
        "per_device_parallelism": -1,
        "per_device_eval_parallelism": -1,
    },
}

# Select TPU type here
TPU_TYPE = "v5p-64"

# Fixed hyperparameters (best config from old runs)
BETA = 1.0
GAMMA_BETA_RATIO = 0.5

# Ablation grid
ABLATION_CONFIGS = [
    # (lr_schedule, max_grad_norm, tag)
    ("cosine", 1.0, "cosine_gradnorm1"),  # A: old config baseline
    # ("linear", 1.0, "linear_gradnorm1"),    # B: isolate lr_schedule
    # ("cosine", None, "cosine_gradnormNone"),  # C: isolate grad_norm
    # ("linear", None, "linear_gradnormNone"),  # D: new config
]

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

BASE_SIMPO_KWARGS = dict(
    **TPU_CONFIGS[TPU_TYPE],
    num_train_steps=2150,
    learning_rate=6e-7,
    warmup=0.1,
    cooldown=None,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path=LLAMA3_8B_HF_PATH,
    train_seq_len=4096,
    max_seq_len=4096,
    validation_split_fraction=None,
    steps_per_eval=200,
    steps_per_checkpoint=1000,
    steps_per_hf_export=1000,
    seed=0,
    # Fixed hyperparameters
    beta=BETA,
    gamma_beta_ratio=GAMMA_BETA_RATIO,
)

training_steps = []
for lr_schedule, max_grad_norm, tag in ABLATION_CONFIGS:
    run_name = f"simpo_ablate/ultrafeedback_llama3_8b_beta1_gamma0p5_{tag}"
    simpo_config = SimpleSimPOConfig(
        **BASE_SIMPO_KWARGS,
        lr_schedule=lr_schedule,
        max_grad_norm=max_grad_norm,
    )
    training_steps.append(
        default_simpo(
            name=run_name,
            tokenized=tokenized_preferences,
            model_config=llama_8b,
            simpo_config=simpo_config,
            tags=[
                "ultrafeedback",
                "llama3",
                "ablation",
                f"beta={BETA:g}",
                f"gamma_beta_ratio={GAMMA_BETA_RATIO:g}",
                f"lr_schedule={lr_schedule}",
                f"max_grad_norm={max_grad_norm}",
            ],
        )
    )


if __name__ == "__main__":
    executor_main(
        steps=[
            preference_dataset,
            tokenized_train_preferences,
            tokenized_test_preferences,
            *training_steps,
        ]
    )
