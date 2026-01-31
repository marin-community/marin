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
Run a SimPO sweep on the Ultrafeedback preference dataset.

Configure BETAS and GAMMAS below to control the sweep grid.
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
# Global train_batch_size stays at 128, per_device_parallelism adjusts for memory
TPU_CONFIGS = {
    "v5p-8": {
        "resources": ResourceConfig.with_tpu("v5p-8"),
        "train_batch_size": 128,  # Keep global batch size constant
        "per_device_parallelism": 16,  # 4 devices: 128/4=32 per device, 32/16=2 microbatch
        "per_device_eval_parallelism": 16,
    },
    "v5p-16": {
        "resources": ResourceConfig.with_tpu("v5p-16"),
        "train_batch_size": 128,  # Baseline
        "per_device_parallelism": 8,  # 8 devices: 128/8=16 per device, 16/8=2 microbatch
        "per_device_eval_parallelism": 8,
    },
    "v5p-32": {
        "resources": ResourceConfig.with_tpu("v5p-32"),
        "train_batch_size": 128,  # Keep global batch size constant
        "per_device_parallelism": 4,  # 16 devices: 128/16=8 per device, 8/4=2 microbatch
        "per_device_eval_parallelism": 4,
    },
    "v5p-64": {
        "resources": ResourceConfig.with_tpu("v5p-64"),
        "train_batch_size": 128,  # Keep global batch size constant
        "per_device_parallelism": -1,  # Auto-determine based on memory constraints
        "per_device_eval_parallelism": -1,
    },
}

# Select TPU type here
TPU_TYPE = "v5p-64"  # Change to "v5p-8", "v5p-32", "v5p-64", etc.

# Sweep configuration.
BETAS = [0.1, 1.0, 2.0]
GAMMA_BETA_RATIOS = [0.2, 0.5, 0.8]


def _format_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _validate_sweep_values(values: list[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if any(v <= 0 for v in values):
        raise ValueError(f"{name} must be positive")


_validate_sweep_values(BETAS, "BETAS")
_validate_sweep_values(GAMMA_BETA_RATIOS, "GAMMA_BETA_RATIOS")

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
    lr_schedule="linear",
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
)

training_steps = []
for beta in BETAS:
    for gamma_beta_ratio in GAMMA_BETA_RATIOS:
        beta_tag = _format_float(beta)
        gamma_tag = _format_float(gamma_beta_ratio)
        run_name = f"simpo/ultrafeedback_llama3_8b_beta{beta_tag}_gamma{gamma_tag}"
        simpo_config = SimpleSimPOConfig(
            **BASE_SIMPO_KWARGS,
            beta=beta,
            gamma_beta_ratio=gamma_beta_ratio,
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
                    f"beta={beta:g}",
                    f"gamma_beta_ratio={gamma_beta_ratio:g}",
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
