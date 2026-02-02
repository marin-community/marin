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
Train a 1B Llama model on Nemotron CC high-quality data (real + synthetic).
"""

import dataclasses

from experiments.defaults import default_train
from experiments.evals.evals import default_base_eval
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.llama import llama3_tokenizer, llama_3_2_1b
from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from fray.cluster import ResourceConfig


################################################################
# Nemotron HQ Data Configuration
################################################################

# Get all Nemotron tokenized datasets
nemotron_tokenized = tokenize_nemotron(tokenizer=llama3_tokenizer)

# Only use high-quality data (real + synthetic)
nemotron_hq_components = {
    "nemotron_cc/hq_actual": nemotron_tokenized["nemotron_cc/hq_actual"],
    "nemotron_cc/hq_synth": nemotron_tokenized["nemotron_cc/hq_synth"],
}

# Weights based on dataset sizes (in TiB)
nemotron_hq_weights = {
    "nemotron_cc/hq_actual": 0.91351,
    "nemotron_cc/hq_synth": 2.72,
}

# Normalize weights to sum to 1
total_weight = sum(nemotron_hq_weights.values())
nemotron_hq_weights_normalized = {k: v / total_weight for k, v in nemotron_hq_weights.items()}

nemotron_hq_data_config = lm_mixture_data_config(
    components=nemotron_hq_components,
    weights=nemotron_hq_weights_normalized,
)


################################################################
# 1B Model Training Configuration
################################################################

tootsie_1b_train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v4-64"),
    train_batch_size=512,
    num_train_steps=100_000,
    learning_rate=3e-4,
    weight_decay=0.1,
    warmup=1000,
    decay=0.1,
    lr_schedule="linear",
    steps_per_eval=5000,
    steps_per_export=10000,
)

llama_1b_tootsie = dataclasses.replace(
    default_train(
        name="llama-1b-tootsie",
        tokenized=nemotron_hq_data_config,
        model_config=llama_3_2_1b,
        train_config=tootsie_1b_train_config,
        tags=["llama", "1b", "nemotron-hq", "exp600"],
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    ),
    override_output_path="checkpoints/llama-1b-tootsie",
)


if __name__ == "__main__":
    executor_main(
        steps=[
            llama_1b_tootsie,
            *default_base_eval(llama_1b_tootsie),
        ],
        description="Train 1B Llama model on Nemotron CC high-quality data (real + synthetic).",
    )
