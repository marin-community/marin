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

"""Reference: Single-run pretraining → midtraining → SFT pipeline.

Demonstrates that pretrain/midtrain/SFT are all just data mixing phases.
The entire pipeline is one training run with time-varying mixture weights:

  1. Pretrain (steps 0-20k): DCLM baseline
  2. Midtrain (steps 20k-25k): Blend DCLM + Dolmino math
  3. SFT (steps 25k-26k): SmolTalk instruction data
"""

import dataclasses

from levanter.data.text import ChatLmDatasetFormat
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_tokenize, default_train
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.pretraining_datasets.dolmino import tokenize_dolmino
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

# --- Model: 600M LLaMA ---
model = LlamaConfig(
    max_seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3584,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
    cross_entropy_block_size=32000,  # blockwise CE to reduce memory spike
)

# --- Schedule ---
PRETRAIN_STEPS = 20_000
MIDTRAIN_STEPS = 5_000
SFT_STEPS = 1_000
TOTAL_STEPS = PRETRAIN_STEPS + MIDTRAIN_STEPS + SFT_STEPS

# --- Data components ---
pretrain = {"dclm": dclm_components_llama3["dclm_baseline"]}

dolmino = tokenize_dolmino()
midtrain = {"dolmino_math": dolmino["dolmino/math/metamath-owmfilter"]}

smoltalk = get_instruction_dataset("HuggingFaceTB/smoltalk", splits=["train"])
sft = {
    "smoltalk": default_tokenize(
        name="smoltalk_marin",
        dataset=smoltalk / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )
}

# --- Time-varying mixture weights ---
data = lm_varying_mixture_data_config(
    components={**pretrain, **midtrain, **sft},
    weights_list=[
        (0, {"dclm": 1.0, "dolmino_math": 0.0, "smoltalk": 0.0}),
        (PRETRAIN_STEPS, {"dclm": 0.7, "dolmino_math": 0.3, "smoltalk": 0.0}),
        (PRETRAIN_STEPS + MIDTRAIN_STEPS, {"dclm": 0.0, "dolmino_math": 0.0, "smoltalk": 1.0}),
    ],
)
# Override tokenizer to use marin_tokenizer (same vocab as llama3 but with chat template for SFT)
data = dataclasses.replace(data, tokenizer=marin_tokenizer)

# --- Training ---
train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-16"),
    train_batch_size=512,
    num_train_steps=TOTAL_STEPS,
    learning_rate=3e-3,
    weight_decay=0.1,
    warmup=0.05,
    decay=0.2,
    steps_per_eval=500,
)

training_step = default_train(
    name="reference-pipeline",
    tokenized=data,
    model_config=model,
    train_config=train_config,
    tags=["reference", "pipeline"],
    eval_harness_tasks=[],
)

if __name__ == "__main__":
    executor_main(steps=[training_step])
