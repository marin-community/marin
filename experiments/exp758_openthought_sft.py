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

from experiments.defaults import default_sft, default_tokenize
from experiments.exp964_custom_chat_tokenizer import llama3_instruct_chat_format
from experiments.llama import llama3_tokenizer, llama_8b
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.processing.tokenize import lm_data_config

# Get instruction dataset
openthoughts_dataset = get_instruction_dataset("open-r1/OpenThoughts-114k-math")

# TODO: tune this for a good number of steps
NUM_TRAIN_STEPS = 2500

# Add tokenization step
openthoughts_llama_tokenize_step = default_tokenize(
    name="openthoughts_llama3_tokenizer",
    dataset=openthoughts_dataset / "**/*.jsonl.gz",
    tokenizer=llama3_tokenizer,
    format=llama3_instruct_chat_format,
)
openthoughts_sft_config = SimpleSFTConfig(
    train_batch_size=128,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=5e-6,
    resources=TpuPodConfig(tpu_type="v4-128"),
    tokenizer=llama3_tokenizer,
    model_name_or_path="meta-llama/Llama-3.1-8B",
    max_seq_len=4096,
    seed=1,
    # Additional parameters from original config
    weight_decay=0.0,
    warmup=0.03,
    cooldown=0.0,
    min_lr_ratio=0.0,
    lr_schedule="linear",
    steps_per_hf_export=500,
)

# Create the SFT training step using the pre-defined 8B model config
sft_step = default_sft(
    name="openthoughts_llama3_sft",
    tokenized=lm_data_config(openthoughts_llama_tokenize_step, permutation_type="linear"),
    model_config=llama_8b,
    sft_config=openthoughts_sft_config,
    tags=["openthoughts", "llama", "sft"],
)

if __name__ == "__main__":
    executor_main(steps=[openthoughts_llama_tokenize_step, sft_step])
