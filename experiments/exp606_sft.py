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
SFT dataset definitions for Tulu-3 instruction tuning.

This module provides the `tulu3_llama_data_old` dataset configuration and
`tulu_sft_config` used by various SFT training experiments.
"""

from fray.cluster import ResourceConfig

from experiments.defaults import default_tokenize
from experiments.llama import llama3_instruct_chat_format, llama3_instruct_tokenizer
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.simple_sft_config import SimpleSFTConfig
from marin.processing.tokenize import lm_data_config

# Get instruction dataset
tulu_3_dataset = get_instruction_dataset("allenai/tulu-3-sft-mixture")

# Number of tokens is 670,426,314
NUM_TRAIN_TOKENS = 670426314
# number of epochs over the dataset set to reproduce Olmo SFT v2
# or Tulu 3 starting from Llama 3.1 8B. This script
# is used to reproduce the Tulu 3 SFT model.
# Link: https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (128 * 4096) * 3  # 2 epochs

# Create tokenization step for Tulu-3 dataset
tulu3_llama_tokenize_step = default_tokenize(
    name="tulu_sft_v3_llama3_instruct_tokenizer",
    dataset=tulu_3_dataset / "**/*.jsonl.gz",
    tokenizer=llama3_instruct_tokenizer,
    format=llama3_instruct_chat_format,
)

# This dataset should only by used for older runs. Don't use this in new experiments
tulu3_llama_data_old = lm_data_config(tulu3_llama_tokenize_step, permutation_type="linear")

tulu_sft_config = SimpleSFTConfig(
    train_batch_size=128,
    num_train_steps=NUM_TRAIN_STEPS,  # Adjust as needed.
    learning_rate=5e-6,
    resources=ResourceConfig.with_tpu("v4-128", slice_count=1),
    tokenizer=llama3_instruct_tokenizer,
    initialize_from_hf="meta-llama/Llama-3.1-8B",
    max_seq_len=4096,
    seed=1,
)
