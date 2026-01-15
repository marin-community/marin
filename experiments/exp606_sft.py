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
from experiments.llama import llama3_instruct_tokenizer, llama_8b
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, step
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

tulu_sft_config = SimpleSFTConfig(
    train_batch_size=128,
    num_train_steps=NUM_TRAIN_STEPS,  # Adjust as needed.
    learning_rate=5e-6,
    resources=ResourceConfig.with_tpu("v4-128", slice_count=1),
    tokenizer=llama3_instruct_tokenizer,
    model_name_or_path="meta-llama/Llama-3.1-8B",
    max_seq_len=4096,
    seed=1,
)


def tokenize_tulu3_llama():
    return default_tokenize(
        name="tulu_sft_v3_llama3_instruct_tokenizer",
        dataset=tulu_3_dataset / "**/*.jsonl.gz",
        tokenizer=llama3_instruct_tokenizer,
        format=llama3_instruct_chat_format,
    )


@step(name="exp606-sft/all")
def run_tulu3_sft():
    """Entry point for Tulu-3 SFT training."""
    tulu3_llama_tokenize_step = tokenize_tulu3_llama()
    tulu3_llama_data = lm_data_config(tulu3_llama_tokenize_step, permutation_type="linear")

    sft_step = default_sft(
        name="tulu3_llama3_sft", tokenized=tulu3_llama_data, model_config=llama_8b, sft_config=tulu_sft_config
    )

    return sft_step


if __name__ == "__main__":
    executor_main(steps=[run_tulu3_sft()])
