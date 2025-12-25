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

# This script is a helper script to download, transform, tokenize, and count
# tokens for SFT datasets.
# In addition to previous SFT datasets, we include Nemotron SFT and OpenThoughts3-1.2M.

import logging

from levanter.data.text import ChatLmDatasetFormat

from experiments.data_utils.count_dataset import compile_and_store_num_rows_step, compile_and_store_num_tokens_step
from experiments.defaults import default_tokenize

# Define datasets
from experiments.exp808_sft_mixture import DATASETS as EXP808_DATASETS
# from experiments.marin_models import marin_tokenizer as tokenizer
# chat_template = None
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    download_dataset_step,
    get_instruction_dataset,
    transform_dataset_step,
)
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
)

# Qwen tokenizer from HuggingFace - use the model's tokenizer to match training
# This matches the tokenizer used by Qwen2.5-7B-Instruct model
tokenizer = "Qwen/Qwen2.5-7B-Instruct"
# Qwen-compatible chat template with {%generation%} tag for Levanter
# Based on Qwen's typical <|im_start|>/<|im_end|> format
chat_template = """{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] | trim }}<|im_end|>
{%- elif message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] | trim }}<|im_end|>
{%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{% generation %}{{ message['content'] | trim }}<|im_end|>{% endgeneration %}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- endif -%}""".strip()

logger = logging.getLogger("ray")


########### Tokenization ###########
def create_tokenization_step(dataset_name: str) -> ExecutorStep:
    # This is a modified version of the `create_tokenization_step` function in exp808_sft_mixture.py
    """
    Creates a tokenization ExecutorStep for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'TIGER-Lab/AceCode-89K', 'HuggingFaceTB/smoltalk')

    Returns:
        ExecutorStep configured for tokenizing the specified dataset
    """
    # Get the dataset with only train split
    if dataset_name == "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT":
        dataset = get_instruction_dataset(dataset_name, splits=["chat", "code", "math", "science", "safety"])
    elif dataset_name == "nvidia/Nemotron-Post-Training-Dataset-v2-SFT":
        dataset = get_instruction_dataset(dataset_name, splits=["chat", "code", "math", "stem"])
    else:
        dataset = get_instruction_dataset(dataset_name, splits=["train"])

    # Get the last part of the path and clean it up
    short_name = dataset_name.split("/")[-1].lower().replace("-", "_")

    # Use .jsonl.gz extension since transform_and_write_batch produces .jsonl.gz files
    dataset_path = dataset / "**/*.jsonl.gz"

    return default_tokenize(
        name=f"{short_name}_qwen_tokenizer",
        dataset=dataset_path,
        tokenizer=tokenizer,
        format=ChatLmDatasetFormat(chat_template=chat_template),
    )


DATASETS = {
    **EXP808_DATASETS,
    "nemotron_sft": "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT",
    # "nemotron_v2_sft": "nvidia/Nemotron-Post-Training-Dataset-v2-SFT",
    "openthoughts3": "open-thoughts/OpenThoughts3-1.2M",
}


def download_transform_tokenize_count_steps():
    ALL_STEPS = []
    TOKENIZATION_STEPS = dict()
    for short_ds_name, full_ds_name in DATASETS.items():
        # Download the dataset
        config = INSTRUCTION_DATASET_NAME_TO_CONFIG[full_ds_name]
        data_download_step = download_dataset_step(config)
        # Transform the dataset
        data_transform_step = transform_dataset_step(config, data_download_step)
        # Tokenize the dataset
        data_tokenize_step = create_tokenization_step(full_ds_name)

        ALL_STEPS += [data_download_step, data_transform_step, data_tokenize_step]
        TOKENIZATION_STEPS[short_ds_name] = [data_tokenize_step]

    # Compile token counts
    ALL_STEPS.append(
        compile_and_store_num_rows_step(TOKENIZATION_STEPS, "experiments/exp905a_nemotron_sft_dstc/num_rows")
    )
    ALL_STEPS.append(
        compile_and_store_num_tokens_step(TOKENIZATION_STEPS, "experiments/exp905a_nemotron_sft_dstc/num_tokens")
    )

    return ALL_STEPS


########### Main ###########
if __name__ == "__main__":
    ALL_STEPS = download_transform_tokenize_count_steps()
    executor_main(steps=ALL_STEPS)
