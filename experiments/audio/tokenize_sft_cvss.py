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

# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
Custom instruction dataset functions for audio fine-tuning experiments.

This module provides functionality to prepare custom HuggingFace datasets
for Supervised Fine-Tuning (SFT), specifically designed for audio-related
instruction following tasks.
"""

import dataclasses
from collections.abc import Sequence

from experiments.defaults import default_tokenize
from experiments.posttrain.instruction_datasets import (
    InstructionDatasetConfig,
    instruction_response_adapter,
    transform_dataset_step,
)
from levanter.data.text import ChatLmDatasetFormat
from marin.execution.executor import ExecutorStep

CUSTOM_INSTRUCTION_DATASET_CONFIGS = {
    "rma9248/cvss_method1": InstructionDatasetConfig(
        hf_dataset_id="rma9248/cvss_method1",
        revision="fc5e8f53dbb6aa870e3a8cd0153706b4247eb786",
        adapter=instruction_response_adapter(
            instruction_column="instruction",
            response_column="response",
        ),
        metadata_columns=["id", "lang"],
        name="rma9248/cvss_method1",
    ),
    "rma9248/cvss_method2": InstructionDatasetConfig(
        hf_dataset_id="rma9248/cvss_method2",
        revision="6c24a365d28fe1abd87979fd32c998042e9ee454",
        adapter=instruction_response_adapter(
            instruction_column="instruction",
            response_column="response",
        ),
        metadata_columns=["id", "lang"],
        name="rma9248/cvss_method2",
    ),
}


def get_instruction_dataset_custom(hf_dataset_id: str, splits: Sequence[str] | None = None) -> ExecutorStep:
    """
    Get a custom instruction dataset for SFT.

    This function creates an ExecutorStep that processes a custom HuggingFace dataset
    for supervised fine-tuning. It supports datasets with instruction/response format
    and additional metadata fields.

    Args:
        hf_dataset_id: The HuggingFace dataset repository ID (e.g., "rma9248/cvss_method1")
        splits: Optional list of dataset splits to use. Defaults to ["train"]

    Returns:
        ExecutorStep that transforms the dataset into the appropriate format for SFT

    Raises:
        AssertionError: If the dataset ID is not found in CUSTOM_INSTRUCTION_DATASET_CONFIGS

    Example:
        >>> # Get the cvss_method1 dataset for training
        >>> dataset_step = get_instruction_dataset_custom("rma9248/cvss_method1")
        >>>
        >>> # Get specific splits
        >>> dataset_step = get_instruction_dataset_custom(
        ...     "rma9248/cvss_method2",
        ...     splits=["train", "validation"]
        ... )
    """
    # Validate that the dataset is supported
    assert hf_dataset_id in CUSTOM_INSTRUCTION_DATASET_CONFIGS, (
        f"Unknown custom instruction dataset: {hf_dataset_id}. "
        f"Available datasets: {list(CUSTOM_INSTRUCTION_DATASET_CONFIGS.keys())}"
    )

    # Get the original configuration
    original_config = CUSTOM_INSTRUCTION_DATASET_CONFIGS[hf_dataset_id]

    # Use provided splits or default to train
    if splits is None:
        splits = original_config.splits

    # Create a new config with the specified splits
    config = InstructionDatasetConfig(
        **{k: v for k, v in original_config.__dict__.items() if k != "splits"}, splits=splits
    )

    return transform_dataset_step(config)


def get_available_custom_datasets() -> list[str]:
    """
    Get a list of all available custom instruction datasets.

    Returns:
        List of dataset IDs that can be used with get_instruction_dataset_custom
    """
    return list(CUSTOM_INSTRUCTION_DATASET_CONFIGS.keys())


def update_dataset_revision(hf_dataset_id: str, revision: str) -> None:
    """
    Update the revision hash for a custom dataset configuration.

    This is useful when you want to pin to a specific commit for reproducibility.

    Args:
        hf_dataset_id: The dataset ID to update
        revision: The commit hash or branch name to use

    Raises:
        KeyError: If the dataset ID is not found
    """
    if hf_dataset_id not in CUSTOM_INSTRUCTION_DATASET_CONFIGS:
        raise KeyError(f"Dataset {hf_dataset_id} not found in custom configurations")

    # Create a new config with updated revision
    old_config = CUSTOM_INSTRUCTION_DATASET_CONFIGS[hf_dataset_id]
    new_config = dataclasses.replace(old_config, revision=revision)
    CUSTOM_INSTRUCTION_DATASET_CONFIGS[hf_dataset_id] = new_config


# Simple chat template for instruction-response format with masking
# Format: <|begin_of_text|>instruction+response<|end_of_text|>
# Only the response portion (+ <|end_of_text|> token) is used for training loss
SIMPLE_INSTRUCTION_CHAT_TEMPLATE = """<|begin_of_text|>
{%- if messages[0]['role'] == 'user' -%}
{{- messages[0]['content'] | trim }}
{%- endif -%}
{%- if messages[1]['role'] == 'assistant' -%}
{% generation %}{{ messages[1]['content'] | trim }}<|end_of_text|>{% endgeneration %}
{%- endif -%}"""


def tokenize_instruction_dataset(
    tokenizer_name: str, dataset_name: str, splits: Sequence[str] | None = None
) -> ExecutorStep:
    """
    Tokenize a custom instruction dataset for SFT with proper masking.

    This function creates a tokenized version of the instruction dataset where:
    - The format is: <|begin_of_text|>instruction+response<|end_of_text|>
    - Only the response portion (+ <|end_of_text|> token) is used for training loss
    - The instruction portion is masked out from the training loss

    The chat template explicitly includes <|begin_of_text|> and <|end_of_text|> tokens,
    so there's no need for the tokenizer to have bos_token or eos_token defined.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier (e.g., "potsawee/marin-mimi-bpe-8cb-16k-tokenizer")
        dataset_name: Dataset name from CUSTOM_INSTRUCTION_DATASET_CONFIGS (e.g., "rma9248/cvss_method1")
        splits: Optional list of dataset splits to use. Defaults to ["train"]

    Returns:
        ExecutorStep that produces the tokenized dataset with proper masking

    Example:
        >>> # Tokenize cvss_method1 with custom tokenizer
        >>> tokenize_step = tokenize_instruction_dataset(
        ...     "potsawee/marin-mimi-bpe-8cb-16k-tokenizer",
        ...     "rma9248/cvss_method1"
        ... )
    """
    # Get the raw instruction dataset
    dataset_step = get_instruction_dataset_custom(dataset_name, splits=splits)

    # Create a clean name for the tokenized output
    dataset_clean_name = dataset_name.replace("/", "--").replace(".", "-")
    tokenizer_clean_name = tokenizer_name.split("/")[-1].lower().replace("-", "_")

    # Use ChatLmDatasetFormat with masking enabled
    chat_format = ChatLmDatasetFormat(
        messages_field="messages",
        chat_template=SIMPLE_INSTRUCTION_CHAT_TEMPLATE,
        pack=True,
        mask_user_turns=True,  # This masks the instruction (user turn) from loss
    )

    # Create tokenization step
    # Note: We don't pass enforce_bos/enforce_eos because:
    # 1. They don't apply to ChatLmDatasetFormat (only to raw text)
    # 2. The chat template explicitly includes <|begin_of_text|> and <|end_of_text|>
    return default_tokenize(
        name=f"{dataset_clean_name}_{tokenizer_clean_name}_sft",
        dataset=dataset_step / "**/*.jsonl.gz",
        tokenizer=tokenizer_name,
        format=chat_format,
    )


def create_tokenization_steps_for_audio_sft(tokenizer_name: str) -> dict[str, ExecutorStep]:
    """
    Create tokenization steps for all available custom instruction datasets.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier

    Returns:
        Dictionary mapping dataset names to their tokenization steps

    Example:
        >>> steps = create_tokenization_steps_for_audio_sft("potsawee/marin-mimi-bpe-8cb-16k-tokenizer")
        >>> cvss_method1_step = steps["rma9248/cvss_method1"]
    """
    steps = {}
    for dataset_name in get_available_custom_datasets():
        steps[dataset_name] = tokenize_instruction_dataset(tokenizer_name, dataset_name)
    return steps


if __name__ == "__main__":
    from marin.execution.executor import executor_main

    TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"

    # Create tokenization steps for all datasets
    tokenization_steps = create_tokenization_steps_for_audio_sft(TOKENIZER)

    # Also include the raw dataset preparation steps
    dataset_preparation_steps = [
        get_instruction_dataset_custom("rma9248/cvss_method1"),
        get_instruction_dataset_custom("rma9248/cvss_method2"),
    ]

    executor_main(
        steps=[*dataset_preparation_steps, *tokenization_steps.values()],
        description="Process and tokenize custom CVSS instruction datasets for SFT",
    )
