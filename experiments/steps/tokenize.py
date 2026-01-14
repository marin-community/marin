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
Step wrappers for tokenizing datasets.

This module provides step definitions that wrap the library functions in
marin.processing.tokenize.
"""

from levanter.data.text import LmDatasetFormatBase, TextLmDatasetFormat

from marin.execution import ExecutorStep, StepContext, step
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfig, tokenize


@step(name="{name}", fn=tokenize)
def tokenize_from_paths(
    ctx: StepContext,
    name: str,
    train_paths: list[str],
    validation_paths: list[str],
    tokenizer: str,
    tags: list[str] | None = None,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),
    sample_count: int | None = None,
    window_size_bytes: int = 10_000_000_000,
    allow_test_in_train: bool = False,
):
    """
    Create a step that tokenizes datasets from raw file paths.

    Args:
        ctx: Step context (automatically provided by @step decorator)
        name: Name for this tokenization step (used in output path)
        train_paths: List of paths to training data files
        validation_paths: List of paths to validation data files
        tokenizer: HuggingFace tokenizer name
        tags: Optional tags to be added to config
        format: The format of the dataset (default: TextLmDatasetFormat)
        sample_count: Number of samples to tokenize. If None, tokenize all samples.
        window_size_bytes: Window size for bundling files
        allow_test_in_train: If True, allows 'test' or 'validation' in train_paths
    """
    return TokenizeConfig(
        train_paths=train_paths,
        validation_paths=validation_paths,
        cache_path=ctx.output,
        tokenizer=tokenizer,
        tags=tags or [],
        sample_count=sample_count,
        format=format,
        window_size_bytes=window_size_bytes,
        allow_test_in_train=allow_test_in_train,
    )


@step(name="{name}", fn=tokenize)
def tokenize_from_step(
    ctx: StepContext,
    name: str,
    input_step: ExecutorStep,
    tokenizer: str,
    tags: list[str] | None = None,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),
    sample_count: int | None = None,
    window_size_bytes: int = 10_000_000_000,
    allow_test_in_train: bool = False,
    is_validation: bool = False,
):
    """
    Create a step that tokenizes a dataset from another ExecutorStep's output.

    Args:
        ctx: Step context (automatically provided by @step decorator)
        name: Name for this tokenization step (used in output path)
        input_step: The step whose output to tokenize
        tokenizer: HuggingFace tokenizer name
        tags: Optional tags to be added to config
        format: The format of the dataset (default: TextLmDatasetFormat)
        sample_count: Number of samples to tokenize. If None, tokenize all samples.
        window_size_bytes: Window size for bundling files
        allow_test_in_train: If True, allows 'test' or 'validation' in train_paths
        is_validation: Whether the dataset is a validation set
    """
    resolved_path = ctx.require(input_step)

    return TokenizeConfig(
        train_paths=[resolved_path] if not is_validation else [],
        validation_paths=[resolved_path] if is_validation else [],
        cache_path=ctx.output,
        tokenizer=tokenizer,
        tags=tags or [],
        sample_count=sample_count,
        format=format,
        window_size_bytes=window_size_bytes,
        allow_test_in_train=allow_test_in_train,
    )


@step(name="{name}", fn=tokenize)
def tokenize_hf_dataset(
    ctx: StepContext,
    name: str,
    hf_dataset_id: str,
    tokenizer: str,
    revision: str | None = None,
    hf_dataset_name: str | None = None,
    tags: list[str] | None = None,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),
    sample_count: int | None = None,
    window_size_bytes: int = 10_000_000_000,
):
    """
    Create a step that tokenizes a HuggingFace dataset directly.

    Args:
        ctx: Step context (automatically provided by @step decorator)
        name: Name for this tokenization step (used in output path)
        hf_dataset_id: HuggingFace dataset ID (e.g., "org/dataset")
        tokenizer: HuggingFace tokenizer name
        revision: HF dataset revision (commit hash, branch, or tag)
        hf_dataset_name: HF dataset name (subset name)
        tags: Optional tags to be added to config
        format: The format of the dataset (default: TextLmDatasetFormat)
        sample_count: Number of samples to tokenize. If None, tokenize all samples.
        window_size_bytes: Window size for bundling files
    """
    return HfTokenizeConfig(
        id=hf_dataset_id,
        cache_path=ctx.output,
        tokenizer=tokenizer,
        revision=revision,
        name=hf_dataset_name,
        tags=tags or [],
        format=format,
        sample_count=sample_count,
        window_size_bytes=window_size_bytes,
    )
