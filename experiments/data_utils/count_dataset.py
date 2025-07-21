"""
Token counting utilities for Marin experiments.

This module provides functions to count tokens and rows in tokenized datasets,
which is essential for calculating training steps, epochs, and dataset sampling.
"""

import json
import logging
from dataclasses import dataclass

import fsspec

from levanter.data.text import cached_token_count
from marin.execution.executor import ExecutorStep, this_output_path

logger = logging.getLogger("ray")


########### Compiling token counts ###########
@dataclass
class CompileCountsConfig:
    """
    Configuration for compiling counts from tokenized datasets.

    This configuration class is used for both token counts and row counts,
    providing a unified interface for count compilation operations.

    Attributes:
        tokenization_steps: Dictionary mapping dataset short names to their GCS tokenized paths
        output_path: Path where the compiled counts will be stored
    """

    tokenization_steps: dict[str, str]
    output_path: str = this_output_path()


def get_num_rows_from_tokenized_datasets(transform_executor_steps: dict[str, str]) -> dict[str, int]:
    """
    Extract the number of rows from tokenized datasets by reading their shard ledgers.

    This function reads the `shard_ledger.json` file from each tokenized dataset
    to get the total number of rows, which is useful for calculating training steps
    and understanding dataset sizes.

    Args:
        transform_executor_steps: Dictionary mapping dataset short names to their GCS tokenized paths

    Returns:
        Dictionary mapping dataset names to their total row counts

    Example:
        >>> steps = {"dataset1": "gs://bucket/path1", "dataset2": "gs://bucket/path2"}
        >>> counts = get_num_rows_from_tokenized_datasets(steps)
        >>> print(counts)
        {'dataset1': 1000, 'dataset2': 2000}
    """
    size_dict = dict()
    for ds_short_name, gcs_tokenized_path in transform_executor_steps.items():
        json_path = f"{gcs_tokenized_path}/train/shard_ledger.json"
        # Use fsspec to read from GCS
        with fsspec.open(json_path, "r") as f:
            shard_ledger = json.load(f)
        size_dict[ds_short_name] = shard_ledger["total_num_rows"]
    return size_dict


def _compile_and_store_num_rows(
    config: CompileCountsConfig,
) -> str:
    """
    Helper function to compile row counts and store as JSON.

    This function takes the configuration, extracts row counts from all datasets,
    and stores the results as a JSON file in the specified output path.

    Args:
        config: Configuration object containing tokenization steps and output path

    Returns:
        Path to the created JSON file containing row counts

    Note:
        This is an internal helper function used by compile_and_store_num_rows_step.
    """
    import json

    import fsspec

    # Get token counts
    token_counts = get_num_rows_from_tokenized_datasets(config.tokenization_steps)

    # Store as JSON using fsspec for GCS compatibility
    output_path = config.output_path
    output_filepath = f"{output_path}/row_counts.json"

    # Create directory if it doesn't exist
    fs, path = fsspec.core.url_to_fs(output_path)
    fs.makedirs(path, exist_ok=True)

    # Write JSON file using fsspec
    with fsspec.open(output_filepath, "w") as f:
        json.dump(token_counts, f, indent=2)
        logger.info(f"Wrote row counts to {output_filepath}")

    return output_filepath


def compile_and_store_num_rows_step(
    tokenization_steps: dict[str, list[ExecutorStep]],
    output_dir: str,
) -> ExecutorStep:
    """
    Creates an ExecutorStep that compiles row counts from tokenized datasets.

    This function creates an ExecutorStep that can be used in experiment pipelines
    to automatically compute and store row counts from tokenized datasets. This is
    essential for calculating training steps and epochs in end-to-end experiments.

    Previously, row counts were manually computed and compiled, which made it
    impossible to run experiments end-to-end. This function automates that process.

    Args:
        tokenization_steps: Dictionary mapping dataset short names to their tokenization ExecutorSteps.
                           Each value should be a list containing one ExecutorStep.
        output_dir: Custom output path. Recommend that the experiment
                    folder be used since every experiment should have different
                    data mixes. rows_counts.json will be created in the output_dir.

    Returns:
        ExecutorStep that computes and stores row counts as a JSON file

    Example:
        >>> steps = {"dataset1": [tokenization_step1], "dataset2": [tokenization_step2]}
        >>> row_count_step = compile_and_store_num_rows_step(steps)
        >>> # Use in experiment pipeline
    """
    # Flatten the tokenization steps (each value is a list with one step)
    flattened_steps = {name: steps[0] for name, steps in tokenization_steps.items()}

    return ExecutorStep(
        name="compile_row_counts",
        fn=_compile_and_store_num_rows,
        config=CompileCountsConfig(
            tokenization_steps=flattened_steps,
        ),
        override_output_path=output_dir,
    )


def get_num_tokens_from_tokenized_datasets(
    transform_executor_steps: dict[str, str],
) -> dict[str, int]:
    """
    Get the number of tokens from tokenized datasets stored in GCS.

    This function loads each tokenized dataset using Levanter's cache system and
    extracts the total number of tokens from the input_ids store. This is useful
    for calculating training budgets, understanding dataset token distributions,
    and determining sampling strategies.

    Args:
        transform_executor_steps: Dictionary mapping dataset short names to their GCS tokenized paths

    Returns:
        Dictionary mapping dataset names to their total token counts

    """

    # Load the actual tokenizer object
    token_counts = {}

    for ds_short_name, gcs_tokenized_path in transform_executor_steps.items():
        # Construct the cache path for the train split
        cache_path = f"{gcs_tokenized_path}/train"

        total_tokens = cached_token_count(cache_path, field="input_ids")
        token_counts[ds_short_name] = total_tokens

        logger.info(f"Dataset {ds_short_name}: {total_tokens:,} tokens")

    return token_counts


def _compile_and_store_num_tokens(
    config: CompileCountsConfig,
) -> str:
    """
    Helper function to compile token counts and store as JSON.

    This function takes the configuration, extracts token counts from all datasets,
    and stores the results as a JSON file in the specified output path.

    Args:
        config: Configuration object containing tokenization steps and output path

    Returns:
        Path to the created JSON file containing token counts

    Note:
        This is an internal helper function used by compile_and_store_num_tokens_step.
    """
    token_counts = get_num_tokens_from_tokenized_datasets(config.tokenization_steps)

    # Store as JSON using fsspec for GCS compatibility
    output_dir = config.output_path
    output_filepath = f"{output_dir}/token_counts.json"

    # Create directory if it doesn't exist
    fs, path = fsspec.core.url_to_fs(output_dir)
    fs.makedirs(path, exist_ok=True)

    # Write JSON file using fsspec
    with fsspec.open(output_filepath, "w") as f:
        json.dump(token_counts, f, indent=2)
        logger.info(f"Wrote token counts to {output_filepath}")

    return output_filepath


def compile_and_store_num_tokens_step(
    tokenization_steps: dict[str, list[ExecutorStep]],
    output_dir: str,
) -> ExecutorStep:
    """
    Creates an ExecutorStep that compiles token counts from tokenized datasets.

    This function creates an ExecutorStep that can be used in experiment pipelines
    to automatically compute and store token counts from tokenized datasets. This is
    essential for calculating training steps, epochs, and determining how to sample
    datasets given a token budget.

    Previously, token counts were manually computed and compiled, which made it
    impossible to run experiments end-to-end. This function automates that process.

    Args:
        tokenization_steps: Dictionary mapping dataset short names to their tokenization ExecutorSteps.
                           Each value should be a list containing one ExecutorStep.
        output_dir: Custom output path. Recommend that the experiment
                    folder be used since every experiment should have different
                    data mixes. token_counts.json will be created in the output_dir.

    Returns:
        ExecutorStep that computes and stores token counts as a JSON file

    Example:
        >>> steps = {"dataset1": [tokenization_step1], "dataset2": [tokenization_step2]}
        >>> token_count_step = compile_and_store_num_tokens_step(steps)
        >>> # Use in experiment pipeline

    Note:
        This function creates an ExecutorStep that will load each dataset's cache
        and count tokens, which can be memory-intensive for large datasets.
    """
    # Flatten the tokenization steps (each value is a list with one step)
    flattened_steps = {name: steps[0] for name, steps in tokenization_steps.items()}

    return ExecutorStep(
        name="compile_token_counts",
        fn=_compile_and_store_num_tokens,
        config=CompileCountsConfig(
            tokenization_steps=flattened_steps,
        ),
        override_output_path=output_dir,
    )
