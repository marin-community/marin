# This script is a helper script to download, transform, tokenize, and compile
# token counts for SFT datasets.
# In addition to previous SFT datasets, we include Nemotron SFT and OpenThoughts3-1.2M.

from instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    transform_dataset_step,
    download_dataset_step,
    get_instruction_dataset,
)
from levanter.data.text import ChatLmDatasetFormat
from dataclasses import dataclass

from experiments.defaults import default_tokenize, this_output_path
from experiments.marin_models import marin_tokenizer
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
)
import json
import fsspec

import logging
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
        dataset = get_instruction_dataset(dataset_name, splits=['chat', 'code', 'math', 'science', 'safety'])
    else:
        dataset = get_instruction_dataset(dataset_name, splits=['train'])

    # Get the last part of the path and clean it up
    short_name = dataset_name.split("/")[-1].lower().replace("-", "_")
    
    # Use .jsonl.gz extension since transform_and_write_batch produces .jsonl.gz files
    dataset_path = dataset / "**/*.jsonl.gz"
        
    return default_tokenize(
        name=f"{short_name}_marin_tokenizer",
        dataset=dataset_path,
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )


########### Compiling token counts ###########
@dataclass
class CompileTokenCountsConfig:
    tokenization_steps: dict[str, str]
    output_path: str = this_output_path()

def get_num_rows_from_tokenized_datasets(transform_executor_steps: dict[str, str]) -> dict[str, int]:
    size_dict = dict()
    for ds_short_name, gcs_tokenized_path in transform_executor_steps.items():
        json_path = f"{gcs_tokenized_path}/train/shard_ledger.json"
        # Use fsspec to read from GCS
        with fsspec.open(json_path, 'r') as f:
            shard_ledger = json.load(f)
        size_dict[ds_short_name] = shard_ledger['total_num_rows']
    return size_dict

def _compile_and_store_num_rows(config: CompileTokenCountsConfig) -> str:
    """Helper function to compile counts and store as JSON"""
    import json
    import fsspec
    
    # Get token counts
    token_counts = get_num_rows_from_tokenized_datasets(config.tokenization_steps)
    
    # Store as JSON using fsspec for GCS compatibility
    output_path = config.output_path
    output_file_path = f"{output_path}/row_counts.json"
    
    # Create directory if it doesn't exist
    fs, path = fsspec.core.url_to_fs(output_path)
    fs.makedirs(path, exist_ok=True)
    
    # Write JSON file using fsspec
    with fsspec.open(output_file_path, 'w') as f:
        json.dump(token_counts, f, indent=2)
        logger.info(f"Wrote row counts to {output_file_path}")
    
    return output_file_path

def compile_and_store_num_rows_step(tokenization_steps: dict[str, list[ExecutorStep]]) -> ExecutorStep:
    """
    Creates an ExecutorStep that compiles token counts from tokenized datasets.
    We need this to 1) calculate number of epochs, 2) decide how to sample given a token budget
    
    Previously, we manually compute and compile this dict, which makes it impossible to run
    experiments end-to-end.
    
    Args:
        tokenization_steps: Dictionary mapping dataset short names to their tokenization ExecutorSteps
        
    Returns:
        ExecutorStep that computes and returns token counts as dictionary
    """

    # Flatten the tokenization steps (each value is a list with one step)
    flattened_steps = {name: steps[0] for name, steps in tokenization_steps.items()}
    
    return ExecutorStep(
        name="scratch/thinking_sft/compile_row_counts",
        fn=_compile_and_store_num_rows,
        config=CompileTokenCountsConfig(tokenization_steps=flattened_steps),
    )


def get_num_tokens_from_tokenized_datasets(transform_executor_steps: dict[str, str]) -> dict[str, int]:
    """
    Get the number of tokens from tokenized datasets stored in GCS.
    
    Args:
        transform_executor_steps: Dictionary mapping dataset short names to their GCS tokenized paths
        
    Returns:
        Dictionary mapping dataset names to their total token counts
    """
    from levanter.data.text import load_lm_dataset_cache, TextLmDatasetFormat
    from levanter.compat.hf_checkpoints import load_tokenizer
    from experiments.marin_models import marin_tokenizer
    
    # Load the actual tokenizer object
    tokenizer = load_tokenizer(marin_tokenizer)
    
    token_counts = {}
    
    for ds_short_name, gcs_tokenized_path in transform_executor_steps.items():
        # Construct the cache path for the train split
        cache_path = f"{gcs_tokenized_path}/train"
        
        # Load the cache using Levanter's load_lm_dataset_cache function
        # We use TextLmDatasetFormat as the default format since we're just counting tokens
        cache = load_lm_dataset_cache(
            cache_path,
            format=TextLmDatasetFormat(),
            tokenizer=tokenizer,
            enforce_eos=True
        )
        
        # Wait for the cache to be fully loaded
        cache.await_finished()
        
        # Get the total number of tokens from the input_ids store
        total_tokens = cache.store.tree["input_ids"].data_size
        token_counts[ds_short_name] = total_tokens
        
        logger.info(f"Dataset {ds_short_name}: {total_tokens:,} tokens")
    
    return token_counts

def _compile_and_store_num_tokens(config: CompileTokenCountsConfig) -> str:
    """Helper function to compile token counts and store as JSON"""
    token_counts = get_num_tokens_from_tokenized_datasets(config.tokenization_steps)
    
    # Store as JSON using fsspec for GCS compatibility
    output_path = config.output_path
    output_file_path = f"{output_path}/token_counts.json"
    
    # Create directory if it doesn't exist
    fs, path = fsspec.core.url_to_fs(output_path)
    fs.makedirs(path, exist_ok=True)
    
    # Write JSON file using fsspec
    with fsspec.open(output_file_path, 'w') as f:
        json.dump(token_counts, f, indent=2)
        logger.info(f"Wrote token counts to {output_file_path}")
    
    return output_file_path

def compile_and_store_num_tokens_step(tokenization_steps: dict[str, list[ExecutorStep]]) -> ExecutorStep:
    """
    Creates an ExecutorStep that compiles token counts from tokenized datasets.
    We need this to 1) calculate number of epochs, 2) decide how to sample given a token budget
    
    Previously, we manually compute and compile this dict, which makes it impossible to run
    experiments end-to-end.
    
    Args:
        tokenization_steps: Dictionary mapping dataset short names to their tokenization ExecutorSteps
        
    Returns:
        ExecutorStep that computes and returns token counts as dictionary
    """
    # Flatten the tokenization steps (each value is a list with one step)
    flattened_steps = {name: steps[0] for name, steps in tokenization_steps.items()}
    
    return ExecutorStep(
        name="scratch/thinking_sft/compile_token_counts",
        fn=_compile_and_store_num_tokens,
        config=CompileTokenCountsConfig(tokenization_steps=flattened_steps),
    )




# Define datasets
from exp808_sft_mixture import DATASETS as EXP808_DATASETS
DATASETS = {
    **EXP808_DATASETS,
    "nemotron_sft": "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT",
    "openthoughts3": "open-thoughts/OpenThoughts3-1.2M",
}

def download_transform_tokenize_compile_steps():
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
        
        ALL_STEPS += [data_download_step] + [data_transform_step] + [data_tokenize_step]
        TOKENIZATION_STEPS[short_ds_name] = [data_tokenize_step]
    
    # Compile token counts
    ALL_STEPS.append(compile_and_store_num_rows_step(TOKENIZATION_STEPS))
    ALL_STEPS.append(compile_and_store_num_tokens_step(TOKENIZATION_STEPS))
    return ALL_STEPS

########### Main ###########
if __name__ == "__main__":
    ALL_STEPS = download_transform_tokenize_compile_steps()
    executor_main(steps=ALL_STEPS)
