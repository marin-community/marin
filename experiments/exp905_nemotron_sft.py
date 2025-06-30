from instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    transform_dataset_step,
    download_dataset_step,
    get_instruction_dataset,
    InstructionDatasetConfig,
)
from levanter.data.text import ChatLmDatasetFormat
from urllib.parse import urlparse
import os
import shutil
import json
import ray
from google.cloud import storage
import hashlib
from dataclasses import dataclass

from marin.transform.conversation.transform_conversation import (
    TransformSFTDatasetConfig,
    create_shard_output_directory,
    get_shard_dir,
    transform_and_write_batch
)

from experiments.defaults import default_tokenize, default_sft, default_train, this_output_path
from experiments.exp606_sft import tulu3_llama_tokenize_step
from experiments.marin_models import marin_tokenizer
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of
)
import logging
logger = logging.getLogger("ray")

########### Nemotron SFT ###########
def list_jsonl_files_in_gcs(bucket_name: str, gcs_directory_path: str) -> list[str]:
    """
    List all .jsonl files in a GCS directory.
    
    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_directory_path (str): The path to the directory in GCS (excluding the bucket name).
        
    Returns:
        list[str]: List of full GCS paths to .jsonl files.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # List all the blobs (files) with the specified prefix
    blobs = bucket.list_blobs(prefix=gcs_directory_path)
    
    jsonl_files = []
    for blob in blobs:
        if blob.name.endswith('.jsonl') and "provenance.json" not in blob.name:
            jsonl_files.append(blob.name)
    
    return jsonl_files


def download_single_file_from_gcs(bucket_name: str, gcs_file_path: str, local_file_path: str) -> None:
    """
    Download a single file from GCS to a local path.
    
    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_file_path (str): The path to the file in GCS (excluding the bucket name).
        local_file_path (str): The local file path where the file will be saved.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_path)
    
    # Create local directory if it doesn't exist
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    
    # Download the blob to the local file path
    blob.download_to_filename(local_file_path)
    logger.info(f"Downloaded gs://{bucket_name}/{gcs_file_path} to {local_file_path}")


def custom_transform_nemotron(cfg: TransformSFTDatasetConfig):
    """We need a custom transform function because Nemotron is too large (~140GB in total).
    Downloading the entire dataset to disk can fill up the disk and cause disk failure.
    Even downloading splits will cause failure (code split is 50GB+)
    
    This approach:
    1. Lists all .jsonl files in the GCS directory
    2. Downloads each file individually
    3. Processes each file immediately
    4. Deletes the file after processing to save disk space
    """
    assert len(cfg.subsets) == 1, "This script only supports the SFT subset"
    assert len(cfg.splits) > 0, "Nemotron requires splits to be specified"

    # parse gs://my-bucket/path/to/mmlu into "my-bucket", "path/to/mmlu", and "mmlu"
    parsed_url = urlparse(cfg.input_path)
    bucket = parsed_url.netloc
    gcp_path = parsed_url.path.lstrip("/")
    temp_dir = os.path.join("tmp", "nemotron_processing")

    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # 3. For each subset...
    write_ops = []
    try:
        for subset in cfg.subsets:
            for split in cfg.splits:
                # List all .jsonl files in the GCS directory
                jsonl_files = list_jsonl_files_in_gcs(bucket, f"{gcp_path}/{subset}/{split}")
                # Should be gs://nemotron-x/SFT/code/code_v1.jsonl, gs://nemotron-x/SFT/code/code_v1.1.jsonl, etc.
                # For chat: gs://nemotron-x/SFT/chat/chat.jsonl

                for gcs_file_path in jsonl_files:
                    # Extract filename for local path
                    filename = os.path.basename(gcs_file_path)
                    local_file_path = os.path.join(temp_dir, filename)
                    
                    try:
                        # Download the single file
                        logger.info(f"Downloading file: {gcs_file_path}")
                        download_single_file_from_gcs(bucket, gcs_file_path, local_file_path)
                        
                        # Create GCP target directory
                        subset_output_path = get_shard_dir(cfg.output_path, subset, split)
                        if len(jsonl_files) > 1:
                            # Extract version suffix from filename (e.g., "chat_v1.1" from "chat_v1.1.jsonl")
                            suffix = filename.replace(".jsonl", "").replace(".","_").strip()
                            # Make the new output path be e.g. nemotron-x/SFT/code/code_v1.1
                            # For chat, it will remain as nemotron-x/SFT/chat
                            subset_output_path += '/' + suffix
                        output_path = create_shard_output_directory(subset_output_path)

                        # Process the downloaded file
                        with open(local_file_path, 'r') as f:
                            batch = []
                            shard_idx = 0
                            for line in f:
                                try:
                                    row = json.loads(line)
                                    # Validate required fields
                                    if "input" not in row or "output" not in row:
                                        logger.error(f"Missing required fields: {row}")
                                        raise ValueError(f"Skipping row - missing required fields: {row}")
                                        
                                    # Convert input to string if it's a list
                                    if isinstance(row["input"], list):
                                        row["input"] = "\n".join(str(x) for x in row["input"])
                                    elif not isinstance(row["input"], str):
                                        row["input"] = str(row["input"])
                                        
                                    # Ensure output is a string
                                    if not isinstance(row["output"], str):
                                        row["output"] = str(row["output"])
                                        
                                    # Ensure metadata fields exist
                                    for col in cfg.metadata_columns:
                                        if col not in row:
                                            row[col] = ""  # Set empty string for missing metadata
                                            
                                    batch.append(row)
                                    
                                    # When batch reaches shard size, process and write it
                                    if len(batch) >= cfg.shard_size:
                                        # Queue the batch for writing
                                        write_ops.append(
                                            transform_and_write_batch.remote(
                                                batch.copy(),  # need .copy() or else ray will fail
                                                shard_idx,
                                                output_path,
                                                cfg,
                                            )
                                        )
                                        # Clear batch and increment shard index
                                        batch = []
                                        shard_idx += 1
                                        
                                except json.JSONDecodeError as e:
                                    logger.error(f"Error decoding JSON from line: {e}")
                                    raise e
                                except Exception as e:
                                    logger.error(f"Error processing row: {e}")
                                    raise e

                            # Write any remaining rows in the final batch
                            if batch:
                                write_ops.append(
                                    transform_and_write_batch.remote(
                                        batch.copy(),  # need .copy() or else ray will fail
                                        shard_idx,
                                        output_path,
                                        cfg,
                                    )
                                )
                        
                        logger.info(f"Processed file: {local_file_path}")
                        
                    except Exception as e:
                        logger.error(f"Error processing file {gcs_file_path}: {e}")
                        raise e
                    finally:
                        # Always clean up the local file to save disk space
                        if os.path.exists(local_file_path):
                            os.remove(local_file_path)
                            logger.info(f"Deleted local file: {local_file_path}")
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            logger.info(f"Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    # Wait for all write operations to complete
    ray.get(write_ops)
    return cfg.output_path

########### Custom function to transform dataset in order to accomodate Nemotron ###########
def custom_transform_dataset_step(dataset_cfg: InstructionDatasetConfig, data_download_step: ExecutorStep) -> ExecutorStep:
    """We need a custom transform function because Nemotron is too large (~140GB in total).
    Downloading the entire dataset to disk can fill up the disk and cause disk failure.
    Even downloading splits will cause failure (code split is 50GB+)
    
    This approach:
    1. Lists all .jsonl files in the GCS directory
    2. Downloads each file individually
    3. Processes each file immediately
    4. Deletes the file after processing to save disk space
    """
    # Convert InstructionDatasetConfig to TransformSFTDatasetConfig (same as transform_dataset_step)
    adapter_name = dataset_cfg.adapter_name if dataset_cfg.adapter_name is not None else dataset_cfg.hf_dataset_id
    dataset_name = adapter_name.split("/")[-1].lower().replace("-", "_")
    download_data_path = output_path_of(data_download_step)

    config_str = f"{dataset_name}-\
        {sorted(dataset_cfg.subsets)}\
        -{sorted(dataset_cfg.splits)}"
    hashed_config_str = hashlib.md5(config_str.encode()).hexdigest()[:6]

    # Create TransformSFTDatasetConfig from InstructionDatasetConfig
    transform_config = TransformSFTDatasetConfig(
        input_path=download_data_path,
        output_path=this_output_path(),
        shard_size=5000,
        metadata_columns=dataset_cfg.metadata_columns,
        filetype=dataset_cfg.filetype,
        source=dataset_cfg.hf_dataset_id,
        subsets=dataset_cfg.subsets,
        splits=dataset_cfg.splits,
        adapter_name=adapter_name,
    )

    return ExecutorStep(
        name=f"documents/{dataset_name}",
        fn=custom_transform_nemotron,
        config=transform_config,
        override_output_path=f"documents/{dataset_name}-{dataset_cfg.revision}-{hashed_config_str}",
    )


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
    tokenization_paths: dict[str, str]
    output_path: str = this_output_path()

def get_num_tokens_from_tokenized_datasets(transform_executor_steps: dict[str, ExecutorStep]) -> dict[str, int]:
    from levanter.store.jagged_array import JaggedArrayStore
    size_dict = dict()
    for ds_short_name, step in transform_executor_steps.items():
        gcs_tokenized_path = output_path_of(step)
        b = JaggedArrayStore.open(
            f"{gcs_tokenized_path}/input_ids",
            dtype=int,
        )
        size_dict[ds_short_name] = b.offsets.shape[0]
    return size_dict

def _compile_and_store_counts(config: CompileTokenCountsConfig) -> str:
    """Helper function to compile counts and store as JSON"""
    import json
    import fsspec
    
    # Flatten the tokenization steps (each value is a list with one step)
    flattened_steps = {name: steps[0] for name, steps in config.tokenization_paths.items()}
    
    # Get token counts
    token_counts = get_num_tokens_from_tokenized_datasets(flattened_steps)
    
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

def compile_and_store_count_step(tokenization_steps: dict[str, list[ExecutorStep]]) -> ExecutorStep:
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

    # Create InputName references to establish dependencies
    tokenization_paths = {name: output_path_of(steps[0]) for name, steps in tokenization_steps.items()}
    
    return ExecutorStep(
        name="scratch/thinking_sft/compile_token_counts",
        fn=_compile_and_store_counts,
        config=CompileTokenCountsConfig(tokenization_paths=tokenization_paths),
    )

########### Main ###########

if __name__ == "__main__":
    
    # Define datasets
    from exp808_sft_mixture import DATASETS as EXP808_DATASETS, mixture_weights as EXP808_mixture_weights
    DATASETS = {
        **EXP808_DATASETS,
        "nemotron_sft": "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT",
        "openthoughts3": "open-thoughts/OpenThoughts3-1.2M",
    }
    
    ALL_STEPS = []
    TOKENIZATION_STEPS = dict()
    for short_ds_name, full_ds_name in DATASETS.items():
        # Download the dataset
        config = INSTRUCTION_DATASET_NAME_TO_CONFIG[full_ds_name]
        data_download_step = download_dataset_step(config)
        # Transform the dataset
        if full_ds_name == "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT":
            data_transform_step = custom_transform_dataset_step(config, data_download_step)
        else:
            data_transform_step = transform_dataset_step(config, data_download_step)
            
        # Tokenize the dataset
        data_tokenize_step = create_tokenization_step(full_ds_name)
        
        ALL_STEPS += [data_download_step] + [data_transform_step] + [data_tokenize_step]
        TOKENIZATION_STEPS[short_ds_name] = [data_tokenize_step]
    
    # Add the compile token counts step
    ALL_STEPS.append(compile_and_store_count_step(TOKENIZATION_STEPS))
    
    

    executor_main(steps=ALL_STEPS)
