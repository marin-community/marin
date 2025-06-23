from instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    transform_dataset_step,
    InstructionDatasetConfig,
    download_dataset_step,
    get_instruction_dataset,
    get_directory_friendly_dataset_name
)
from levanter.data.text import ChatLmDatasetFormat
from urllib.parse import urlparse
import os
import shutil
import json
import ray
from google.cloud import storage

from marin.transform.conversation.transform_conversation import (
    TransformSFTDatasetConfig,
    create_shard_output_directory,
    get_shard_dir,
    transform_and_write_batch
)

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
)
import logging
logger = logging.getLogger("ray")


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


# This is a modification of the create_tokenization_step function in exp808_sft_mixture.py
def create_tokenization_step(dataset_name: str) -> ExecutorStep:
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

# Dataset configurations
DATASETS = {
    "openthoughts3_1pt2m": "open-thoughts/OpenThoughts3-1.2M",
    "nemotron_sft": "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT",
}

if __name__ == "__main__":
    all_steps = []
    
    # Download, transform, and tokenize datasets
    for dataset_name in DATASETS.values():
        # Download the dataset
        config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_name]
        downloaded_dataset = download_dataset_step(config)
        all_steps.append(downloaded_dataset)
        # Transform the dataset
        transformed_dataset = transform_dataset_step(config, downloaded_dataset)
        all_steps.append(transformed_dataset)
        # Tokenize the dataset
        tokenized_dataset = create_tokenization_step(dataset_name)
        all_steps.append(tokenized_dataset)

    executor_main(steps=all_steps)
