from instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    InstructionDatasetConfig,
    download_dataset_step,
    get_instruction_dataset,
    transform_dataset_step,
    get_directory_friendly_dataset_name
)
from levanter.data.text import ChatLmDatasetFormat
from urllib.parse import urlparse
import os
import shutil
import json
import ray

from marin.transform.conversation.transform_conversation import (
    TransformSFTDatasetConfig,
    download_directory_from_gcs,
    create_shard_output_directory,
    get_shard_dir,
    transform_and_write_batch
)

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
import hashlib
import logging
logger = logging.getLogger("ray")


def custom_transform_nemotron(cfg: TransformSFTDatasetConfig):
    """We need a custom transform function because Nemotron is too large (~140GB in total). Downloading the entire
    dataset to disk can fill up the disk and cause OOM issues.
    """
    assert len(cfg.subsets) == 1, "This script only supports the SFT subset"
    assert len(cfg.splits) > 0, "Nemotron requires splits to be specified"

    # parse gs://my-bucket/path/to/mmlu into "my-bucket", "path/to/mmlu", and "mmlu"
    parsed_url = urlparse(cfg.input_path)
    bucket = parsed_url.netloc
    gcp_path = parsed_url.path.lstrip("/")
    local_dir = os.path.join("/tmp", os.path.basename(gcp_path))

    # 3. For each subset...
    write_ops = []
    for subset in cfg.subsets:
        for split in cfg.splits:
            try:
                logger.info(f"Downloading dataset from GCP: {gcp_path}/{subset}/{split}")
                try:
                    download_directory_from_gcs(bucket, f"{gcp_path}/{subset}/{split}", local_dir)
                except Exception as e:
                    logger.error(f"Error downloading dataset from GCP: {e}. \nRemoving local directory `{local_dir}`")
                    shutil.rmtree(local_dir)
                    raise e

                # Get full paths of all jsonl files
                # File: e.g., "/tmp/nvidia--Llama-Nemotron-Post-Training-Dataset-v1-SFT-ab2a40d-6aa704/SFT/code/v1.jsonl"
                files = [os.path.join(local_dir, file) for file in os.listdir(local_dir) if file.endswith(".jsonl")]

                for file in files:
                    # b. Create GCP target directory
                    subset_output_path = get_shard_dir(cfg.output_path, subset, split)
                    if len(files) > 1:
                        suffix = (
                            file.split("/")[-1]
                            .replace(split, "")
                            .strip("_")
                            .replace(".jsonl", "")
                            .strip() #should be v1 or v1.1
                        )
                        subset_output_path += '/' + suffix
                    output_path = create_shard_output_directory(subset_output_path)

                    # Read the file
                    jsonl_file_path = os.path.join(local_dir, file)
                    # Read the file
                    with open(jsonl_file_path, 'r') as f:
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
            finally:
                if os.path.exists(local_dir):
                    logger.info(f"Deleting local directory `{local_dir}`")
                    shutil.rmtree(local_dir)

    # Wait for all write operations to complete
    ray.get(write_ops)
    return cfg.output_path


def transform_dataset_step(dataset_cfg: InstructionDatasetConfig, download_step: ExecutorStep) -> ExecutorStep:
    """ExecutorStep that preprocesses and shards the input dataset.

    ===========================================================================
    dataset_cfg: {
        ...
        "hf_dataset_id": "cognitivecomputations/dolphin-r1",
        "subsets": ["reasoning-flash"],
        "splits": ['train', 'validation'],
        ...
    }
    output_path_of(download_step) --> gs://.../raw/dolphin-r1-[revision_number]-[hash]

    Expected files written: [
        gs://.../dolphin_r1__[revision_number]_[hash]/reasoning_flash/train/shard_00001.json.gz,
        ...
        gs://.../dolphin_r1__[revision_number]_[hash]/reasoning_flash/train/shard_00055.json.gz,
        gs://.../dolphin_r1__[revision_number]_[hash]/reasoning_flash/validation/shard_00001.json.gz,
        ...
        gs://.../dolphin_r1__[revision_number]_[hash]/reasoning_flash/validation/shard_00023.json.gz,
    ]
    ===========================================================================
    """
    adapter_name = dataset_cfg.adapter_name if dataset_cfg.adapter_name is not None else dataset_cfg.hf_dataset_id
    dataset_name = get_directory_friendly_dataset_name(adapter_name)
    download_data_path = output_path_of(download_step)

    config_str = f"{dataset_name}-\
        {dataset_cfg.revision}\
        -{sorted(dataset_cfg.subsets)}\
        -{sorted(dataset_cfg.splits)}"
    hashed_config_str = hashlib.md5(config_str.encode()).hexdigest()[:6]

    transform_step = ExecutorStep(
        name=f"documents/{dataset_name}",
        fn=custom_transform_nemotron,
        config=TransformSFTDatasetConfig(
            input_path=download_data_path,
            output_path=this_output_path(),
            shard_size=versioned(5000), # Context is long in nemotron
            metadata_columns=versioned(dataset_cfg.metadata_columns),
            filetype=dataset_cfg.filetype,
            source=dataset_cfg.hf_dataset_id,
            subsets=dataset_cfg.subsets,
            splits=dataset_cfg.splits,
            adapter_name=adapter_name,
        ),
        override_output_path=f"documents/{dataset_name}-{dataset_cfg.revision}-{hashed_config_str}",
    )

    return transform_step


def create_tokenization_step(dataset_name: str) -> ExecutorStep:
    """
    Creates a tokenization ExecutorStep for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'TIGER-Lab/AceCode-89K', 'HuggingFaceTB/smoltalk')

    Returns:
        ExecutorStep configured for tokenizing the specified dataset
    """
    # Get the dataset with only train split
    dataset = get_instruction_dataset(dataset_name, splits=['chat', 'code', 'math', 'science', 'safety'])

    # Get the last part of the path and clean it up
    short_name = dataset_name.split("/")[-1].lower().replace("-", "_")
    return default_tokenize(
        name=f"{short_name}_marin_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )


def main():
    return


if __name__ == "__main__":
    dataset_names = [
        # "open-thoughts/OpenThoughts3-1.2M",
        "nvidia/Llama-Nemotron-Post-Training-Dataset-v1-SFT",
    ]
    all_steps = []
    for dataset_name in dataset_names:
        # Download the dataset
        config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_name]
        downloaded_dataset = download_dataset_step(config)
        all_steps.append(downloaded_dataset)
        # Transform the dataset
        transformed_dataset = transform_dataset_step(config, downloaded_dataset)
        all_steps.append(transformed_dataset)

    executor_main(steps=all_steps)
