"""
Transform any HuggingFace dataset to OpenAI messages format.

Usage Examples:
1. Download the dataset from HuggingFace which is used in the input_path in the TransformSFTDatasetConfig.
2. Register your adapter in adapters.py
3. Run the script, filling out the TransformSFTDatasetConfig.

Check out experiments/instruction_datasets.py to see how to run this script using the Executor.
"""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import datasets
import draccus
import fsspec
import ray
from google.cloud import storage

from marin.core.conversation import DolmaConversationOutput, OpenAIChatMessage
from marin.core.runtime import fsspec_mkdirs

from .adapters import TransformAdapter, get_adapter

logger = logging.getLogger("ray")


@dataclass(frozen=True)
class TransformSFTDatasetConfig:
    """Base configuration to transform a conversation dataset from huggingface json to OpenAI format.

    Args:
        input_path (str): The path to the input file.
        output_path (str): The path to the output file.
        shard_size (int): The number of rows per shard.
        metadata_columns (list[str]): The columns to include in the metadata. Check the HuggingFace dataset
            for the columns to use.
        source (str): The name of the HuggingFace dataset. This is used to get the correct adapter.
        filetype (str): The filetype of the input file. Currently supports jsonl, json, and parquet.
        subsets: Data subsets (from HuggingFace config) to use. Empty list indicates to use all/default subset(s).
        splits: Data splits (e.g., `train`, `validation`) to use. Empty list indicates to use all splits.
                Defaults to `train` only
    """

    input_path: str
    output_path: str
    shard_size: int
    metadata_columns: list[str]
    source: str
    filetype: str
    adapter_name: str
    subsets: list[str] = field(default_factory=lambda: [])  # Default behavior is to use all subsets
    splits: list[str] = field(default_factory=lambda: ["train"])  # Set to train; empty set means everything


def generate_hash_from_messages(messages: list[dict[str, str]]) -> str:
    """Generate a hash from a list of messages.

    Args:
        messages (List[Dict[str, str]]): A list of messages.

    Returns:
        str: A hash of the messages.
    """
    return hashlib.sha256(str(messages).encode()).hexdigest()


def transform_row(row: dict, cfg: TransformSFTDatasetConfig, adapter: TransformAdapter):
    transformed_row_messages: list[OpenAIChatMessage] = adapter.transform_conversation_to_openai_format(row)

    if transformed_row_messages is None:
        logger.warning(f"{cfg.adapter_name} returning no valid messages")
        return None

    transformed_row_messages = [message.model_dump() for message in transformed_row_messages]

    # Create a unique ID for the row based on the text
    row_idx = generate_hash_from_messages(transformed_row_messages)
    metadata = {col: row.get(col, "") for col in cfg.metadata_columns}
    return DolmaConversationOutput(
        id=row_idx,
        source=cfg.source,
        messages=transformed_row_messages,
        added=datetime.now(timezone.utc).isoformat(),
        created="",  # Not available in the dataset
        metadata=metadata,
    )


def transform_rows(rows: list[dict], cfg: TransformSFTDatasetConfig):
    """Transform a list of rows from the conversation dataset to a list of formatted OpenAI Messages jsonl rows.

    Args:
        rows (list[dict]): A list of rows from the conversationdataset.

    Returns:
        list[dict]: A list of dolma formatted jsonl rows.
    """
    transformed_rows = []
    adapter = get_adapter(cfg.adapter_name)
    for row in rows:
        transformed_row: DolmaConversationOutput = transform_row(row, cfg, adapter)
        if transformed_row is not None:
            transformed_rows.append(transformed_row.model_dump())

    return transformed_rows


def create_shard_output_directory(output_filename: str) -> str:
    """Given an output filename, remove the suffix of the filename and create a directory for the shards.

    Example:
        [A] output_filename = "gs://A/B.jsonl.gz" -> [B] output_path = "gs://A/B"

    Args:
        output_filename (str): The path to the output file.

    Returns:
        str: The path to the directory containing the shards.
    """
    _, path = fsspec.core.url_to_fs(output_filename)
    protocol = fsspec.core.split_protocol(output_filename)[0]

    path_without_suffix = Path(path)
    while path_without_suffix.suffix:
        path_without_suffix = path_without_suffix.with_suffix("")

    output_path = f"{protocol}://{path_without_suffix}"
    fsspec_mkdirs(output_path)
    return output_path


def download_directory_from_gcs(bucket_name: str, gcs_directory_path: str, local_directory_path: str) -> None:
    """
    Download an entire directory from a GCS bucket to a local directory.
    Note: function mostly copied from marin/raw2json/huggingface/qa/raw2json.py. Added lines to skip provenance.json
        since it is an added file that will cause `datasets` to fail.

    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_directory_path (str): The path to the directory in GCS (excluding the bucket name).
        local_directory_path (str): The local directory path where the files will be saved.
    """
    # Make download dir
    if not os.path.exists(local_directory_path):
        os.makedirs(local_directory_path)

    # Initialize the client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # List all the blobs (files) with the specified prefix
    blobs = bucket.list_blobs(prefix=gcs_directory_path)

    # Download each blob to the local directory
    for blob in blobs:
        if "provenance.json" in blob.name:
            continue

        # Construct the relative path of the file
        relative_path = os.path.relpath(blob.name, gcs_directory_path)
        local_file_path = os.path.join(local_directory_path, relative_path)

        # Skip if file already exists locally and has the same size
        if os.path.exists(local_file_path):
            local_size = os.path.getsize(local_file_path)
            if local_size == blob.size:
                logger.info(f"Skipping {blob.name} - already exists with same size")
                continue
            else:
                logger.info(f"File {blob.name} exists but size differs, downloading again")
        else:
            # Create local directories if they do not exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the blob to the local file path
        blob.download_to_filename(local_file_path)
        logger.info(f"Downloaded gs://{blob.name} to local:{local_file_path}")


def copy_dataset_from_gcp_to_local(input_gcp_path: os.PathLike) -> os.PathLike:
    """
    Download the data from GCP onto local instance.

    Note: function modified from marin/raw2json/huggingface/qa/raw2json.py
    """
    # set up input path which can be GCP path, HF Hub path, or local path
    # handle case of gs:// path which requires downloading resource from GCP to local for processing
    if input_gcp_path.startswith("gs://"):
        # parse gs://my-bucket/path/to/mmlu into "my-bucket", "path/to/mmlu", and "mmlu"
        parsed_url = urlparse(input_gcp_path)
        bucket = parsed_url.netloc
        gcp_path = parsed_url.path.lstrip("/")
        local_dir_path = os.path.join("/dev/shm", os.path.basename(gcp_path))
        # download the repo from GCP path into local directory which is basename of provided path (e.g. mmlu)
        try:
            download_directory_from_gcs(bucket, gcp_path, local_dir_path)
        except Exception as e:
            logger.error(f"Error downloading dataset from GCP: {e}. \nRemoving local directory `{local_dir_path}`")
            shutil.rmtree(local_dir_path)
            raise e
        return local_dir_path
    else:
        raise Exception("Input is not a GCP path")


def get_shard_dir(dir_name: os.PathLike, subset_name: str | None, split: str) -> os.PathLike:
    """Creates a new path with the subset and split names.
    e.g., create_subset_name('gs://thisserver/testfolder-a982374', 'subset', 'train') -> 'gs://thisserver/testfolder-a982374/subset/train'
    """
    if (subset_name == "default") or (subset_name is None):
        return os.path.join(dir_name, split)
    return os.path.join(dir_name, subset_name, split)


@ray.remote(num_cpus=1, num_gpus=0)  # No need for GPUs
def transform_and_write_batch(
    batch: list[dict], shard_idx: int, output_path: str, cfg: TransformSFTDatasetConfig
) -> None:
    """Write a batch of transformed data to a compressed JSONL file.

    Args:
        batch: List of data rows to transform and write
        shard_idx: Index of the current shard
        output_path: Directory to write the shard file to
        cfg: Configuration for transformation
    """
    shard_filename = os.path.join(output_path, f"shard_{shard_idx:05d}.jsonl.gz")
    logger.info(f"Writing shard {shard_idx} to {shard_filename}")
    with fsspec.open(shard_filename, "wt", compression="gzip") as f:
        transformed_batch = transform_rows(batch, cfg)
        for transformed_row in transformed_batch:
            f.write(f"{json.dumps(transformed_row)}\n")


def transform_hf_dataset(cfg: TransformSFTDatasetConfig):
    """Shards the dataset; copies datafiles from GCP to instance, loads
    data using the `datasets` package, and write shards to target directory.
    We now implement checks to ensure that the temporary files are deleted
    after the script is done to free up disk space for others.

    Adding some documentation so that others won't attempt the same things.
    - There is no way to pass in the gs:// path to the `datasets` package in Ray. The main issue is
      that the `datasets` package always resolve the relative path, which resolves to a weird
      `/tmp/...` path when executed in Ray and the `datasets` package will fail.
    - This is the main reason why we need to copy the data from GCP to local instance.
    - However, when we do this, we cannot a-priori know the splits that are available in the dataset.
    - We use the /dev/shm (RAMDisk) to store the data. Please change this to /tmp if it fails. On marin
      cluster, it seems that our RAMDisk is not the typical 50% of memory but way smaller.
    """
    # 1. Copy data from GCP to local instance
    local_data_dir = copy_dataset_from_gcp_to_local(cfg.input_path)

    try:
        # 2. Identify subsets
        if cfg.subsets:
            # Process only given subsets
            subsets = cfg.subsets
        else:
            # No subset is defined, so process all subsets
            subsets = [x for x in datasets.get_dataset_infos(path=local_data_dir)]

        # 3. For each subset...
        write_ops = []
        for subset in subsets:
            # Validate splits
            split_values = [x for x in datasets.get_dataset_infos(path=local_data_dir)[subset].splits.values()]
            if isinstance(split_values[0], dict):
                # Dict obj;
                data_splits = [x["name"] for x in split_values]
            else:
                # SplitInfo obj;
                data_splits = [x.name for x in split_values]

            if cfg.splits:
                # Splits are defined, process only these splits
                splits = cfg.splits
                # Warn when defined splits are not available
                extra_splits = list(set(splits).symmetric_difference(data_splits))
                if extra_splits:
                    logging.log(logging.WARNING, f"Requested split(s) {extra_splits} for {cfg.source} skipped.")
                    splits = list(set(splits).intersection(data_splits))
            else:
                # Splits are not defined, we will load everything (default behavior)
                splits = data_splits

            for split in splits:
                # a. Load dataset
                dataset = datasets.load_dataset(path=local_data_dir, name=subset, split=split, streaming=True)

                # b. Create GCP target directory
                subset_output_path = get_shard_dir(cfg.output_path, subset, split)
                output_path = create_shard_output_directory(subset_output_path)

                # c. Process and write in batches
                batch = []
                shard_idx = 0

                try:
                    for row in dataset:
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
                except Exception as e:
                    logger.error(f"Error processing subset {subset}, split {split}: {e}")
        # Wait for all write operations to complete
        ray.get(write_ops)
    finally:
        # 4. Delete local data for others
        logger.info(f"Deleting local data directory `{local_data_dir}`")
        if os.path.exists(local_data_dir):
            shutil.rmtree(local_data_dir)

    return cfg.output_path


@draccus.wrap()
def main(cfg: TransformSFTDatasetConfig):
    transform_hf_dataset(cfg)
