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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import datasets
import re

from urllib.parse import urlparse
from google.cloud import storage

import draccus
import fsspec
import pandas as pd
import ray

from marin.core.conversation import DolmaConversationOutput, OpenAIChatMessage
from marin.core.runtime import TaskConfig, cached_or_construct_output, fsspec_mkdirs, map_files_in_directory

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
    """

    input_path: str
    output_path: str
    shard_size: int
    metadata_columns: list[str]
    source: str
    filetype: str
    subsets: list[str] = field(default_factory=lambda: ["all"])


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
    adapter = get_adapter(cfg.source)
    for row in rows:
        transformed_row: DolmaConversationOutput = transform_row(row, cfg, adapter)
        transformed_rows.append(transformed_row.model_dump())

    return transformed_rows


def load_dataset(input_path: str) -> list[dict]:
    """Load a list of rows from the file. Currently supports jsonl, json, and parquet.

    Args:
        input_path (str): The path to the input file.

    Returns:
        list[dict]: A list of rows from the input file.
    """
    if input_path.endswith(".jsonl"):
        with fsspec.open(input_path, "rt") as f:
            return [json.loads(line) for line in f]
    elif input_path.endswith(".json"):
        with fsspec.open(input_path, "rt") as f:
            return json.load(f)
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path, engine="pyarrow")
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file type: {input_path}")


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


@ray.remote(memory=4 * 1024 * 1024 * 1024)
@cached_or_construct_output(success_suffix="SUCCESS")
def transform_file(input_filename: str, output_filename: str, cfg: TransformSFTDatasetConfig):
    """
    Transforms the input dataset file and writes the transformed data into shards.
    Args:
        input_filename (str): The path to the input dataset file.
        output_filename (str): The base path for the output shard files.
        cfg (TransformSFTDatasetConfig): Configuration object containing transformation parameters.
    Returns:
        None

    Note: we assume regardlss of filetype that every directory will have a 'provenance.json' file
    which we use for our own metadata. We check this explicitly because *json is a valid
    file type, but 'provenance.json' is not a valid dataset file.
    """

    if "provenance.json" in input_filename:
        return
    rows = load_dataset(input_filename)
    logger.info(f"Transforming {len(rows)} rows from {input_filename} to {output_filename}")

    output_path = create_shard_output_directory(output_filename)

    for idx, shard in enumerate(range(0, len(rows), cfg.shard_size)):
        shard_rows = rows[shard : min(shard + cfg.shard_size, len(rows))]
        shard_filename = os.path.join(output_path, f"shard_{idx:05d}.jsonl.gz")
        logger.info(f"Writing shard {idx} to {shard_filename}")
        with fsspec.open(shard_filename, "wt", compression="gzip") as f:
            transformed_shard_rows = transform_rows(shard_rows, cfg)
            for row in transformed_shard_rows:
                f.write(f"{json.dumps(row)}\n")


@ray.remote
def transform_dataset(cfg: TransformSFTDatasetConfig):
    responses = map_files_in_directory(
        transform_file.remote, cfg.input_path, f"**/*.{cfg.filetype}", cfg.output_path, TaskConfig(), False, cfg
    )

    ray.get(responses)



def download_directory_from_gcs(bucket_name: str, gcs_directory_path: str, local_directory_path: str) -> None:
    """
    Download an entire directory from a GCS bucket to a local directory.

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
        # Construct the relative path of the file
        relative_path = os.path.relpath(blob.name, gcs_directory_path)
        local_file_path = os.path.join(local_directory_path, relative_path)

        # Create local directories if they do not exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the blob to the local file path
        blob.download_to_filename(local_file_path)
        logger.info(f"Downloaded {blob.name} to {local_file_path}")

def copy_dataset_from_gcp_to_local(input_gcp_path: os.PathLike) -> os.PathLike:
    """
    Load the dataset from GCP.
    We will download the data from GCP onto local, and then operate on local
    """
    # set up input path which can be GCP path, HF Hub path, or local path
    # handle case of gs:// path which requires downloading resource from GCP to local for processing
    if input_gcp_path.startswith("gs://"):
        # parse gs://my-bucket/path/to/mmlu into "my-bucket", "path/to/mmlu", and "mmlu"
        parsed_url = urlparse(input_gcp_path)
        bucket = parsed_url.netloc
        gcp_path = parsed_url.path.lstrip("/")
        dir_name = os.path.basename(gcp_path)
        # download the repo from GCP path into local directory which is basename of provided path (e.g. mmlu)
        download_directory_from_gcs(bucket, gcp_path, dir_name)
        input_path = dir_name
    else:
        raise Exception('Input is not a GCP path')

    return input_path

def create_subset_name(dir_name: os.PathLike, subset_name: str) -> os.PathLike:
    """ Creates a new dir name that incorporates the subset name.
    e.g., create_subset_name('gs://thisserver/testfolder-a982374', 'testsubset') -> 'gs://thisserver/testfolder-testsubset-a982374'  
    """
    end_name = dir_name.split('/')[-1]
    _matches = re.match('(.*-)(.*)$', end_name)
    prefix = _matches[1]
    suffix = _matches[2]
    new_end_name = f"{prefix}{subset_name}-{suffix}"
    return dir_name.replace(end_name, new_end_name)

@ray.remote
def transform_hf_dataset(cfg: TransformSFTDatasetConfig):
    local_dir = copy_dataset_from_gcp_to_local(cfg.input_path)
    
    if not cfg.subsets:
        # No subset is defined, so process all subsets
        cfg.subsets = [x for x in datasets.get_dataset_infos(path=local_dir)]
    
    for subset in cfg.subsets:
        dataset = datasets.load_dataset(path=local_dir, name=subset, split='train')
        subset_output_path = create_subset_name(cfg.output_path, subset)
        output_path = create_shard_output_directory(subset_output_path)
        
        rows = [r for r in dataset] 
        del dataset # saves memory
        
        for idx, shard in enumerate(range(0, len(rows), cfg.shard_size)):
            shard_rows = rows[shard : min(shard + cfg.shard_size, len(rows))]
            shard_filename = os.path.join(output_path, f"shard_{idx:05d}.jsonl.gz")
            logger.info(f"Writing shard {idx} to {shard_filename}")
            with fsspec.open(shard_filename, "wt", compression="gzip") as f:
                transformed_shard_rows = transform_rows(shard_rows, cfg)
                for row in transformed_shard_rows:
                    f.write(f"{json.dumps(row)}\n")


@draccus.wrap()
def main(cfg: TransformSFTDatasetConfig):
    ray.get(transform_dataset.remote(cfg))


if __name__ == "__main__":
    main()
