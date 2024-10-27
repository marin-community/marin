"""Remove unnecessary columns while streaming data from huggingface."""

import logging
import os
from dataclasses import dataclass

import draccus
import ray
from datasets import Dataset, IterableDataset, load_dataset
from tqdm import tqdm

logger = logging.getLogger("ray")


@ray.remote(memory=10 * 1024 * 1024 * 1024, max_retries=5)  # 10 GB
def save_parquet_to_gcs(chunk_buffer: list[dict], output_path: str, shard_idx: int, sub_idx: int):
    """
    Save a chunk of data to a parquet file on GCS.

    Args:
        chunk_buffer (list[dict]): The chunk of data to save
        output_path (str): The output path to save the parquet file
        shard_idx (int): The shard index of the parquet file
        sub_idx (int): The sub-index of the parquet file
    """
    try:
        chunk_dataset = Dataset.from_list(chunk_buffer)
        chunk_path = os.path.join(output_path, f"{shard_idx:03d}_{sub_idx:05d}.parquet")

        logger.info(f"Saving chunk {shard_idx}_{sub_idx} to {chunk_path}")
        chunk_dataset.to_parquet(chunk_path)

        return True

    except Exception as e:
        raise e


@ray.remote(memory=10 * 1024 * 1024 * 1024, max_retries=5)  # 10 GB
def prune_stream_and_save(
    dataset: IterableDataset, keep_columns: list[str], output_path: str, chunk_size: int = 1000000
):
    """
    Prunes and saves a streaming dataset by removing un-specified columns.

    This function takes a streaming dataset, removes columns not specified in keep_columns,
    and saves the pruned dataset to disk in parquet format. The dataset is processed and saved
    in chunks to avoid memory issues with large subsets.

    Args:
        dataset (IterableDataset): The input streaming dataset to prune
        keep_columns (list[str]): List of column names to retain
        output_path (str): Path where the pruned dataset will be saved
        chunk_size (int): Number of rows to process and save in each chunk (default: 10000)
    """
    drop_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(drop_columns)

    logger.info(f"Pruned dataset to columns: {keep_columns}")

    # Create output directory if it doesn't exist
    ray_chunk_refs = []
    os.makedirs(output_path, exist_ok=True)

    chunk_buffer = []
    chunk_idx = 0

    for item in tqdm(dataset):
        chunk_buffer.append(item)

        if len(chunk_buffer) >= chunk_size:
            shard_idx = chunk_idx // 50
            sub_idx = chunk_idx % 50

            ray_chunk_refs.append(save_parquet_to_gcs.remote(chunk_buffer, output_path, shard_idx, sub_idx))

            # Clear buffer and increment counter
            chunk_buffer = []
            chunk_idx += 1

    # Save any remaining items in the final chunk
    if chunk_buffer:
        shard_idx = chunk_idx // 50
        sub_idx = chunk_idx % 50

        ray_chunk_refs.append(save_parquet_to_gcs.remote(chunk_buffer, output_path, shard_idx, sub_idx))

    try:
        ray.get(ray_chunk_refs)
        logger.info("Successfully saved the chunk")
    except Exception as e:
        raise Exception(f"Failed to save dataset to {output_path}: {str(e)}")  # noqa


@ray.remote(memory=10 * 1024 * 1024 * 1024, max_retries=5)  # 10 GB
def process_hf_subset(
    hf_dataset_id: str, revision, subset: str, splits: list[str], output_path: str, keep_columns: list[str]
):
    """
    Process a subset of a HuggingFace dataset by pruning columns and saving to disk.

    Args:
        hf_dataset_id (str): The HuggingFace dataset ID to load
        revision (str): The revision of the dataset to load
        subset (str): The subset of the dataset to process
        splits (list[str]): The splits of the dataset to process
        output_path (str): The output path to save the pruned dataset
        keep_columns (list[str]): The columns to keep in the pruned dataset
    """

    logger.info(f"Loading dataset {hf_dataset_id} subset {subset} revision {revision}")
    dataset = load_dataset(hf_dataset_id, subset, streaming=True, revision=revision)
    logger.info(f"Successfully loaded dataset {hf_dataset_id} subset {subset}")

    ray_waitables = []
    for split in splits:
        split_output_path = output_path
        if len(splits) > 1:
            split_output_path = os.path.join(output_path, split)

        logger.info(f"Processing split {split} to {split_output_path}")
        ray_waitables.append(prune_stream_and_save.remote(dataset[split], keep_columns, split_output_path))

    for ray_waitable in ray_waitables:
        try:
            ray.get(ray_waitable)
            logger.info("Successfully processed split")
        except Exception as e:
            logger.exception(f"Error processing split for {output_path}: {e!s}")
            raise e


@dataclass
class DatasetConfig:
    hf_dataset_id: str
    revision: str
    subsets: list[str]
    splits: list[str]
    output_path: str
    keep_columns: list[str]


@draccus.wrap()
def prune_hf_dataset(cfg: DatasetConfig):
    logger.info(f"Starting dataset pruneing for {cfg.hf_dataset_id}")
    result_refs = []

    for subset in cfg.subsets:
        output_path = os.path.join(cfg.output_path, subset)
        logger.info(f"Processing subset {subset} to {output_path}")

        try:
            result_refs.append(
                process_hf_subset.remote(
                    cfg.hf_dataset_id, cfg.revision, subset, cfg.splits, output_path, cfg.keep_columns
                )
            )
        except Exception as e:
            logger.exception(f"Error processing subset {subset}: {e!s}")
            continue

    try:
        logger.info("Waiting for all subset processing to complete")
        ray.get(result_refs)
        logger.info("Successfully processed all subsets")
    except Exception as e:
        logger.exception(f"Error in final processing: {e!s}")
