"""Remove unnecessary columns while streaming data from huggingface."""

import logging
import os
from dataclasses import dataclass

import draccus
import pandas as pd
import pyarrow.parquet as pq
import ray
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
logger = logging.getLogger("ray")


@ray.remote(memory=10 * 1024 * 1024 * 1024, max_retries=5)  # 10 GB
def prune_stream_and_save(input_file: str, output_file: str, keep_columns: list[str]):
    """
    Prunes and saves a streaming dataset by removing un-specified columns.

    This function takes a streaming dataset, removes columns not specified in keep_columns,
    and saves the pruned dataset to disk in parquet format. The dataset is processed and saved
    in chunks to avoid memory issues with large subsets.

    Args:
        dataset (IterableDataset): The input streaming dataset to prune
        keep_columns (list[str]): List of column names to retain
        output_path (str): Path where the pruned dataset will be saved
    """
    try:
        parquet_file = pq.ParquetFile(fs.open(input_file))

        full_df_list = []
        for batch in parquet_file.iter_batches(batch_size=10000):
            df = batch.to_pandas()

            drop_columns = [col for col in df.columns if col not in keep_columns]
            df = df.drop(columns=drop_columns)

            full_df_list.append(df)

        full_df = pd.concat(full_df_list)
        logger.info(f"Saving pruned dataset of shape {full_df.shape} to {output_file}")
        full_df.to_parquet(output_file, index=False)
    except Exception as e:
        logger.exception(f"Error processing {input_file}")
        raise e


@ray.remote(memory=10 * 1024 * 1024 * 1024, max_retries=5)  # 10 GB
def process_hf_subset(hf_path: str, output_path: str, keep_columns: list[str]):
    """
    Process a subset of a HuggingFace dataset by pruning columns and saving to disk.

    Args:
        hf_path (str): The HuggingFace dataset path to load
        output_path (str): The output path to save the pruned dataset
        keep_columns (list[str]): The columns to keep in the pruned dataset
    """

    logger.info(f"Loading dataset from {hf_path}")
    parquet_list = fs.glob(f"{hf_path}/*.parquet")

    ray_waitables = []

    for file in parquet_list:
        output_file = os.path.join(output_path, os.path.basename(file))
        ray_waitables.append(prune_stream_and_save.remote(file, output_file, keep_columns))

    for ray_waitable in ray_waitables:
        try:
            ray.get(ray_waitable)
            logger.info("Successfully processed split")
        except Exception as e:
            logger.exception(f"Error processing split for {output_path}: {e!s}")
            raise e


@dataclass
class DatasetConfig:
    hf_repo_id: str
    hf_revision: str
    hf_paths: list[str]
    output_path: str
    keep_columns: list[str]


@draccus.wrap()
def prune_hf_dataset(cfg: DatasetConfig):
    logger.info(f"Starting dataset pruning for {cfg.hf_paths}")

    result_refs = []

    for path in cfg.hf_paths:
        # HF Path form: hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
        hf_path = f"hf://datasets/{cfg.hf_repo_id}@{cfg.hf_revision}/{path}"
        logger.info(f"Processing subset {hf_path}")
        output_path = os.path.join(cfg.output_path, path)

        try:
            result_refs.append(process_hf_subset.remote(hf_path, output_path, cfg.keep_columns))
        except Exception as e:
            logger.exception(f"Error processing subset {hf_path}: {e!s}")
            continue

    try:
        logger.info("Waiting for all subset processing to complete")
        ray.get(result_refs)
        logger.info("Successfully processed all subsets")
    except Exception as e:
        logger.exception(f"Error in final processing: {e!s}")
