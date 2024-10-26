"""Remove unnecessary columns while streaming data from huggingface."""

from functools import partial
import logging
import os

import draccus
import ray

from dataclasses import dataclass
from datasets import load_dataset, IterableDataset, Dataset

logger = logging.getLogger("ray")


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


@ray.remote(memory=10 * 1024 * 1024 * 1024, max_retries=5)  # 10 GB
def filter_stream_and_save(
    dataset: IterableDataset,
    keep_columns: list[str],
    output_path: str
):
    """
    Filters and saves a streaming dataset by removing unnecessary columns.

    This function takes a streaming dataset, removes columns not specified in keep_columns,
    and saves the filtered dataset to disk in parquet format.

    Args:
        dataset (IterableDataset): The input streaming dataset to filter
        keep_columns (list[str]): List of column names to retain
        output_path (str): Path where the filtered dataset will be saved
    """
    
    drop_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(drop_columns)

    dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset), features=dataset.features)

    try:
        logger.info(f"Saving filtered dataset to {output_path}")
        dataset.save_to_disk(output_path, format="parquet")
        logger.info("Successfully saved filtered dataset")
    except Exception as e:
        raise Exception(f"Failed to save dataset to {output_path}: {str(e)}")
    return True


@ray.remote(memory=10 * 1024 * 1024 * 1024, max_retries=5)  # 10 GB
def process_hf_subset(hf_dataset_id: str, revision, subset: str, splits: list[str], output_path: str, keep_columns: list[str]):
    """
    Process a subset of a HuggingFace dataset by filtering columns and saving to disk.

    Args:
        hf_dataset_id (str): The HuggingFace dataset ID to load
        revision (str): The revision of the dataset to load
        subset (str): The subset of the dataset to process
        splits (list[str]): The splits of the dataset to process
        output_path (str): The output path to save the filtered dataset
        keep_columns (list[str]): The columns to keep in the filtered dataset
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
        ray_waitables.append(
            filter_stream_and_save.remote(
                dataset[split],
                keep_columns,
                split_output_path
            )
        )

    for ray_waitable in ray_waitables:
        try:
            ray.get(ray_waitable)
            logger.info("Successfully processed split")
        except Exception as e:
            logger.exception(f"Error processing split for {output_path}: {str(e)}")


@dataclass
class DatasetConfig:
    hf_dataset_id: str
    revision: str
    subsets: list[str]
    splits: list[str]
    output_path: str
    keep_columns: list[str]


@draccus.wrap()
def filter_hf_dataset(cfg: DatasetConfig):
    logger.info(f"Starting dataset filtering for {cfg.hf_dataset_id}")
    result_refs = []

    for subset in cfg.subsets:
        output_path = os.path.join(cfg.output_path, subset)
        logger.info(f"Processing subset {subset} to {output_path}")

        try:
            result_refs.append(
                process_hf_subset.remote(cfg.hf_dataset_id, cfg.revision, subset, cfg.splits, output_path, cfg.keep_columns)
            )
        except Exception as e:
            logger.exception(f"Error processing subset {subset}: {str(e)}")
            continue

    try:
        logger.info("Waiting for all subset processing to complete")
        ray.get(result_refs)
        logger.info("Successfully processed all subsets")
    except Exception as e:
        logger.exception(f"Error in final processing: {str(e)}")
