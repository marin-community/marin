"""Remove unnecessary columns while streaming data from huggingface."""

import logging
import os
from dataclasses import dataclass

import draccus
import ray
from datasets import Dataset, IterableDataset, load_dataset

logger = logging.getLogger("ray")


@ray.remote(memory=10 * 1024 * 1024 * 1024, max_retries=5)  # 10 GB
def prune_stream_and_save(dataset: IterableDataset, keep_columns: list[str], output_path: str):
    """
    Prunes and saves a streaming dataset by removing un-specified columns.

    This function takes a streaming dataset, removes columns not specified in keep_columns,
    and saves the pruned dataset to disk in parquet format.

    NOTE:
    Data is being streamed and the column removal is done streamingly too, but to save the
    dataset we need to convert it to a cached dataset, while doing this we fit the dataset in memory.

    However since we removed some columns the subset will be smaller. so it doesn't need as
    much memory as the full subset.

    The memory needed for this operation should atleast be the size of the dataset
    in memory after removing the columns.

    Args:
        dataset (IterableDataset): The input streaming dataset to prune
        keep_columns (list[str]): List of column names to retain
        output_path (str): Path where the pruned dataset will be saved
    """

    drop_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(drop_columns)

    logger.info(f"Pruned dataset to columns: {keep_columns}")
    dataset = Dataset.from_generator(lambda: (yield from dataset), features=dataset.features)

    try:
        logger.info(f"Saving pruned dataset to {output_path}")
        dataset.save_to_disk(output_path, format="parquet")
        logger.info("Successfully saved pruned dataset")
    except Exception:
        raise Exception(f"Failed to save dataset to {output_path}")  # noqa
    return True


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
