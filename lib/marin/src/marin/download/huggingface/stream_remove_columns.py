# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remove unnecessary columns while streaming data from huggingface."""

import logging
import os
from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tqdm import tqdm
from zephyr import Dataset, ZephyrContext

hf_fs = HfFileSystem()
logger = logging.getLogger(__name__)


def prune_stream_and_save(input_file: str, output_file: str, keep_columns: list[str]):
    """
    Prunes and saves a parquet file by removing un-specified columns.

    Reads the input parquet file in batches, removes columns not in keep_columns,
    and writes the result to output_file. Processing in batches avoids memory issues.

    Args:
        input_file (str): Path to input parquet file on HuggingFace
        output_file (str): Path where pruned parquet file will be saved
        keep_columns (list[str]): List of column names to retain
    """
    parquet_file = pq.ParquetFile(hf_fs.open(input_file))

    full_df_list = []
    for batch in tqdm(parquet_file.iter_batches(batch_size=10000), desc=f"Processing {input_file}"):
        df = batch.to_pandas()

        drop_columns = [col for col in df.columns if col not in keep_columns]
        df = df.drop(columns=drop_columns)

        full_df_list.append(df)

    full_df = pd.concat(full_df_list)
    logger.info(f"Saving pruned dataset of shape {full_df.shape} to {output_file}")
    full_df.to_parquet(output_file, index=False)


def get_file_tasks(hf_path: str, output_path: str, keep_columns: list[str]):
    """
    Generate file processing tasks for a HuggingFace subset.

    Args:
        hf_path (str): The HuggingFace dataset path to load
        output_path (str): The output path to save the pruned dataset
        keep_columns (list[str]): The columns to keep in the pruned dataset

    Yields:
        Dict with input_file, output_file, and keep_columns for each parquet file
    """
    logger.info(f"Loading dataset from {hf_path}")
    parquet_list = hf_fs.glob(f"{hf_path}/*.parquet")

    for file in parquet_list:
        output_file = os.path.join(output_path, os.path.basename(file))
        yield {"input_file": file, "output_file": output_file, "keep_columns": keep_columns}


@dataclass
class DatasetConfig:
    hf_repo_id: str
    hf_revision: str
    hf_paths: list[str]
    output_path: str
    keep_columns: list[str]


def prune_hf_dataset(cfg: DatasetConfig):
    logger.info(f"Starting dataset pruning for {cfg.hf_paths}")

    # Build list of subset paths to process
    subset_tasks = []
    for path in cfg.hf_paths:
        # HF Path form: hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
        hf_path = f"hf://datasets/{cfg.hf_repo_id}@{cfg.hf_revision}/{path}"
        logger.info(f"Processing subset {hf_path}")
        output_path = os.path.join(cfg.output_path, path)
        subset_tasks.append({"hf_path": hf_path, "output_path": output_path})

    # Build pipeline with nested parallelism:
    # - Outer level: process subsets (MAX_CONCURRENT_WORKERS=1)
    # - Inner level: process files within each subset
    pipeline = (
        Dataset.from_list(subset_tasks)
        .flat_map(lambda task: get_file_tasks(task["hf_path"], task["output_path"], cfg.keep_columns))
        .map(lambda task: prune_stream_and_save(task["input_file"], task["output_file"], cfg.keep_columns))
    )

    logger.info("Executing pipeline")
    with ZephyrContext() as ctx:
        ctx.execute(pipeline)
    logger.info("Successfully processed all subsets")
