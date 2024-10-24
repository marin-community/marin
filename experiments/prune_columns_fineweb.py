"""Prune unnecessary columns from FineWeb parquet files."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

import draccus
import fsspec
import pandas as pd
import ray

from marin.utils import fsspec_exists, fsspec_glob

logger = logging.getLogger("ray")


@ray.remote(memory=10 * 1024 * 1024 * 1024, max_task_retries=5)  # 10 GB
def prune_file_and_save(
    input_file: str,
    output_file: str,
    keep_columns: list[str],
):
    """
    Prunes unnecessary columns from a FineWeb parquet file and save the result.

    This function reads a parquet file from the input path, drops unnecessary columns
    to reduce file size, and saves the pruned dataframe to the output path. It retains
    only the essential columns: 'id', 'url', 'file_path', 'language_score', and 'token_count'.

    Args:
        input_file (str): Path to the input FineWeb parquet file.
        output_file (str): Path where the pruned parquet file will be saved.
        keep_columns (list[str]): Columns to keep in the pruned parquet file.
    """

    # Example of input_path = gs://marin-data/raw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000.parquet
    # Example of output_path = gs://marin-data/raw/fineweb/fw-v1.0-pruned/CC-MAIN-2024-10/000_00000.parquet
    logger.info(f"Processing {input_file = }, {output_file = }")
    success_file = output_file.replace(".parquet", ".SUCCESS")
    datetime_start = datetime.utcnow()

    if fsspec_exists(success_file):
        logger.info(f"Output file already processed. Skipping {input_file}")
        return True

    try:
        df = pd.read_parquet(input_file)
    except FileNotFoundError as e:
        logger.exception(f"Error reading the parquet file: {e}")
        raise e

    # reserve columns: id, url, file_path, language_score, token_count
    drop_columns = [col for col in df.columns if col not in keep_columns]
    df.drop(columns=drop_columns, inplace=True)

    try:
        df.to_parquet(output_file, index=False)
    except Exception as e:
        logger.exception(f"Error processing the file {input_file}: {e}")
        return False

    logger.info(f"Processed {input_file} and saved to {output_file}")

    datetime_end = datetime.utcnow()

    with fsspec.open(success_file, "w") as f:
        metadata = {
            "input_path": input_file,
            "output_file_path": output_file,
            "datetime_start": str(datetime_start),
            "datetime_end": str(datetime_end),
        }
        print(json.dumps(metadata), file=f)

    return True


@dataclass
class PruningConfig:
    input_path: str
    output_path: str
    cc_dumps: list[str] | None = None
    max_files: int | None = None
    keep_columns: list[str] = field(default_factory=lambda: ["id", "url", "file_path", "language_score", "token_count"])


@draccus.wrap()
def prune_fineweb_parquet(cfg: PruningConfig):
    file_ctr = 0
    end_processing = False

    cc_dumps = cfg.cc_dumps or fsspec_glob(f"{cfg.input_path}/*")

    for cc_dump in cc_dumps:
        logger.info(f"Processing {cc_dump}")

        files = fsspec_glob(os.path.join(cfg.input_path, cc_dump, "*.parquet"))

        if not files:
            logger.info(f"No files found in {cc_dump}, Skipping")
            continue

        result_refs = []
        for file in files:
            # Get the input file name
            # Example of file = gs://marin-data/raw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000.parquet
            # input_file_name = 000_00000.parquet
            input_file_name = os.path.basename(file)
            output_file_name = os.path.join(
                cfg.output_path,
                cc_dump,
                input_file_name,
            )

            result_refs.append(prune_file_and_save.remote(file, output_file_name, cfg.keep_columns))

            if cfg.max_files and file_ctr >= cfg.max_files:
                end_processing = True
                break

            file_ctr += 1
        # Wait for all the tasks to finish
        try:
            ray.get(result_refs)
        except Exception as e:
            logger.exception(f"Error processing the group: {e}")

        if end_processing:
            break


if __name__ == "__main__":
    prune_fineweb_parquet()
