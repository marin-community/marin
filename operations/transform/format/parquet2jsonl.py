"""
Utility to convert Parquet files to JSONL format.
"""

import json
import logging
import os
from dataclasses import dataclass, field

import draccus
import fsspec
import pandas as pd
import ray

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_glob

logger = logging.getLogger("ray")


@dataclass
class JsonlConversionConfig:
    input_path: str
    output_path: str
    rename_key_from: str | None = field(default=None)
    rename_key_to: str | None = field(default=None)


@ray.remote(memory=0.5 * 1024 * 1024 * 1024)
@cached_or_construct_output(success_suffix="SUCCESS")
def convert_parquet_to_jsonl(
    input_path: str, output_path: str, rename_key_from: str | None = None, rename_key_to: str | None = None
) -> bool:
    """
    Convert a Parquet file to JSONL format.

    Args:
        input_path: Path to the input Parquet file
        output_path: Path to save the JSONL file (will be compressed with zstandard)
        rename_key_from: Optional key name in the Parquet file to rename.
        rename_key_to: Optional new key name for the JSONL output.

    Returns:
        True if successful
    """
    logger.info(f"Converting {input_path} to {output_path}")

    # Read the Parquet file
    df = pd.read_parquet(input_path)

    # Optional: Rename a key
    if rename_key_from and rename_key_to:
        if rename_key_from not in df.columns:
            logger.warning(f"Key '{rename_key_from}' not found in columns of {input_path}. Skipping renaming.")
            # Optionally raise an exception here if it's critical:
            # raise KeyError(f"Key '{rename_key_from}' not found in columns of {input_path}")
        else:
            logger.info(f"Renaming column '{rename_key_from}' to '{rename_key_to}' in {input_path}")
            df.rename(columns={rename_key_from: rename_key_to}, inplace=True)

    # Convert records to JSON strings
    records = df.to_dict(orient="records")

    # Write to JSONL with zstandard compression
    import io

    from zstandard import ZstdCompressor

    with fsspec.open(output_path, "wb") as f:
        cctx = ZstdCompressor(level=3)  # Adjust compression level as needed
        with cctx.stream_writer(f) as writer:
            text_buffer = io.StringIO()
            for record in records:
                json_line = json.dumps(record) + "\n"
                text_buffer.write(json_line)

            writer.write(text_buffer.getvalue().encode("utf-8"))

    logger.info(f"Successfully converted {input_path} to {output_path}")
    return True


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def convert_shards_to_jsonl(input_path: str, output_path: str) -> bool:
    """
    Convert all Parquet files in the input path to JSONL format.
    """
    MAX_CONCURRENT_SHARDS = 50

    shards = fsspec_glob(os.path.join(input_path, "*.parquet"))
    result_refs = []

    for shard in shards:
        if len(result_refs) > MAX_CONCURRENT_SHARDS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        shard_output_path = os.path.join(output_path, os.path.basename(shard).replace(".parquet", ".jsonl.zst"))
        result_refs.append(convert_parquet_to_jsonl.remote(shard, shard_output_path))

    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")

    logger.info(f"Successfully converted {input_path} to {output_path}")
    return True


@draccus.wrap()
def convert_dataset_parquet2jsonl(cfg: JsonlConversionConfig) -> bool:
    """
    Convert any Parquet-based dataset under cfg.input_path (including nested directories)
    into JSONL (zst-compressed) under cfg.output_path, preserving subdirectory structure.
    """
    logger.info(f"Starting conversion from {cfg.input_path} to {cfg.output_path}")
    # Recursively locate all Parquet files
    parquet_files = fsspec_glob(os.path.join(cfg.input_path, "**", "*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found under {cfg.input_path}")
        return True

    futures = []
    for path in parquet_files:
        # Compute the relative path under the input root
        rel_path = os.path.relpath(path, start=cfg.input_path)
        # Replace extension and build output path
        out_rel = rel_path.replace(".parquet", ".jsonl.zst")
        out_path = os.path.join(cfg.output_path, out_rel)
        futures.append(
            convert_parquet_to_jsonl.remote(
                path, out_path, rename_key_from=cfg.rename_key_from, rename_key_to=cfg.rename_key_to
            )
        )

    if futures:
        ray.get(futures)
    logger.info(f"Completed conversion of {len(futures)} shards")
    return True
