import os
import sys
import logging

import ray
import draccus

from operations.transform.format.parquet2jsonl import JsonlConversionConfig, convert_parquet_to_jsonl
from marin.utils import fsspec_glob
from marin.execution.executor import ExecutorStep, executor_main

logger = logging.getLogger(__name__)

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
        futures.append(convert_parquet_to_jsonl.remote(path, out_path))

    if futures:
        ray.get(futures)
    logger.info(f"Completed conversion of {len(futures)} shards")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_dataset_parquet2jsonl.py <dataset_relative_path_under_raw>")
        sys.exit(1)

    dataset_relpath = sys.argv[1].strip().rstrip("/")
    input_path = os.path.join("raw", dataset_relpath)
    output_path = os.path.join("documents", dataset_relpath)

    step = ExecutorStep(
        name=f"convert_parquet2jsonl/{dataset_relpath}",
        fn=convert_dataset_parquet2jsonl,
        config=JsonlConversionConfig(
            input_path=input_path,  
            output_path=output_path,
        ),
    )
    executor_main(steps=[step], description=f"Convert dataset {dataset_relpath} from Parquet to JSONL") 