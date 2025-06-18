#!/usr/bin/env python3
import dataclasses
import logging
import os
from dataclasses import dataclass

import ray

from experiments.midtraining_datasets import finemath_3_plus
from experiments.train_test_overlap.utils import clean_shard_basename, find_dataset_shards
from marin.core.runtime import simple_backpressure
from marin.execution.executor import ExecutorStep, executor_main, get_executor_step, this_output_path
from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe
from marin.utils import fsspec_glob

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Determine MARIN prefix (must point to GCS or local root)
prefix = os.environ.get("MARIN_PREFIX")
if not prefix:
    raise ValueError("MARIN_PREFIX environment variable must be set to your GCS prefix")

# Base dedupe configuration - modify this to change n-gram settings, processes, etc.
BASE_DEDUPE_CONFIG = DedupeConfig(
    input_path="gs://marin-us-central2/decontamination/",
    output_path="",  # Will be replaced per shard
    attribute_name="ngram_overlap",
    false_positive_rate=0.0001,
    ngram=NGramConfig(
        ngram_length=[10, 15],  # Multiple n-gram sizes - modify this to change n-grams
        overlap_threshold=1e-6,
        stride=0,
    ),
    processes=16,  # Modify this to change number of processes
    mode=DedupMode.TRAIN_TEST_OVERLAP,
    decontaminate_source="",  # Will be replaced per shard
)


def make_task(shard_path: str, base_output_path: str, base_config: DedupeConfig = BASE_DEDUPE_CONFIG) -> DedupeConfig:
    """Create a DedupeConfig for a single shard using the base config."""
    # Derive a unique output path from the shard file
    shard_basename = clean_shard_basename(shard_path)
    output_path = os.path.join(base_output_path, f"debug/{shard_basename}")

    # Use dataclasses.replace to create a new config with the shard-specific values
    return dataclasses.replace(
        base_config,
        output_path=output_path,
        decontaminate_source=shard_path,
    )


# Identify dataset directory from the executor step or InputName
# Extract the underlying ExecutorStep
base_step = get_executor_step(finemath_3_plus)
# Use override_output_path if pinned, otherwise use the step name
base_rel = base_step.override_output_path or base_step.name
# Append any subpath from the InputName (e.g., 'finemath-3plus')
subpath = finemath_3_plus.name or ""

# Use globbing to find the actual versioned directory
if base_step.override_output_path:
    # If override is set, use it directly
    dataset_dir = os.path.join(prefix, base_rel, subpath) if subpath else os.path.join(prefix, base_rel)
else:
    # Glob for the versioned directory (e.g., raw/finemath-*)
    base_pattern = os.path.join(prefix, f"{base_rel}-*")
    matching_dirs = fsspec_glob(base_pattern)
    if not matching_dirs:
        raise FileNotFoundError(f"No directories matching pattern {base_pattern}")
    if len(matching_dirs) > 1:
        logger.warning(f"Multiple directories match {base_pattern}, using first: {matching_dirs[0]}")
    base_dir = matching_dirs[0]
    dataset_dir = os.path.join(base_dir, subpath) if subpath else base_dir


@dataclass(frozen=True)
class ShardedDedupeConfig:
    """Configuration for running dedupe across multiple shards with backpressure."""

    dataset_dir: str
    output_path: str
    max_in_flight: int = 16


@ray.remote
def run_all_shards(config: ShardedDedupeConfig) -> str:
    """
    Discover all dataset shards and launch dedupe tasks with backpressure.
    """
    logger.info(f"Looking for dataset shards in {config.dataset_dir}")

    # Find all supported dataset shards under root (Parquet or compressed JSONL)
    shard_paths = find_dataset_shards(config.dataset_dir)
    # Generator of arguments for each Ray task
    task_generator = ((make_task(shard_path, config.output_path),) for shard_path in shard_paths)

    # Launch tasks with simple backpressure
    for ref in simple_backpressure(
        dedupe,  # Use dedupe directly - it's already a Ray remote function
        task_generator,
        max_in_flight=config.max_in_flight,
        fetch_local=True,
    ):
        ray.get(ref)

    return f"Sharded dedupe pipeline completed! Processed {len(shard_paths)} shards."


# Configure the sharded dedupe pipeline
config = ShardedDedupeConfig(
    dataset_dir=dataset_dir,
    output_path=this_output_path(),
    max_in_flight=64,
)

dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/shm_test",
    fn=run_all_shards,
    config=config,
    description="Run per-shard Dolma dedupe against MMLU test set with backpressure",
)

if __name__ == "__main__":
    executor_main(
        steps=[dedupe_step],
        description="Run per-shard Dolma dedupe against MMLU test set, reading Parquet or JSONL directly",
    )
