#!/usr/bin/env python3
import logging
import os

from experiments.midtraining_datasets import finemath_3_plus
from experiments.train_test_overlap.utils import clean_shard_basename, find_dataset_shards
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

logger.info(f"Looking for dataset shards in {dataset_dir}")

# Find all supported dataset shards under root (Parquet or compressed JSONL)
shard_paths = find_dataset_shards(dataset_dir)

# Build one dedupe step per shard
steps: list[ExecutorStep] = []
for shard_path in shard_paths:
    # derive a unique name from the shard file
    shard_basename = clean_shard_basename(shard_path)

    step_name = f"train_test_overlap/dolma/parquet_finemath3plus_dedupe/{shard_basename}"
    cfg = DedupeConfig(
        input_path="gs://marin-us-central2/decontamination/",
        output_path=this_output_path(),
        attribute_name="ngram_overlap",
        false_positive_rate=0.0001,
        ngram=NGramConfig(
            ngram_length=[10, 15],
            overlap_threshold=1e-6,
            stride=0,
        ),
        processes=16,
        mode=DedupMode.TRAIN_TEST_OVERLAP,
        decontaminate_source=shard_path,
    )
    steps.append(ExecutorStep(name=step_name, fn=dedupe, config=cfg))

if __name__ == "__main__":
    executor_main(
        steps=steps,
        description="Run per-shard Dolma dedupe against MMLU test set, reading Parquet or JSONL directly",
    )
