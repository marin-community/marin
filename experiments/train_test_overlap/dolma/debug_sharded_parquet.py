#!/usr/bin/env python3
import logging

from experiments.midtraining_datasets import finemath_3_plus
from experiments.train_test_overlap.utils import run_all_shards, ShardedDedupeConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure the sharded dedupe pipeline
config = ShardedDedupeConfig(
    dataset_dir=finemath_3_plus,
    output_path=this_output_path(),
    max_in_flight=64,
)

finemath_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/shm_test",
    fn=run_all_shards,
    config=config,
    description="Run per-shard Dolma dedupe against MMLU test set with backpressure",
)

if __name__ == "__main__":
    executor_main(
        steps=[finemath_dedupe_step],
        description="Run per-shard Dolma dedupe against MMLU test set, reading Parquet or JSONL directly",
    )
