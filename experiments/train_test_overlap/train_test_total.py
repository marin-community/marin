#!/usr/bin/env python3
"""

Run Dolma-based *train test overlap* detection (a.k.a. contamination
checking) over **every shard** of a set of large training datasets.

For each dataset listed in `DATASET_CONFIGS` this script constructs a single
`ExecutorStep` that:
    • Discovers all input shards (Parquet / JSONL / …).
    • Launches `run_all_shards` (from `train_test_overlap.utils`) with
      back-pressure so only `max_in_flight` shards are processed in parallel.
    • Writes attribute files containing n-gram overlap annotations under
      `<prefix>/train_test_overlap/dolma/total/<dataset_name>/**/15/…`.

Usage (local example):

    python experiments/train_test_overlap/train_test_total.py --prefix gs://my-bucket

Notes
-----
1. The heavy lifting is performed by Dolma via `marin.processing.classification.dedupe.dedupe`.
2. Shard discovery and Ray back-pressure logic lives in
   `experiments.train_test_overlap.utils`.
3. To add a new dataset simply append a `(name, path, max_in_flight)` tuple
   to `DATASET_CONFIGS`.
"""
import logging

from experiments.midtraining_datasets import finemath_3_plus
from experiments.pretraining_datasets import (
    dclm_baseline,
    dolmino,
    nemotron_cc,
    proofpile_2,
    starcoderdata,
)
from experiments.train_test_overlap.utils import ShardedDedupeConfig, run_all_shards
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_CONFIGS = [
    ("finemath", finemath_3_plus, 64),
    ("dclm", dclm_baseline, 64),
    ("starcoder", starcoderdata, 64),
    ("proofpile", proofpile_2, 128),
    ("dolmino_1e-12", dolmino, 128),
    ("nemotron_cc", nemotron_cc, 64),
]


def build_step(name: str, dataset_dir: str, max_in_flight: int) -> ExecutorStep:
    cfg = ShardedDedupeConfig(
        dataset_dir=dataset_dir,
        output_path=this_output_path(),
        max_in_flight=max_in_flight,
    )
    return ExecutorStep(
        name=f"train_test_overlap/dolma/total/{name}",
        fn=run_all_shards,
        config=cfg,
        description=f"Run dedupe train-test overlap on {name} shards",
    )


STEPS = [build_step(n, d, m) for n, d, m in DATASET_CONFIGS]

if __name__ == "__main__":
    executor_main(
        steps=STEPS,
        description="Run train-test-overlap dedupe across all pretraining and midtraining datasets",
    )
