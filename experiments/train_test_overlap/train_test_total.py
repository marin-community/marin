#!/usr/bin/env python3
"""

Run Dolma-based *train test overlap* detection (a.k.a. contamination
checking) over **every shard** of a set of large training datasets.

For each dataset listed in `DATASET_CONFIGS` this script constructs a single
`ExecutorStep` that:
    • Discovers all input shards (Parquet / JSONL / …).
    • Launches `run_all_shards` (from `train_test_overlap.utils`) with
      back-pressure so only `max_in_flight` shards are processed in parallel.
    • Automatically resolves evaluation dataset paths using the executor framework.
    • Writes attribute files containing n-gram overlap annotations under
      `<prefix>/train_test_overlap/dolma/total/<dataset_name>/**/15/…`.

The evaluation datasets are automatically imported from `eval_datasets_overlap.py`
and their paths are resolved dynamically by the executor framework, removing the
need for hardcoded paths.

Usage (local example):

    python experiments/train_test_overlap/train_test_total.py --prefix gs://my-bucket

Notes
-----
1. The heavy lifting is performed by Dolma via `marin.processing.classification.dedupe.dedupe`.
2. Shard discovery and Ray back-pressure logic lives in
   `experiments.train_test_overlap.utils`.
3. To add a new dataset simply append a DatasetConfig to `DATASET_CONFIGS`.
4. Evaluation datasets are automatically resolved from EVAL_DATASET_STEPS in utils.py.
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
from experiments.train_test_overlap.utils import EVAL_DATASET_STEPS, DatasetConfig, ShardedDedupeConfig, run_all_shards
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_IN_FLIGHT = 32

DATASET_CONFIGS = [
    DatasetConfig(name="finemath", path=finemath_3_plus, max_in_flight=MAX_IN_FLIGHT),
    DatasetConfig(name="dclm", path=dclm_baseline, max_in_flight=MAX_IN_FLIGHT),
    DatasetConfig(name="starcoder", path=starcoderdata, max_in_flight=MAX_IN_FLIGHT),
    DatasetConfig(name="proofpile", path=proofpile_2, max_in_flight=MAX_IN_FLIGHT),
    DatasetConfig(name="dolmino", path=dolmino, max_in_flight=MAX_IN_FLIGHT),
    DatasetConfig(name="nemotron_cc", path=nemotron_cc, max_in_flight=MAX_IN_FLIGHT),
]


def build_step(dataset_config: DatasetConfig) -> ExecutorStep:
    cfg = ShardedDedupeConfig(
        dataset_dir=dataset_config.path,
        output_path=this_output_path(),
        max_in_flight=dataset_config.max_in_flight,
        eval_dataset_steps=EVAL_DATASET_STEPS,
    )
    return ExecutorStep(
        name=f"train_test_overlap/dolma/total/{dataset_config.name}",
        fn=run_all_shards,
        config=cfg,
        description=f"Run dedupe train-test overlap on {dataset_config.name} shards",
    )


STEPS = [build_step(config) for config in DATASET_CONFIGS]

if __name__ == "__main__":
    executor_main(
        steps=STEPS,
        description="Run train-test-overlap dedupe across all pretraining and midtraining datasets",
    )
