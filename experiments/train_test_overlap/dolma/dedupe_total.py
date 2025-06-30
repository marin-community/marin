#!/usr/bin/env python3
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
