#!/usr/bin/env python3
import logging

from experiments.midtraining_datasets import finemath_3_plus
from experiments.pretraining_datasets import dclm_baseline, dolmino, nemotron_cc, proofpile_2, starcoderdata
from experiments.train_test_overlap.utils import ShardedDedupeConfig, run_all_shards
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

finemath_config = ShardedDedupeConfig(
    dataset_dir=finemath_3_plus,
    output_path=this_output_path(),
    max_in_flight=64,
)
finemath_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total/finemath",
    fn=run_all_shards,
    config=finemath_config,
    description="Run dedupe train-test overlap on Finemath-3+ shards",
)

dclm_config = ShardedDedupeConfig(
    dataset_dir=dclm_baseline,
    output_path=this_output_path(),
    max_in_flight=64,
)
dclm_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total/dclm",
    fn=run_all_shards,
    config=dclm_config,
    description="Run dedupe train-test overlap on DCLM baseline shards",
)

star_config = ShardedDedupeConfig(
    dataset_dir=starcoderdata,
    output_path=this_output_path(),
    max_in_flight=64,
)
starcoder_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total/starcoder",
    fn=run_all_shards,
    config=star_config,
    description="Run dedupe train-test overlap on StarCoder shards",
)

proofpile_config = ShardedDedupeConfig(
    dataset_dir=proofpile_2,
    output_path=this_output_path(),
    max_in_flight=128,
)
proofpile_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total/proofpile",
    fn=run_all_shards,
    config=proofpile_config,
    description="Run dedupe train-test overlap on Proof-Pile shards",
)

dolmino_config = ShardedDedupeConfig(
    dataset_dir=dolmino,
    output_path=this_output_path(),
    max_in_flight=128,
)
dolmino_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total/dolmino_1e-12",
    fn=run_all_shards,
    config=dolmino_config,
    description="Run dedupe train-test overlap on Dolmino shards",
)


nemotron_config = ShardedDedupeConfig(
    dataset_dir=nemotron_cc,
    output_path=this_output_path(),
    max_in_flight=64,
)
nemotron_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total/nemotron_cc",
    fn=run_all_shards,
    config=nemotron_config,
    description="Run dedupe train-test overlap on Nemotron-CC shards",
)

if __name__ == "__main__":
    executor_main(
        steps=[
            finemath_dedupe_step,
            dclm_dedupe_step,
            starcoder_dedupe_step,
            proofpile_dedupe_step,
            nemotron_dedupe_step,
            dolmino_dedupe_step,
        ],
        description="Run train-test-overlap dedupe across all pretraining and midtraining datasets",
    )
