#!/usr/bin/env python3
import logging
import os

from experiments.midtraining_datasets import finemath_3_plus
from experiments.pretraining_datasets import dclm_baseline, starcoderdata, proofpile_2, nemotron_cc
from experiments.train_test_overlap.dolma.debug_sharded_parquet import run_all_shards, ShardedDedupeConfig
from marin.execution.executor import ExecutorStep, executor_main, get_executor_step, this_output_path, output_path_of
from marin.utils import fsspec_glob

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Determine MARIN prefix (must point to GCS or local root)
prefix = os.environ.get("MARIN_PREFIX")
if not prefix:
    raise ValueError("MARIN_PREFIX environment variable must be set to your GCS prefix")

# 1) Finemath-3+
fm_step = get_executor_step(finemath_3_plus)
fm_base = fm_step.override_output_path or fm_step.name
fm_subpath = finemath_3_plus.name or ""
if fm_step.override_output_path:
    finemath_dir = os.path.join(prefix, fm_base, fm_subpath) if fm_subpath else os.path.join(prefix, fm_base)
else:
    pattern = os.path.join(prefix, f"{fm_base}-*")
    matches = fsspec_glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No directories matching {pattern}")
    finemath_dir = os.path.join(matches[0], fm_subpath) if fm_subpath else matches[0]

finemath_config = ShardedDedupeConfig(
    dataset_dir=finemath_dir,
    output_path=this_output_path(),
    max_in_flight=64,
)
finemath_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total_finemath",
    fn=run_all_shards,
    config=finemath_config,
    description="Run dedupe train-test overlap on Finemath-3+ shards",
)

# 2) DCLM baseline
clm_step = get_executor_step(dclm_baseline)
# Build the DCLM dataset path under the HuggingFace structure via InputName chaining
dclm_input = (
    dclm_baseline
    .cd("huggingface.co")
    .cd("datasets")
    .cd(clm_step.config.hf_dataset_id)
    .cd("resolve")
    .cd(clm_step.config.revision)
)
dclm_config = ShardedDedupeConfig(
    dataset_dir=dclm_input,
    output_path=this_output_path(),
    max_in_flight=64,
)
dclm_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total_dclm",
    fn=run_all_shards,
    config=dclm_config,
    description="Run dedupe train-test overlap on DCLM baseline shards",
)

# 3) StarCoder
star_config = ShardedDedupeConfig(
    dataset_dir=starcoderdata,
    output_path=this_output_path(),
    max_in_flight=64,
)
starcoder_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total_starcoder",
    fn=run_all_shards,
    config=star_config,
    description="Run dedupe train-test overlap on StarCoder shards",
)

# 4) Proof-Pile
proofpile_config = ShardedDedupeConfig(
    dataset_dir=proofpile_2,
    output_path=this_output_path(),
    max_in_flight=64,
)
proofpile_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total_proofpile",
    fn=run_all_shards,
    config=proofpile_config,
    description="Run dedupe train-test overlap on Proof-Pile shards",
)

# 5) Nemotron-CC
nemotron_config = ShardedDedupeConfig(
    dataset_dir=output_path_of(nemotron_cc).cd("contrib/Nemotron/Nemotron-CC/data-jsonl"),
    output_path=this_output_path(),
    max_in_flight=64,
)
nemotron_dedupe_step = ExecutorStep(
    name="train_test_overlap/dolma/total_nemotron_cc",
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
        ],
        description="Run train-test-overlap dedupe across all pretraining and midtraining datasets",
    ) 