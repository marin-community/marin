#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from experiments.pretraining_datasets import dclm_baseline, starcoderdata, proofpile_2, dolmino, nemotron_cc
from experiments.train_test_overlap.utils import (
    EVAL_DATASET_STEPS,
    DatasetConfig,
    ShardedDedupeConfig,
    UnifiedResources,
    run_all_shards,
)
from marin.execution.executor import ExecutorStep, executor_main, this_output_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_IN_FLIGHT = 8
MAX_PER_WORKER = 8

# Custom temporary directory for deduplication processing
# default uses ram, but let's use gcsfuse for speed
TEMP_DIR = "/dev/shm"
# TODO: debug open file descriptor limit on gcsfuse mount
# TEMP_DIR = "/opt/gcsfuse_mount/dedupe/"

# starcoder is parquet with 'content' as text key
# finemath is parquet with 'text' as text key
#
# TPU resource note:
# - Only the following TPU types are supported for gating concurrency via the
#   head resource key: "v4-8", "v5p-8", "v6e-4".
#   See experiments/evals/resource_configs.py for the same canonical IDs.
# - We do not require a TPU device for dedupe itself; we use the
#   "TPU-<type>-head" fractional resource solely to control scheduling
#   on clusters
ALLOWED_TPU_TYPES = ("v4-8", "v5p-8", "v6e-4")
TPU_TYPE = "v6e-4"  # choose from ALLOWED_TPU_TYPES
DATASET_CONFIGS = [
    DatasetConfig(name="finemath", path=finemath_3_plus, max_in_flight=MAX_IN_FLIGHT, text_field="text"),
    DatasetConfig(name="dclm", path=dclm_baseline, max_in_flight=MAX_IN_FLIGHT),
    DatasetConfig(name="starcoder", path=starcoderdata, max_in_flight=MAX_IN_FLIGHT, text_field="content"),
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
        text_field=dataset_config.text_field,
        temp_dir=TEMP_DIR,
        # internal Dolma parallelism for each shard task
        processes=15,
        # Unified resource specification: schedule against v4 fleet with fractional head resource
        unified_resources=UnifiedResources(
            tpu_type=TPU_TYPE,
            tpu_head_fraction=1 / MAX_PER_WORKER,
            num_cpus=15,
        ),
    )
    return ExecutorStep(
        name=f"train_test_overlap/dolma/total/{dataset_config.name}",
        fn=run_all_shards,
        config=cfg,
        description=f"Run dedupe train-test overlap on {dataset_config.name} shards",
        pip_dependency_groups=["quality_dedup_consolidate"],
    )


STEPS = [build_step(config) for config in DATASET_CONFIGS]

if __name__ == "__main__":
    executor_main(
        steps=STEPS,
        description="Run train-test-overlap dedupe across all pretraining and midtraining datasets",
    )
