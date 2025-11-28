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

Run Zephyr-based *train test overlap* detection (a.k.a. contamination
checking) over large training datasets.

For each dataset listed in `DATASET_CONFIGS` this script constructs a single
`ExecutorStep` that:
    • Passes the entire dataset directory to dedupe
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
1. To add a new dataset simply append a DatasetConfig to `DATASET_CONFIGS`.
2. Evaluation datasets are automatically resolved from EVAL_DATASET_STEPS in eval_datasets_overlap.py.
"""

import logging
from dataclasses import dataclass

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe

from experiments.midtraining_datasets import finemath_3_plus
from experiments.pretraining_datasets.simple import downloads
from experiments.pretraining_datasets.dolmino import downloads as dolmino_downloads
from experiments.pretraining_datasets.nemotron import downloads as nemotron_downloads
from experiments.train_test_overlap.eval_datasets_overlap import EVAL_DATASET_STEPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# N-gram configuration for train-test overlap detection
DEFAULT_NGRAM_CONFIG = NGramConfig(
    ngram_length=[15],  # Multiple n-gram sizes - modify this to change n-grams
    overlap_threshold=1e-6,
    stride=0,
)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a single dataset to process for train-test overlap detection."""

    name: str
    """Human-readable name for the dataset (used in output paths)."""

    path: str
    """Path to the dataset directory (local, GCS, or S3)."""

    text_field: str = "text"
    """Name of the text field in the parquet file."""


def run_train_test_overlap(config: DedupeConfig) -> str:
    logger.info(f"Starting train-test overlap dedupe with config: {config}")
    dedupe(config)
    logger.info(f"Train-test overlap completed! Results written to {config.output_path}")
    return config.output_path


# starcoder is parquet with 'content' as text key
# finemath is parquet with 'text' as text key
DATASET_CONFIGS = [
    DatasetConfig(name="finemath", path=finemath_3_plus, text_field="text"),
    DatasetConfig(name="dclm", path=downloads["dclm_baseline"]),
    DatasetConfig(name="starcoder", path=downloads["starcoderdata"], text_field="content"),
    DatasetConfig(name="proofpile", path=downloads["proofpile_2"]),
    DatasetConfig(name="dolmino", path=dolmino_downloads["dolmino"]),
    DatasetConfig(name="nemotron_cc", path=nemotron_downloads["nemotron_cc"]),
]


def build_step(dataset_config: DatasetConfig) -> ExecutorStep:
    dedupe_config = DedupeConfig(
        input_path=dataset_config.path,
        output_path=this_output_path(),
        decontaminate_source=EVAL_DATASET_STEPS,
        attribute_name="ngram_overlap",
        false_positive_rate=1e-20,
        ngram=DEFAULT_NGRAM_CONFIG,
        processes=1024,
        mode=DedupMode.TRAIN_TEST_OVERLAP,
        text_field=dataset_config.text_field,
    )

    return ExecutorStep(
        name=f"train_test_overlap/dolma/total/{dataset_config.name}",
        fn=run_train_test_overlap,
        config=dedupe_config,
        description=f"Run dedupe train-test overlap on {dataset_config.name}",
        pip_dependency_groups=["quality_dedup_consolidate"],
    )


STEPS = [build_step(config) for config in DATASET_CONFIGS]

if __name__ == "__main__":
    executor_main(
        steps=STEPS,
        description="Run train-test-overlap dedupe across all pretraining and midtraining datasets",
    )
