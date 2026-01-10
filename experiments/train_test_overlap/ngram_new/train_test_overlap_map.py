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
Run Zephyr-based map stage for old-style train/test overlap detection.

This map stage builds a test-side n-gram index and streams training data,
emitting overlap records that include training provenance (file + doc id).
"""

import logging
from dataclasses import dataclass

from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.midtraining_datasets import finemath_3_plus
from experiments.train_test_overlap.eval_datasets_overlap import EVAL_DATASET_STEPS
from experiments.train_test_overlap.ngram_new.overlap_map import OverlapMapConfig, run_overlap_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


DEFAULT_NGRAM_LENGTHS = [15]
DEFAULT_MAX_PARALLELISM = 16
DEFAULT_NUM_SHARDS = 64


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a single training dataset."""

    name: str
    path: str
    text_field: str = "text"
    num_shards: int = DEFAULT_NUM_SHARDS


DATASET_CONFIGS = [
    DatasetConfig(name="finemath", path=finemath_3_plus, text_field="text"),
    # DatasetConfig(name="dclm", path=downloads["dclm_baseline"]),
    # DatasetConfig(name="starcoder", path=downloads["starcoderdata"], text_field="content"),
    # DatasetConfig(name="proofpile", path=downloads["proofpile_2"]),
    # DatasetConfig(name="dolmino", path=dolmino_downloads["dolmino"]),
    # DatasetConfig(name="nemotron_cc", path=nemotron_downloads["nemotron_cc"]),
]


def build_map_step(dataset_config: DatasetConfig) -> ExecutorStep:
    config = OverlapMapConfig(
        input_path=dataset_config.path,
        eval_dataset_paths=EVAL_DATASET_STEPS,
        output_path=this_output_path(),
        ngram_lengths=DEFAULT_NGRAM_LENGTHS,
        processes=DEFAULT_MAX_PARALLELISM,
        num_shards=dataset_config.num_shards,
        text_field=dataset_config.text_field,
    )

    return ExecutorStep(
        name=f"train_test_overlap/ngram_new/map/{dataset_config.name}",
        fn=run_overlap_map,
        config=config,
        description=f"Run n-gram overlap map stage for {dataset_config.name} (with provenance)",
    )


MAP_STEPS = [build_map_step(cfg) for cfg in DATASET_CONFIGS]

if __name__ == "__main__":
    executor_main(
        steps=MAP_STEPS,
        description="Run n-gram overlap map stage across training datasets",
    )
