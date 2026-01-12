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
Run Zephyr-based reduce stage for old-style train/test overlap detection.
"""

import logging

from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.train_test_overlap.eval_datasets_overlap import EVAL_DATASET_STEPS
from experiments.train_test_overlap.ngram_new.overlap_reduce import OverlapReduceConfig, run_overlap_reduce
from experiments.train_test_overlap.ngram_new.train_test_overlap_map import (
    DATASET_CONFIGS,
    DEFAULT_MAX_PARALLELISM,
    DEFAULT_NGRAM_LENGTHS,
    build_map_step,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_WRITE_DETAILS = False


def build_reduce_step(dataset_config, map_step: ExecutorStep) -> ExecutorStep:
    config = OverlapReduceConfig(
        map_output_path=map_step,
        output_path=this_output_path(),
        eval_dataset_paths=EVAL_DATASET_STEPS,
        ngram_lengths=DEFAULT_NGRAM_LENGTHS,
        processes=DEFAULT_MAX_PARALLELISM,
        write_details=True,
    )

    return ExecutorStep(
        name=f"train_test_overlap/ngram_new/reduce/{dataset_config.name}",
        fn=run_overlap_reduce,
        config=config,
        description=f"Run n-gram overlap reduce stage for {dataset_config.name}",
    )


MAP_STEPS = [build_map_step(cfg) for cfg in DATASET_CONFIGS]
REDUCE_STEPS = [build_reduce_step(cfg, step) for cfg, step in zip(DATASET_CONFIGS, MAP_STEPS, strict=True)]
STEPS = [*MAP_STEPS, *REDUCE_STEPS]


if __name__ == "__main__":
    executor_main(
        steps=STEPS,
        description="Run n-gram overlap reduce stage across training datasets",
    )
