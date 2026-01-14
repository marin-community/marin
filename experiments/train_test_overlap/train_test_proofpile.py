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
Run train-test overlap detection for Proofpile against evaluation datasets.

This script creates a single ExecutorStep that compares Proofpile against
all evaluation datasets defined in eval_datasets_overlap.py.

Usage:
    python experiments/train_test_overlap/train_test_proofpile.py --prefix gs://my-bucket
"""

import logging

from marin.execution.executor import ExecutorStep, executor_main, StepRef, step, StepContext
from marin.processing.classification.decon import DeconConfig, DeconMode, NGramConfig, decontaminate

from experiments.pretraining_datasets.simple import tokenized
from experiments.train_test_overlap.eval_datasets_overlap import EVAL_DATASET_STEPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# N-gram configuration for train-test overlap detection
DEFAULT_NGRAM_CONFIG = NGramConfig(
    ngram_length=[5, 10, 15],
    overlap_threshold=1e-6,
    stride=0,
)


def run_train_test_overlap(config: DeconConfig) -> str:
    logger.info(f"Starting train-test overlap dedupe with config: {config}")
    decontaminate(config)
    logger.info(f"Train-test overlap completed! Results written to {config.output_path}")
    return config.output_path


@step(name="tmp/train_test_overlap/proofpile", fn=run_train_test_overlap, description="Run dedupe train-test overlap on Proofpile")
def build_proofpile_step(ctx: StepContext):
    return DeconConfig(
        input_path=ctx.require(tokenized["proofpile_2"]),
        output_path=ctx.output,
        decontaminate_source=[ctx.require(s) for s in EVAL_DATASET_STEPS],
        attribute_name="ngram_overlap",
        false_positive_rate=1e-20,
        ngram=DEFAULT_NGRAM_CONFIG,
        processes=1024,
        mode=DeconMode.TRAIN_TEST_OVERLAP,
        text_field="text",
    )


STEPS = [build_proofpile_step()]

if __name__ == "__main__":
    executor_main(
        steps=STEPS,
        description="Run train-test-overlap dedupe for Proofpile against evaluation datasets",
    )
