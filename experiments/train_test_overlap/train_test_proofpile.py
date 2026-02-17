#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Run train-test overlap detection for Proofpile against evaluation datasets.

This script creates a single ExecutorStep that compares Proofpile against
all evaluation datasets defined in eval_datasets_overlap.py.

Usage:
    python experiments/train_test_overlap/train_test_proofpile.py --prefix gs://my-bucket
"""

import logging

from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner
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


def build_proofpile_step() -> StepSpec:
    return StepSpec(
        name="tmp/train_test_overlap/proofpile",
        hash_attrs={
            "input_path": tokenized["proofpile_2"],
            "ngram_length": [5, 10, 15],
            "mode": DeconMode.TRAIN_TEST_OVERLAP,
        },
        deps=EVAL_DATASET_STEPS,
        fn=lambda output_path: run_train_test_overlap(
            DeconConfig(
                input_path=tokenized["proofpile_2"],
                output_path=output_path,
                decontaminate_source=EVAL_DATASET_STEPS,
                attribute_name="ngram_overlap",
                false_positive_rate=1e-20,
                ngram=DEFAULT_NGRAM_CONFIG,
                processes=1024,
                mode=DeconMode.TRAIN_TEST_OVERLAP,
                text_field="text",
            )
        ),
    )


STEPS = [build_proofpile_step()]

if __name__ == "__main__":
    StepRunner().run(STEPS)
