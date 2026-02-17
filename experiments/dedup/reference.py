# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner
from marin.processing.classification.deduplication.dedup_commons import DedupConfig, DedupMode, deduplicate

from experiments.pretraining_datasets.simple import downloads

logger = logging.getLogger(__name__)


def build_dedup_step(dataset: str, dataset_name: str, max_parallelism: int) -> StepSpec:
    """
    Builds a deduplication step for the given dataset.

    Args:
        dataset: The input dataset path.
        dataset_name: A human-readable name for the dataset.
        max_parallelism: Maximum parallelism for Zephyr tasks.
    """
    return StepSpec(
        name=f"dedup_{dataset_name}",
        hash_attrs={
            "input_paths": dataset,
            "mode": DedupMode.FUZZY_DOCUMENT,
            "processes": max_parallelism,
        },
        fn=lambda output_path: deduplicate(
            DedupConfig(
                input_paths=dataset,
                mode=DedupMode.FUZZY_DOCUMENT,
                processes=max_parallelism,
            )
        ),
    )


STEPS = [
    build_dedup_step(downloads["fineweb_edu_sample_10bt"], "fineweb_edu_sample_10bt", max_parallelism=1024),
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    StepRunner().run(STEPS)
