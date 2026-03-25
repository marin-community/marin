# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document

from experiments.pretraining_datasets.simple import downloads

logger = logging.getLogger(__name__)


def build_dedup_step(dataset: InputName, max_parallelism: int) -> ExecutorStep:
    return ExecutorStep(
        name=f"dedup_{dataset.name}",
        fn=lambda op: dedup_fuzzy_document(
            input_paths=dataset,
            output_path=op,
            max_parallelism=max_parallelism,
        ),
        description=f"Run dedupe on {dataset.name}",
    )


STEPS = [
    build_dedup_step(downloads["fineweb_edu_sample_10bt"], max_parallelism=1024),
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    executor_main(
        steps=STEPS,
    )
