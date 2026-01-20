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

import logging

from experiments.pretraining_datasets.simple import downloads
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.processing.classification.deduplication.dedup_commons import DedupConfig, DedupMode, deduplicate

logger = logging.getLogger(__name__)


def build_dedup_step(dataset: InputName, max_parallelism: int) -> ExecutorStep:
    """
    Builds a deduplication step for the given dataset.

    Args:
        dataset: The input dataset to deduplicate.
        max_parallelism: Maximum parallelism for Zephyr tasks.
    """
    config = DedupConfig(
        input_paths=dataset,
        mode=DedupMode.FUZZY_DOCUMENT_DUPLICATE,
        processes=max_parallelism,
    )

    return ExecutorStep(
        name=f"dedup_{dataset.name}",
        fn=deduplicate,
        config=config,
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
