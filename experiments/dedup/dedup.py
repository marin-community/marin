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
Run deduplication on fineweb-edu.

Usage:
    python experiments/dedup/dedup.py --prefix gs://my-bucket
"""

import logging

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.processing.classification.deduplication.pipeline import DedupeConfig, DedupMode, deduplicate

logger = logging.getLogger(__name__)


fineweb_edu_small_2 = ExecutorStep(
    name="raw_fineweb_edu_small_2",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        hf_urls_glob=["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"],
    ),
)

fineweb_edu_small_1 = ExecutorStep(
    name="raw_fineweb_edu_small_1",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        hf_urls_glob=["sample/10BT/000_00000.parquet"],
    ),
)

fineweb_edu_small_10bt = ExecutorStep(
    name="raw_fineweb_edu_small_10bt",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        hf_urls_glob=["sample/10BT/*.parquet"],
    ),
)


def run_dedup(config: DedupeConfig) -> str:
    logger.info(f"Starting dedupe with config: {config}")

    deduplicate(config)

    logger.info(f"Dedupe completed! Results written to {config.output_path}")
    return config.output_path


def build_dedup_step(dataset: InputName, max_parallelism: int) -> ExecutorStep:
    """
    Builds a deduplication step for the given dataset.

    Args:
        dataset: The input dataset to deduplicate.
        max_parallelism: Maximum parallelism for Zephyr tasks.
    """
    input_path = dataset.cd("sample/10BT")

    config = DedupeConfig(
        input_path=input_path,
        attribute_name="is_duplicate",
        mode=DedupMode.PARAGRAPH_DEDUPLICATE,
        processes=max_parallelism,
    )

    return ExecutorStep(
        name=f"dedup_{dataset.name}",
        fn=run_dedup,
        config=config,
        description=f"Run dedupe on {dataset.name}",
        pip_dependency_groups=["quality_dedup_consolidate"],
    )


STEPS = [
    build_dedup_step(fineweb_edu_small_1, max_parallelism=7),
    build_dedup_step(fineweb_edu_small_2, max_parallelism=7),
    build_dedup_step(fineweb_edu_small_10bt, max_parallelism=1024),
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    executor_main(
        steps=STEPS,
        description="Run dedupe",
    )
