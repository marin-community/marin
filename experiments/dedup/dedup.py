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
import ray
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe

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

# N-gram configuration for train-test overlap detection
DEFAULT_NGRAM_CONFIG = NGramConfig(
    ngram_length=[5, 10, 15],
    overlap_threshold=1e-6,
    stride=0,
)


@ray.remote(runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu", "PJRT_DEVICE": "cpu"}})
def run_dedup(config: DedupeConfig) -> str:
    logger.info(f"Starting dedupe with config: {config}")
    dedupe(config)
    logger.info(f"Dedupe completed! Results written to {config.output_path}")
    return config.output_path


def build_dedup_step(dataset: InputName) -> ExecutorStep:
    config = DedupeConfig(
        input_path=dataset.cd("sample/10BT"), attribute_name="is_duplicate", mode=DedupMode.EXACT_DOC_DEDUPLICATE
    )

    return ExecutorStep(
        name=f"dedup_{dataset.name}",
        fn=run_dedup,
        config=config,
        description=f"Run dedupe on {dataset.name}",
        pip_dependency_groups=["quality_dedup_consolidate"],
    )


STEPS = [build_dedup_step(fineweb_edu_small_1), build_dedup_step(fineweb_edu_small_2)]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    executor_main(
        steps=STEPS,
        description="Run dedupe",
    )
