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

from dataclasses import dataclass, field, replace
from enum import StrEnum, auto
import logging

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
import ray
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.classification.dedupe import DedupeConfig, DedupMode, NGramConfig, dedupe

logger = logging.getLogger(__name__)


class Dataset(StrEnum):
    FINEWEB_EDU_SMALL_1 = auto()
    FINEWEB_EDU_SMALL_2 = auto()


@dataclass(frozen=True)
class DedupPipelineConfig:
    # TODO: need to copy these from DedupeConfig to avoid default argument issues
    dataset: Dataset = Dataset.FINEWEB_EDU_SMALL_1
    dedup_config: DedupeConfig = field(default_factory=DedupeConfig)


fineweb_edu_small_2 = ExecutorStep(
    name="raw/fineweb-edu-small-2",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        hf_urls_glob=["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"],
    ),
).cd("sample")

fineweb_edu_small_1 = ExecutorStep(
    name="raw/fineweb-edu-small-1",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        hf_urls_glob=["sample/10BT/000_00000.parquet"],
    ),
).cd("sample")

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


def build_dedup_step(config: DedupPipelineConfig = DedupPipelineConfig()) -> ExecutorStep:
    input_path = fineweb_edu_small_1 if config.dataset == Dataset.FINEWEB_EDU_SMALL_1 else fineweb_edu_small_2
    dedupe_config = replace(
        config.dedup_config, input_path=input_path, mode=DedupMode.EXACT_DOC_DEDUPLICATE, processes=8
    )

    return ExecutorStep(
        name=f"tmp/dedup/{config.dataset}",
        fn=run_dedup,
        config=dedupe_config,
        description=f"Run dedupe on {config.dataset}",
        pip_dependency_groups=["quality_dedup_consolidate"],
    )


STEPS = [build_dedup_step()]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    executor_main(
        steps=STEPS,
        description="Run dedupe",
    )
