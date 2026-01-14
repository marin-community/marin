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
from marin.execution import step, StepContext, StepRef, executor_main
from marin.processing.classification.deduplication.pipeline import DedupeConfig, DedupMode, deduplicate

logger = logging.getLogger(__name__)


@step(name="raw_fineweb_edu_small_2", fn=download_hf)
def raw_fineweb_edu_small_2(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        hf_urls_glob=["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"],
    )


@step(name="raw_fineweb_edu_small_1", fn=download_hf)
def raw_fineweb_edu_small_1(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        hf_urls_glob=["sample/10BT/000_00000.parquet"],
    )


@step(name="raw_fineweb_edu_small_10bt", fn=download_hf)
def raw_fineweb_edu_small_10bt(ctx: StepContext):
    return DownloadConfig(
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="3c452cb",
        hf_urls_glob=["sample/10BT/*.parquet"],
    )


def run_dedup(config: DedupeConfig) -> str:
    logger.info(f"Starting dedupe with config: {config}")

    deduplicate(config)

    logger.info(f"Dedupe completed! Results written to {config.output_path}")
    return config.output_path


@step(name="dedup_raw_fineweb_edu_small_1", fn=run_dedup)
def dedup_fineweb_edu_small_1(ctx: StepContext):
    dataset = ctx.require(raw_fineweb_edu_small_1())
    input_path = dataset.cd("sample/10BT")

    return DedupeConfig(
        input_path=input_path,
        mode=DedupMode.EXACT_PARAGRAPH_DEDUPLICATE,
        processes=7,
    )


@step(name="dedup_raw_fineweb_edu_small_2", fn=run_dedup)
def dedup_fineweb_edu_small_2(ctx: StepContext):
    dataset = ctx.require(raw_fineweb_edu_small_2())
    input_path = dataset.cd("sample/10BT")

    return DedupeConfig(
        input_path=input_path,
        mode=DedupMode.EXACT_PARAGRAPH_DEDUPLICATE,
        processes=7,
    )


@step(name="dedup_raw_fineweb_edu_small_10bt", fn=run_dedup)
def dedup_fineweb_edu_small_10bt(ctx: StepContext):
    dataset = ctx.require(raw_fineweb_edu_small_10bt())
    input_path = dataset.cd("sample/10BT")

    return DedupeConfig(
        input_path=input_path,
        mode=DedupMode.EXACT_PARAGRAPH_DEDUPLICATE,
        processes=1024,
    )


STEPS = [
    dedup_fineweb_edu_small_1(),
    dedup_fineweb_edu_small_2(),
    dedup_fineweb_edu_small_10bt(),
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    executor_main(
        steps=STEPS,
        description="Run dedupe",
    )
