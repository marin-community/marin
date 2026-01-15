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

from marin.download.huggingface.download_hf import DownloadConfig
from marin.download.huggingface.download_hf import download_hf as _download_hf
from marin.execution import deferred, executor_main, output, step
from marin.processing.classification.deduplication.pipeline import DedupeConfig, DedupMode
from marin.processing.classification.deduplication.pipeline import deduplicate as _deduplicate

logger = logging.getLogger(__name__)

# Mark library functions as deferred
download_hf = deferred(_download_hf)
deduplicate = deferred(_deduplicate)


@step(name="raw_fineweb_edu_small_2")
def raw_fineweb_edu_small_2():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="HuggingFaceFW/fineweb-edu",
            revision="3c452cb",
            hf_urls_glob=["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"],
        )
    )


@step(name="raw_fineweb_edu_small_1")
def raw_fineweb_edu_small_1():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="HuggingFaceFW/fineweb-edu",
            revision="3c452cb",
            hf_urls_glob=["sample/10BT/000_00000.parquet"],
        )
    )


@step(name="raw_fineweb_edu_small_10bt")
def raw_fineweb_edu_small_10bt():
    return download_hf(
        DownloadConfig(
            hf_dataset_id="HuggingFaceFW/fineweb-edu",
            revision="3c452cb",
            hf_urls_glob=["sample/10BT/*.parquet"],
        )
    )


@step(name="dedup_raw_fineweb_edu_small_1")
def dedup_fineweb_edu_small_1():
    dataset = raw_fineweb_edu_small_1()
    input_path = dataset.cd("sample/10BT")

    return deduplicate(
        DedupeConfig(
            input_path=input_path,
            output_path=output(),
            mode=DedupMode.EXACT_PARAGRAPH_DEDUPLICATE,
            processes=7,
        )
    )


@step(name="dedup_raw_fineweb_edu_small_2")
def dedup_fineweb_edu_small_2():
    dataset = raw_fineweb_edu_small_2()
    input_path = dataset.cd("sample/10BT")

    return deduplicate(
        DedupeConfig(
            input_path=input_path,
            output_path=output(),
            mode=DedupMode.EXACT_PARAGRAPH_DEDUPLICATE,
            processes=7,
        )
    )


@step(name="dedup_raw_fineweb_edu_small_10bt")
def dedup_fineweb_edu_small_10bt():
    dataset = raw_fineweb_edu_small_10bt()
    input_path = dataset.cd("sample/10BT")

    return deduplicate(
        DedupeConfig(
            input_path=input_path,
            output_path=output(),
            mode=DedupMode.EXACT_PARAGRAPH_DEDUPLICATE,
            processes=1024,
        )
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
