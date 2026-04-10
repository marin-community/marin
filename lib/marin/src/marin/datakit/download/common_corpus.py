# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and filter PleIAs/common_corpus from HuggingFace."""

import os

from fray.v2 import ResourceConfig

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec
from zephyr import Dataset, ZephyrContext
from zephyr.expr import col

HF_DATASET_ID = "PleIAs/common_corpus"
OPEN_TYPES = ["Open Science", "Open Government", "Open Culture"]


def download_common_corpus_raw_step() -> StepSpec:
    """Download the raw PleIAs/common_corpus parquet files to GCS."""
    return download_hf_step(
        "raw/common_corpus",
        hf_dataset_id=HF_DATASET_ID,
        revision="b78a5c1",
        hf_urls_glob=["common_corpus_*/*.parquet"],
    )


def filter_common_corpus(input_path: str, output_path: str) -> None:
    """Filter common_corpus to English + open types, writing parquet."""
    pipeline = (
        Dataset.from_files(os.path.join(input_path, "**/*.parquet"))
        .load_parquet()
        .filter(col("language") == "English")
        .filter(
            (col("open_type") == "Open Science")
            | (col("open_type") == "Open Government")
            | (col("open_type") == "Open Culture")
        )
        .write_parquet(
            os.path.join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            skip_existing=True,
        )
    )

    ctx = ZephyrContext(name="filter-common-corpus", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def filter_common_corpus_step(raw_step: StepSpec) -> StepSpec:
    return StepSpec(
        name="raw/common_corpus_english",
        fn=lambda output_path: filter_common_corpus(raw_step.output_path, output_path),
        deps=[raw_step],
    )
