# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download, filter, and normalize PleIAs/common_corpus from HuggingFace."""

import os

from fray.v2 import ResourceConfig

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from zephyr import Dataset, ZephyrContext, counters

HF_DATASET_ID = "PleIAs/common_corpus"
OPEN_TYPES = ["Open Science", "Open Government", "Open Culture"]


def _count_total(record: dict) -> dict:
    counters.increment("common_corpus/total")
    return record


def _language_is_english(record: dict) -> bool:
    if record.get("language") != "English":
        counters.increment("common_corpus/dropped_non_english")
        return False
    return True


def _open_type_allowed(record: dict) -> bool:
    if record.get("open_type") not in OPEN_TYPES:
        counters.increment("common_corpus/dropped_wrong_open_type")
        return False
    return True


def _count_kept(record: dict) -> dict:
    counters.increment("common_corpus/kept")
    return record


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
        .map(_count_total)
        .filter(_language_is_english)
        .filter(_open_type_allowed)
        .map(_count_kept)
        .write_parquet(
            os.path.join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            skip_existing=True,
        )
    )

    ctx = ZephyrContext(name="filter-common-corpus", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def filter_common_corpus_step(raw_step: StepSpec) -> StepSpec:
    return StepSpec(
        name="raw/common_corpus_english_filtered",
        fn=lambda output_path: filter_common_corpus(raw_step.output_path, output_path),
        deps=[raw_step],
    )


def normalize_common_corpus_step(filtered_step: StepSpec) -> StepSpec:
    """Normalize filtered common_corpus: generate content-hash IDs, dedup, sort."""
    return normalize_step(
        name="normalized/common_corpus_english_filtered",
        download=filtered_step,
        id_field="identifier",
    )
