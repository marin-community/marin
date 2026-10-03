# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild a compact, source-confirmed Snowball replay without family-wide downloads."""

from functools import cache

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.datakit.sources import DatakitSource
from marin.execution.step_runner import StepRunner
from rigging.log_setup import configure_logging

NEMOTRON_SPECIALIZED_REPOSITORY = "nvidia/Nemotron-Pretraining-Specialized-v1"
NEMOTRON_SPECIALIZED_REVISION = "9ed3718"
MATH_TEXTBOOKS_GLOB = "Nemotron-Pretraining-Math-Textbooks/**/*.parquet"


@cache
def snowball_math_textbooks_source() -> DatakitSource:
    """Return only the pinned math-textbook subset confirmed in Snowball replay."""
    download = download_hf_step(
        "raw/snowball_replay/nemotron_specialized/math_textbooks",
        hf_dataset_id=NEMOTRON_SPECIALIZED_REPOSITORY,
        revision=NEMOTRON_SPECIALIZED_REVISION,
        hf_urls_glob=[MATH_TEXTBOOKS_GLOB],
        zephyr_max_parallelism=4,
    )
    normalized = normalize_step(
        name="normalized/snowball_replay/nemotron_specialized/math_textbooks",
        download=download,
        file_extensions=(".parquet",),
    )
    return DatakitSource(
        name="nemotron_specialized/math_textbooks",
        normalize_steps=(download, normalized),
        rough_token_count_b=25.59,
    )


if __name__ == "__main__":
    configure_logging()
    StepRunner().run([snowball_math_textbooks_source().normalized])
