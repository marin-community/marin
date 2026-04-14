# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download fineweb-edu 10BT sample (~10GB) and run exact paragraph dedup locally.

Usage:
    MARIN_PREFIX=/tmp/marin uv run iris --config=lib/iris/examples/local.yaml job run -- \
        python experiments/dedup/fineweb_10bt_exact.py
"""

import logging
import os

from rigging.log_setup import configure_logging
from rigging.filesystem import marin_prefix

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph

logger = logging.getLogger(__name__)

OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "exact-para-dedup-fineweb-10bt")


def build_steps() -> list[StepSpec]:
    download = download_hf_step(
        "raw/fineweb-edu",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="87f0914",
        hf_urls_glob=["sample/10BT/*.parquet"],
    )

    dedup_step = StepSpec(
        name="exact_dedup_fineweb_10bt",
        output_path_prefix=f"{marin_prefix()}/tmp/{OUTPUT_PREFIX}",
        deps=[download],
        fn=lambda op: dedup_exact_paragraph(
            input_paths=os.path.join(download.output_path, "sample/10BT"),
            output_path=op,
            max_parallelism=4,
        ),
    )
    return [download, dedup_step]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_steps())
