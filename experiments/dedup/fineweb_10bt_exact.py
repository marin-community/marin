# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download fineweb-edu 10BT sample (~10GB) and run exact paragraph dedup locally.

Usage:
    MARIN_PREFIX=/tmp/marin uv run iris --config=lib/iris/examples/local.yaml job run -- \\
        python experiments/dedup/fineweb_10bt_exact.py [--max-parallelism N]
"""

import argparse
import logging
import os

from marin.datakit.canonical import fineweb_edu
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph
from rigging.filesystem import marin_prefix
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "exact-para-dedup-fineweb-10bt")


def build_steps(max_parallelism: int) -> list[StepSpec]:
    download = fineweb_edu.download(
        hf_urls_glob=["sample/10BT/*.parquet"],
    )

    dedup_step = StepSpec(
        name="exact_dedup_fineweb_10bt",
        output_path_prefix=f"{marin_prefix()}/tmp/{OUTPUT_PREFIX}",
        deps=[download],
        fn=lambda op: dedup_exact_paragraph(
            input_paths=os.path.join(download.output_path, "sample/10BT"),
            output_path=op,
            max_parallelism=max_parallelism,
        ),
    )
    return [download, dedup_step]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=128,
        metavar="N",
        help="Maximum parallelism passed to dedup_exact_paragraph (default: %(default)s).",
    )
    args = parser.parse_args()
    configure_logging(logging.INFO)
    StepRunner().run(build_steps(max_parallelism=args.max_parallelism))


if __name__ == "__main__":
    main()
