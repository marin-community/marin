# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact paragraph dedup on quality=high of Nemotron-CC.

Runs the full exact paragraph dedup pipeline on the quality=high split.
Faster wall-time than fuzzy dedup while still exercising shuffle at scale.

Usage:
    uv run iris --config=lib/iris/examples/marin.yaml job run -- python experiments/dedup/nemotron_1split_exact.py
    MAX_FILES=1000 uv run iris ... -- python experiments/dedup/nemotron_1split_exact.py
"""

import logging
import os

from rigging.log_setup import configure_logging
from rigging.filesystem import marin_temp_bucket

from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.exact import dedup_exact_paragraph

logger = logging.getLogger(__name__)

NEMOTRON_HIGH = "gs://marin-eu-west4/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high"

OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "arrow-scatter-exact-bench-fast")
MAX_FILES = int(os.environ.get("MAX_FILES", "0"))  # 0 = all files


def _maybe_truncate_inputs(input_path: str, max_files: int) -> str | list[str]:
    """If max_files > 0, glob the input path and return a truncated file list."""
    if max_files <= 0:
        return input_path
    from marin.utils import fsspec_glob

    files = sorted(fsspec_glob(f"{input_path.rstrip('/')}/**/*.{{jsonl.gz,jsonl,json.gz,json,parquet,vortex}}"))
    truncated = files[:max_files]
    logger.info("Truncated input to %d / %d files (max_files=%d)", len(truncated), len(files), max_files)
    return truncated


def build_steps() -> list[StepSpec]:
    input_paths = _maybe_truncate_inputs(NEMOTRON_HIGH, MAX_FILES)

    dedup_step = StepSpec(
        name="exact_dedup_nemotron_high_arrow",
        output_path_prefix=marin_temp_bucket(ttl_days=1, prefix=OUTPUT_PREFIX),
        fn=lambda op: dedup_exact_paragraph(
            input_paths=input_paths,
            output_path=op,
            max_parallelism=2048,
        ),
    )
    return [dedup_step]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_steps())
