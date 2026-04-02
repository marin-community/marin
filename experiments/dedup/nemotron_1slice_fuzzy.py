# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fuzzy dedup on a single CC crawl slice of Nemotron-CC (quality=high).

Runs the full fuzzy dedup pipeline (MinHash LSH -> connected components ->
dedup tagging) on CC-MAIN-2013-20 (~43 files, ~15 GB compressed). Use this
to validate the Arrow scatter/reduce optimization on real data.

Usage:
    # Submit as an Iris job (requires cluster connection):
    uv run lib/marin/src/marin/run/ray_run.py -- python experiments/dedup/nemotron_1slice_fuzzy.py

    # Or run directly if gcloud auth is configured:
    uv run python experiments/dedup/nemotron_1slice_fuzzy.py
"""

import logging

from fray.v2 import ResourceConfig
from rigging.log_setup import configure_logging
from rigging.filesystem import marin_temp_bucket

from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document
from marin.processing.classification.deduplication.dedup_commons import _collect_input_files, DEFAULT_FILETYPES

logger = logging.getLogger(__name__)

NEMOTRON_HIGH = "gs://marin-eu-west4/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high"

# Single CC crawl slice prefix
SLICE_PREFIX = "CC-MAIN-2013-20"


def _collect_slice_files() -> list[str]:
    """Collect only files matching SLICE_PREFIX from the quality=high directory."""
    all_files = _collect_input_files(input_paths=NEMOTRON_HIGH, filetypes=DEFAULT_FILETYPES)
    slice_files = [f for f in all_files if SLICE_PREFIX in f]
    logger.info("Selected %d files for slice %s (out of %d total)", len(slice_files), SLICE_PREFIX, len(all_files))
    return slice_files


def build_steps() -> list[StepSpec]:
    slice_files = _collect_slice_files()

    dedup_step = StepSpec(
        name="fuzzy_dedup_nemotron_1slice_rust_arrow",
        output_path_prefix=marin_temp_bucket(ttl_days=1, prefix="arrow-scatter-bench-fast"),
        fn=lambda op: dedup_fuzzy_document(
            input_paths=slice_files,
            output_path=op,
            max_parallelism=32,
            worker_resources=ResourceConfig(cpu=5, ram="32g", disk="5g"),
        ),
    )
    return [dedup_step]


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run(build_steps())
