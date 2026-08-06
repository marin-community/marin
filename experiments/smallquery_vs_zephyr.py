# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment comparing DataFusion/Ballista vs Zephyr versions of normalize_step."""

import logging
import os

from fray.types import ResourceConfig
from marin.datakit.normalize import normalize_step as zephyr_normalize_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging
from rigging.timing import log_time
from smallquery.datakit.normalize import normalize_step as ballista_normalize_step

logger = logging.getLogger(__name__)

NEMOTRON_RAW_PATH = "gs://marin-eu-west4/scratch/wmoss"
NEMOTRON_DATA_SUBDIR = "nemotron_parquet_subset"
NEMOTRON_QUALITY_DIR = ""


def _verify_nemotron_quality_present(output_path: str) -> None:
    """Confirm the quality split is staged at ``output_path``; never downloads."""
    quality_dir = f"{output_path}/{NEMOTRON_DATA_SUBDIR}/{NEMOTRON_QUALITY_DIR}"
    if quality_dir.endswith("/"):
        quality_dir = quality_dir[:-1]
    fs, _ = url_to_fs(quality_dir)
    if not fs.exists(quality_dir):
        raise RuntimeError(
            f"Nemotron-CC {NEMOTRON_QUALITY_DIR} not found at {quality_dir}. " "Stage the raw dump externally first."
        )
    sample = fs.glob(f"{quality_dir}/**/*.parquet", maxdepth=4)
    if not sample:
        sample = fs.glob(f"{quality_dir}/*.parquet", maxdepth=4)
    if not sample:
        raise RuntimeError(f"Nemotron-CC {NEMOTRON_QUALITY_DIR} at {quality_dir} contains no .parquet files.")
    logger.info("Nemotron-CC %s confirmed at %s", NEMOTRON_QUALITY_DIR, quality_dir)


def main() -> None:
    configure_logging()

    run_id = os.environ.get("SMOKE_RUN_ID", "test-comparison")
    base = f"gs://marin-eu-west4/scratch/wmoss/datakit-nemotron-smoke/{run_id}"

    download = StepSpec(
        name="datakit-nemotron-smoke/download",
        fn=_verify_nemotron_quality_present,
        override_output_path=NEMOTRON_RAW_PATH,
    )

    normalized_zephyr = zephyr_normalize_step(
        name="datakit-nemotron-smoke/normalize-zephyr",
        download=download,
        text_field="text",
        id_field="id",
        relative_input_path=f"{NEMOTRON_DATA_SUBDIR}/{NEMOTRON_QUALITY_DIR}",
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="5g"),
        max_workers=100,
        override_output_path=f"{base}/normalize-zephyr",
    )

    normalized_ballista = ballista_normalize_step(
        name="datakit-nemotron-smoke/normalize-ballista",
        download=download,
        text_field="text",
        id_field="id",
        relative_input_path=f"{NEMOTRON_DATA_SUBDIR}/{NEMOTRON_QUALITY_DIR}",
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="5g"),
        max_workers=100,
        override_output_path=f"{base}/normalize-ballista",
    )

    run_mode = os.environ.get("RUN_MODE", "both")
    steps = [download]
    if run_mode in ("zephyr", "both"):
        steps.append(normalized_zephyr)
    if run_mode in ("ballista", "both"):
        steps.append(normalized_ballista)

    logger.info("Starting Smallquery vs Zephyr comparison run (mode: %s)...", run_mode)
    with log_time("Comparison total wall time"):
        StepRunner().run(steps)
    logger.info("Comparison completed successfully!")


if __name__ == "__main__":
    main()
