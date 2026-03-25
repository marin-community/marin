# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
import time

from iris.marin_fs import url_to_fs
from marin.execution.step_spec import StepSpec
from zephyr import Dataset, ZephyrContext

from marin.utils import fsspec_exists, fsspec_glob

logger = logging.getLogger(__name__)


def transfer_files(
    input_path: str,
    output_path: str,
    *,
    num_random_files: int | None = None,
    filetype: str = "jsonl.zst",
) -> None:
    """Transfer files from input_path to output_path.

    When num_random_files is None, copies all matching files.
    When specified, randomly samples that many files.
    """
    input_path = input_path.rstrip("/")

    logger.info("Transferring %s to %s", input_path, output_path)
    start_time = time.time()
    fs, _ = url_to_fs(input_path)
    if not fs.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist.")

    filenames = fsspec_glob(os.path.join(input_path, f"**/*.{filetype}"))

    if num_random_files is not None:
        random.seed(42)
        random.shuffle(filenames)
        filenames = filenames[:num_random_files]

    def copy_file(filename: str) -> None:
        output_filename = os.path.join(output_path, os.path.basename(filename))
        if not fsspec_exists(output_filename):
            fs.makedirs(output_path, exist_ok=True)
            fs.copy(filename, output_filename)

    pipeline = Dataset.from_list(filenames).map(copy_file)
    ctx = ZephyrContext(name="fs-transfer")
    ctx.execute(pipeline)

    elapsed = time.time() - start_time
    logger.info("Transferred %s to %s (%.1fs)", input_path, output_path, elapsed)


def transfer_step(
    name: str,
    *,
    input_path: str,
    num_random_files: int | None = None,
    filetype: str = "jsonl.zst",
    deps: list[StepSpec] | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that transfers files between fsspec paths."""

    def _run(output_path: str) -> None:
        transfer_files(input_path, output_path, num_random_files=num_random_files, filetype=filetype)

    return StepSpec(
        name=name,
        fn=_run,
        deps=deps or [],
        hash_attrs={"input_path": input_path, "num_random_files": num_random_files, "filetype": filetype},
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
