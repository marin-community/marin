# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time
from dataclasses import dataclass

from iris.marin_fs import url_to_fs
from marin.execution.step_spec import StepSpec
from zephyr import Dataset, ZephyrContext

from marin.utils import fsspec_exists, fsspec_glob


@dataclass
class TransferConfig:
    input_path: str
    output_path: str

    # Selectively choose the number of random files to transfer. None means all files
    num_random_files: int | None = None
    filetype: str = "jsonl.zst"


def transfer_files(config: TransferConfig) -> None:
    """Transfers files from the input path to the output path.

    When num_random_files is None, copies the entire directory recursively.
    When num_random_files is specified, randomly samples that many files and
    copies them in parallel using zephyr.
    """
    if config.input_path.endswith("/"):
        input_path = config.input_path[:-1]
    else:
        input_path = config.input_path

    print(f"Downloading {input_path} from GCS.")
    start_time: float = time.time()
    fs, _ = url_to_fs(input_path)
    if not fs.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist.")

    # Glob all matching files
    filenames = fsspec_glob(os.path.join(input_path, f"**/*.{config.filetype}"))

    # Select files: either random sample or all files
    if config.num_random_files is None:
        selected_files = filenames
    else:
        random.seed(42)
        random.shuffle(filenames)
        selected_files = filenames[: config.num_random_files]

    def copy_file(filename: str) -> None:
        """Copy a single file if it doesn't already exist at destination."""
        output_filename = os.path.join(config.output_path, os.path.basename(filename))
        if not fsspec_exists(output_filename):
            # Ensure output directory exists
            fs.makedirs(config.output_path, exist_ok=True)
            fs.copy(filename, output_filename)

    # Always use parallel copying via zephyr
    pipeline = Dataset.from_list(selected_files).map(copy_file)
    ctx = ZephyrContext(name="fs-transfer")
    ctx.execute(pipeline)

    elapsed_time_seconds: float = time.time() - start_time
    print(f"Downloaded {input_path} to {config.output_path} ({elapsed_time_seconds}s).")


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
        transfer_files(
            TransferConfig(
                input_path=input_path,
                output_path=output_path,
                num_random_files=num_random_files,
                filetype=filetype,
            )
        )

    return StepSpec(
        name=name,
        fn=_run,
        deps=deps or [],
        hash_attrs={"input_path": input_path, "num_random_files": num_random_files, "filetype": filetype},
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
