# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import time
from dataclasses import dataclass

import fsspec
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
    fs, _ = fsspec.core.url_to_fs(input_path)
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
    with ZephyrContext(name="fs-transfer") as ctx:
        ctx.execute(pipeline)

    elapsed_time_seconds: float = time.time() - start_time
    print(f"Downloaded {input_path} to {config.output_path} ({elapsed_time_seconds}s).")
