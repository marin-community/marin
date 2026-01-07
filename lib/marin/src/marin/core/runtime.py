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

import functools
import json
import logging
from dataclasses import dataclass
from datetime import datetime

import fsspec
from marin.utils import fsspec_exists

logger = logging.getLogger("ray")


@dataclass
class RayConfig:
    address: str | None = None


def cached_or_construct_output(success_suffix="success", verbose=True):
    """
    Decorator to make a function idempotent. This decorator will check if the success file exists, if it does then it
    will skip the function. If the success file does not exist, then it will execute the function and write
    the success file.

    Args:
        success_suffix: The suffix of the success file.
                        The path for the success file will be output_file_path + "." + success_suffix
        verbose: If true, print logs for each function invocation.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(input_file_path, output_file_path, *args, **kwargs):
            # output_file is file for md output and success_file is the Ledger file to
            success_file = output_file_path + f".{success_suffix}"

            # If the ledger file exists, then we do not process the file again
            if fsspec_exists(success_file):
                if verbose:
                    logger.info(f"Output file already processed. Skipping {input_file_path}")
                return True

            datetime_start = datetime.utcnow()
            # Execute the main function
            if verbose:
                logger.info(f"Processing {input_file_path} to {output_file_path}")
            response = func(input_file_path, output_file_path, *args, **kwargs)
            datetime_end = datetime.utcnow()

            # Write the success file, so that we don't have to process it next time
            with fsspec.open(success_file, "w") as f:
                metadata = {
                    "input_file_path": input_file_path,
                    "output_file_path": output_file_path,
                    "datetime_start": str(datetime_start),
                    "datetime_end": str(datetime_end),
                }
                f.write(json.dumps(metadata))

            if verbose:
                logger.info(f"Processed {input_file_path}")
            return response

        return wrapper

    return decorator


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for controlling tasks run with Ray."""

    max_in_flight: int | None = 1000
    task_options: dict | None = None
