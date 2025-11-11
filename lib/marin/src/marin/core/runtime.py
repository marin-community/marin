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
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

import fsspec
import ray
from marin.utils import fsspec_exists
from ray import ObjectRef

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


def workflow_cached(success_suffix="SUCCESS", verbose=True):
    """
    Decorator to make a workflow function idempotent by checking for a SUCCESS file
    at config.output_path before execution. Functions must return a dict with success info.

    Args:
        success_suffix: The suffix of the success file.
        verbose: If true, print logs for each function invocation.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(config, *args, **kwargs):
            success_file = f"{config.output_path.rstrip('/')}.{success_suffix}"

            # If the success file exists, skip execution
            if fsspec_exists(success_file):
                if verbose:
                    logger.info(f"Output already exists at {config.output_path}. Skipping {func.__name__}")
                return {"success": True, "reason": "already_exists", "skipped": True}

            datetime_start = datetime.now(timezone.utc)
            if verbose:
                logger.info(f"Running {func.__name__} with output to {config.output_path}")

            # Execute the main function
            response = func(config, *args, **kwargs)

            datetime_end = datetime.now(timezone.utc)

            # Write the success file with merged metadata
            with fsspec.open(success_file, "w") as f:
                metadata = {
                    "output_path": config.output_path,
                    "datetime_start": str(datetime_start),
                    "datetime_end": str(datetime_end),
                    "function_name": func.__name__,
                    **response,  # Merge all the function's return data
                }
                f.write(json.dumps(metadata, indent=2))

            if verbose:
                logger.info(f"Completed {func.__name__}")
            return response

        return wrapper

    return decorator


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for controlling tasks run with Ray."""

    max_in_flight: int | None = 1000
    task_options: dict | None = None


def simple_backpressure(
    remote_func, task_generator: Iterator, max_in_flight: int | None, fetch_local: bool, *args, **kwargs
) -> Iterator[ObjectRef]:
    """
    Simple backpressure implementation for ray.remote functions.

    This function will return a list of refs *in order* of the tasks that are being executed.
    (The usual ray.wait returns the refs in the order of completion, or at least when they're
    determined to be completed.)

    Parameters:
    - remote_func: The Ray remote function to execute.
    - task_generator: An iterator that generates the tasks to be executed.
    - max_in_flight: The maximum number of tasks to run concurrently.
    - fetch_local: Whether to fetch the results locally before returning.

    Returns:
    - An iterator of refs in the order of the tasks that are being executed.
    """
    refs = []
    in_flight = []

    for task in task_generator:
        if max_in_flight is not None:
            while len(in_flight) >= max_in_flight:
                _, in_flight = ray.wait(in_flight, fetch_local=fetch_local, num_returns=1)

        ref = remote_func.remote(*task, *args, **kwargs)
        refs.append(ref)

        if max_in_flight is not None:
            in_flight.append(ref)

    yield from refs
