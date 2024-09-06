import functools
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterator, List, Optional

import fsspec
import ray
from ray import ObjectRef
from ray.remote_function import RemoteFunction

from marin.utils import fsspec_exists, fsspec_glob, fsspec_mkdirs, rebase_file_path, fsspec_get_curr_subdirectories

logger = logging.getLogger("ray")


@dataclass
class RayConfig:
    address: Optional[str] = None

    def initialize(self):
        ray.init(address=self.address)


def cached_or_construct_output(success_suffix="success"):
    """
    Decorator to make a function idempotent. This decorator will check if the success file exists, if it does then it
    will skip the function. If the success file does not exist, then it will execute the function and write
    the success file.

    Args:
        success_suffix: The suffix of the success file.
                        The path for the success file will be output_file_path + "." + success_suffix
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(input_file_path, output_file_path, *args, **kwargs):
            # output_file is file for md output and success_file is the Ledger file to
            success_file = output_file_path + f".{success_suffix}"

            # If the ledger file exists, then we do not process the file again
            if fsspec_exists(success_file):
                logger.info(f"Output file already processed. Skipping {input_file_path}")
                return True

            datetime_start = datetime.utcnow()
            # Execute the main function
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

            logger.info(f"Processed {input_file_path}")
            return response

        return wrapper

    return decorator


@dataclass(frozen=True)
class TaskConfig:
    """
    Configuration for controlling tasks run with Ray
    """
    max_in_flight: Optional[int] = 1000  # Maximum number of tasks to run concurrently
    task_options: Optional[dict] = None  # Options to pass to ray.remote decorator


def map_files_in_directory(
    func: Callable | RemoteFunction,
    input_dir: os.PathLike | str,
    pattern: str,
    output_dir: os.PathLike | str,
    task_config: TaskConfig = TaskConfig(),  # noqa
    empty_glob_ok: bool = False,
    *args,
    **kwargs,
):
    """
    Map a function to all files in a directory.
    If the function is a ray.remote function, then it will be executed in parallel.

    Args:
        func: The function to map
        input_dir: The input directory
        pattern: Input file pattern to glob on
        output_dir: The output directory
        task_config: TaskConfig object

        empty_glob_ok: If True, then an empty glob will not raise an error.

    Returns:
        List: A list of outputs from the function.
    """
    # Get a list of all files in the input directory
    files = fsspec_glob(os.path.join(input_dir, pattern))

    file_pairs = []
    for file in files:
        output_file = rebase_file_path(input_dir, file, output_dir)
        dir_name = os.path.dirname(output_file)
        fsspec_mkdirs(dir_name)
        file_pairs.append([file, output_file])

    if len(file_pairs) == 0:
        logger.error(f"No files found in {input_dir} with pattern {pattern}!!! This is likely an error.")
        if not empty_glob_ok:
            raise FileNotFoundError(f"No files found in {input_dir} with pattern {pattern}")

    if isinstance(func, ray.remote_function.RemoteFunction):
        # If the function is a ray.remote function, then execute it in parallel
        responses = simple_backpressure(func, iter(file_pairs), task_config.max_in_flight, fetch_local=True,
                                        *args, **kwargs)
        return responses
    else:
        # Map the function to all files
        outputs = []
        for file in file_pairs:
            outputs.append(func(*file, *args, **kwargs))

    return outputs

def map_directories_in_directory(
    func: Callable | RemoteFunction,
    input_dir: str,
    output_dir: str,
    task_config: TaskConfig = TaskConfig(),  # noqa
    *args,
    **kwargs,
):
    # Gets all the directories in a directory
    directories = fsspec_get_curr_subdirectories(input_dir)

    if len(directories) == 0:
        return []

    def func_to_call(input_subdir):
        # Construct the output directory
        output_subdir = rebase_file_path(input_dir, input_subdir, output_dir)
        fsspec_mkdirs(output_subdir)
        return func(input_subdir, output_subdir, *args, **kwargs)

    if isinstance(func, ray.remote_function.RemoteFunction):
        # If the function is a ray.remote function, then execute it in parallel
        responses = simple_backpressure(func_to_call, iter(directories), task_config.max_in_flight, fetch_local=True)
        return responses
    else:
        # Map the function to all files
        outputs = []
        for directory in directories:
            outputs.append(func_to_call(directory))

    return outputs


def simple_backpressure(remote_func, task_generator: Iterator, max_in_flight: Optional[int], fetch_local: bool,
                        *args, **kwargs) -> Iterator[ObjectRef]:
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
                done, in_flight = ray.wait(in_flight, fetch_local=fetch_local, num_returns=1)

        ref = remote_func.remote(*task, *args, **kwargs)
        refs.append(ref)

        if max_in_flight is not None:
            in_flight.append(ref)

    yield from refs


