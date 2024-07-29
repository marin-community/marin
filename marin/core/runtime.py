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

from marin.utils import fsspec_exists, fsspec_glob, fsspec_mkdirs, rebase_file_path

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
    input_dir: os.PathLike,
    pattern: str,
    output_dir: os.PathLike,
    task_config: TaskConfig = TaskConfig(),  # noqa
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

    Returns:
        List: A list of outputs from the function.
    """
    # Get a list of all files in the input directory
    files = fsspec_glob(os.path.join(input_dir, pattern))

    def func_to_call(input_file):
        # Construct the output file path
        output_file = rebase_file_path(input_dir, input_file, output_dir)
        dir_name = os.path.dirname(output_file)
        fsspec_mkdirs(dir_name)
        return func(input_file, output_file, *args, **kwargs)

    if isinstance(func, ray.remote_function.RemoteFunction):
        # If the function is a ray.remote function, then execute it in parallel
        responses = simple_backpressure(func_to_call, iter(files), task_config.max_in_flight, fetch_local=True)
        return responses
    else:
        # Map the function to all files
        outputs = []
        for file in files:
            outputs.append(func_to_call(file))

    return outputs


def map_dolma_documents(func, input_dir, output_dir, task_config: TaskConfig = TaskConfig(), *args, **kwargs):
    """
    Convenience wrapper around map_files_in_directory for processing directories that are already in
    [Dolma format](https://github.com/allenai/dolma/blob/main/docs/data-format.md) (or similar).

    The Dolma format looks like this:

    ```
    |-- dataset-name/
        |-- documents/
            |-- 2019-09/
                |-- 0933_uk_all.jsonl.gz        (1GB)
                |-- 0933_vi_all.jsonl.gz        (1GB)
                |-- 0106_uk_all.jsonl.gz        (1GB)
                |-- 0106_vi_all.jsonl.gz        (1GB)
            |-- 2019-08/
                |-- ...
        |-- attributes/
            |-- toxicity-0/
                |-- 2019-09/
                    |-- 0933_uk_all.jsonl.gz    (..MB)
                    |-- 0933_vi_all.jsonl.gz    (..MB)
                    |-- 0106_uk_all.jsonl.gz    (..MB)
                    |-- 0106_vi_all.jsonl.gz    (..MB)
                |-- 2019-08/
                    |-- ...
            |-- paragraph_duplicates/
                |-- ...
    ```


    Each directory (e.g. dataset-name/documents/) contains a list of jsonl files, each of which is a Dolma document.
    This function will process each shard in the directory in parallel, producing a corresponding output file in the
    output directory. It will call the provided function with each document in the corpus. It should either
    return a new/modified document or an attributes file. It should in general not "split" or "skip" documents.

    Dolma documents have this structure:

    ```
        {
        "id": "...",             # MANDATORY: source-specific identifier
        "text": "foo",           # MANDATORY: textual content of the document
        "source": "...",         # MANDATORY: source of the data, such as peS2o, common-crawl, etc.
        "added": "...",          # OPTIONAL: timestamp ai2 acquired this data
        "created": "..."         # OPTIONAL: timestamp when orig document was created (best-guess if not available)
        "metadata": {...}        # OPTIONAL: source-specific metadata
        }
    ```

    Quality classification decisions/annotations have this structure:

    ```
    {
    "source": "...",
    "id": "...",
        "attributes": {
           "toxicity": 0.7  # this should be a unique label per classifier.
       }
    }
    ```

    The content of an attribute can be arbitrary json. Ideally it should be a score, a label, or spans labels:
    ```
    {
    "source": "...",
    "id": "...",
    attributes: {
        "olmo_mix_v1_taggers__ft_lang_id_en_paragraph_with_doc_score_v2__en": [
            [0, 300, 0.9],         # this means text[0:300] is tagged with score 0.9
            [300, 540, 0.3],       # this means text[300:540] is tagged with score 0.3
            ...
        ],
        ...
        }
    }
    ```

    Span labels should be w.r.t. python string offsets.

    Example:
    ```
    def simple_quality_classifier(document):
        return {
            "source": document["source"],
            "id": document["id"],
            "attributes": {
                "toxicity": 0.7
            }
        }

    map_dolma_documents(simple_quality_classifier, "gs://my-bucket/dataset/documents", "gs://my-bucket/dataset/attributes/toxicity/0")
    ```
    """

    # TODO: use Ray Data to autoscale this nicer

    raise NotImplementedError("This function is not yet implemented")

    # def handle_single_file(input_file, output_file):
    #     with fsspec.open(input_file, "rb", compression="infer") as f, \
    #             fsspec.open(output_file, "wb", compression="gzip") as output:
    #         if isinstance(func, ray.remote_function.RemoteFunction):
    #             for line in f:
    #                 data = json.loads(line)
    #                 new_data = func.remote(data, *args, **kwargs)
    #                 output.write(json.dumps(new_data) + "\n")
    #         else:
    #             for line in f:
    #                 data = json.loads(line)
    #                 if isinstance(func, ray.remote_function.RemoteFunction):
    #                     new_data = func.remote(data, *args, **kwargs)
    #                 else:
    #                     new_data = func(data, *args, **kwargs)
    #                 output.write(json.dumps(new_data) + "\n")


def simple_backpressure(remote_func, task_generator: Iterator, max_in_flight: Optional[int], fetch_local: bool) -> Iterator[ObjectRef]:
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
                num_to_await = len(in_flight) - max_in_flight + 1
                done, in_flight = ray.wait(in_flight, fetch_local=fetch_local, num_returns=num_to_await)

        ref = remote_func.remote(*task)
        refs.append(ref)

        if max_in_flight is not None:
            in_flight.append(ref)

    yield from refs


