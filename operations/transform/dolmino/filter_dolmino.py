import dataclasses
import os

import draccus
import pandas as pd
import ray

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_glob, rebase_file_path
from operations.transform.conversation.transform_conversation import create_shard_output_directory


@dataclasses.dataclass
class FilterDolminoConfig:
    """Configuration to filter the dolmino dataset.

    Attributes:
        input_path: The path to the input dolmino dataset.
        output_path: The path to the output of the filtered dolmino dataset.
        split: The split of the dolmino dataset to filter (e.g. "wiki", "stackexchange", "pes2o").
        min_length: The minimum length for each document to filter by.
    """

    input_path: str
    output_path: str
    split: str
    min_length: int | None = None


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def _process_file(input_file_path: str, output_dir: str, config: FilterDolminoConfig):
    """Filters the dolmino dataset.

    Input format is dolma formatted jsonl.gz (lines). Output format is dolma formatted jsonl.gz (lines).
    Processes the file in chunks and writes multiple sharded output files.
    """
    shard_size = 10000
    shard_count = 0

    # Process file in chunks
    for chunk in pd.read_json(input_file_path, lines=True, compression="gzip", chunksize=shard_size):
        if config.min_length is not None:
            chunk["length"] = chunk["metadata"].apply(lambda x: x.get("length", 0))
            chunk = chunk[chunk["length"] >= config.min_length]

        if not chunk.empty:
            shard_path = os.path.join(output_dir, f"000_{shard_count:05d}.jsonl.gz")
            chunk.to_json(shard_path, orient="records", lines=True, compression="gzip")
            shard_count += 1


@ray.remote
def _process_dataset(config: FilterDolminoConfig):
    file_paths = fsspec_glob(os.path.join(config.input_path, f"data/{config.split}/**/*.json.gz"))
    max_task_in_flight = 1000
    responses = []
    for input_filepath in file_paths:
        if len(responses) > max_task_in_flight:
            ready_refs, responses = ray.wait(responses, num_returns=1)
            ray.get(ready_refs)

        output_filepath = rebase_file_path(config.input_path, input_filepath, config.output_path)
        output_dir = create_shard_output_directory(output_filepath)

        # To bypass the current error where we still allow running code on the head node which should not be allowed.
        result_ref = _process_file.options(memory=10 * 1024 * 1024 * 1024, num_cpus=2).remote(
            input_filepath, output_dir, config
        )

        responses.append(result_ref)

    ray.get(responses)


@draccus.wrap()
def filter_dolmino(config: FilterDolminoConfig):
    ray.get(_process_dataset.remote(config))


if __name__ == "__main__":
    filter_dolmino()
