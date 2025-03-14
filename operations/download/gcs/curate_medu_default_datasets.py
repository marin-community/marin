"""This operation curates the default datasets for the MEDU pipeline.

The overarching goal of this file is to selectively sample some tokens from the DCLM datasets
that will later be used to train the encoder models as well as serve as the default pretraining
corpus for training.

We need to perform this sampling because the DCLM datasets are too large and we only need roughly
250,000 examples to train the encoder models and roughly 500B tokens to filter from to yield us around
50B tokens for pretraining.
"""

import os
import time
from dataclasses import dataclass

import fsspec
import ray


@dataclass
class TransferConfig:
    input_path: str
    output_path: str

    # Selectively choose the number of files to transfer. None means all files
    num_files: int | None = None


@ray.remote
def transfer_files(config: TransferConfig) -> None:
    """
    Transfers the files from the input path to the output path.
    """
    assert config.gcs_path.endswith(
        "/"
    ), "GCS path must not end with a slash. If this is a directory, please pass in without the trailing slash"

    print(f"Downloading {config.input_path} from GCS.")
    start_time: float = time.time()
    fs = fsspec.filesystem("gcs")
    if not fs.exists(config.input_path):
        raise FileNotFoundError(f"{config.input_path} does not exist in GCS.")

    if config.num_files is None:
        fs.copy(config.input_path + "/", config.output_path, recursive=True)
    else:
        for i in range(config.num_files):
            # shard_00000000_processed.jsonl.zst
            filename = f"shard_{i:08d}_processed.jsonl.zst"
            fs.copy(os.path.join(config.input_path, filename), os.path.join(config.output_path, filename))

    elapsed_time_seconds: float = time.time() - start_time
    print(f"Downloaded {config.input_path} to {config.output_path} ({elapsed_time_seconds}s).")
