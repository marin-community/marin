import os
import random
import time
from dataclasses import dataclass

import fsspec
import ray

from marin.utils import fsspec_exists, fsspec_glob


@dataclass
class TransferConfig:
    input_path: str
    output_path: str

    # Selectively choose the number of random files to transfer. None means all files
    num_random_files: int | None = None
    filetype: str = "jsonl.zst"


@ray.remote
def transfer_files(config: TransferConfig) -> None:
    """
    Transfers the files from the input path to the output path.
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

    if config.num_random_files is None:
        fs.copy(input_path + "/", config.output_path, recursive=True)
    else:
        random.seed(42)
        filenames = fsspec_glob(os.path.join(input_path, f"**/*.{config.filetype}"))
        random.shuffle(filenames)
        for i in range(config.num_random_files):
            output_filename = os.path.join(config.output_path, os.path.basename(filenames[i]))
            if not fsspec_exists(output_filename):
                fs.copy(filenames[i], output_filename)

    elapsed_time_seconds: float = time.time() - start_time
    print(f"Downloaded {input_path} to {config.output_path} ({elapsed_time_seconds}s).")
