import os
import random
import time
from dataclasses import dataclass

import fsspec
import ray

from marin.utils import fsspec_glob


@dataclass
class TransferConfig:
    input_path: str
    output_path: str

    # Selectively choose the number of files to transfer. None means all files
    num_files: int | None = None
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
    fs = fsspec.filesystem("gcs")
    if not fs.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist in GCS.")

    if config.num_files is None:
        fs.copy(input_path + "/", config.output_path, recursive=True)
    else:
        random.seed(42)
        filenames = fsspec_glob(os.path.join(input_path, f"**/*.{config.filetype}"))
        random.shuffle(filenames)
        for i in range(config.num_files):
            fs.copy(filenames[i], os.path.join(config.output_path, os.path.basename(filenames[i])))

    elapsed_time_seconds: float = time.time() - start_time
    print(f"Downloaded {input_path} to {config.output_path} ({elapsed_time_seconds}s).")
