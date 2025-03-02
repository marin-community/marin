import time
from dataclasses import dataclass

import fsspec
import ray


@dataclass
class DownloadFromGCSConfig:
    gcs_path: str
    destination_path: str


@ray.remote
def download_model_from_gcs(config: DownloadFromGCSConfig) -> None:
    """
    Downloads the folder at `gcs_path` to `destination_path`.
    """
    if config.gcs_path.endswith("/"):
        config.gcs_path = config.gcs_path[:-1]

    print(f"Downloading {config.gcs_path} from GCS.")
    start_time: float = time.time()
    fs = fsspec.filesystem("gcs")
    if not fs.exists(config.gcs_path):
        raise FileNotFoundError(f"{config.gcs_path} does not exist in GCS.")

    fs.copy(config.gcs_path + "/", config.destination_path, recursive=True)

    elapsed_time_seconds: float = time.time() - start_time
    print(f"Downloaded {config.gcs_path} to {config.destination_path} ({elapsed_time_seconds}s).")
