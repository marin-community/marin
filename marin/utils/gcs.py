"""
gcs.py

Generic utilities for interacting with GCS buckets (e.g., `gs://`) including reading files, listing directories, etc.
"""
import fsspec


def read_gcs_file(gcs_filepath: str) -> str:
    with fsspec.open(gcs_filepath, "r") as f:
        return f.read()
