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

"""
NOTE(chris): Remove the usage of read_dataset and write_dataset.
Right now, this code utilizes read_dataset and write_dataset but this is
probably not needed because we could theoretically stream the data in as well.
The reason why we need it currently is for two reasons:
1. We have attributes in one file and documents in another file. However, we are not
guaranteed that they are read in the same order. So, our solution is to just
read in all of the attributes and then map the id of the document to the id of the attribute.
We can then filter based on the value. However, this means we cannot simply stream the data in
since we have to store the mapping of id -> attributes.
2. We use some of the builtin Huggingface Dataset .map and .filter functions which may not work with
the streaming data paradigm (it might but not sure).
"""
import pandas as pd
import datasets
import datetime
import numpy as np


# TODO(chris): Consolidate this with other make json serializable functions
def make_json_serializable(row: dict) -> dict:
    """Make a row JSON serializable"""
    for key, value in row.items():
        if isinstance(value, dict):
            row[key] = make_json_serializable(value)
        if isinstance(value, datetime.datetime):
            row[key] = value.isoformat()
        if isinstance(value, np.ndarray):
            row[key] = value.tolist()
        if isinstance(value, np.float32 | np.float64):
            row[key] = float(value)
    return row


def read_dataset_streaming(input_filename: str, columns: list[str] | None = None):
    """Read in a dataset as a streaming iterator using datasets library

    Args:
        input_filename: str
            The path to the input file. Currently supports .jsonl.gz, .jsonl.zst, and .parquet

    Returns:
        Iterator: An iterator over the dataset rows
    """
    import datasets

    # Disable caching for streaming
    datasets.disable_caching()
    datasets.logging.set_verbosity_warning()

    # Determine file type and load with streaming
    if input_filename.endswith((".jsonl.gz", ".jsonl.zst", ".jsonl")):
        # Load as JSON lines with streaming
        dataset = datasets.load_dataset("json", data_files=input_filename, streaming=True, split="train")
    elif input_filename.endswith(".parquet"):
        # Load parquet with streaming
        dataset = datasets.load_dataset("parquet", data_files=input_filename, streaming=True, split="train")
    else:
        raise ValueError(f"Unsupported filetype: {input_filename}")

    # Filter columns if specified
    if columns:
        dataset = dataset.select_columns(columns)

    # Yield rows from the streaming dataset
    yield from dataset


def write_dataset_streaming(rows_iterator, output_filename: str, append: bool = False):
    """Writes rows to a file in streaming fashion

    JSONL behavior:
      - For JSONL(.gz/.zst), write to a deterministic temp file under /tmp in append-mode,
        then upload the full file to the destination (e.g., gs://...). This provides safe
        append semantics for object stores.
      - Checkpoint restoration should continue to read from the remote path.
    """
    import json
    import hashlib
    import os
    import shutil

    import fsspec

    mode = "ab" if append else "wb"

    if ".jsonl" in output_filename:
        # Build deterministic temp path for this output file
        file_hash = hashlib.sha256(output_filename.encode("utf-8")).hexdigest()
        if output_filename.endswith(".jsonl.gz"):
            tmp_path = f"/tmp/marin_{file_hash}.jsonl.gz"
        elif output_filename.endswith(".jsonl.zst"):
            tmp_path = f"/tmp/marin_{file_hash}.jsonl.zst"
        else:
            tmp_path = f"/tmp/marin_{file_hash}.jsonl"

        # If appending and local temp doesn't exist, hydrate it from remote (if present)
        if append and not os.path.exists(tmp_path):
            fs, _ = fsspec.core.url_to_fs(output_filename)
            if fs.exists(output_filename):
                with fsspec.open(output_filename, "rb") as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        # Turn on compression inference to have fsspec auto-compress files according to the suffix
        with fsspec.open(tmp_path, mode, compression="infer") as f:
            for row in rows_iterator:
                row = make_json_serializable(row)
                f.write((json.dumps(row) + "\n").encode("utf-8"))

        # Upload temp file to destination (overwrite remote with full content)
        with fsspec.open(output_filename, "wb") as dst, open(tmp_path, "rb") as src:
            shutil.copyfileobj(src, dst)
        return
    if output_filename.endswith(".parquet"):
        # For parquet, we need to collect rows and write in batches
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = list(rows_iterator)
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)

            fs, _ = fsspec.core.url_to_fs(output_filename)
            if append and fs.exists(output_filename):
                # Read existing parquet and append
                with fsspec.open(output_filename, "rb") as f:
                    existing_table = pq.read_table(f)
                table = pa.concat_tables([existing_table, table])

            with fsspec.open(output_filename, "wb") as f:
                pq.write_table(table, f)
        return

    raise ValueError(f"Unsupported filetype: {output_filename}")


def read_dataset(input_filename: str, columns: list[str] | None = None):
    """Read in a data source and return as a Huggingface Dataset

    Args:
        input_filename: str
            The path to the input file. Currently supports .jsonl.gz and .parquet

    Returns:
        datasets.Dataset: A Huggingface Dataset in-memory without using the disk
    """
    datasets.disable_caching()  # Disabling caching or else it spills to disk to cache
    datasets.logging.set_verbosity_warning()
    # We use pandas to read in the file so that we don't have to materialize
    # the entire dataset in disk since we have limited disk space.
    # Huggingface datasets loads the dataset into disk first and mmaps.
    if input_filename.endswith(".jsonl.gz"):
        df = pd.read_json(input_filename, compression="gzip", lines=True)
    elif input_filename.endswith(".jsonl.zst"):
        df = pd.read_json(input_filename, compression="zstd", lines=True)
    elif input_filename.endswith(".parquet"):
        df = pd.read_parquet(input_filename, columns=columns)
    else:
        raise ValueError(f"Unsupported filetype: {input_filename}")

    return datasets.Dataset.from_pandas(df)


def write_dataset(dataset, output_filename: str):
    """Writes a Huggingface Dataset to a file (remote or local)"""
    if output_filename.endswith(".jsonl.gz"):
        dataset.to_json(output_filename, compression="gzip")
    elif output_filename.endswith(".jsonl.zst"):
        df_pandas = dataset.to_pandas()
        df_pandas.to_json(output_filename, orient="records", compression="zstd", lines=True)
        # dataset.to_json(output_filename, to_json_kwargs={"compression": "zstd", "lines": True})
    elif output_filename.endswith(".parquet"):
        dataset.to_parquet(output_filename)
    else:
        raise ValueError(f"Unsupported filetype: {output_filename}")
