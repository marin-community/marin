# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
import datetime
import json
import logging
import os
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd
from rigging.filesystem import open_url, url_to_fs

logger = logging.getLogger(__name__)


class Document(TypedDict):
    id: str
    source: str
    text: str


class Attribute(TypedDict):
    id: str
    source: str
    attributes: dict


@dataclass
class DatasetConfig:
    """Configuration for curating a dataset for training a quality classifier

    Attributes:
        input_doc_path (str): Path to the input dataset directory (Dolma format).
        label (str): Label for the dataset. This should be in the format "<label>"
            where <label> is the label for the dataset. For example, "hq" or "lq", respectively.
        sampling_rate (Optional[float]): Subsampling fraction to construct the dataset.
        max_sample_size (Optional[int]): Maximum number of examples to include in the dataset.
    """

    input_doc_path: str
    label: str
    sampling_rate: float = 1.0
    max_sample_size: int | None = None


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
    """Read in a dataset as a streaming iterator.

    Uses fsspec + json directly instead of HuggingFace datasets to avoid
    the datasets CompressionFilesystem injecting aiohttp kwargs
    (requote_redirect_url) into botocore's create_client, which breaks
    on S3-compatible backends.

    Args:
        input_filename: Path to the input file (.jsonl.gz, .jsonl.zst, .jsonl, or .parquet).

    Returns:
        Iterator over dataset rows as dicts.
    """
    if input_filename.endswith((".jsonl.gz", ".jsonl.zst", ".jsonl")):
        with open_url(input_filename, "rb", compression="infer") as f:
            for line in f:
                row = json.loads(line)
                if columns:
                    row = {k: row[k] for k in columns if k in row}
                yield row
    elif input_filename.endswith(".parquet"):
        import pyarrow.parquet as pq

        with open_url(input_filename, "rb") as f:
            table = pq.read_table(f, columns=columns)
        for row in table.to_pylist():
            yield row
    else:
        raise ValueError(f"Unsupported filetype: {input_filename}")


def write_dataset_streaming(rows_iterator, output_filename: str, append: bool = False):
    """Writes rows to a file in streaming fashion

    JSONL behavior:
      - For JSONL(.gz/.zst), write to a deterministic temp file under /tmp in append-mode,
        then upload the full file to the destination (e.g., gs://...). This provides safe
        append semantics for object stores.
      - Checkpoint restoration should continue to read from the remote path.
    """
    import hashlib
    import shutil

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
            fs, _ = url_to_fs(output_filename)
            if fs.exists(output_filename):
                with open_url(output_filename, "rb") as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        # Turn on compression inference to have fsspec auto-compress files according to the suffix
        with open_url(tmp_path, mode, compression="infer") as f:
            for row in rows_iterator:
                row = make_json_serializable(row)
                f.write((json.dumps(row) + "\n").encode("utf-8"))

        # Upload temp file to destination (overwrite remote with full content)
        with open_url(output_filename, "wb") as dst, open(tmp_path, "rb") as src:
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

            fs, _ = url_to_fs(output_filename)
            if append and fs.exists(output_filename):
                # Read existing parquet and append
                with open_url(output_filename, "rb") as f:
                    existing_table = pq.read_table(f)
                table = pa.concat_tables([existing_table, table])

            with open_url(output_filename, "wb") as f:
                pq.write_table(table, f)
        return

    raise ValueError(f"Unsupported filetype: {output_filename}")
