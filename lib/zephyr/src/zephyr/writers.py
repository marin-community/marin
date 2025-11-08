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

"""Writers for common output formats."""

from __future__ import annotations

from collections.abc import Iterable

import fsspec
import msgspec
from tqdm import tqdm


def ensure_parent_dir(path: str) -> None:
    """Create directories for `path` if necessary."""
    import os

    # Use os.path.dirname for local paths, otherwise use fsspec
    if "://" in path:
        output_dir = path.rsplit("/", 1)[0]
        fs, dir_path = fsspec.core.url_to_fs(output_dir)
        if not fs.exists(dir_path):
            fs.mkdirs(dir_path, exist_ok=True)
    else:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


def write_jsonl_file(records: Iterable, output_path: str) -> dict:
    """Write records to a JSONL file with automatic compression.

    Compression is automatically inferred from the file extension (.gz and .zst supported).
    Useful for writing to custom output paths in map operations where you need control over
    the destination path.

    Args:
        records: Records to write (each should be JSON-serializable)
        output_path: Path to output file (supports local and cloud storage)

    Returns:
        Dict with metadata: {"path": output_path, "count": num_records}

    Example:
        def process_task(task):
            records = (transform(row) for row in load_data(task))
            return write_jsonl_file(records, f"{task['output_dir']}/data.jsonl.gz")

        pipeline = Dataset.from_list(tasks).map(process_task)
    """
    ensure_parent_dir(output_path)

    count = 0
    encoder = msgspec.json.Encoder()

    if output_path.endswith(".zst"):
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=1)
        with fsspec.open(output_path, "wb", block_size=64 * 1024 * 1024) as raw_f:
            with cctx.stream_writer(raw_f) as f:
                for record in tqdm(records, desc=f"write_json {output_path}", mininterval=10):
                    f.write(encoder.encode(record) + b"\n")
                    count += 1
    elif output_path.endswith(".gz"):
        with fsspec.open(output_path, "wb", compression="gzip", compresslevel=1, block_size=64 * 1024 * 1024) as f:
            for record in tqdm(records, desc=f"write_json {output_path}", mininterval=10):
                f.write(encoder.encode(record) + b"\n")
                count += 1
    else:
        with fsspec.open(output_path, "wb", block_size=64 * 1024 * 1024) as f:
            for record in tqdm(records, desc=f"write_json {output_path}", mininterval=10):
                f.write(encoder.encode(record) + b"\n")
                count += 1

    return {"path": output_path, "count": count}
