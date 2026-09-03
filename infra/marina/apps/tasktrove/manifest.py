# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the source manifest the browser reads before it touches Hugging Face.

One parquet footer read per source file, from which the page gets the list of
sources and how many tasks each holds. The row-group figures describe the file's
layout for anyone sizing a direct download; the page itself reads rows from
Hugging Face's datasets-server.

Usage:
    uv run --with pyarrow --with huggingface_hub manifest.py
"""

import json
import statistics
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

DATASET = "datasets/open-thoughts/TaskTrove"
# The app's data directory: uploaded to the Marina bucket, served at /tasktrove/data/.
DATA_DIR = Path(__file__).parents[2] / ".data" / "tasktrove"
OUT = DATA_DIR / "files.json"


def main() -> None:
    fs = HfFileSystem()
    entries = [info for info in fs.ls(DATASET, detail=True) if isinstance(info, dict)]
    files = sorted(
        info["name"] for info in entries if info["type"] == "directory" and not info["name"].endswith("/deprecated")
    )
    sources = []
    for directory in files:
        path = f"{directory}/tasks.parquet"
        info = fs.info(path)
        with fs.open(path, "rb") as f:
            metadata = pq.ParquetFile(f).metadata
        groups = [metadata.row_group(i) for i in range(metadata.num_row_groups)]
        binary = [
            next(g.column(c) for c in range(g.num_columns) if g.column(c).path_in_schema == "task_binary")
            for g in groups
        ]
        name = directory.rsplit("/", 1)[1]
        sources.append(
            {
                "source": name,
                "file": path.removeprefix(DATASET + "/"),
                "size": info["size"],
                "rows": metadata.num_rows,
                "groups": len(groups),
                "group_rows": int(statistics.median(g.num_rows for g in groups)),
                "largest_group_bytes": max(c.total_compressed_size for c in binary),
            }
        )
        print(name, metadata.num_rows, len(groups), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sources))


if __name__ == "__main__":
    main()
