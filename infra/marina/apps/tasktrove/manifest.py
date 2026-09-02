# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the parquet manifest the browser reads before it touches Hugging Face.

One footer read per source file. The page reads a footer itself when it opens
a source; this is what it shows before that: how many tasks each source holds,
how the file is cut into row groups, and how large the largest group's task
column is, which is what opening one task from that source downloads.

Usage:
    uv run --with pyarrow --with huggingface_hub manifest.py
"""

import json
import statistics
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

DATASET = "datasets/open-thoughts/TaskTrove"
OUT = Path(__file__).parent / "web" / "public" / "corpus" / "files.json"


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
