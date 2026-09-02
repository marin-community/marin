# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the audit overlay the site shows beside the live dataset.

The audit sampled about eleven tasks from each of the 96 sources, classified
each one, and reviewed each source. This copies that into the two files the
page reads: one row per source, one row per sampled task.

Each sampled task also gets its row in the dataset, which is where the page
fetches it from: the source's offset from `files.json` (run `manifest.py`
first) plus its position in the source's `path` column.

Usage:
    uv run --with pyarrow --with huggingface_hub overlay.py AUDIT_DIR

where AUDIT_DIR holds `tasks_clean.json` and `sources/<source>.json`, which is
what `shellsim/eval/tasktrove_audit` is.
"""

import gzip
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

DATASET = "datasets/open-thoughts/TaskTrove"
OUT = Path(__file__).parent / "web" / "public" / "corpus"


def current_name(source: str, manifest: set[str]) -> str:
    """The manifest's name for a source the audit knew under an earlier version.

    A source is re-published under a bumped `-vN` suffix; the review carries
    over, and a sampled task carries over when its path is still in the file.
    """
    if source in manifest:
        return source
    stem = source.removesuffix(source.split("-v")[-1]) if "-v" in source else source + "-v"
    matches = sorted(name for name in manifest if name.startswith(stem))
    if len(matches) != 1:
        raise ValueError(f"{source} is not in the manifest and {matches} is not one replacement")
    return matches[-1]


def row_indices(sources: list[str]) -> dict[str, dict[str, int]]:
    """Each source's task paths, in file order, read from the `path` column alone."""
    fs = HfFileSystem()
    found = {}
    for source in sources:
        with fs.open(f"{DATASET}/{source}/tasks.parquet", "rb") as f:
            paths = pq.ParquetFile(f).read(columns=["path"]).column(0).to_pylist()
        found[source] = {path: i for i, path in enumerate(paths)}
        print(source, len(paths), flush=True)
    return found


def main(audit: Path) -> None:
    tasks = json.loads((audit / "tasks_clean.json").read_text())
    manifest = json.loads((OUT / "files.json").read_text())
    offsets = {}
    total = 0
    for entry in manifest:
        offsets[entry["source"]] = total
        total += entry["rows"]
    names = set(offsets)
    for task in tasks:
        task["path"] = task["id"].removeprefix(task["source"] + "__")
        task["source"] = current_name(task["source"], names)
    indices = row_indices(sorted({task["source"] for task in tasks}))
    kept = []
    for task in tasks:
        index = indices[task["source"]].get(task["path"])
        if index is None:
            print(f"dropped {task['id']}: not in {task['source']}", flush=True)
            continue
        task["row"] = offsets[task["source"]] + index
        kept.append(task)
    tasks = kept
    sources = [json.loads(path.read_text()) for path in sorted((audit / "sources").glob("*.json"))]
    for review in sources:
        review["source"] = current_name(review["source"], names)
    OUT.mkdir(parents=True, exist_ok=True)
    # Gzipped: the labels run to nearly a megabyte and the kernel serves `x.json` from
    # `x.json.gz` with a Content-Encoding header.
    with gzip.open(OUT / "labels.json.gz", "wt", compresslevel=9) as f:
        json.dump(tasks, f)
    (OUT / "sources.json").write_text(json.dumps(sources))
    print(f"{len(tasks)} labels, {len(sources)} source reviews -> {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
