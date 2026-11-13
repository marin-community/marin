# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract a pinned, stratified TaskTrove Clean sample for runtime comparison."""

import argparse
import hashlib
import io
import json
import random
import tarfile
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

DATASET_REVISION = "9065fa568394f286dab0081e43dc76fc87c48984"
PARQUET_URL = (
    "https://huggingface.co/datasets/open-athena/task-trove/resolve/" f"{DATASET_REVISION}/data/part-00000.parquet"
)
ROW_GROUP = 0
SEED = 20260924
COHORTS = (
    ("math", "7ebf98ecd2fd"),
    ("mcq", "7aee1955b736"),
    ("ifeval", "7aee1955b736"),
    ("json-schema", "49879a65e274"),
    ("pytest", "6bbc92a19d6e"),
)
TASKS_PER_COHORT = 20


def extract_archive(data: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(destination, filter="data")


def sample(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    with fsspec.open(PARQUET_URL, "rb", block_size=8 * 1024 * 1024) as source:
        parquet = pq.ParquetFile(source)
        rows = parquet.read_row_group(ROW_GROUP).to_pylist()

    selected = []
    for mode, dockerfile_id in COHORTS:
        candidates = [
            (index, row)
            for index, row in enumerate(rows)
            if row["mode"] == mode and row["dockerfile_id"] == dockerfile_id
        ]
        if len(candidates) < TASKS_PER_COHORT:
            raise ValueError(f"Too few {mode}/{dockerfile_id} candidates: {len(candidates)}")
        selected.extend(rng.sample(candidates, TASKS_PER_COHORT))

    manifest = []
    for index, row in selected:
        task = output / "tasks" / f"{index:05d}"
        extract_archive(row["task_binary"], task)
        if row["solution_binary"] is not None:
            extract_archive(row["solution_binary"], task)
        manifest.append(
            {
                "row_group": ROW_GROUP,
                "row_index": index,
                "path": row["path"],
                "source": row["source"],
                "mode": row["mode"],
                "dockerfile_id": row["dockerfile_id"],
                "has_solution": row["has_solution"],
                "task_binary_sha256": hashlib.sha256(row["task_binary"]).hexdigest(),
            }
        )
    (output / "sample.json").write_text(
        json.dumps({"revision": DATASET_REVISION, "seed": SEED, "tasks": manifest}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    sample(parser.parse_args().output)
