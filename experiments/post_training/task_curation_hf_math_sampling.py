# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Acquire bounded pinned training math prefixes and record their sampling provenance."""

import argparse
import codecs
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests
from taskcompendium.pipeline.datasets import deepscaler, hardmath, hendrycks_math

from experiments.post_training.task_curation_partitions import assign_partitions
from experiments.post_training.task_curation_prefix_sampling import BudgetedHfFileSystem, TransferBudget

SHARDS = {
    "hardmath": "data/train-00000-of-00001.parquet",
    "hendrycks_math": "algebra/train-00000-of-00001.parquet",
    "deepscaler": "deepscaler.json",
}
RECIPES = {"hardmath": hardmath.recipe, "hendrycks_math": hendrycks_math.recipe, "deepscaler": deepscaler.recipe}
BLOCK_BYTES = 65536


def json_array_prefix(url: str, count: int, budget: TransferBudget) -> list[dict[str, Any]]:
    """Stop reading a JSON array once the requested records are decoded."""
    rows = []
    buffer = ""
    position = None
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        for chunk in response.iter_content(BLOCK_BYTES):
            budget.transferred += len(chunk)
            if budget.transferred > budget.maximum:
                raise ValueError("Transfer budget exceeded")
            buffer += utf8.decode(chunk)
            if position is None:
                if not buffer.strip():
                    continue
                if not buffer.lstrip().startswith("["):
                    raise ValueError("Expected a top-level JSON array")
                position = buffer.index("[") + 1
            while len(rows) < count:
                while position < len(buffer) and buffer[position] in " \r\n\t,":
                    position += 1
                try:
                    row, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    # A record may span multiple transfer blocks.
                    break
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object record")
                rows.append(row)
                position = end
            if len(rows) == count:
                return rows
    raise ValueError(f"Source ended after {len(rows)} records; requested {count}")


def sample_source(
    name: str, output: Path, count: int, seed: int, *, max_transfer_bytes: int = 16 * 1024 * 1024
) -> dict[str, Any]:
    """Persist a reproducible train prefix without downloading the full large dataset."""
    directory = output / name
    snapshot = directory / "sample.jsonl"
    recipe = RECIPES[name](snapshot)
    source = recipe.source
    shard = SHARDS[name]
    budget = TransferBudget(max_transfer_bytes)
    filesystem = BudgetedHfFileSystem(budget=budget)
    if shard.endswith(".parquet"):
        path = f"datasets/{source.dataset}@{source.revision}/{shard}"
        with filesystem.open(path, cache_type="none") as stream:
            parquet = pq.ParquetFile(stream)
            first_group = parquet.read_row_group(0)
            rows = first_group.slice(0, count).to_pylist()
            source_rows = parquet.metadata.num_rows
            schema = str(parquet.schema_arrow)
    else:
        url = f"https://huggingface.co/datasets/{source.dataset}/resolve/{source.revision}/{shard}"
        rows = json_array_prefix(url, count, budget)
        source_rows = None
        schema = "problem,solution,answer JSON strings"
    if len(rows) != count:
        raise ValueError(f"Read {len(rows)} records; requested {count}")
    for index, row in enumerate(rows):
        problem = row.get("problem", row.get("question"))
        row.update(
            sample_index=index,
            path=f"{source.config}/{source.split}/{index}",
            sample_group=hashlib.sha256(str(problem).encode()).hexdigest(),
        )
    assign_partitions(rows, random.Random(f"{seed}:{name}"), "sample_index")
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(serialized)
    manifest = {
        "name": name,
        "dataset": source.dataset,
        "revision": source.revision,
        "config": source.config,
        "split": source.split,
        "shard": shard,
        "source_rows": source_rows,
        "sample_rows": len(rows),
        "seed": seed,
        "sampling": (
            "First N pinned train records; seed assigns grouped development/holdout partitions, "
            "not population-uniform sampling"
        ),
        "snapshot_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "transfer_budget_bytes": budget.maximum,
        "transferred_bytes": budget.transferred,
        "http_ranges": budget.ranges,
        "schema": schema,
        "atlas_is_benchmark": False,
        "intended_use": "train",
        "source_solution_visibility": "private",
    }
    (directory / "sample-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=SHARDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=6501)
    parser.add_argument("--max-transfer-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    print(
        json.dumps(
            sample_source(args.name, args.output, args.count, args.seed, max_transfer_bytes=args.max_transfer_bytes),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
