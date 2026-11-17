# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure parent checkpoint geometry using sources from a merge recipe."""

import argparse
import json
import logging
from pathlib import Path

import torch
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from marin.merging.geometry import weight_update_gram
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--chunk-elements", type=int, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    recipe = json.loads(args.recipe.read_text())
    sources = [recipe["anchor"], *recipe["donors"]]
    readers = [CheckpointReader(CheckpointSource(**source)) for source in sources]
    names = set(readers[0].weight_map)
    if any(set(reader.weight_map) != names for reader in readers[1:]):
        raise ValueError("Checkpoint tensor keys differ")
    fs, output = filesystem_for(args.output)
    if fs.exists(output) and fs.ls(output):
        raise FileExistsError(f"Output must be empty: {args.output}")
    fs.makedirs(output, exist_ok=True)
    manifest = {
        "sources": sources,
        "code_revision": args.code_revision,
        "chunk_elements": args.chunk_elements,
        "vector_order": [
            "anchor",
            *[f"donor_{i}" for i in range(len(readers) - 1)],
            *[f"donor_{i}-anchor" for i in range(len(readers) - 1)],
        ],
        "tensors": [],
    }
    ordered = sorted(names, key=lambda name: (readers[0].weight_map[name], name))
    for index, name in enumerate(ordered, 1):
        tensors = [reader.tensor(name) for reader in readers]
        gram = weight_update_gram(tensors[0], tensors[1:], chunk_elements=args.chunk_elements)
        row = {
            "name": name,
            "shape": list(tensors[0].shape),
            "dtype": str(tensors[0].dtype),
            "numel": tensors[0].numel(),
            "gram": gram.tolist(),
        }
        filename = f"tensor-{index:05d}.json"
        fs.pipe_file(prefix_join(output, filename), json.dumps(row, allow_nan=False).encode())
        manifest["tensors"].append({"name": name, "file": filename})
        del tensors, gram
        logger.info("Measured %d/%d: %s", index, len(ordered), name)
    fs.pipe_file(prefix_join(output, "manifest.json"), json.dumps(manifest, indent=2).encode())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
