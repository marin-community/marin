# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge sharded Hugging Face checkpoints without staging full models on disk."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import safetensors.torch
import torch
from rigging.filesystem.buckets import filesystem_for

from marin.merging.arithmetic import MergeParameters, merge_tensor

logger = logging.getLogger(__name__)
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "merge-manifest.json"
METADATA_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class CheckpointSource:
    path: str
    revision: str


class CheckpointReader:
    """Keep only the most recently requested shard of one checkpoint in memory."""

    def __init__(self, source: CheckpointSource):
        self.source = source
        self.fs, self.path = filesystem_for(source.path)
        self.weight_map = json.loads(self.fs.cat_file(f"{self.path}/{INDEX_NAME}"))["weight_map"]
        self.shard_name = ""
        self.tensors: dict[str, torch.Tensor] = {}

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard != self.shard_name:
            self.tensors.clear()
            self.tensors = safetensors.torch.load(self.fs.cat_file(f"{self.path}/{shard}"))
            self.shard_name = shard
        return self.tensors[name]


def merge_checkpoint(
    anchor: CheckpointSource,
    donors: list[CheckpointSource],
    output: str,
    parameters: MergeParameters,
    *,
    code_revision: str,
    preserve_rows: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Write an immutable merged checkpoint, publishing its manifest last.

    Inputs must already be aligned and their tokenizer/config semantics reviewed.
    Each output tensor is one safetensors shard, bounding output serialization
    memory independently of input shard layout. A partial failure is retained
    for audit and must not be served without the completion manifest.
    """
    readers = [CheckpointReader(source) for source in [anchor, *donors]]
    names = set(readers[0].weight_map)
    if any(set(reader.weight_map) != names for reader in readers[1:]):
        raise ValueError("Checkpoint tensor keys differ")
    if not set(preserve_rows) <= names:
        raise ValueError("Preserved rows refer to missing tensors")
    if len(donors) != len(parameters.coefficients):
        raise ValueError("Each donor needs one coefficient")
    fs, output_path = filesystem_for(output)
    if fs.exists(output_path) and fs.ls(output_path):
        raise FileExistsError(f"Output must be empty: {output}")
    fs.makedirs(output_path, exist_ok=True)
    manifest: dict[str, Any] = {
        "anchor": asdict(anchor),
        "donors": [asdict(source) for source in donors],
        "parameters": asdict(parameters),
        "code_revision": code_revision,
        "preserve_rows": preserve_rows,
        "objects": [],
    }

    def write(name: str, data: bytes) -> None:
        fs.pipe_file(f"{output_path}/{name}", data)
        manifest["objects"].append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    # Establish provenance before expensive work; the manifest remains the final gate.
    write("merge-recipe.json", json.dumps(manifest, indent=2).encode())
    weight_map = {}
    total_size = 0
    ordered = sorted(names, key=lambda name: (readers[0].weight_map[name], name))
    for index, name in enumerate(ordered, 1):
        merged = merge_tensor(
            readers[0].tensor(name),
            [reader.tensor(name) for reader in readers[1:]],
            parameters,
            tensor_name=name,
        )
        if name in preserve_rows:
            rows = list(preserve_rows[name])
            merged[rows] = readers[0].tensor(name)[rows]
        shard = f"model-{index:05d}-of-{len(ordered):05d}.safetensors"
        write(shard, safetensors.torch.save({name: merged}, metadata={"format": "pt"}))
        weight_map[name] = shard
        total_size += merged.numel() * merged.element_size()
        del merged
        logger.info("Merged %d/%d: %s", index, len(ordered), name)
    for name in METADATA_NAMES:
        path = f"{readers[0].path}/{name}"
        if readers[0].fs.exists(path):
            write(name, readers[0].fs.cat_file(path))
    write(INDEX_NAME, json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}).encode())
    manifest["tensor_count"] = len(weight_map)
    manifest["total_size"] = total_size
    fs.pipe_file(f"{output_path}/{MANIFEST_NAME}", json.dumps(manifest, indent=2).encode())
    return manifest
