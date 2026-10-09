# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge sharded Hugging Face checkpoints without staging full models on disk."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from typing import Any

import safetensors.torch
import torch
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from marin.merging.arithmetic import MergeParameters, merge_tensor
from marin.merging.curvature import ota_merge_tensor
from marin.merging.learned import differentiable_weight_blend

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


@dataclass(frozen=True)
class RowMerge:
    """Donor coefficients for one first-axis slice of a tensor."""

    row: int
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class ChunkMerge:
    coefficients: dict[str, tuple[tuple[float, ...], ...]]
    block_elements: int
    calibration_artifact: str
    calibration_sha256: str


@dataclass(frozen=True)
class CurvatureMerge:
    second_moments: tuple[CheckpointSource, ...]
    density: float
    epsilon: float


class CheckpointReader:
    """Keep only the most recently requested shard of one checkpoint in memory."""

    def __init__(self, source: CheckpointSource):
        self.source = source
        self.fs, self.path = filesystem_for(source.path)
        self.weight_map = json.loads(self.fs.cat_file(prefix_join(self.path, INDEX_NAME)))["weight_map"]
        self.shard_name = ""
        self.tensors: dict[str, torch.Tensor] = {}

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard != self.shard_name:
            self.tensors.clear()
            self.tensors = safetensors.torch.load(self.fs.cat_file(prefix_join(self.path, shard)))
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
    tensor_coefficients: dict[str, tuple[float, ...]],
    row_overrides: dict[str, tuple[RowMerge, ...]],
    calibration: ChunkMerge | CurvatureMerge | None = None,
) -> dict[str, Any]:
    """Write an immutable merged checkpoint, publishing its manifest last.

    Inputs must already be aligned and their tokenizer/config semantics reviewed.
    Each output tensor is one safetensors shard, bounding output serialization
    memory independently of input shard layout. A partial failure is retained
    for audit and must not be served without the completion manifest. Row
    overrides merge first-axis slices independently and cannot overlap protected rows.
    """
    readers = [CheckpointReader(source) for source in [anchor, *donors]]
    names = set(readers[0].weight_map)
    if any(set(reader.weight_map) != names for reader in readers[1:]):
        raise ValueError("Checkpoint tensor keys differ")
    if not set(preserve_rows) <= names:
        raise ValueError("Preserved rows refer to missing tensors")
    if len(donors) != len(parameters.coefficients):
        raise ValueError("Each donor needs one coefficient")
    if not set(tensor_coefficients) <= names:
        raise ValueError("Tensor coefficients refer to missing tensors")
    if not set(row_overrides) <= names:
        raise ValueError("Row overrides refer to missing tensors")
    for name, overrides in row_overrides.items():
        rows = [override.row for override in overrides]
        if len(rows) != len(set(rows)) or any(row < 0 for row in rows):
            raise ValueError(f"Row overrides must have unique nonnegative indices: {name}")
        if set(rows) & set(preserve_rows.get(name, ())):
            raise ValueError(f"Row overrides overlap protected rows: {name}")
        if any(len(override.coefficients) != len(donors) for override in overrides):
            raise ValueError(f"Each row override needs one coefficient per donor: {name}")
    moment_readers = []
    if calibration is not None and (tensor_coefficients or row_overrides):
        raise ValueError("Calibrated merges define their own tensor coefficients")
    if isinstance(calibration, ChunkMerge) and not set(calibration.coefficients) <= names:
        raise ValueError("Chunk coefficients refer to missing tensors")
    if isinstance(calibration, CurvatureMerge):
        if len(calibration.second_moments) != len(donors):
            raise ValueError("Each donor needs one second-moment checkpoint")
        moment_readers = [CheckpointReader(source) for source in calibration.second_moments]
        if any(set(reader.weight_map) != names for reader in moment_readers):
            raise ValueError("Second-moment tensor keys differ from model weights")
    selected_parameters = {}
    for name, coefficients in tensor_coefficients.items():
        if len(coefficients) != len(donors):
            raise ValueError(f"Each donor needs one coefficient for {name}")
        selected_parameters[name] = replace(parameters, coefficients=coefficients)
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
        "tensor_coefficients": tensor_coefficients,
        "row_overrides": {
            name: [asdict(override) for override in overrides] for name, overrides in row_overrides.items()
        },
        "calibration": asdict(calibration) if calibration is not None else None,
        "objects": [],
    }

    def write(name: str, data: bytes) -> None:
        fs.pipe_file(prefix_join(output_path, name), data)
        manifest["objects"].append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    # Establish provenance before expensive work; the manifest remains the final gate.
    write("merge-recipe.json", json.dumps(manifest, indent=2).encode())
    weight_map = {}
    total_size = 0
    ordered = sorted(names, key=lambda name: (readers[0].weight_map[name], name))
    for index, name in enumerate(ordered, 1):
        base_tensor = readers[0].tensor(name)
        donor_tensors = [reader.tensor(name) for reader in readers[1:]]
        if isinstance(calibration, ChunkMerge):
            if name in calibration.coefficients:
                with torch.no_grad():
                    merged = differentiable_weight_blend(
                        base_tensor.contiguous(),
                        tuple(tensor.contiguous() for tensor in donor_tensors),
                        torch.tensor(calibration.coefficients[name], dtype=torch.float32),
                        block_elements=calibration.block_elements,
                    )
            else:
                merged = base_tensor.clone()
        elif isinstance(calibration, CurvatureMerge):
            merged = ota_merge_tensor(
                base_tensor,
                donor_tensors,
                [reader.tensor(name) for reader in moment_readers],
                density=calibration.density,
                epsilon=calibration.epsilon,
            )
        else:
            merged = merge_tensor(
                base_tensor, donor_tensors, selected_parameters.get(name, parameters), tensor_name=name
            )
        for override in row_overrides.get(name, ()):
            if merged.ndim == 0 or override.row >= merged.shape[0]:
                raise ValueError(f"Row override outside tensor shape: {name}[{override.row}]")
            merged[override.row] = merge_tensor(
                readers[0].tensor(name)[override.row],
                [reader.tensor(name)[override.row] for reader in readers[1:]],
                replace(parameters, coefficients=override.coefficients),
                tensor_name=f"{name}[{override.row}]",
            )
        if name in preserve_rows:
            rows = list(preserve_rows[name])
            merged[rows] = readers[0].tensor(name)[rows]
        if not torch.isfinite(merged).all():
            raise ValueError(f"Nonfinite merged tensor: {name}")
        shard = f"model-{index:05d}-of-{len(ordered):05d}.safetensors"
        write(shard, safetensors.torch.save({name: merged}, metadata={"format": "pt"}))
        weight_map[name] = shard
        total_size += merged.numel() * merged.element_size()
        del merged, base_tensor, donor_tensors
        logger.info("Merged %d/%d: %s", index, len(ordered), name)
    for name in METADATA_NAMES:
        path = prefix_join(readers[0].path, name)
        if readers[0].fs.exists(path):
            write(name, readers[0].fs.cat_file(path))
    write(INDEX_NAME, json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}).encode())
    manifest["tensor_count"] = len(weight_map)
    manifest["total_size"] = total_size
    fs.pipe_file(prefix_join(output_path, MANIFEST_NAME), json.dumps(manifest, indent=2).encode())
    return manifest
