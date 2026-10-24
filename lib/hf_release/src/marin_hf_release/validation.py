# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded-memory validation of completed local safetensors releases."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path

from marin_hf_release.policy import ReleasePolicy, TensorSpec, dtype_size

INDEX_NAME = "model.safetensors.index.json"
MAX_HEADER_BYTES = 100 * 2**20
SAFETENSORS_LENGTH_PREFIX_BYTES = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_header_specs(path: Path) -> dict[str, TensorSpec]:
    """Read one safetensors header without loading tensor payloads."""
    with path.open("rb") as source:
        prefix = source.read(SAFETENSORS_LENGTH_PREFIX_BYTES)
        if len(prefix) != SAFETENSORS_LENGTH_PREFIX_BYTES:
            raise ValueError(f"truncated safetensors length prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes > MAX_HEADER_BYTES:
            raise ValueError(f"safetensors header exceeds {MAX_HEADER_BYTES} bytes: {path}")
        header = source.read(header_bytes)
        if len(header) != header_bytes:
            raise ValueError(f"truncated safetensors header: {path}")
    parsed = json.loads(header)
    if not isinstance(parsed, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    payload_bytes = path.stat().st_size - SAFETENSORS_LENGTH_PREFIX_BYTES - header_bytes
    tensors = {}
    offsets = []
    for name, entry in parsed.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"invalid safetensors entry {name!r}: {path}")
        dtype = entry["dtype"]
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        if not isinstance(name, str) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise ValueError(f"invalid safetensors tensor {name!r}: {path}")
        element_count = 1
        for dim in shape:
            element_count *= dim
        if end - start != element_count * dtype_size(dtype) or start < 0 or end > payload_bytes:
            raise ValueError(f"invalid safetensors byte range for {name}: {path}")
        offsets.append((start, end))
        tensors[name] = TensorSpec(shape, dtype)
    next_offset = 0
    for start, end in sorted(offsets):
        if start != next_offset:
            raise ValueError(f"safetensors payload has overlapping or uncovered bytes: {path}")
        next_offset = end
    if next_offset != payload_bytes:
        raise ValueError(f"safetensors payload has uncovered bytes: {path}")
    return tensors


def validate_release_shard(path: Path, specs: Mapping[str, TensorSpec], policy: ReleasePolicy) -> str:
    """Validate a staged shard before upload, then return its SHA-256 hash."""
    actual = read_safetensors_header_specs(path)
    if actual.keys() != specs.keys():
        missing = sorted(specs.keys() - actual.keys())
        extra = sorted(actual.keys() - specs.keys())
        raise ValueError(f"HF tensor names differ from schema: missing={missing[:5]}, extra={extra[:5]}")
    for name, expected in specs.items():
        observed = actual[name]
        target_dtype = policy.target_dtype(name, expected)
        if observed.shape != expected.shape or observed.source_dtype != target_dtype:
            raise ValueError(
                f"{name}: observed {observed.shape}/{observed.source_dtype}, "
                f"expected {expected.shape}/{target_dtype}"
            )
    return _sha256(path)


def validate_release_index(
    index: Mapping[str, object], weight_map: Mapping[str, str], specs: Mapping[str, TensorSpec], policy: ReleasePolicy
) -> None:
    """Compare the published index with already validated shard headers."""
    if index.get("weight_map") != weight_map:
        raise ValueError("HF safetensors index weight_map differs from shards")
    total_size = 0
    for name, expected in specs.items():
        target_dtype = policy.target_dtype(name, expected)
        count = 1
        for dim in expected.shape:
            count *= dim
        total_size += count * dtype_size(target_dtype)
    if index.get("metadata", {}).get("total_size") != total_size:
        raise ValueError("HF safetensors index total_size differs from tensor bytes")


def validate_release_weights(
    root: Path,
    specs: Mapping[str, TensorSpec],
    policy: ReleasePolicy,
) -> dict[str, str]:
    """Check tensor layout and index totals, returning shard and index hashes."""
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"HF release has no safetensors shards: {root}")
    index_path = root / INDEX_NAME
    if len(shards) > 1 and not index_path.is_file():
        raise ValueError(f"sharded HF release requires {INDEX_NAME}: {root}")
    actual: dict[str, TensorSpec] = {}
    weight_map: dict[str, str] = {}
    for shard in shards:
        for name, spec in read_safetensors_header_specs(shard).items():
            if name in actual:
                raise ValueError(f"duplicate safetensors tensor {name}: {root}")
            actual[name] = spec
            weight_map[name] = shard.name
    if actual.keys() != specs.keys():
        missing = sorted(specs.keys() - actual.keys())
        extra = sorted(actual.keys() - specs.keys())
        raise ValueError(f"HF tensor names differ from schema: missing={missing[:5]}, extra={extra[:5]}")
    hashes = {
        shard.name: validate_release_shard(
            shard, {name: specs[name] for name, file in weight_map.items() if file == shard.name}, policy
        )
        for shard in shards
    }
    if index_path.is_file():
        validate_release_index(json.loads(index_path.read_text()), weight_map, specs, policy)
        hashes[index_path.name] = _sha256(index_path)
    return hashes
