# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply bounded edits while retaining the original safetensors shard structure."""

import hashlib
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import safetensors.torch
import torch
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.merging.checkpoint import INDEX_NAME, CheckpointReader


@dataclass(frozen=True)
class TensorSupport:
    name: str
    rows: tuple[int, ...] | None


def changed_elements(left: torch.Tensor, right: torch.Tensor) -> int:
    """Count bitwise scalar differences, including signed-zero differences."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Cannot compare tensors of different shape or dtype")
    a = left.contiguous().reshape(-1).view(torch.uint8).reshape(-1, left.element_size())
    b = right.contiguous().reshape(-1).view(torch.uint8).reshape(-1, right.element_size())
    return int((a != b).any(dim=1).sum().item())


def export_constrained(
    base: CheckpointReader,
    output: str,
    support: tuple[TensorSupport, ...],
    cap: float,
    transform: Callable[[str, int | None, torch.Tensor], torch.Tensor],
) -> dict:
    """Export bounded edits and independently read back every object before returning an audit.

    The caller must save the returned audit outside the model directory and only
    expose the model after this function returns successfully. A failed export
    remains incomplete and cannot be reused as a serial input.
    """
    if not 0 <= cap <= 0.25:
        raise ValueError("Changed-weight cap must be within [0, 0.25]")
    selections = {item.name: item.rows for item in support}
    if len(selections) != len(support) or not selections.keys() <= base.weight_map.keys():
        raise ValueError("Support contains duplicate or unknown tensors")
    headers = {}
    total = 0
    selected = 0
    for shard in sorted(set(base.weight_map.values())):
        with base.fs.open(prefix_join(base.path, shard), "rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(length))
        headers[shard] = header
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if base.weight_map.get(name) != shard:
                raise ValueError(f"Shard/index disagreement: {name}")
            count = math.prod(spec["shape"])
            total += count
            if name not in selections:
                continue
            rows = selections[name]
            if rows is None:
                selected += count
            else:
                shape = spec["shape"]
                if not shape or len(rows) != len(set(rows)) or any(r < 0 or r >= shape[0] for r in rows):
                    raise ValueError(f"Invalid support rows: {name}")
                selected += len(rows) * math.prod(shape[1:])
    if selected > math.floor(cap * total):
        raise ValueError(f"Support exceeds scalar budget: {selected}/{total}")
    fs, path = filesystem_for(output)
    if fs.exists(path) and fs.ls(path):
        raise FileExistsError(output)
    if str(StoragePath.parse(output)).rstrip("/") == str(StoragePath.parse(base.source.path)).rstrip("/"):
        raise ValueError("Output cannot replace Base")
    fs.makedirs(path, exist_ok=True)
    files = {}
    changes = 0
    shards = set(headers)
    for source_path in sorted(base.fs.find(base.path)):
        relative = source_path[len(base.path.rstrip("/")) + 1 :]
        if not relative or relative.startswith("../"):
            raise ValueError(f"Invalid source path: {source_path}")
        original = base.fs.cat_file(source_path)
        payload = original
        if relative in shards:
            tensors = safetensors.torch.load(original)
            offset = 8 + int.from_bytes(original[:8], "little")
            edited = bytearray(original)
            for name, original_tensor in tensors.items():
                if name not in selections:
                    continue
                tensor = original_tensor.clone()
                rows = selections[name]
                if rows is None:
                    result = transform(name, None, original_tensor)
                    changed_elements(original_tensor, result)
                    tensor.copy_(result)
                else:
                    for row in rows:
                        result = transform(name, row, original_tensor[row])
                        changed_elements(original_tensor[row], result)
                        tensor[row].copy_(result)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite output: {name}")
                start, end = headers[relative][name]["data_offsets"]
                raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                if len(raw) != end - start:
                    raise ValueError(f"Tensor byte size changed: {name}")
                edited[offset + start : offset + end] = raw
            payload = bytes(edited)
        destination = prefix_join(path, relative)
        fs.makedirs(str(StoragePath.parse(destination).parent), exist_ok=True)
        fs.pipe_file(destination, payload)
        stored = fs.cat_file(destination)
        if stored != payload:
            raise ValueError(f"Stored object differs from export: {relative}")
        changed = 0
        if relative in shards:
            after = safetensors.torch.load(stored)
            before = safetensors.torch.load(original)
            for name, value in before.items():
                tensor_changes = changed_elements(value, after[name])
                if name not in selections and tensor_changes:
                    raise ValueError(f"Changed unselected tensor: {name}")
                rows = selections.get(name)
                if rows is not None:
                    allowed_changes = sum(changed_elements(value[r], after[name][r]) for r in rows)
                    if tensor_changes != allowed_changes:
                        raise ValueError(f"Changed unselected rows: {name}")
                changed += tensor_changes
        elif stored != original:
            raise ValueError(f"Non-weight file changed: {relative}")
        changes += changed
        logging.info("Verified %s: %s changed scalars", relative, changed)
        files[relative] = {"sha256": hashlib.sha256(stored).hexdigest(), "bytes": len(stored), "changed": changed}
    if set(base.weight_map.values()) - files.keys() or INDEX_NAME not in files:
        raise ValueError("Missing checkpoint objects")
    if changes > math.floor(cap * total):
        raise ValueError("Stored checkpoint exceeds scalar budget")
    actual_files = {p[len(path.rstrip("/")) + 1 :] for p in fs.find(path)}
    if actual_files != files.keys():
        raise ValueError("Export contains unexpected or missing files")
    return {
        "base": base.source.path,
        "base_revision": base.source.revision,
        "output": output,
        "total_elements": total,
        "supported_elements": selected,
        "changed_elements": changes,
        "changed_percent": 100 * changes / total,
        "cap": cap,
        "files": files,
        "passed": True,
    }
