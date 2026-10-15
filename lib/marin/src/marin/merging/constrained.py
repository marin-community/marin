# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply bounded edits while retaining the original safetensors shard structure."""

import hashlib
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import safetensors.torch
import torch
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.merging.checkpoint import INCOMPLETE_NAME, INDEX_NAME, CheckpointReader


@dataclass(frozen=True)
class TensorSupport:
    name: str
    rows: tuple[int, ...] | None


def validate_audit_location(audit: str, output: str) -> None:
    """Require the audit to be outside the checkpoint, including local path aliases."""
    audit_location = StoragePath.parse(audit)
    output_location = StoragePath.parse(output)
    if audit_location.scheme in (None, "file"):
        audit_location = StoragePath.parse(
            str(Path("/" + audit_location.key if audit_location.scheme else audit).resolve())
        )
    if output_location.scheme in (None, "file"):
        output_location = StoragePath.parse(
            str(Path("/" + output_location.key if output_location.scheme else output).resolve())
        )
    try:
        audit_location.relative_to(output_location)
    except ValueError:
        return
    raise ValueError("Audit must be outside the model directory")


def changed_elements(left: torch.Tensor, right: torch.Tensor) -> int:
    """Count bitwise scalar differences, including signed-zero differences."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Cannot compare tensors of different shape or dtype")
    a = left.contiguous().reshape(-1).view(torch.uint8).reshape(-1, left.element_size())
    b = right.contiguous().reshape(-1).view(torch.uint8).reshape(-1, right.element_size())
    return int((a != b).any(dim=1).sum().item())


def finalize_constrained_export(audit: dict, audit_url: str, change_cap: float) -> None:
    """Publish and verify the external audit before making the checkpoint readable."""
    validate_audit_location(audit_url, audit["output"])
    if not 0 <= change_cap <= 0.30:
        raise ValueError("Changed-weight cap must be within [0, 0.30]")
    if audit["changed_elements"] > math.floor(change_cap * audit["total_elements"]):
        raise ValueError("Actual changed elements exceed the experiment cap")
    fs, path = filesystem_for(audit_url)
    if fs.exists(path):
        raise FileExistsError(audit_url)
    output_fs, output_path = filesystem_for(audit["output"])
    marker = prefix_join(output_path, INCOMPLETE_NAME)
    if not output_fs.exists(marker):
        raise ValueError("Checkpoint has no pending constrained export")
    payload = json.dumps(audit, indent=2).encode()
    fs.makedirs(str(StoragePath.parse(path).parent), exist_ok=True)
    fs.pipe_file(path, payload)
    if fs.cat_file(path) != payload:
        raise ValueError("Stored audit differs from the completed export")
    output_fs.rm(marker)


def export_constrained(
    base: CheckpointReader,
    output: str,
    support: tuple[TensorSupport, ...],
    support_cap: float,
    change_cap: float,
    transform: Callable[[str, int | None, torch.Tensor], torch.Tensor],
) -> dict:
    """Export bounded edits and independently read back every object before returning an audit.

    The caller must pass the returned audit to ``finalize_constrained_export``.
    The checkpoint remains unreadable until that external audit is saved and
    verified. A failed export retains its incomplete marker.

    ``support_cap`` bounds the scalars that the transform may inspect or edit.
    ``change_cap`` independently bounds stored scalars that differ from Base.
    This permits wider support when selected donor values equal Base.
    """
    if not 0 <= support_cap <= 1:
        raise ValueError("Support cap must be within [0, 1]")
    if not 0 <= change_cap <= 0.30:
        raise ValueError("Changed-weight cap must be within [0, 0.30]")
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
    if selected > math.floor(support_cap * total):
        raise ValueError(f"Support exceeds scalar budget: {selected}/{total}")
    fs, path = filesystem_for(output)
    if fs.exists(path) and fs.ls(path):
        raise FileExistsError(output)
    if str(StoragePath.parse(output)).rstrip("/") == str(StoragePath.parse(base.source.path)).rstrip("/"):
        raise ValueError("Output cannot replace Base")
    source_paths = sorted(base.fs.find(base.path))
    fs.makedirs(path, exist_ok=True)
    fs.pipe_file(prefix_join(path, INCOMPLETE_NAME), b"")
    files = {}
    changes = 0
    shards = set(headers)
    for source_path in source_paths:
        relative = StoragePath.parse(source_path).relative_to(StoragePath.parse(base.path))
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
    if changes > math.floor(change_cap * total):
        raise ValueError("Stored checkpoint exceeds scalar budget")
    actual_files = {StoragePath.parse(p).relative_to(StoragePath.parse(path)) for p in fs.find(path)}
    actual_files.discard(INCOMPLETE_NAME)
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
        "support_cap": support_cap,
        "change_cap": change_cap,
        "files": files,
        "passed": True,
    }
