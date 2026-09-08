# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content identities for region-local Delphi TPP40 evaluation data."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import fsspec

TABLE9_MANIFEST_FILE = "manifest.json"
TABLE9_REQUESTS_FILE = "requests.jsonl"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _object_identity(fs, path: str) -> dict[str, int | str]:
    info = fs.info(path)
    crc32c = info.get("crc32c")
    if not isinstance(crc32c, str) or not crc32c:
        raise ValueError(f"GCS object lacks CRC32C identity: {path}")
    size = info.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"GCS object has invalid size: {path}")
    return {"size": size, "crc32c": crc32c}


def validation_payload_identity(
    cache_paths: dict[str, str],
    *,
    excluded_suffixes: tuple[str, ...] = ("shard_ledger.json",),
) -> dict[str, Any]:
    """Return a path-independent identity for tokenized validation payloads."""
    caches: list[dict[str, Any]] = []
    total_objects = 0
    total_bytes = 0
    for name, cache_path in sorted(cache_paths.items()):
        fs, _, roots = fsspec.get_fs_token_paths(cache_path)
        if len(roots) != 1:
            raise ValueError(f"Expected one validation cache root for {cache_path!r}, got {roots}")
        validation_root = roots[0].rstrip("/") + "/validation"
        objects: list[dict[str, Any]] = []
        for object_path in sorted(fs.find(validation_root)):
            relative_path = object_path.removeprefix(validation_root.rstrip("/") + "/")
            if not relative_path or relative_path.endswith(excluded_suffixes):
                continue
            identity = _object_identity(fs, object_path)
            objects.append({"path": relative_path, **identity})
            total_objects += 1
            total_bytes += int(identity["size"])
        if not objects or ".stats.json" not in {item["path"] for item in objects}:
            raise ValueError(f"Validation cache lacks payload or .stats.json: {cache_path}")
        caches.append({"name": name, "objects": objects})
    payload = {"caches": caches, "excluded_suffixes": list(excluded_suffixes)}
    return {
        "payload_sha256": _canonical_sha256(payload),
        "objects": total_objects,
        "bytes": total_bytes,
        **payload,
    }


def table9_request_set_identity(request_set_dir: str) -> dict[str, Any]:
    """Return a path-independent identity for a native Table-9 request set."""
    root = request_set_dir.rstrip("/")
    manifest_path = f"{root}/{TABLE9_MANIFEST_FILE}"
    requests_path = f"{root}/{TABLE9_REQUESTS_FILE}"
    with fsspec.open(manifest_path, "rb") as handle:
        manifest_bytes = handle.read()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError(f"Table-9 manifest is not a JSON object: {manifest_path}")
    fs, _, request_paths = fsspec.get_fs_token_paths(requests_path)
    if len(request_paths) != 1:
        raise ValueError(f"Expected one Table-9 requests object, got {request_paths}")
    requests_identity = _object_identity(fs, request_paths[0])
    payload = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest": manifest,
        "requests": requests_identity,
    }
    return {"payload_sha256": _canonical_sha256(payload), **payload}


def tree_payload_identity(
    root_path: str,
    *,
    excluded_relative_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a path-independent object identity for one GCS tree."""
    fs, _, roots = fsspec.get_fs_token_paths(root_path)
    if len(roots) != 1:
        raise ValueError(f"Expected one GCS tree root for {root_path!r}, got {roots}")
    root = roots[0].rstrip("/")
    objects: list[dict[str, Any]] = []
    total_bytes = 0
    for object_path in sorted(fs.find(root)):
        relative_path = object_path.removeprefix(root + "/")
        if not relative_path or relative_path in excluded_relative_paths:
            continue
        identity = _object_identity(fs, object_path)
        objects.append({"path": relative_path, **identity})
        total_bytes += int(identity["size"])
    if not objects:
        raise ValueError(f"GCS tree is empty: {root_path}")
    payload: object = objects
    if excluded_relative_paths:
        payload = {
            "excluded_relative_paths": sorted(excluded_relative_paths),
            "objects": objects,
        }
    return {
        "payload_sha256": _canonical_sha256(payload),
        "objects": len(objects),
        "bytes": total_bytes,
    }
