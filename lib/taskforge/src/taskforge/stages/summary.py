# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small prompt views of workspace artifacts; full payloads remain in files."""

import json

from taskcompendium.models import TaskSpec


def manifest(value: object, role: str = "") -> object:
    if isinstance(value, dict):
        source = value.get("source")
        if isinstance(source, dict) and source.get("kind") == "inline_file":
            encoded = source["content_base64"]
            return {
                "path": value["path"],
                "bytes": len(encoded) * 3 // 4 - len(encoded) + len(encoded.rstrip("=")),
                "role": role,
            }
        return {key: manifest(child, f"{role}.{key}".strip(".")) for key, child in value.items()}
    if isinstance(value, list):
        return [manifest(child, role) for child in value]
    return value


def task_manifest(task: TaskSpec) -> str:
    return json.dumps(manifest(task.model_dump(mode="json")), separators=(",", ":"))
