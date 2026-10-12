# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the catalog generated from the task-curation source registry."""

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).with_name("catalog_data.py")


@dataclass(frozen=True)
class Snapshot:
    origin: str
    revision: str
    revised_at: str | None
    rows: list[dict[str, Any]]


def catalog_snapshots(path: Path) -> list[Snapshot]:
    """Validate the complete artifact before any saved inventory is replaced."""
    content = path.read_text()
    if path.suffix == ".py":
        # Marina extracts only server Python files for backend execution.
        content = ast.literal_eval(content)
    catalog = json.loads(content)
    if catalog["schema_version"] != 1:
        raise ValueError(f"Unsupported catalog schema version: {catalog['schema_version']}")
    revision = catalog["revision"]
    if not isinstance(revision, str) or not revision:
        raise ValueError("Generated catalog has no revision")
    content_revision = hashlib.sha256(
        json.dumps(catalog["sources"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if revision != content_revision:
        raise ValueError("Catalog revision does not match its sources")
    grouped: dict[str, list[dict[str, Any]]] = {}
    ids: set[str] = set()
    for row in catalog["sources"]:
        source_id, origin = row["id"], row["origin"]
        if not isinstance(source_id, str) or not source_id or not isinstance(origin, str) or not origin:
            raise ValueError("Catalog source requires a nonempty id and origin")
        if source_id in ids:
            raise ValueError(f"Duplicate catalog source: {source_id}")
        ids.add(source_id)
        grouped.setdefault(origin, []).append(row)
    return [Snapshot(origin, revision, catalog.get("revised_at"), rows) for origin, rows in sorted(grouped.items())]
