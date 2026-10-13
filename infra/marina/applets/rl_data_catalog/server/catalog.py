# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the curation catalog's published inventory and evidence."""

import hashlib
import json
from typing import Any

from finestore.reader import ReadView

CATALOG_ROOT = "gs://marin-us-central2/rl-data-curation-assessment/2026-10-12-campaign/campaigns/catalog"


def validate_catalog(document: dict[str, Any]) -> dict[str, Any]:
    if document["schema_version"] != 2:
        raise ValueError(f"Unsupported catalog schema version: {document['schema_version']}")
    revision = hashlib.sha256(
        json.dumps(
            document["sources"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if document["revision"] != revision:
        raise ValueError("Catalog revision does not match its sources")
    identifiers = set()
    for row in document["sources"]:
        if not row["id"] or not row["origin"]:
            raise ValueError("Catalog sources require an ID and origin")
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate catalog source: {row['id']}")
        identifiers.add(row["id"])
    return document


def published_catalog(root: str, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the current snapshot, reusing a saved copy while its immutable ID matches."""
    view = ReadView(root)
    current = view.point("catalog_current", name="latest")
    if current is None:
        raise ValueError(f"No published curation catalog at {root}")
    if previous is not None and current["snapshot_id"] == previous["snapshot_id"]:
        return previous
    data = view.resolve(current["document_uri"])
    if data is None:
        raise ValueError("Published catalog document is unavailable")
    return validate_catalog(json.loads(data))


def evidence_document(root: str, uri: str) -> dict[str, Any]:
    data = ReadView(root).resolve(uri)
    if data is None:
        raise ValueError("Archived evidence is unavailable")
    return json.loads(data)
