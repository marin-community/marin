# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load production catalogs through the committed reference index.

The catalog JSON files are pinned in a private Hugging Face dataset
(open-athena/glm-tool-conversion-production-data) instead of committed to the
repository (large-data guard). The loader resolves a catalog by name through
experiments/glm_tool_serving/production_data/reference-index.json, verifies the
downloaded bytes against the pinned SHA-256, and caches them beside the index so
a pod pays the download at most once per workspace. The returned digest is the
sha256 of the original file bytes, so producer cursors stay bound to the same
catalog_sha256 values as before the migration.
"""

import hashlib
import json
import time
from pathlib import Path

from experiments.post_training.task_curation.reference_data import read_referenced_document

INDEX_NAME = "reference-index.json"


def load_catalog(data_dir: Path, name: str) -> tuple[dict, str]:
    """Return (catalog, sha256-of-original-bytes) for a production catalog."""
    path = data_dir / name
    if path.exists():
        raw = path.read_bytes()
    else:
        raw = None
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                raw = read_referenced_document(data_dir / INDEX_NAME, name)
                break
            except Exception as error:
                last_error = error
                time.sleep(5 * (attempt + 1))
        if raw is None:
            raise RuntimeError(f"Catalog fetch failed after retries: {name}: {last_error}") from last_error
        path.write_bytes(raw)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()
