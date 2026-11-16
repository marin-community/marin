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
from pathlib import Path

from experiments.post_training.task_curation.reference_data import read_referenced_document

INDEX_NAME = "reference-index.json"


def load_catalog(data_dir: Path, name: str) -> tuple[dict, str]:
    """Return (catalog, sha256-of-original-bytes) for a production catalog."""
    path = data_dir / name
    if path.exists():
        raw = path.read_bytes()
    else:
        raw = read_referenced_document(data_dir / INDEX_NAME, name)
        path.write_bytes(raw)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()
