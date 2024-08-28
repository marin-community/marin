"""
validation_utils.py

Helpful (and semi-standardized) functions for maintaining and validating dataset provenance and statistics (both for
raw and processed data).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fsspec


def write_provenance_json(gcs_output_path: Path, gcs_bucket: str, metadata: dict[str, Any]) -> None:
    print(f"[*] Writing Dataset `provenance.json` to `gs://{gcs_bucket}/{gcs_output_path}`")
    metadata["access_time"] = datetime.now(timezone.utc).isoformat()

    with fsspec.open(f"gs://{gcs_bucket}/{gcs_output_path!s}/provenance.json", "wt") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)
