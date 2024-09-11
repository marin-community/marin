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
from pydantic import BaseModel


# === General Pydantic Schema for Quick & Easy Ser/De + Validation ==
class DolmaDocument(BaseModel):
    id: str
    source: str
    text: str


# === Raw Data Download Utilities ===
def write_provenance_json(gcs_output_path: Path, gcs_bucket: str, metadata: dict[str, Any]) -> None:
    print(f"[*] Writing Dataset `provenance.json` to `gs://{gcs_bucket}/{gcs_output_path}`")
    metadata["access_time"] = datetime.now(timezone.utc).isoformat()

    with fsspec.open(f"gs://{gcs_bucket}/{gcs_output_path!s}/provenance.json", "wt") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)


# === Sharding Utilities ===
def compute_global_mean_std(
    shard_n_examples: list[int], shard_means: list[float], shard_stds: list[float]
) -> tuple[float, float]:
    n_examples = sum(shard_n_examples)
    global_mean = sum(n * mean for n, mean in zip(shard_n_examples, shard_means)) / n_examples
    global_variance = (
        sum(n * (std**2 + mean**2) for n, mean, std in zip(shard_n_examples, shard_means, shard_stds)) / n_examples
    )

    return global_mean, global_variance**0.5


# === Dolma-Formatted Data Validation Utilities ===
def parse_document_json(json_blob: str) -> DolmaDocument:
    return DolmaDocument.model_validate_json(json_blob)


def get_size_bytes(blob: str) -> int:
    return len(blob.encode("utf-8"))


def summarize_document_from_json(json_blob: str) -> dict[str, int]:
    """Validate that a JSON blob (str) is in valid Dolma-format, and return summary statistics."""
    doc = parse_document_json(json_blob)
    doc_bytes, text_bytes = get_size_bytes(json_blob), get_size_bytes(doc.text)

    return doc_bytes, text_bytes
