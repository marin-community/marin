"""
validation_utils.py

Helpful (and semi-standardized) functions for maintaining and validating dataset provenance and statistics (both for
raw and processed data).
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import fsspec
from pydantic import BaseModel


# === General Pydantic Schema for Quick & Easy Ser/De + Validation ==
class DolmaDocument(BaseModel):
    id: str
    source: str
    text: str


# === Utility Dataclasses & Functions ===
@dataclass
class DocumentSummary:
    document_bytes: int
    text_bytes: int


@dataclass
class SummaryStatistics:
    count: int
    mean: float
    std: float


def get_size_bytes(blob: str) -> int:
    return len(blob.encode("utf-8"))


# === Raw Data Download Utilities ===
def write_provenance_json(output_path, metadata: dict[str, Any]) -> None:
    print(f"[*] Writing Dataset `provenance.json` to `{output_path}`")
    metadata["access_time"] = datetime.now(timezone.utc).isoformat()

    with fsspec.open(f"{output_path}/provenance.json", "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)


# === Sharding Utilities ===
def compute_global_mean_std(
    shard_num_examples: list[int], shard_means: list[float], shard_stds: list[float]
) -> SummaryStatistics:
    """Compute global mean/std given lists of (num_examples, mean, std) for individual dataset shards."""
    num_examples = sum(shard_num_examples)
    global_mean = sum(n * mean for n, mean in zip(shard_num_examples, shard_means, strict=False)) / num_examples
    global_variance = (
        sum(n * (std**2 + mean**2) for n, mean, std in zip(shard_num_examples, shard_means, shard_stds, strict=False))
        / num_examples
    ) - (global_mean**2)

    return SummaryStatistics(count=num_examples, mean=global_mean, std=global_variance**0.5)


# === Dolma-Formatted Data Validation Utilities ===
def parse_document_json(json_blob: str) -> DolmaDocument:
    return DolmaDocument.model_validate_json(json_blob)


def summarize_document_from_json(json_blob: str) -> DocumentSummary:
    """Validate that a JSON blob (str) is in valid Dolma-format, and return summary (e.g., footprint in bytes, etc.)."""
    doc = parse_document_json(json_blob)

    return DocumentSummary(document_bytes=get_size_bytes(json_blob), text_bytes=get_size_bytes(doc.text))
