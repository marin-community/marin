# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
validation_utils.py

Helpful (and semi-standardized) functions for maintaining and validating dataset provenance and statistics (both for
raw and processed data).
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from rigging.filesystem import StoragePath

logger = logging.getLogger(__name__)


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
    logger.info("Writing Dataset `.provenance.json` to `%s`", output_path)
    metadata["access_time"] = datetime.now(UTC).isoformat()

    # Dot-prefix keeps the sidecar out of data-discovery passes that match by
    # extension (e.g. ``normalize._discover_files`` would otherwise read it as
    # JSONL — see #5864).
    StoragePath(f"{output_path}/.provenance.json").write_text(json.dumps(metadata, indent=4, sort_keys=True))


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


def summarize_document(doc: dict) -> DocumentSummary:
    """Validate that a document dict is in valid Dolma-format, and return summary (e.g., footprint in bytes, etc.)."""
    validated_doc = DolmaDocument.model_validate(doc)
    json_blob = json.dumps(doc)
    return DocumentSummary(document_bytes=get_size_bytes(json_blob), text_bytes=get_size_bytes(validated_doc.text))
