# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stratified document sampling from JSONL files.

Reads documents from multiple sources (e.g. Nemotron quality buckets, Dolma source splits)
and samples N documents per source, preserving a label field for downstream evaluation.
"""

import glob
import json
import logging
import os
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Artifact returned by sample_documents."""

    path: str
    """Path to the output JSONL file."""
    n_docs: int
    """Total number of sampled documents."""
    n_per_label: dict[str, int]
    """Number of documents sampled per label."""


def sample_documents(
    output_path: str,
    source_paths: dict[str, str],
    n_per_source: int,
    seed: int = 42,
) -> SampleResult:
    """Sample documents from multiple JSONL sources with stratification.

    Args:
        output_path: Directory to write the sampled documents.
        source_paths: Mapping from label → glob pattern for JSONL files.
            Each matched file is read, and up to n_per_source documents are sampled
            for that label.
        n_per_source: Target number of documents to sample per source/label.
        seed: Random seed for reproducibility.

    Returns:
        SampleResult with path to the output JSONL and counts.
    """
    rng = random.Random(seed)
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "sampled_docs.jsonl")

    all_sampled: list[dict] = []
    n_per_label: dict[str, int] = {}

    for label, pattern in sorted(source_paths.items()):
        files = sorted(glob.glob(pattern))
        if not files:
            logger.warning(f"No files matched pattern {pattern!r} for label {label!r}")
            continue

        docs = []
        for fpath in files:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    doc["label"] = label
                    docs.append(doc)

        if len(docs) <= n_per_source:
            sampled = docs
        else:
            sampled = rng.sample(docs, n_per_source)

        all_sampled.extend(sampled)
        n_per_label[label] = len(sampled)
        logger.info(f"Sampled {len(sampled)} docs for label {label!r} (from {len(docs)} available)")

    with open(out_file, "w") as f:
        for doc in all_sampled:
            f.write(json.dumps(doc) + "\n")

    return SampleResult(path=out_file, n_docs=len(all_sampled), n_per_label=n_per_label)
