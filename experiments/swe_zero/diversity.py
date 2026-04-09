# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Diversity measurement for rollout traces using MinHash / Jaccard similarity.

Implements the near-deduplication approach from the Code World Model paper:
  1. Represent each trajectory by the concatenation of its actions
  2. Encode using MinHash
  3. Compute pairwise Jaccard similarity
  4. Report diversity metrics (and optionally filter duplicates at threshold 0.5)
"""

from __future__ import annotations

import hashlib
import logging
import struct
from dataclasses import dataclass

import numpy as np

from experiments.swe_zero.rollout_generator import Rollout

logger = logging.getLogger(__name__)

# MinHash parameters
NUM_HASHES = 128
SHINGLE_SIZE = 5  # character n-gram size for shingling
MAX_HASH = 2**32 - 1
JACCARD_THRESHOLD = 0.5


def _shingle(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """Convert text to a set of character k-grams."""
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash(shingles: set[str], num_hashes: int = NUM_HASHES) -> np.ndarray:
    """Compute MinHash signature for a set of shingles."""
    if not shingles:
        return np.full(num_hashes, MAX_HASH, dtype=np.uint32)

    signature = np.full(num_hashes, MAX_HASH, dtype=np.uint32)
    for shingle in shingles:
        for i in range(num_hashes):
            # Use different hash functions by mixing seed with index
            h = hashlib.md5(f"{i}:{shingle}".encode()).digest()
            val = struct.unpack("<I", h[:4])[0]
            if val < signature[i]:
                signature[i] = val
    return signature


def jaccard_from_minhash(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return float(np.mean(sig1 == sig2))


@dataclass
class DiversityReport:
    """Summary of rollout diversity."""

    n_rollouts: int
    n_unique: int  # after dedup at threshold
    mean_pairwise_jaccard: float
    median_pairwise_jaccard: float
    min_pairwise_jaccard: float
    max_pairwise_jaccard: float
    std_pairwise_jaccard: float
    pairwise_similarities: list[float]

    def summary(self) -> str:
        return (
            f"Diversity Report:\n"
            f"  Total rollouts: {self.n_rollouts}\n"
            f"  Unique (Jaccard < {JACCARD_THRESHOLD}): {self.n_unique}\n"
            f"  Mean pairwise Jaccard: {self.mean_pairwise_jaccard:.4f}\n"
            f"  Median pairwise Jaccard: {self.median_pairwise_jaccard:.4f}\n"
            f"  Min pairwise Jaccard: {self.min_pairwise_jaccard:.4f}\n"
            f"  Max pairwise Jaccard: {self.max_pairwise_jaccard:.4f}\n"
            f"  Std pairwise Jaccard: {self.std_pairwise_jaccard:.4f}"
        )


def measure_diversity(rollouts: list[Rollout]) -> DiversityReport:
    """
    Measure diversity of rollouts using MinHash-based Jaccard similarity.

    Following the Code World Model paper:
      - Represent trajectory by concatenation of actions
      - Encode with MinHash
      - Compute pairwise Jaccard similarity
    """
    n = len(rollouts)
    if n < 2:
        return DiversityReport(
            n_rollouts=n,
            n_unique=n,
            mean_pairwise_jaccard=0.0,
            median_pairwise_jaccard=0.0,
            min_pairwise_jaccard=0.0,
            max_pairwise_jaccard=0.0,
            std_pairwise_jaccard=0.0,
            pairwise_similarities=[],
        )

    # Step 1: Extract action text from each rollout
    action_texts = [r.actions_text() for r in rollouts]

    # Step 2: Compute MinHash signatures
    signatures = []
    for text in action_texts:
        shingles = _shingle(text)
        sig = _minhash(shingles)
        signatures.append(sig)

    # Step 3: Compute pairwise Jaccard similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_from_minhash(signatures[i], signatures[j])
            similarities.append(sim)

    # Step 4: Greedy deduplication (keep rollouts with pairwise Jaccard < threshold)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            sim = jaccard_from_minhash(signatures[i], signatures[j])
            if sim >= JACCARD_THRESHOLD:
                keep[j] = False

    n_unique = sum(keep)
    sims_arr = np.array(similarities)

    return DiversityReport(
        n_rollouts=n,
        n_unique=n_unique,
        mean_pairwise_jaccard=float(np.mean(sims_arr)),
        median_pairwise_jaccard=float(np.median(sims_arr)),
        min_pairwise_jaccard=float(np.min(sims_arr)),
        max_pairwise_jaccard=float(np.max(sims_arr)),
        std_pairwise_jaccard=float(np.std(sims_arr)),
        pairwise_similarities=similarities,
    )


def filter_diverse(rollouts: list[Rollout], threshold: float = JACCARD_THRESHOLD) -> list[Rollout]:
    """Return subset of rollouts where all pairwise Jaccard < threshold."""
    n = len(rollouts)
    if n < 2:
        return rollouts

    action_texts = [r.actions_text() for r in rollouts]
    signatures = [_minhash(_shingle(text)) for text in action_texts]

    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            sim = jaccard_from_minhash(signatures[i], signatures[j])
            if sim >= threshold:
                keep[j] = False

    return [r for r, k in zip(rollouts, keep, strict=True) if k]
