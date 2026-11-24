# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reciprocal-rank fusion for the hybrid OpenResearcher retriever.

Kept dependency-free so the pinned retrieval runtime, the hosted service and the
tests can all import it without the workspace environment.
"""

from __future__ import annotations

CANDIDATE_DEPTH = 25
RRF_K = 60
TOP_K = 5


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Fuse ranked docid lists with reciprocal rank fusion.

    Each list contributes ``1 / (k + rank)`` for its members (ranks start at 1).
    Ties resolve deterministically by docid so every consumer sees one order.
    """
    if k <= 0:
        raise ValueError("The RRF constant must be positive")
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, docid in enumerate(ranked, start=1):
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
