# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find duplicates within one materialized candidate cluster.

The solver processes documents in descending character count. Input order
determines the order for equal lengths. A surviving representative must meet the
directional word n-gram containment threshold to remove a member.

Small clusters compare each member with earlier survivors. Large clusters use
a rare-n-gram index with limits on posting lists and candidate counts. These
limits can miss duplicates within a cluster. The materializer can also split
large components with a MinHash key, which can separate containment pairs.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import dupekit
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_EMPTY = np.empty(0, dtype=np.uint64)


class ClusterDedupParams(BaseModel):
    """Thresholds and work bounds for one cluster."""

    model_config = ConfigDict(frozen=True)

    ngram_size: int = Field(default=3, ge=1)
    minimum_containment: float = Field(default=0.75, ge=0, le=1)
    """Minimum fraction of member n-grams that a representative must contain."""

    exact_scan_maximum: int = Field(default=256, ge=2)
    """Clusters with at most this many members use an exact scan."""

    probe_ngrams: int = Field(default=32, ge=1)
    """How many of a member's rarest n-grams probe the inverted index."""

    maximum_posting_length: int = Field(default=512, ge=1)
    """Skip longer postings when at least one posting meets this limit."""

    maximum_candidates: int = Field(default=32, ge=1)
    """Candidate limit before the solver excludes removed representatives."""


@dataclass(frozen=True)
class ClusterDocument:
    id: str
    text: str


@dataclass(frozen=True)
class PreparedDocument:
    index: int
    chars: int
    ngrams: np.ndarray
    text: str


@dataclass(frozen=True)
class Removal:
    member_index: int
    representative_index: int
    containment: float
    jaccard: float
    novel_tokens: int
    comparisons: int


def ngram_hashes(text: str, ngram_size: int) -> np.ndarray:
    """Sorted unique 64-bit hashes of the case-folded word n-grams.

    A 64-bit collision is unlikely for the bounded clusters this solver reads,
    so the hash array stands in for the n-gram set.
    """
    tokens = text.casefold().split()
    if not tokens:
        return _EMPTY
    if len(tokens) < ngram_size:
        shingles = [" ".join(tokens).encode("utf-8", "surrogatepass")]
    else:
        shingles = [
            " ".join(tokens[start : start + ngram_size]).encode("utf-8", "surrogatepass")
            for start in range(len(tokens) - ngram_size + 1)
        ]
    return np.unique(np.asarray(dupekit.hash_xxh3_64_batch(shingles), dtype=np.uint64))


def prepare(documents: Sequence[ClusterDocument], params: ClusterDedupParams) -> list[PreparedDocument]:
    prepared = []
    for index, document in enumerate(documents):
        prepared.append(
            PreparedDocument(
                index=index,
                chars=len(document.text),
                ngrams=ngram_hashes(document.text, params.ngram_size),
                text=document.text,
            )
        )
    return prepared


def novel_token_count(
    member: PreparedDocument,
    representative: PreparedDocument,
    cache: dict[int, frozenset[str]],
) -> int:
    """Words of the member that the representative does not hold."""
    for document in (member, representative):
        if document.index not in cache:
            cache[document.index] = frozenset(document.text.casefold().split())
    return len(cache[member.index] - cache[representative.index])


def _overlap(left: np.ndarray, right: np.ndarray) -> int:
    """Size of the intersection of two sorted unique hash arrays."""
    if left.size == 0 or right.size == 0:
        return 0
    if left.size > right.size:
        left, right = right, left
    position = np.searchsorted(right, left)
    position[position >= right.size] = right.size - 1
    return int(np.count_nonzero(right[position] == left))


@dataclass(frozen=True)
class _NgramIndex:
    """Inverted index from n-gram hash to the documents that hold it."""

    values: np.ndarray
    counts: np.ndarray
    starts: np.ndarray
    owners: np.ndarray


def _build_index(prepared: list[PreparedDocument]) -> _NgramIndex:
    sizes = np.fromiter((document.ngrams.size for document in prepared), dtype=np.int64, count=len(prepared))
    values = np.concatenate([document.ngrams for document in prepared]) if sizes.sum() else _EMPTY
    owners = np.repeat(np.arange(len(prepared), dtype=np.int32), sizes)
    order = np.argsort(values, kind="stable")
    values = values[order]
    owners = owners[order]
    if values.size == 0:
        return _NgramIndex(values=values, counts=_EMPTY.astype(np.int64), starts=_EMPTY.astype(np.int64), owners=owners)
    # ``np.unique`` would sort this array a second time, which dominated the
    # stage on a cluster holding a hundred million n-grams. The array is
    # already sorted, so the run boundaries are one comparison per element.
    boundary = np.empty(values.size, dtype=bool)
    boundary[0] = True
    np.not_equal(values[1:], values[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary)
    counts = np.diff(np.append(starts, values.size))
    return _NgramIndex(values=values[starts], counts=counts, starts=starts, owners=owners)


def _index_candidates(
    member: PreparedDocument,
    index: _NgramIndex,
    rank: np.ndarray,
    params: ClusterDedupParams,
) -> np.ndarray:
    """Select candidates by shared probe count, as in the production rule."""
    if member.ngrams.size == 0:
        return np.empty(0, dtype=np.int32)
    # The index contains every n-gram from every prepared document.
    position = np.searchsorted(index.values, member.ngrams)

    counts = index.counts[position]
    usable = counts <= params.maximum_posting_length
    if np.any(usable):
        position, counts = position[usable], counts[usable]
    if position.size > params.probe_ngrams:
        rarest = np.argpartition(counts, params.probe_ngrams)[: params.probe_ngrams]
        position, counts = position[rarest], counts[rarest]

    total = int(counts.sum())
    offsets = np.repeat(index.starts[position], counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    owners = index.owners[offsets + within]

    # Count only the probed documents to prevent a cluster-sized allocation
    # for every member.
    distinct, shared = np.unique(owners, return_counts=True)
    ahead = rank[distinct] < rank[member.index]
    distinct, shared = distinct[ahead], shared[ahead]
    if distinct.size == 0:
        return np.empty(0, dtype=np.int32)
    if distinct.size > params.maximum_candidates:
        strongest = np.argpartition(-shared, params.maximum_candidates - 1)[: params.maximum_candidates]
        distinct, shared = distinct[strongest], shared[strongest]
    return distinct[np.argsort(-shared)].astype(np.int32)


def find_duplicates(
    documents: Sequence[ClusterDocument],
    params: ClusterDedupParams,
) -> list[Removal]:
    """Find members that meet the containment threshold against an earlier survivor.

    Representatives have at least as many characters as their members.
    Input order determines the representative for equal lengths.
    Removed documents cannot act as representatives.
    """
    prepared = prepare(documents, params)
    order = sorted(range(len(prepared)), key=lambda index: (-prepared[index].chars, index))
    rank = np.empty(len(prepared), dtype=np.int64)
    rank[order] = np.arange(len(order))
    ngram_index = _build_index(prepared) if len(prepared) > params.exact_scan_maximum else None

    removed = np.zeros(len(prepared), dtype=bool)
    removals: list[Removal] = []
    token_cache: dict[int, frozenset[str]] = {}
    for position, member in enumerate(order):
        comparisons = 0
        member_prepared = prepared[member]
        if member_prepared.ngrams.size == 0:
            continue
        candidates = (
            order[:position]
            if ngram_index is None
            else _index_candidates(member_prepared, ngram_index, rank, params).tolist()
        )
        for representative in candidates:
            if removed[representative]:
                continue
            other = prepared[representative]
            comparisons += 1
            shared = _overlap(member_prepared.ngrams, other.ngrams)
            containment = shared / member_prepared.ngrams.size
            if containment < params.minimum_containment:
                continue
            novel_tokens = novel_token_count(member_prepared, other, token_cache)
            union = member_prepared.ngrams.size + other.ngrams.size - shared
            removed[member] = True
            removals.append(
                Removal(
                    member_index=member,
                    representative_index=representative,
                    containment=containment,
                    jaccard=shared / union if union else 1.0,
                    novel_tokens=novel_tokens,
                    comparisons=comparisons,
                )
            )
            break
    return removals
