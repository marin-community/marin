# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-split embedding heads and token-mass selection for quality experiments."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class LabelSplit(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    AUDIT = "audit"


def label_split(duplicate_group: str, split_seed: int) -> LabelSplit:
    """Assign a duplicate group to a fixed 80/10/10 split, independent of the head."""
    value = int.from_bytes(hashlib.sha256(f"{split_seed}:{duplicate_group}".encode()).digest()[:8], "big") % 10
    return LabelSplit.TRAIN if value < 8 else LabelSplit.DEVELOPMENT if value == 8 else LabelSplit.AUDIT


@dataclass(frozen=True)
class LabelledEmbedding:
    source: str
    document_id: str
    duplicate_group: str
    embedding: tuple[float, ...]
    label: float


class QualityScorer(Protocol):
    """Score a batch of embeddings after head fitting."""

    def scores(self, embeddings: NDArray) -> NDArray[np.float64]: ...


class QualityHeadConfig(Protocol):
    """Fit on training rows; identity names the implementation revision and every fit parameter."""

    @property
    def identity(self) -> Mapping[str, str | int | float | bool]: ...

    def fit(self, rows: Sequence[LabelledEmbedding]) -> QualityScorer: ...


@dataclass(frozen=True)
class RidgeHead:
    coefficients: tuple[float, ...]
    intercept: float
    regularization: float

    def scores(self, embeddings: NDArray) -> NDArray[np.float64]:
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.coefficients) or not np.isfinite(values).all():
            raise ValueError("embedding dimensions or values do not match the quality head")
        return values @ np.asarray(self.coefficients) + self.intercept


@dataclass(frozen=True)
class RidgeHeadConfig:
    regularization: float

    @property
    def identity(self) -> Mapping[str, str | int | float | bool]:
        return {"implementation": "ridge", "revision": "ridge-v1", "regularization": self.regularization}

    def fit(self, rows: Sequence[LabelledEmbedding]) -> RidgeHead:
        if not math.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("ridge regularization must be finite and positive")
        if not rows:
            raise ValueError("quality head training rows must not be empty")
        x = np.asarray([row.embedding for row in rows], dtype=np.float64)
        y = np.asarray([row.label for row in rows], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training embeddings and labels must be finite")
        mean_x, mean_y = x.mean(axis=0), float(y.mean())
        centered_x, centered_y = x - mean_x, y - mean_y
        gram = centered_x.T @ centered_x / len(rows)
        gram.flat[:: len(mean_x) + 1] += self.regularization
        coefficients = np.linalg.solve(gram, centered_x.T @ centered_y / len(rows))
        return RidgeHead(tuple(coefficients.tolist()), mean_y - float(mean_x @ coefficients), self.regularization)


@dataclass(frozen=True)
class LabelMetrics:
    documents: int
    mean_squared_error: float
    source_mean_squared_error: dict[str, float]


def split_labelled_embeddings(
    rows: Sequence[LabelledEmbedding], *, split_seed: int
) -> tuple[list[LabelledEmbedding], list[LabelledEmbedding]]:
    """Split labels by duplicate group and keep audit rows out of the returned partitions."""
    keys = [(row.source, row.document_id) for row in rows]
    if len(set(keys)) != len(keys) or any(not row.duplicate_group for row in rows):
        raise ValueError("label rows require unique source/document keys and duplicate groups")
    train = [row for row in rows if label_split(row.duplicate_group, split_seed) is LabelSplit.TRAIN]
    development = [row for row in rows if label_split(row.duplicate_group, split_seed) is LabelSplit.DEVELOPMENT]
    if not train or not development:
        raise ValueError("the frozen labels must contain training and development groups")
    try:
        embeddings = np.asarray([row.embedding for row in train], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("training embeddings must have one fixed dimension") from exc
    labels = np.asarray([row.label for row in train], dtype=np.float64)
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] == 0
        or not np.isfinite(embeddings).all()
        or not np.isfinite(labels).all()
    ):
        raise ValueError("training embeddings and labels must be finite")
    return train, development


def development_metrics(scorer: QualityScorer, rows: Sequence[LabelledEmbedding]) -> LabelMetrics:
    """Measure one scorer on development rows, grouped by source."""
    try:
        x = np.asarray([row.embedding for row in rows], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("development embeddings must have one fixed dimension") from exc
    y = np.asarray([row.label for row in rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("development embeddings and labels must be finite")
    predictions = score_embeddings(scorer, x)
    errors = (predictions - y) ** 2
    if not np.isfinite(errors).all():
        raise ValueError("development scores and labels must be finite")
    by_source: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, errors, strict=True):
        by_source[row.source].append(float(error))
    return LabelMetrics(
        len(rows), float(errors.mean()), {source: float(np.mean(values)) for source, values in by_source.items()}
    )


def quality_head_identity(head: QualityHeadConfig) -> dict[str, str | int | float | bool]:
    """Return the JSON-safe identity that selects a quality-head implementation and config."""
    identity = dict(head.identity)
    if not isinstance(identity.get("implementation"), str) or not isinstance(identity.get("revision"), str):
        raise ValueError("quality head identity requires implementation and revision strings")
    try:
        json.dumps(identity, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("quality head identity must contain finite JSON values") from exc
    return identity


def score_embeddings(scorer: QualityScorer, embeddings: NDArray) -> NDArray[np.float64]:
    """Require one finite score for each finite embedding row."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("scoring embeddings must be a finite matrix")
    scores = np.asarray(scorer.scores(values), dtype=np.float64)
    if scores.shape != (len(values),) or not np.isfinite(scores).all():
        raise ValueError("quality scorer must return one finite score per embedding")
    return scores


@dataclass(frozen=True)
class PoolDocument:
    source: str
    document_id: str
    duplicate_group: str
    token_count: int
    quality_bin: str
    content_type: str
    language: str


@dataclass(frozen=True)
class PoolRequirements:
    """Predeclared coverage limits for a production-weighted pool."""

    source_token_shares: dict[str, float]
    source_share_tolerance: float
    quality_bins: tuple[str, ...]
    min_documents_per_quality_bin: int
    min_duplicate_groups: int
    max_duplicate_token_share: float


@dataclass(frozen=True)
class PoolAudit:
    documents: int
    tokens: int
    duplicate_groups: int
    effective_token_documents: float
    largest_duplicate_token_share: float
    source_token_shares: dict[str, float]
    quality_bin_documents: dict[str, int]
    content_type_tokens: dict[str, int]
    language_tokens: dict[str, int]


def audit_pool(
    documents: Sequence[PoolDocument], *, requirements: PoolRequirements, labelled_groups: set[str]
) -> PoolAudit:
    """Reject label leakage, insufficient coverage, and concentrated duplicate groups."""
    if not documents:
        raise ValueError("quality pool is empty")
    keys = [(row.source, row.document_id) for row in documents]
    if len(set(keys)) != len(keys):
        raise ValueError("quality pool contains repeated source/document keys")
    if any(row.token_count <= 0 or not row.duplicate_group for row in documents):
        raise ValueError("quality pool requires positive token counts and duplicate groups")
    if labelled_groups & {row.duplicate_group for row in documents}:
        raise ValueError("quality pool overlaps the labelled duplicate groups")
    weights = requirements.source_token_shares
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("pool source shares must be finite and positive")
    if not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("pool source shares must sum to one")
    if (
        len(requirements.quality_bins) < 2
        or len(set(requirements.quality_bins)) != len(requirements.quality_bins)
        or requirements.min_documents_per_quality_bin < 1
    ):
        raise ValueError("quality coverage requires at least two distinct populated bins")
    if not 0 <= requirements.source_share_tolerance < 1 or not 0 < requirements.max_duplicate_token_share <= 1:
        raise ValueError("pool coverage tolerances are invalid")
    source_tokens: Counter[str] = Counter()
    group_tokens: Counter[str] = Counter()
    content_tokens: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    for row in documents:
        source_tokens[row.source] += row.token_count
        group_tokens[row.duplicate_group] += row.token_count
        content_tokens[row.content_type] += row.token_count
        language_tokens[row.language] += row.token_count
        quality_counts[row.quality_bin] += 1
    tokens = sum(source_tokens.values())
    shares = {source: count / tokens for source, count in source_tokens.items()}
    if set(shares) != set(weights) or any(
        abs(shares[source] - weights[source]) > requirements.source_share_tolerance for source in weights
    ):
        raise ValueError(f"pool source shares differ from the frozen recipe: {shares}")
    if any(quality_counts[name] < requirements.min_documents_per_quality_bin for name in requirements.quality_bins):
        raise ValueError(f"quality pool lacks the declared score coverage: {dict(quality_counts)}")
    largest_share = max(group_tokens.values()) / tokens
    if len(group_tokens) < requirements.min_duplicate_groups or largest_share > requirements.max_duplicate_token_share:
        raise ValueError("quality pool contains insufficient independent duplicate groups")
    return PoolAudit(
        len(documents),
        tokens,
        len(group_tokens),
        tokens**2 / sum(count**2 for count in group_tokens.values()),
        largest_share,
        shares,
        dict(quality_counts),
        dict(content_tokens),
        dict(language_tokens),
    )


@dataclass(frozen=True)
class QualitySelection:
    indices: tuple[int, ...]
    requested_tokens: int
    selected_tokens: int
    cutoff: float
    cutoff_ties: int
    source_token_shares: dict[str, float]


def select_top_tokens(
    documents: Sequence[PoolDocument], scores: Sequence[float], *, fraction: float, tie_seed: int
) -> QualitySelection:
    """Select whole documents by score until the requested token fraction is reached."""
    if not documents or len(documents) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("selection requires one finite score per pool document")
    if not 0 < fraction <= 1:
        raise ValueError("selection fraction must be greater than zero and at most one")
    target = math.ceil(sum(row.token_count for row in documents) * fraction)
    tie_keys = [hashlib.sha256(f"{tie_seed}:{row.source}:{row.document_id}".encode()).digest() for row in documents]
    order = sorted(range(len(documents)), key=lambda i: (-scores[i], tie_keys[i]))
    selected: list[int] = []
    selected_tokens = 0
    source_tokens: Counter[str] = Counter()
    for index in order:
        selected.append(index)
        selected_tokens += documents[index].token_count
        source_tokens[documents[index].source] += documents[index].token_count
        if selected_tokens >= target:
            break
    cutoff = float(scores[selected[-1]])
    return QualitySelection(
        tuple(selected),
        target,
        selected_tokens,
        cutoff,
        sum(score == cutoff for score in scores),
        {source: count / selected_tokens for source, count in source_tokens.items()},
    )


def selection_token_overlap(
    documents: Sequence[PoolDocument], candidate: QualitySelection, incumbent: QualitySelection
) -> float:
    """Return token intersection divided by token union on one frozen pool."""
    left, right = set(candidate.indices), set(incumbent.indices)
    shared = sum(documents[i].token_count for i in left & right)
    union = sum(documents[i].token_count for i in left | right)
    return shared / union
