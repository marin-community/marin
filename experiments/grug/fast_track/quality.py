# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-split embedding heads and text scorer interfaces for quality experiments."""

import hashlib
import json
import math
from collections import defaultdict
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
    embedding: Sequence[float] | NDArray[np.float32]
    label: float


class QualityScorer(Protocol):
    """Score a batch of embeddings after head fitting."""

    def scores(self, embeddings: NDArray) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class QualityScoringBatch:
    """Document inputs with optional normalized Harrier vectors."""

    texts: Sequence[str]
    document_ids: Sequence[str]
    embeddings: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.document_ids):
            raise ValueError("quality texts and document IDs must have equal length")
        if any(not isinstance(text, str) for text in self.texts):
            raise ValueError("quality scoring texts must be strings")
        if any(not isinstance(document_id, str) for document_id in self.document_ids):
            raise ValueError("quality scoring document IDs must be strings")
        if self.embeddings is not None:
            embeddings = np.asarray(self.embeddings)
            if embeddings.ndim != 2 or len(embeddings) != len(self.texts) or not np.isfinite(embeddings).all():
                raise ValueError("quality embeddings must be a finite matrix with one row per document")


class DocumentQualityScorer(Protocol):
    """Score a batch of documents from text, optional embeddings, or both."""

    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]: ...


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
        if len(rows) < x.shape[1]:
            dual_gram = centered_x @ centered_x.T
            dual_gram.flat[:: len(rows) + 1] += len(rows) * self.regularization
            coefficients = centered_x.T @ np.linalg.solve(dual_gram, centered_y)
        else:
            gram = centered_x.T @ centered_x / len(rows)
            gram.flat[:: len(mean_x) + 1] += self.regularization
            coefficients = np.linalg.solve(gram, centered_x.T @ centered_y / len(rows))
        return RidgeHead(tuple(coefficients.tolist()), mean_y - float(mean_x @ coefficients), self.regularization)


@dataclass(frozen=True)
class FittedQualityHead:
    """A fitted head and its development-only fit report."""

    scorer: QualityScorer
    identity: dict[str, str | int | float | bool]
    training_documents: int
    development: "LabelMetrics"


@dataclass(frozen=True)
class EmbeddingHeadScorer:
    """Adapt a fitted embedding head to the document scorer API."""

    scorer: QualityScorer

    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]:
        if batch.embeddings is None:
            raise ValueError("embedding head requires prepared Harrier embeddings")
        return score_embeddings(self.scorer, batch.embeddings)


@dataclass(frozen=True)
class LabelMetrics:
    documents: int
    mean_squared_error: float
    source_mean_squared_error: dict[str, float]


def fit_quality_head(
    rows: Sequence[LabelledEmbedding], *, head: QualityHeadConfig, split_seed: int
) -> FittedQualityHead:
    """Fit on frozen training embeddings and report development metrics, without audit labels."""
    training, development = split_labelled_embeddings(rows, split_seed=split_seed)
    scorer = head.fit(training)
    metrics = development_metrics(scorer, development)
    return FittedQualityHead(scorer, quality_classifier_identity(head.identity), len(training), metrics)


def frozen_label_duplicate_groups(rows: Sequence[LabelledEmbedding]) -> frozenset[str]:
    """Return all label duplicate groups for exclusion from every scored pool."""
    if any(not row.duplicate_group for row in rows):
        raise ValueError("frozen labels require duplicate groups")
    return frozenset(row.duplicate_group for row in rows)


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


def quality_classifier_identity(
    values: Mapping[str, str | int | float | bool],
) -> dict[str, str | int | float | bool]:
    """Validate and copy the JSON-safe classifier implementation, revision, and parameters."""
    identity = dict(values)
    if not isinstance(identity.get("implementation"), str) or not isinstance(identity.get("revision"), str):
        raise ValueError("classifier identity requires implementation and revision strings")
    try:
        json.dumps(identity, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("classifier identity must contain finite JSON values") from exc
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
