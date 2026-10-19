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


@dataclass(frozen=True)
class QualityScoringBatch:
    """Document inputs with optional normalized Harrier vectors."""

    texts: Sequence[str]
    document_ids: Sequence[str]
    sources: Sequence[str]
    embeddings: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if not len(self.texts) == len(self.document_ids) == len(self.sources):
            raise ValueError("quality texts, document IDs, and sources must have equal length")
        if self.embeddings is not None and len(self.embeddings) != len(self.texts):
            raise ValueError("quality embeddings must have one row per document")


class DocumentQualityScorer(Protocol):
    """Score a batch of documents from text, optional embeddings, or both."""

    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class RidgeHead:
    coefficients: tuple[float, ...]
    intercept: float
    regularization: float

    def scores(self, embeddings: NDArray) -> NDArray[np.float64]:
        return np.asarray(embeddings, dtype=np.float64) @ np.asarray(self.coefficients) + self.intercept


@dataclass(frozen=True)
class RidgeHeadConfig:
    regularization: float

    def fit(self, rows: Sequence[LabelledEmbedding]) -> RidgeHead:
        if not math.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("ridge regularization must be finite and positive")
        x = np.asarray([row.embedding for row in rows], dtype=np.float64)
        y = np.asarray([row.label for row in rows], dtype=np.float64)
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
class LabelMetrics:
    documents: int
    mean_squared_error: float
    source_mean_squared_error: dict[str, float]


@dataclass(frozen=True)
class FittedQualityHead:
    """A fitted ridge head and its development-only fit report."""

    head: RidgeHead
    training_documents: int
    development: LabelMetrics


@dataclass(frozen=True)
class EmbeddingHeadScorer:
    """Adapt a fitted ridge head to the document scorer API."""

    head: RidgeHead

    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]:
        if batch.embeddings is None:
            raise ValueError("embedding head requires prepared Harrier embeddings")
        return self.head.scores(batch.embeddings)


def fit_quality_head(rows: Sequence[LabelledEmbedding], *, head: RidgeHeadConfig, split_seed: int) -> FittedQualityHead:
    """Fit on frozen training embeddings and report development metrics, without audit labels."""
    training, development = split_labelled_embeddings(rows, split_seed=split_seed)
    fitted = head.fit(training)
    return FittedQualityHead(fitted, len(training), development_metrics(fitted, development))


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
    return train, development


def development_metrics(head: RidgeHead, rows: Sequence[LabelledEmbedding]) -> LabelMetrics:
    """Measure one head on development rows, grouped by source."""
    predictions = head.scores(np.asarray([row.embedding for row in rows], dtype=np.float64))
    errors = (predictions - np.asarray([row.label for row in rows], dtype=np.float64)) ** 2
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
