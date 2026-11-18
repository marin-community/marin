# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select bounded token views and score text or jagged replay caches."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.jagged_array import JaggedArrayStore
from levanter.tokenizers import MarinTokenizer
from rigging.filesystem.storage_path import prefix_join

from marin.datakit.low_value.features import document_features
from marin.datakit.low_value.model import LowValueModel, OperatingPoint, load_model_tokenizer
from marin.processing.tokenize._core import text_preprocessor


@dataclass(frozen=True)
class DocumentScorer:
    """A model together with its verified tokenizer backend."""

    model: LowValueModel
    tokenizer: MarinTokenizer


def load_scorer(model: LowValueModel) -> DocumentScorer:
    """Bind the model to the exact tokenizer files named in its pinned metadata."""
    return DocumentScorer(model, load_model_tokenizer(model))


@dataclass(frozen=True)
class DocumentViews:
    tokens: int
    intervals: tuple[tuple[int, int], ...]
    windows: tuple[str, ...]


@dataclass(frozen=True)
class DocumentScore:
    id: str
    tokens: int
    low_value_score: float
    is_low_value: bool


def window_intervals(tokens: int, window_tokens: int) -> tuple[tuple[int, int], ...]:
    """Merge overlapping head, middle and tail views without repeating tokens."""
    starts = sorted({0, max(0, (tokens - window_tokens) // 2), max(0, tokens - window_tokens)})
    intervals: list[tuple[int, int]] = []
    for start in starts:
        end = min(tokens, start + window_tokens)
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))
    return tuple(intervals)


def text_views(text: str, tokenizer: MarinTokenizer, window_tokens: int) -> DocumentViews:
    """Tokenize text once, then select the same views used by cache scoring."""
    processor = text_preprocessor(TextLmDatasetFormat(), tokenizer)
    ids = processor([{"text": text}])[0]["input_ids"]
    intervals = window_intervals(len(ids), window_tokens)
    windows = tuple(tokenizer.decode(ids[start:end], skip_special_tokens=False) for start, end in intervals)
    return DocumentViews(len(ids), intervals, windows)


def cache_views(store: JaggedArrayStore, index: int, tokenizer: MarinTokenizer, window_tokens: int) -> DocumentViews:
    """Read offsets and bounded windows without materializing a giant cache row."""
    start = int(store.offsets[index].read().result()) if index else 0
    end = int(store.offsets[index + 1].read().result())
    intervals = window_intervals(end - start, window_tokens)
    windows = tuple(
        tokenizer.decode(
            np.asarray(store.data[start + left : start + right].read().result()).tolist(),
            skip_special_tokens=False,
        )
        for left, right in intervals
    )
    return DocumentViews(end - start, intervals, windows)


def score_views(documents: Sequence[DocumentViews], model: LowValueModel) -> np.ndarray:
    """Score ordered document views using the structural model contract."""
    if not documents:
        return np.empty(0, dtype=np.float64)
    structural = np.stack([document_features(document.windows, document.tokens) for document in documents])
    return model.predict(structural)


def score_texts(texts: Sequence[str], scorer: DocumentScorer) -> np.ndarray:
    """Score text using the model's verified tokenizer and window policy."""
    documents = [text_views(text, scorer.tokenizer, scorer.model.window_tokens) for text in texts]
    return score_views(documents, scorer.model)


def score_token_cache(
    cache_dir: str,
    *,
    scorer: DocumentScorer,
    operating_point: OperatingPoint,
    indices: Sequence[int],
    batch_size: int = 128,
) -> Iterator[DocumentScore]:
    """Score selected cache rows, emitting their original row indices as IDs.

    The cache's IDs must use the model's tokenizer vocabulary. Historical
    cache ledgers do not pin tokenizer bytes, so callers must establish that
    provenance before selecting a cache; no caller-supplied digest proves it.
    """
    store = JaggedArrayStore.open(prefix_join(cache_dir, "input_ids"), mode="r", dtype=np.int32)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        documents = [cache_views(store, index, scorer.tokenizer, scorer.model.window_tokens) for index in batch]
        scores = score_views(documents, scorer.model)
        for index, document, score, drop in zip(batch, documents, scores, operating_point.drop(scores), strict=True):
            yield DocumentScore(str(index), document.tokens, float(score), bool(drop))
