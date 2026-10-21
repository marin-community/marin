# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Document scorers that check the quality-track selection and judge end to end.

Most scorers move the selected data far from a random selection in a known direction.
``dclm_fasttext_web`` is a known good classifier, the positive control for the judge.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from functools import cache

import fasttext
import numpy as np
from marin.execution.artifact import read_artifact
from numpy.typing import NDArray
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.fast_track.contracts import FAST_TRACK_ROOT
from experiments.grug.fast_track.downstream_eval import collapse_whitespace
from experiments.grug.fast_track.quality import (
    DocumentQualityScorer,
    EmbeddingHeadScorer,
    QualityScoringBatch,
    RidgeHead,
)
from experiments.grug.fast_track.quality_labels import FittedRidgeQualityHead

RIDGE_HEAD_PATH = f"{FAST_TRACK_ROOT}/quality-head/8fc570a91617dbf22066/2026.10.14"
PROBE_TABLES_ROOT = f"{FAST_TRACK_ROOT}/probe-tables"
# Token-weighted quantiles per source of the character fraction that whitespace collapsing removes,
# from 1/24 of the raw-pool shards. Sources missing from the sample use the pooled ``__all__`` table.
FORMAT_SOURCE_QUOTA_TABLE_PATH = f"{PROBE_TABLES_ROOT}/format_excess_source_quota_table.json"
# Token-weighted quantiles per source of minus the UTF-8 byte length, from the same sample.
SHORTEST_SOURCE_QUOTA_TABLE_PATH = f"{PROBE_TABLES_ROOT}/shortest_source_quota_table.json"
POOLED_TABLE = "__all__"
# Token-weighted ridge-score quantiles per source, from the ridge scored pool of the 10.7B-token pool.
RIDGE_SOURCE_QUOTA_TABLE_PATH = f"{PROBE_TABLES_ROOT}/ridge_source_quota_table.json"


def inverted_ridge() -> DocumentQualityScorer:
    """Select the documents that the ridge head scores lowest."""
    head = read_artifact(RIDGE_HEAD_PATH, FittedRidgeQualityHead).head
    return EmbeddingHeadScorer(
        RidgeHead(tuple(-value for value in head.coefficients), -head.intercept, head.regularization)
    )


@dataclass(frozen=True)
class _TextLengthScorer:
    sign: float

    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]:
        return self.sign * np.asarray([len(text.encode("utf-8")) for text in batch.texts], dtype=np.float64)


def longest_documents() -> DocumentQualityScorer:
    return _TextLengthScorer(1.0)


def shortest_documents() -> DocumentQualityScorer:
    return _TextLengthScorer(-1.0)


# Raw (not synthetic, not translated) web sources in the corpus pool, by Nemotron-CC quality tier.
NEMOTRON_CC_TIERS = {
    "nemotron_cc_v2/high_quality": 3,
    "nemotron_cc_v2_1/high_quality": 3,
    "nemotron_cc_v2/medium_high_quality": 2,
    "nemotron_cc_v2_1/medium_high_quality": 2,
    "nemotron_cc_v2/medium_quality": 1,
    "nemotron_cc_v2_1/medium_quality": 1,
}
UNTIERED_WEB_SOURCES = frozenset({"hplt_v3", "common-crawl-focus-2026-22"})
RAW_WEB_SOURCES = frozenset(NEMOTRON_CC_TIERS) | UNTIERED_WEB_SOURCES
DIRTY_WEB_SOURCES = frozenset({"nemotron_cc_v2/medium_quality", "nemotron_cc_v2_1/medium_quality", "hplt_v3"})


@dataclass(frozen=True)
class _SourceSetScorer:
    """Score ``sign`` for a document from one of ``sources`` and 0 otherwise; the tie hash orders each group."""

    sources: frozenset[str]
    sign: float

    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]:
        return self.sign * np.asarray([source in self.sources for source in batch.sources], dtype=np.float64)


def web_only() -> DocumentQualityScorer:
    return _SourceSetScorer(RAW_WEB_SOURCES, 1.0)


def dirty_web() -> DocumentQualityScorer:
    return _SourceSetScorer(DIRTY_WEB_SOURCES, 1.0)


def no_web() -> DocumentQualityScorer:
    return _SourceSetScorer(RAW_WEB_SOURCES, -1.0)


@dataclass(frozen=True)
class _NemotronTierScorer:
    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]:
        return np.asarray([NEMOTRON_CC_TIERS.get(source, 0) for source in batch.sources], dtype=np.float64)


def nemotron_tier() -> DocumentQualityScorer:
    """Prefer the high Nemotron-CC web tier, then lower tiers, then everything else."""
    return _NemotronTierScorer()


@dataclass(frozen=True)
class _SourceQuotaScorer:
    """Replace a score by minus the token fraction of its source that scores higher.

    The top ``B`` of a ``10 * B`` prefix then takes about the top tenth of each source's tokens, so the
    selection keeps the source mix of the raw prefix and changes only which documents each source gives.
    """

    inner: DocumentQualityScorer
    tables: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]

    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]:
        raw = self.inner.scores(batch)
        fractions = np.empty(len(raw), dtype=np.float64)
        for index, (source, document_id, score) in enumerate(zip(batch.sources, batch.document_ids, raw, strict=True)):
            descending_scores, fraction_above = self.tables[source if source in self.tables else POOLED_TABLE]
            ascending = -descending_scores
            first = np.searchsorted(ascending, -score, side="left")
            last = np.searchsorted(ascending, -score, side="right")
            if last - first > 1:
                # Tied scores span a token-fraction interval. A stable document hash spreads them across
                # it, so a source whose documents all tie still gives its quota and no more.
                low, high = fraction_above[first], fraction_above[last - 1]
                fractions[index] = low + (high - low) * _unit_hash(source, document_id)
            else:
                fractions[index] = np.interp(-score, ascending, fraction_above)
        return -fractions


def _unit_hash(source: str, document_id: str) -> float:
    digest = hashlib.sha256(f"quota:{source}:{document_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _quota_tables(path: str) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    with StoragePath(path).open("r") as stream:
        raw_tables = json.load(stream)
    return {
        source: (np.asarray(table["scores"], dtype=np.float64), np.asarray(table["fractions"], dtype=np.float64))
        for source, table in raw_tables.items()
    }


def ridge_source_quota() -> DocumentQualityScorer:
    """Rank by the ridge head within each source, and keep the source mix of the raw prefix."""
    head = read_artifact(RIDGE_HEAD_PATH, FittedRidgeQualityHead).head
    return _SourceQuotaScorer(EmbeddingHeadScorer(head), _quota_tables(RIDGE_SOURCE_QUOTA_TABLE_PATH))


@dataclass(frozen=True)
class _ExcessWhitespaceScorer:
    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]:
        return np.asarray(
            [(len(text) - len(collapse_whitespace(text))) / max(len(text), 1) for text in batch.texts], dtype=np.float64
        )


def format_heavy() -> DocumentQualityScorer:
    """Within each source, prefer the documents with the most whitespace that collapsing would remove."""
    return _SourceQuotaScorer(_ExcessWhitespaceScorer(), _quota_tables(FORMAT_SOURCE_QUOTA_TABLE_PATH))


def shortest_source_quota() -> DocumentQualityScorer:
    """Within each source, prefer the shortest documents, and keep the source mix of the raw prefix."""
    return _SourceQuotaScorer(_TextLengthScorer(-1.0), _quota_tables(SHORTEST_SOURCE_QUOTA_TABLE_PATH))


# The public DCLM fastText classifier (OpenHermes 2.5 and ELI5 against RefinedWeb), copied from
# mlfoundations/fasttext-oh-eli5 at revision cd8b714a90f2.
DCLM_FASTTEXT_MODEL_PATH = (
    f"{FAST_TRACK_ROOT}/models/dclm-fasttext-oh-eli5/cd8b714a90f2/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
)
DCLM_HIGH_QUALITY_LABEL = "__label__hq"


@dataclass(frozen=True)
class _WebOnlyScorer:
    """Score the raw web documents with ``inner`` and every other document 0."""

    inner: DocumentQualityScorer

    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]:
        web = np.flatnonzero([source in RAW_WEB_SOURCES for source in batch.sources])
        out = np.zeros(len(batch.sources), dtype=np.float64)
        if len(web):
            web_batch = QualityScoringBatch(
                [batch.texts[i] for i in web], [batch.document_ids[i] for i in web], [batch.sources[i] for i in web]
            )
            out[web] = self.inner.scores(web_batch)
        return out


@dataclass(frozen=True)
class _FastTextScorer:
    """DCLM's high-quality probability: the document joined to one line, scored by the classifier."""

    model: fasttext.FastText._FastText

    def scores(self, batch: QualityScoringBatch) -> NDArray[np.float64]:
        out = np.empty(len(batch.texts), dtype=np.float64)
        for index, text in enumerate(batch.texts):
            # Score one line at a time: the list form of `predict` returns wrong probabilities under NumPy 2.
            predictions = self.model.f.predict(" ".join(text.strip().splitlines()) + "\n", 2, 0.0, "replace")
            out[index] = {label: probability for probability, label in predictions}[DCLM_HIGH_QUALITY_LABEL]
        return out


@cache
def _dclm_fasttext_model() -> fasttext.FastText._FastText:
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(DCLM_FASTTEXT_MODEL_PATH))
    if not os.path.exists(local_path):
        # Worker processes can share a disk, so each one renames its own completed copy into place.
        partial_path = f"{local_path}.{os.getpid()}.partial"
        StoragePath(DCLM_FASTTEXT_MODEL_PATH).download_to(partial_path)
        os.replace(partial_path, local_path)
    return fasttext.load_model(local_path)


def dclm_fasttext_web() -> DocumentQualityScorer:
    """Rank the web documents by the DCLM fastText classifier, below them every other document.

    The classifier's probability is above 0, so a selection that web tokens can fill takes only web.
    """
    return _WebOnlyScorer(_FastTextScorer(_dclm_fasttext_model()))
