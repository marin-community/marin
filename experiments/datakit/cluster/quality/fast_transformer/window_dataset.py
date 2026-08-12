# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the window-level training set for the scaled quality-scorer retrain.

The scale-up campaign moved labeling from document prefixes to begin/middle/end
512-gemma-token windows (:mod:`bme_windows`). Training examples come from three
places:

* the scale-up window labels — newly mined documents' begin/middle/end windows
  plus middle/end top-ups of the 88k campaign's long documents, each row
  carrying the exact window text the grader saw;
* the 88k label campaign — every label survives as a BEGIN-window grade whose
  training text is the document's first-512-gemma-token window, cut here with
  the committed :func:`bme_windows.doc_windows` cutter rather than the old
  10,500-char excerpt, so every example shares the window contract;
* the two labels-x-embeddings joins — the stored 1024-d harrier document
  embedding that feeds the fusion super token and the MoE router. Windows share
  their document's embedding; middle/end top-ups of 88k documents read theirs
  from the 88k join.

Split discipline: the 88k campaign's seed-0 id-set holdout is the eval set,
unchanged. Every window of a holdout doc id is excluded from training —
including the scale-up's middle/end top-ups of those documents.
"""

import logging
import zlib
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath

from experiments.datakit.cluster.quality.fast_transformer.bme_windows import (
    WINDOW_TOKENS,
    doc_windows,
    encode_documents,
)

logger = logging.getLogger(__name__)

WINDOW_LABELS = "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_scaleup/labels/windows.parquet"
SCALEUP_JOINED = (
    "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_scaleup-x-harrier-oss-v1-0.6b-50m-text-v1"
)
WINDOW_COLUMNS = ["id", "source", "window", "text", "quality", "score_normalized"]
# Characters that always cover 512 gemma tokens (the campaign's graded
# 10,500-char prefixes never fell short); a document whose capped prefix still
# tokenizes under the window is re-tokenized in full.
BEGIN_CHAR_CAP = 20 * WINDOW_TOKENS

BEGIN = "begin"


@dataclass(frozen=True)
class WindowExamples:
    """The assembled window-level training rows, embedding-aligned."""

    ids: list[str]
    texts: list[str]
    sources: np.ndarray
    positions: np.ndarray  # begin / middle / end
    targets: np.ndarray  # float32 normalized quality
    embeddings: list  # int8 rows, one per example (shared across a doc's windows)


@dataclass(frozen=True)
class AssemblyStats:
    """What the assembly kept and why it dropped the rest (the leakage check)."""

    legacy_begin: int
    scaleup_windows: int
    holdout_excluded: int
    missing_embedding: int
    begin_regrades_skipped: int


def load_window_labels(path: str = WINDOW_LABELS) -> dict[str, list]:
    """Scale-up window label rows, deduplicated by ``(id, window)`` keeping the first.

    A duplicated key whose rows disagree on text would mean one corpus id naming
    two different documents; the id-keyed embedding join cannot tell those
    apart, so every window of such an id is dropped. Keys duplicated with
    identical text (the same window graded more than once) collapse to the
    first row, the same discipline as the 88k loader's id dedup.
    """
    with StoragePath(path).open("rb") as fh:
        table = pq.read_table(fh, columns=WINDOW_COLUMNS)
    rows = {c: table.column(c).to_pylist() for c in WINDOW_COLUMNS}
    first_text: dict[tuple[str, str], str] = {}
    ambiguous: set[str] = set()
    for doc_id, window, text in zip(rows["id"], rows["window"], rows["text"], strict=True):
        key = (doc_id, window)
        if key in first_text and first_text[key] != text:
            ambiguous.add(doc_id)
        first_text.setdefault(key, text)
    out: dict[str, list] = {c: [] for c in WINDOW_COLUMNS}
    seen: set[tuple[str, str]] = set()
    dupes = 0
    for i, (doc_id, window) in enumerate(zip(rows["id"], rows["window"], strict=True)):
        key = (doc_id, window)
        if doc_id in ambiguous or key in seen:
            dupes += 1
            continue
        seen.add(key)
        for c in WINDOW_COLUMNS:
            out[c].append(rows[c][i])
    logger.info(
        "window labels: %d rows kept of %d (%d duplicate/ambiguous dropped, %d ambiguous ids)",
        len(out["id"]),
        len(rows["id"]),
        dupes,
        len(ambiguous),
    )
    return out


def begin_window_texts(texts: list[str]) -> list[str]:
    """The first-512-gemma-token window of each document, cut with the bme cutter.

    Tokenizes a capped prefix to bound the gigatoken pass (the cut needs only
    the first ``WINDOW_TOKENS`` ids); a document whose capped prefix tokenizes
    under one window while more text exists is re-tokenized in full, so the
    cap can never shorten a window.
    """
    capped = [t[:BEGIN_CHAR_CAP] for t in texts]
    ids = encode_documents(capped)
    starved = [i for i, row in enumerate(ids) if len(row) < WINDOW_TOKENS and len(texts[i]) > BEGIN_CHAR_CAP]
    if starved:
        logger.info("begin windows: re-tokenizing %d documents whose capped prefix fell short", len(starved))
        full = encode_documents([texts[i] for i in starved])
        for i, row in zip(starved, full, strict=True):
            ids[i] = row
    return [doc_windows(row[:WINDOW_TOKENS])[0].text for row in ids]


def assemble_training_windows(
    windows: dict[str, list],
    legacy: dict[str, list],
    legacy_begin_texts: list[str],
    scaleup: dict[str, list],
    holdout_ids: set[str],
) -> tuple[WindowExamples, AssemblyStats]:
    """Merge the three sources into one embedding-aligned window training set.

    ``legacy``/``scaleup`` are ``load_joined`` outputs; ``legacy_begin_texts``
    is row-aligned with ``legacy`` (:func:`begin_window_texts` of its text
    column). Every window whose id is in ``holdout_ids`` is excluded, whatever
    its source. Scale-up begin rows for 88k ids would re-grade windows the 88k
    label already covers, so they are skipped (none exist today; the guard
    keeps a future re-run from double counting). Windows whose document has no
    stored embedding are dropped and counted.
    """
    embedding_by_id: dict[str, object] = dict(zip(scaleup["id"], scaleup["embedding"], strict=True))
    embedding_by_id.update(zip(legacy["id"], legacy["embedding"], strict=True))
    legacy_ids = set(legacy["id"])

    ids: list[str] = []
    texts: list[str] = []
    sources: list[str] = []
    positions: list[str] = []
    targets: list[float] = []
    embeddings: list = []
    holdout_excluded = 0
    missing_embedding = 0
    begin_regrades = 0

    for i, doc_id in enumerate(legacy["id"]):
        if doc_id in holdout_ids:
            holdout_excluded += 1
            continue
        ids.append(doc_id)
        texts.append(legacy_begin_texts[i])
        sources.append(legacy["glm52_source"][i])
        positions.append(BEGIN)
        targets.append(legacy["glm52_score_normalized"][i])
        embeddings.append(legacy["embedding"][i])
    legacy_begin = len(ids)

    for i, doc_id in enumerate(windows["id"]):
        if doc_id in holdout_ids:
            holdout_excluded += 1
            continue
        if windows["window"][i] == BEGIN and doc_id in legacy_ids:
            begin_regrades += 1
            continue
        embedding = embedding_by_id.get(doc_id)
        if embedding is None:
            missing_embedding += 1
            continue
        ids.append(doc_id)
        texts.append(windows["text"][i])
        sources.append(windows["source"][i])
        positions.append(windows["window"][i])
        targets.append(windows["score_normalized"][i])
        embeddings.append(embedding)

    stats = AssemblyStats(
        legacy_begin=legacy_begin,
        scaleup_windows=len(ids) - legacy_begin,
        holdout_excluded=holdout_excluded,
        missing_embedding=missing_embedding,
        begin_regrades_skipped=begin_regrades,
    )
    logger.info(
        "assembled %d window examples over %d docs (%d legacy begin + %d scale-up windows; "
        "%d holdout windows excluded, %d missing embeddings, %d begin re-grades skipped)",
        len(ids),
        len(set(ids)),
        stats.legacy_begin,
        stats.scaleup_windows,
        stats.holdout_excluded,
        stats.missing_embedding,
        stats.begin_regrades_skipped,
    )
    examples = WindowExamples(
        ids=ids,
        texts=texts,
        sources=np.array(sources),
        positions=np.array(positions),
        targets=np.array(targets, dtype=np.float32),
        embeddings=embeddings,
    )
    return examples, stats


def subsample_mask(ids: list[str], every: int) -> np.ndarray:
    """Deterministic 1-in-``every`` doc subsample by id hash (smoke runs).

    Hashing the id keeps a document's rows aligned across the label tables and
    joins, so a subsampled assembly still finds its embeddings.
    """
    if every <= 1:
        return np.ones(len(ids), dtype=bool)
    return np.array([zlib.crc32(i.encode()) % every == 0 for i in ids])
