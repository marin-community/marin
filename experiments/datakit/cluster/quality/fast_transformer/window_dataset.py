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
import re
import zlib
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath

from experiments.datakit.cluster.quality.fast_transformer.bme_windows import (
    GEOMETRY_512,
    doc_windows,
    encode_documents,
)

logger = logging.getLogger(__name__)

WINDOW_LABELS = "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_scaleup/labels/windows.parquet"
SCALEUP_JOINED = (
    "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_scaleup-x-harrier-oss-v1-0.6b-50m-text-v1"
)
# The bme2048 regrade campaign's grades: 2048-token windows, the excerpt marker
# applied to cut begin windows, drawn with the seed-0 holdout ids excluded.
BME2048_WINDOW_LABELS = (
    "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_bme2048/labels/windows.parquet"
)
WINDOW_COLUMNS = ["id", "source", "window", "text", "quality", "score_normalized", "valid", "why"]
# The window's token offsets say whether it ends before its document does, which
# is what separates a cut window from a whole short document.
BME2048_COLUMNS = [*WINDOW_COLUMNS, "token_end", "doc_tokens"]

# A begin window is cut at exactly one window with no excerpt marker, so the
# grader reads a mid-expression stop as document damage and marks the window
# invalid — the same harness artifact the 88k campaign's excerpt marker fixed
# (93% of invalid new-doc begin code windows cite it). These grades score the
# cut, not the document, and can be dropped before training.
CUT_WHY_PATTERN = re.compile(
    r"truncat|cut[- ]?off|cuts off|abrupt|mid-token|mid-sentence|mid-expression|mid-statement|incomplete", re.I
)
# Characters that always cover 512 gemma tokens (the campaign's graded
# 10,500-char prefixes never fell short); a document whose capped prefix still
# tokenizes under the window is re-tokenized in full.
BEGIN_CHAR_CAP = 20 * GEOMETRY_512.window_tokens

BEGIN = "begin"
# The positions whose grades can contradict a begin-window verdict.
SIBLING_POSITIONS = ("middle", "end")


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


def load_window_labels(path: str = WINDOW_LABELS, columns: list[str] | None = None) -> dict[str, list]:
    """Window label rows, deduplicated by ``(id, window)`` keeping the first.

    A duplicated key whose rows disagree on text would mean one corpus id naming
    two different documents; the id-keyed embedding join cannot tell those
    apart, so every window of such an id is dropped. Keys duplicated with
    identical text (the same window graded more than once) collapse to the
    first row, the same discipline as the 88k loader's id dedup.
    """
    columns = columns or WINDOW_COLUMNS
    with StoragePath(path).open("rb") as fh:
        table = pq.read_table(fh, columns=columns)
    rows = {c: table.column(c).to_pylist() for c in columns}
    first_text: dict[tuple[str, str], str] = {}
    ambiguous: set[str] = set()
    for doc_id, window, text in zip(rows["id"], rows["window"], rows["text"], strict=True):
        key = (doc_id, window)
        if key in first_text and first_text[key] != text:
            ambiguous.add(doc_id)
        first_text.setdefault(key, text)
    out: dict[str, list] = {c: [] for c in columns}
    seen: set[tuple[str, str]] = set()
    dupes = 0
    for i, (doc_id, window) in enumerate(zip(rows["id"], rows["window"], strict=True)):
        key = (doc_id, window)
        if doc_id in ambiguous or key in seen:
            dupes += 1
            continue
        seen.add(key)
        for c in columns:
            out[c].append(rows[c][i])
    logger.info(
        "window labels: %d rows kept of %d (%d duplicate/ambiguous dropped, %d ambiguous ids)",
        len(out["id"]),
        len(rows["id"]),
        dupes,
        len(ambiguous),
    )
    return out


def drop_cut_artifact_grades(windows: dict[str, list]) -> dict[str, list]:
    """Remove invalid grades whose ``why`` blames the window cut, not the text.

    Only rows the grader marked invalid *and* whose rationale matches
    :data:`CUT_WHY_PATTERN` are dropped — a quality-1 verdict for the harness
    cutting mid-expression is a label for the harness. Valid rows that mention
    the cut keep their (mildly depressed) grade: they cannot be separated from
    real flaws without relabeling.
    """
    keep = [
        i
        for i, (valid, why) in enumerate(zip(windows["valid"], windows["why"], strict=True))
        if valid or not CUT_WHY_PATTERN.search(why or "")
    ]
    dropped = len(windows["id"]) - len(keep)
    logger.info("cut-artifact filter: dropped %d invalid windows whose rationale blames the cut", dropped)
    return _keep_rows(windows, keep)


def _keep_rows(windows: dict[str, list], keep: list[int]) -> dict[str, list]:
    return {c: [windows[c][i] for i in keep] for c in windows}


def drop_cross_window_disagreements(windows: dict[str, list], min_sibling_quality: float) -> dict[str, list]:
    """Drop invalid begin grades that the document's own middle/end grades contradict.

    A whole-document judgment cannot say whether an ``invalid`` verdict is about
    the document or about the window the harness cut it to. A document graded at
    three positions can: a begin window called invalid whose middle *and* end
    windows are valid and average at least ``min_sibling_quality`` is a document
    the grader read as fine everywhere it was not cut. Those begin grades score
    the cut rather than the text.

    Only begin grades are dropped, and only where both siblings exist — a
    document with one graded window offers no such evidence, so its verdict
    stands.
    """
    grade_by_key = {
        (doc_id, window): (valid, quality)
        for doc_id, window, valid, quality in zip(
            windows["id"], windows["window"], windows["valid"], windows["quality"], strict=True
        )
    }

    def sibling_mean_quality(doc_id: str) -> float | None:
        """Mean quality of the document's middle and end grades, or None unless both exist and are valid."""
        total = 0.0
        for position in SIBLING_POSITIONS:
            grade = grade_by_key.get((doc_id, position))
            if grade is None or not grade[0]:
                return None
            total += grade[1]
        return total / len(SIBLING_POSITIONS)

    keep: list[int] = []
    for i, (doc_id, window, valid) in enumerate(zip(windows["id"], windows["window"], windows["valid"], strict=True)):
        if window != BEGIN or valid:
            keep.append(i)
            continue
        siblings = sibling_mean_quality(doc_id)
        if siblings is None or siblings < min_sibling_quality:
            keep.append(i)
    logger.info(
        "cross-window filter: dropped %d invalid begin grades contradicted by valid siblings "
        "averaging quality >= %.1f",
        len(windows["id"]) - len(keep),
        min_sibling_quality,
    )
    return _keep_rows(windows, keep)


def drop_cut_window_invalids(windows: dict[str, list]) -> dict[str, list]:
    """Drop every invalid grade on a window that ends before its document does.

    The aggressive counterpart to :func:`drop_cross_window_disagreements`: it
    needs no siblings, so it reaches the cut begin windows of documents too
    short for three, but it also discards invalid verdicts that were about the
    text. Whether the trade is worth it is what the arm measures.
    """
    keep = [
        i
        for i, (valid, token_end, doc_tokens) in enumerate(
            zip(windows["valid"], windows["token_end"], windows["doc_tokens"], strict=True)
        )
        if valid or token_end >= doc_tokens
    ]
    logger.info(
        "cut-invalid filter: dropped %d invalid grades on windows that end mid-document", len(windows["id"]) - len(keep)
    )
    return _keep_rows(windows, keep)


def begin_window_texts(texts: list[str]) -> list[str]:
    """The first-512-gemma-token window of each document, cut with the bme cutter.

    Tokenizes a capped prefix to bound the gigatoken pass (the cut needs only
    the first ``GEOMETRY_512.window_tokens`` ids); a document whose capped prefix tokenizes
    under one window while more text exists is re-tokenized in full, so the
    cap can never shorten a window.
    """
    capped = [t[:BEGIN_CHAR_CAP] for t in texts]
    ids = encode_documents(capped)
    starved = [
        i for i, row in enumerate(ids) if len(row) < GEOMETRY_512.window_tokens and len(texts[i]) > BEGIN_CHAR_CAP
    ]
    if starved:
        logger.info("begin windows: re-tokenizing %d documents whose capped prefix fell short", len(starved))
        full = encode_documents([texts[i] for i in starved])
        for i, row in zip(starved, full, strict=True):
            ids[i] = row
    return [doc_windows(row[: GEOMETRY_512.window_tokens], GEOMETRY_512)[0].text for row in ids]


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
