# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontaminate normalized data against eval sources.

Reads datakit-normalized Parquet (``id``, ``text``), builds an in-memory
bloom filter from the eval text, and emits a co-partitioned Parquet
attributes dataset marking which records overlap with eval text.

Schema of the emitted Parquet attributes (flat Datakit attribute convention,
consumable by :func:`marin.processing.classification.consolidate.consolidate`):

    id                       : string         — matches source document id
    partition_id             : int            — source partition index (from sorted file order)
    contaminated             : bool           — one paragraph meets the overlap and evidence thresholds
    max_overlap              : float          — highest paragraph overlap fraction in [0, 1]
    matched_hashes           : list[uint64]   — bloom-hit hashes that caused the mark

Build also emits ``<output>/_bloom/eval_hash_index.parquet`` with columns
``hash: uint64, eval_id: string`` (flattened, one row per (hash, eval_id) pair).
Join ``matched_hashes`` against this sidecar to attribute
contamination back to specific eval records.

Output follows the normalize job's layout: main attributes land in
``<output>/outputs/main/`` and (when ``flagged_sample_size`` > 0) a sample of
flagged docs with text lands in ``<output>/outputs/flagged_sample/``. The main
output is co-partitioned with the source — one ``part-NNNNN-of-MMMMM.parquet``
per input partition, preserving the source filenames so consolidate can
sorted-merge-join without a shuffle.

The bloom can also be built once and shared across many corpus marks via
:func:`build_eval_bloom` (single-source) and :func:`merge_eval_blooms`
(combine pre-built per-eval blooms). Pass the resulting directory to
:func:`decon_to_parquet` as ``prebuilt_bloom_dir`` to skip the inline build.
"""

import hashlib
import json
import logging
import os
import random
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any, Protocol

import dupekit
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.input_file import InputFileSpec
from zephyr.readers import SUPPORTED_EXTENSIONS, compute_parquet_splits, load_file
from zephyr.writers import write_parquet_file

from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)


class _FeatureMembership(Protocol):
    def __contains__(self, value: int, /) -> bool: ...


# Bump when the ngram feature-extraction policy changes. Both the bloom build and
# the corpus mark fold this into their step hash_attrs, so a policy change
# re-addresses cached blooms/marks instead of silently reusing incompatible
# features. v2 added the no-alphabetic-character ngram filter (marin#6852 cluster D).
# v3 added an exact feature for short alphabetic paragraphs with at least three
# tokens. This keeps required eval records matchable without restoring the
# one-token punctuation and label collisions removed in v2.
# v4 limits that exact feature to a complete short record. It does not index a
# short paragraph from a longer record because common labels can then mark an
# unrelated complete document. With the default blank-line policy, a record-level
# n-gram fallback keeps records matchable when all paragraphs are short.
FEATURE_FILTER_VERSION = 4
DECON_ATTRIBUTES_VERSION = 5
BLOOM_BUILD_VERSION = 2
# v3 increases sampling fan-out so large sources do not hold the pipeline on a
# small number of long-running shards.
DROP_SET_BUILD_VERSION = 3
MIN_SHORT_EXACT_TOKENS = 3
DEFAULT_PARAGRAPH_DELIMITER = "\n\n"
DROP_SET_SAMPLE_SHARD_BYTES = 64 * 1024 * 1024
DROP_SET_SHARDS_PER_SOURCE = 128


@dataclass(frozen=True)
class NGramConfig:
    """Word-ngram matching parameters.

    Attributes:
        ngram_length: Size of each ngram in whitespace-split word tokens.
        stride: Step between successive ngrams. 0 = contiguous (every position).
        overlap_threshold: Minimum fraction of paragraph ngrams that must hit
            the filter for the paragraph to count as contaminated.
        min_matched_features: Minimum number of distinct matched features that
            a paragraph must contain. A complete document with exactly one
            feature can still match.
        paragraph_delimiter: String the text is split on to form paragraphs (the
            unit the overlap fraction is computed over). Defaults to ``"\n\n"``, a
            blank-line-delimited block, so ngrams span single line breaks — this
            both dilutes isolated-line coincidences (precision) and lets short-line
            / inline-embedded eval text be matched (recall). ``"\n"`` instead treats
            each line as its own paragraph. See marin#6852.
    """

    ngram_length: int = 13
    stride: int = 0
    overlap_threshold: float = 0.5
    min_matched_features: int = 2
    paragraph_delimiter: str = DEFAULT_PARAGRAPH_DELIMITER

    def __post_init__(self) -> None:
        if self.min_matched_features < 1:
            raise ValueError("min_matched_features must be at least 1")


class DeconAttributes(BaseModel):
    """Outcome of :func:`decon_to_parquet`: a co-partitioned attributes dataset.

    Persisted as the step's ``.artifact`` so downstream consumers can locate
    the output without re-running the pipeline.

    Attributes:
        main_output_dir: Directory of ``part-NNNNN-of-MMMMM.parquet`` attribute
            files (``<output>/outputs/main``, mirroring the normalize job).
        flagged_output_dir: Directory of the mark-time flagged-doc sample sidecar
            (``<output>/outputs/flagged_sample``); empty when no sample was taken.
        num_partitions: Number of output partitions; matches the source.
        eval_hash_index_path: Path to the ``hash → eval_id`` sidecar Parquet.
            Join the causal per-record ``matched_hashes`` column against this
            to attribute contamination to specific eval records.
        counters: Aggregated zephyr counters from the marking pipeline.
    """

    version: str = f"v{DECON_ATTRIBUTES_VERSION}"
    main_output_dir: DatakitArtifactPath
    flagged_output_dir: DatakitArtifactPath
    num_partitions: int
    eval_hash_index_path: DatakitArtifactPath
    counters: dict[str, int | float]


_BLOOM_FILENAME = "filter.bin"
_INDEX_FILENAME = "eval_hash_index.parquet"
_GLOBAL_DROP_SET_DIRECTORY = "_global"


def bloom_paths(bloom_dir: str) -> tuple[str, str]:
    """Return ``(bloom_path, eval_hash_index_path)`` for a bloom directory.

    A "bloom directory" is any directory under which a bloom + sidecar live at
    ``<bloom_dir>/_bloom/filter.bin`` and ``<bloom_dir>/_bloom/eval_hash_index.parquet``.
    This is the layout written by :func:`build_eval_bloom`,
    :func:`merge_eval_blooms`, and the inline-build path of
    :func:`decon_to_parquet`.
    """
    return (
        os.path.join(bloom_dir, "_bloom", _BLOOM_FILENAME),
        os.path.join(bloom_dir, "_bloom", _INDEX_FILENAME),
    )


class EvalBloom(BaseModel):
    """Artifact describing a pre-built eval bloom filter + hash index sidecar.

    Persisted as the step's ``.artifact`` so downstream consumers can locate
    the bloom without re-running the build. Pass the producing step's
    ``output_path`` to :func:`decon_to_parquet`'s ``prebuilt_bloom_dir`` to
    skip the inline build.

    Attributes:
        bloom_dir: Directory containing ``_bloom/filter.bin`` and
            ``_bloom/eval_hash_index.parquet``. Equal to the producing step's
            ``output_path``.
        bloom_path, eval_hash_index_path: Resolved leaf paths (redundant with
            ``bloom_dir`` + the layout convention; included for convenience).
        estimated_doc_count, false_positive_rate: Sizing parameters the bloom
            was built with. Per-eval blooms intended for merging must share
            both values — ``dupekit.Bloom.update`` requires identical sizing.
        n_eval_records: Total eval records that contributed at least one
            feature. For a merged bloom, the sum across inputs.
    """

    version: str = "v1"
    bloom_dir: DatakitArtifactPath
    bloom_path: DatakitArtifactPath
    eval_hash_index_path: DatakitArtifactPath
    estimated_doc_count: int
    false_positive_rate: float
    n_eval_records: int = 0


def _bloom_hash(x: str) -> int:
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "big")


def _has_alpha(ngram: str) -> bool:
    """True if *ngram* contains any alphabetic character.

    Cluster-D filter (marin#6852): a 13-gram with no letters — pure numeric
    sequences (``1 , 2 , 3 …``), punctuation runs, form-field/index boilerplate —
    carries no distinctive contamination signal but collides with number-list
    eval items (HLE / MMLU-Pro math). Skipping these on *both* the bloom and the
    mark side (both go through :func:`_extract_ngrams`) keeps the overlap
    denominator consistent. Trade-off: drops recall on purely-numeric
    contamination, which is acceptable — a bare number run is never a leak we can
    attribute anyway.
    """
    return any(c.isalpha() for c in ngram)


def _extract_token_ngrams(tokens: list[str], n: int, stride: int) -> Iterator[str]:
    token_has_alpha = bytearray(_has_alpha(token) for token in tokens)
    for i in range(0, len(tokens) - n + 1, stride + 1):
        if any(token_has_alpha[i : i + n]):
            yield " ".join(tokens[i : i + n])


def _extract_ngrams(text: str, n: int, stride: int) -> Iterator[str]:
    yield from _extract_token_ngrams(text.split(), n, stride)


def _short_exact_feature_from_tokens(tokens: list[str], n: int) -> str | None:
    if MIN_SHORT_EXACT_TOKENS <= len(tokens) < n and any(_has_alpha(token) for token in tokens):
        return " ".join(tokens)
    return None


def _short_exact_feature(text: str, n: int) -> str | None:
    """Return one normalized feature for guarded short exact matching."""
    return _short_exact_feature_from_tokens(text.split(), n)


def _extract_paragraph_features(text: str, n: int, stride: int) -> Iterator[str]:
    """Yield n-grams, or one guarded exact feature for a short paragraph."""
    tokens = text.split()
    short_exact = _short_exact_feature_from_tokens(tokens, n)
    if short_exact is not None:
        yield short_exact
        return
    yield from _extract_token_ngrams(tokens, n, stride)


def _extract_features(text: str, ngram: NGramConfig | None) -> Iterator[str]:
    """Yield matchable features from a complete record.

    N-gram mode emits n-grams from each paragraph. It does not emit exact
    features for short paragraphs in a longer record. With the default
    blank-line policy, the complete record emits n-grams when all paragraphs
    are short. A complete short record emits one guarded exact feature.
    """
    delimiter = ngram.paragraph_delimiter if ngram is not None else "\n"
    paragraphs = [paragraph for paragraph in text.split(delimiter) if paragraph]
    if ngram is None:
        for para in paragraphs:
            yield para
        return

    short_exact = _short_exact_feature(text, ngram.ngram_length)
    if short_exact is not None:
        yield short_exact
        return

    yielded = False
    for para in paragraphs:
        for feature in _extract_ngrams(para, ngram.ngram_length, ngram.stride):
            yielded = True
            yield feature
    if not yielded and ngram.paragraph_delimiter == DEFAULT_PARAGRAPH_DELIMITER:
        yield from _extract_ngrams(text, ngram.ngram_length, ngram.stride)


def _paragraph_overlap_and_matches(
    paragraph: str,
    bf: _FeatureMembership,
    ngram: NGramConfig | None,
    drop_hashes: frozenset[int] = frozenset(),
) -> tuple[float, list[int]]:
    """Return ``(overlap_score, matched_hashes)`` for a single paragraph.

    Score is 0.0 or 1.0 in exact-paragraph mode and the fraction of bloom-hit
    ngrams otherwise. *matched_hashes* is the list of ngram hashes that hit
    the bloom (in iteration order, with duplicates if the same ngram repeats).

    *drop_hashes* are removed from *both* the numerator and denominator.
    Corpus-common boilerplate carries no contamination signal, so an
    all-boilerplate paragraph collapses to zero ngrams and scores 0. Remaining
    distinctive leak ngrams stay matchable; a leak made entirely of dropped
    ngrams is intentionally suppressed.

    Alphabetic paragraphs with at least three but fewer than ``ngram_length``
    tokens use one exact feature. Shorter and non-alphabetic paragraphs return
    ``(0.0, [])``.
    """
    score, matched, _has_features, _feature_count, _has_ngram_features = _paragraph_overlap_matches_and_presence(
        paragraph, bf, ngram, drop_hashes
    )
    return score, matched


def _paragraph_overlap_matches_and_presence(
    paragraph: str,
    bf: _FeatureMembership,
    ngram: NGramConfig | None,
    drop_hashes: frozenset[int] = frozenset(),
) -> tuple[float, list[int], bool, int, bool]:
    """Return overlap details, feature counts, and n-gram presence."""
    if ngram is None:
        h = _bloom_hash(paragraph)
        if h in drop_hashes:
            return 0.0, [], True, 0, False
        return (1.0, [h], True, 1, False) if h in bf else (0.0, [], True, 1, False)

    has_features = False
    has_ngram_features = False
    feature_count = 0
    matched: list[int] = []
    tokens = paragraph.split()
    short_exact = _short_exact_feature_from_tokens(tokens, ngram.ngram_length)
    features: Iterator[str]
    if short_exact is not None:
        features = iter((short_exact,))
    else:
        features = _extract_token_ngrams(tokens, ngram.ngram_length, ngram.stride)
    for feature in features:
        has_features = True
        has_ngram_features = short_exact is None
        hash_value = _bloom_hash(feature)
        if hash_value in drop_hashes:
            continue
        feature_count += 1
        if hash_value in bf:
            matched.append(hash_value)
    if feature_count == 0:
        return 0.0, [], has_features, feature_count, has_ngram_features
    return len(matched) / feature_count, matched, has_features, feature_count, has_ngram_features


def _document_overlap_and_matches(
    text: str,
    bf: _FeatureMembership,
    ngram: NGramConfig | None,
    drop_hashes: frozenset[int] = frozenset(),
) -> tuple[float, list[int]]:
    """Return the highest overlap and the hashes that cause a document mark."""
    minimum = ngram.min_matched_features if ngram is not None else 1
    max_score, matches = _document_overlap_matches_by_minimum(text, bf, ngram, (minimum,), drop_hashes)
    return max_score, matches[minimum]


def _document_overlap_matches_by_minimum(
    text: str,
    bf: _FeatureMembership,
    ngram: NGramConfig | None,
    minimums: tuple[int, ...],
    drop_hashes: frozenset[int] = frozenset(),
) -> tuple[float, dict[int, list[int]]]:
    """Score a document once and return causal hashes for each evidence minimum."""
    if not minimums or any(minimum < 1 for minimum in minimums):
        raise ValueError("minimums must contain positive integers")
    threshold = ngram.overlap_threshold if ngram is not None else 0.0
    delimiter = ngram.paragraph_delimiter if ngram is not None else "\n"
    paragraphs = [paragraph for paragraph in text.split(delimiter) if paragraph]
    max_score = 0.0
    matched = {minimum: set() for minimum in minimums}
    has_paragraph_ngrams = False

    for paragraph in paragraphs:
        score, hits, _has_features, feature_count, paragraph_has_ngrams = _paragraph_overlap_matches_and_presence(
            paragraph, bf, ngram, drop_hashes
        )
        if ngram is not None and paragraph_has_ngrams:
            has_paragraph_ngrams = True
        max_score = max(max_score, score)
        if not hits:
            continue
        if ngram is None:
            for hashes in matched.values():
                hashes.update(hits)
            continue

        complete_single_feature_document = len(paragraphs) == 1 and feature_count == 1 and score == 1.0
        distinct_hits = len(set(hits))
        for minimum, hashes in matched.items():
            if score >= threshold and (distinct_hits >= minimum or complete_single_feature_document):
                hashes.update(hits)

    if ngram is not None and not has_paragraph_ngrams:
        use_record_fallback = (
            ngram.paragraph_delimiter == DEFAULT_PARAGRAPH_DELIMITER
            or _short_exact_feature(text, ngram.ngram_length) is not None
        )
        if use_record_fallback:
            score, hits, _has_features, feature_count, _has_ngrams = _paragraph_overlap_matches_and_presence(
                text, bf, ngram, drop_hashes
            )
            max_score = max(max_score, score)
            distinct_hits = len(set(hits))
            complete_single_feature_document = feature_count == 1 and score == 1.0
            for minimum, hashes in matched.items():
                if score >= threshold and (distinct_hits >= minimum or complete_single_feature_document):
                    hashes.update(hits)

    return max_score, {minimum: sorted(hashes) for minimum, hashes in matched.items()}


def _record_feature_status(text: str, ngram: NGramConfig | None) -> tuple[bool, bool]:
    """Return feature presence and exact self-match status for one record."""
    feature_hashes = {_bloom_hash(feature) for feature in _extract_features(text, ngram)}
    if not feature_hashes:
        return False, False
    _score, matched_hashes = _document_overlap_and_matches(text, feature_hashes, ngram)
    return True, bool(matched_hashes)


def _is_hidden_dir(root: str, resolved: str) -> bool:
    """Return True if any path segment between *resolved* and *root* starts with a dot.

    Skips ``.metrics/``, ``.executor_info/``, and other hidden sidecar directories
    that show up routinely in normalize / executor outputs.
    """
    rel = os.path.relpath(root, resolved)
    if rel == ".":
        return False
    return any(p.startswith(".") for p in rel.split(os.sep))


def _discover_eval_files(eval_paths: list[str], exclude_dir_names: frozenset[str] = frozenset()) -> Iterator[str]:
    """Walk all *eval_paths* recursively and yield zephyr-readable data files.

    Filters by ``zephyr.readers.SUPPORTED_EXTENSIONS`` so common sidecars
    (``README``, ``_SUCCESS``, ``provenance.json``, ``.executor_info``, …)
    that live alongside eval data don't kill the whole decon step when
    ``load_file`` later rejects their extension. Mirrors ``normalize._discover_files``.

    *exclude_dir_names* skips any file whose immediate parent directory name is
    in the set (the eval-corpus layout is ``<root>/<split>/<task>/<file>``, so the
    task name is the parent dir). This lets a caller drop specific eval tasks from
    the bloom *at read time*, so an already-materialized eval corpus that still
    contains those task dirs is excluded without regenerating it.
    """
    for source in eval_paths:
        fs, resolved = url_to_fs(source)
        protocol = source.split("://")[0] if "://" in source else ""
        if fs.isfile(resolved):
            filename = os.path.basename(resolved)
            parent_name = os.path.basename(os.path.dirname(resolved).rstrip("/"))
            if (
                parent_name not in exclude_dir_names
                and not filename.startswith(".")
                and filename.endswith(SUPPORTED_EXTENSIONS)
            ):
                yield source
            continue
        for root, _dirs, files in fs.walk(resolved):
            if _is_hidden_dir(root, resolved):
                continue
            if os.path.basename(root.rstrip("/")) in exclude_dir_names:
                continue
            for fname in files:
                if fname.startswith(".") or not fname.endswith(SUPPORTED_EXTENSIONS):
                    continue
                full = os.path.join(root, fname)
                yield f"{protocol}://{full}" if protocol else full


_INDEX_SCHEMA = pa.schema([pa.field("hash", pa.uint64()), pa.field("eval_id", pa.string())])


@dataclass
class _EvalIndexStats:
    n_records: int = 0
    n_index_rows: int = 0


@dataclass(frozen=True)
class _EvalIndexPart:
    shard_idx: int
    path: str
    n_records: int
    n_index_rows: int


def _emit_eval_index_rows(
    eval_paths: list[str],
    text_field: str,
    ngram: NGramConfig | None,
    stats: _EvalIndexStats,
) -> Iterator[dict[str, Any]]:
    for path in eval_paths:
        for idx, record in enumerate(load_file(path)):
            text = record.get(text_field)
            if not text:
                continue
            eval_id = str(record.get("id") or f"{path}::{idx}")
            seen_in_record: set[int] = set()
            for feature in _extract_features(str(text), ngram):
                hash_value = _bloom_hash(feature)
                if hash_value in seen_in_record:
                    continue
                seen_in_record.add(hash_value)
                stats.n_index_rows += 1
                yield {"hash": hash_value, "eval_id": eval_id}
            if seen_in_record:
                stats.n_records += 1


def _build_filter(
    eval_paths: list[str],
    bloom_path: str,
    index_path: str,
    text_field: str,
    ngram: NGramConfig | None,
    estimated_doc_count: int,
    false_positive_rate: float,
    exclude_dir_names: frozenset[str] = frozenset(),
) -> int:
    """Build a bloom filter and a streaming hash → eval_id sidecar.

    The hash index is written incrementally via :func:`write_parquet_file` so
    build-time memory stays bounded to the writer's buffer (~64 MB) plus a
    per-record dedup set (~10 KB). The eval suite size does not bound memory.

    Sidecar schema: ``hash: uint64, eval_id: string`` (flattened — one row per
    ``(hash, eval_id)`` pair, with the hash deduped *within* a single eval
    record). Inter-record duplicates are allowed; joins handle them naturally.

    This local path supports inline builds in :func:`decon_to_parquet`.
    :func:`build_eval_bloom` uses Zephyr for reusable Bloom artifacts.
    """
    bf = dupekit.Bloom(estimated_doc_count, false_positive_rate)
    stats = _EvalIndexStats()

    def emit_index_rows() -> Iterator[dict[str, Any]]:
        files = list(_discover_eval_files(eval_paths, exclude_dir_names))
        for row in _emit_eval_index_rows(files, text_field, ngram, stats):
            bf.add(row["hash"])
            yield row

    # Stream the index parquet; this iteration also fills the bloom.
    idx_dir = os.path.dirname(index_path)
    if idx_dir:
        StoragePath(idx_dir).mkdirs()
    write_parquet_file(emit_index_rows(), output_path=index_path, schema=_INDEX_SCHEMA)

    # Persist the populated bloom.
    bloom_dir = os.path.dirname(bloom_path)
    if bloom_dir:
        StoragePath(bloom_dir).mkdirs()
    StoragePath(bloom_path).write_bytes(bf.save_bytes())

    logger.info(
        "decon: built bloom + index from %d eval records (%d index rows) → bloom=%s, index=%s",
        stats.n_records,
        stats.n_index_rows,
        bloom_path,
        index_path,
    )
    return stats.n_records


def _build_eval_index_part(
    eval_paths: Iterator[str],
    shard: ShardInfo,
    *,
    parts_dir: str,
    text_field: str,
    ngram: NGramConfig | None,
) -> Iterator[_EvalIndexPart]:
    stats = _EvalIndexStats()
    path = prefix_join(parts_dir, f"part-{shard.shard_idx:05d}-of-{shard.total_shards:05d}.parquet")
    write_parquet_file(
        _emit_eval_index_rows(list(eval_paths), text_field, ngram, stats),
        output_path=path,
        schema=_INDEX_SCHEMA,
    )
    yield _EvalIndexPart(
        shard_idx=shard.shard_idx,
        path=path,
        n_records=stats.n_records,
        n_index_rows=stats.n_index_rows,
    )


def _merge_eval_index_parts(
    parts: list[_EvalIndexPart],
    *,
    bloom_path: str,
    index_path: str,
    estimated_doc_count: int,
    false_positive_rate: float,
) -> int:
    bf = dupekit.Bloom(estimated_doc_count, false_positive_rate)
    StoragePath(os.path.dirname(index_path)).mkdirs()
    with StoragePath(index_path).open("wb") as destination:
        with pq.ParquetWriter(destination, _INDEX_SCHEMA) as writer:
            for part in sorted(parts, key=lambda item: item.shard_idx):
                with StoragePath(part.path).open("rb") as source:
                    for batch in pq.ParquetFile(source).iter_batches():
                        for hash_value in batch.column("hash").to_pylist():
                            bf.add(hash_value)
                        writer.write_batch(batch)

    StoragePath(os.path.dirname(bloom_path)).mkdirs()
    StoragePath(bloom_path).write_bytes(bf.save_bytes())
    n_records = sum(part.n_records for part in parts)
    n_index_rows = sum(part.n_index_rows for part in parts)
    logger.info(
        "decon: merged %d index shards from %d eval records (%d index rows) → bloom=%s, index=%s",
        len(parts),
        n_records,
        n_index_rows,
        bloom_path,
        index_path,
    )
    return n_records


def _merge_eval_index_shard(
    parts: Iterator[_EvalIndexPart],
    _shard: ShardInfo,
    *,
    bloom_path: str,
    index_path: str,
    parts_dir: str,
    estimated_doc_count: int,
    false_positive_rate: float,
) -> Iterator[int]:
    n_records = _merge_eval_index_parts(
        list(parts),
        bloom_path=bloom_path,
        index_path=index_path,
        estimated_doc_count=estimated_doc_count,
        false_positive_rate=false_positive_rate,
    )
    StoragePath(parts_dir).rmtree()
    yield n_records


# Flat attribute columns keep all Datakit sidecars directly selectable by name.
_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("partition_id", pa.int64()),
        pa.field("contaminated", pa.bool_()),
        pa.field("max_overlap", pa.float64()),
        pa.field("matched_hashes", pa.list_(pa.uint64())),
    ]
)


_FLAGGED_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("max_overlap", pa.float64()),
        pa.field("matched_hashes", pa.list_(pa.uint64())),
    ]
)


def _make_marker(
    bloom_path: str,
    output_dir: str,
    text_field: str,
    ngram: NGramConfig | None,
    drop_hashes: frozenset[int] = frozenset(),
    flagged_sample_size: int = 0,
) -> Callable[[Iterator[str], ShardInfo], Iterator[dict[str, Any]]]:
    """Return a ``map_shard`` function that processes one input parquet → one output parquet.

    *drop_hashes* is the source's common-ngram set, excluded from every
    paragraph's overlap (see :func:`_paragraph_overlap_and_matches`).

    *flagged_sample_size* > 0 reservoir-samples that many contaminated docs per
    shard — with their text and matched hashes — into
    ``<output_dir>/_flagged/part-<shard>.parquet``. The mark already reads every
    doc, so this makes reports O(sample) instead of O(corpus): a viewer reads the
    small sidecar rather than rescanning the full attributes to find flags.
    """

    def mark_shard(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict[str, Any]]:
        # Load bloom once per shard.
        bf = dupekit.Bloom.load_bytes(StoragePath(bloom_path).read_bytes())
        reservoir: list[dict[str, Any]] = []
        n_flagged = 0
        rng = random.Random(shard.shard_idx)

        for input_path in paths:

            def rows_for(p: str) -> Iterator[dict[str, Any]]:
                nonlocal n_flagged
                for record in load_file(p):
                    text = str(record.get(text_field, "") or "")
                    max_score, matched = _document_overlap_and_matches(text, bf, ngram, drop_hashes)
                    contaminated = bool(matched)
                    counters.pipeline.update_counter("decon/contaminated" if contaminated else "decon/clean", 1)
                    if contaminated and flagged_sample_size:
                        n_flagged += 1
                        row = {
                            "id": record["id"],
                            "text": text,
                            "max_overlap": max_score,
                            "matched_hashes": matched,
                        }
                        if len(reservoir) < flagged_sample_size:
                            reservoir.append(row)
                        elif (j := rng.randint(0, n_flagged - 1)) < flagged_sample_size:
                            reservoir[j] = row
                    # Dataset.from_list yields one shard per item in input order, so shard.shard_idx
                    # matches the input's "part-NNNNN-of-NNNNN" partition number on a sorted file list.
                    yield {
                        "id": record["id"],
                        "partition_id": shard.shard_idx,
                        "contaminated": contaminated,
                        "max_overlap": max_score,
                        "matched_hashes": matched,
                    }

            # Follow the normalize job's output layout: main attributes under
            # outputs/main/, the flagged-doc sample under outputs/flagged_sample/,
            # co-partitioned by the source filename.
            shard_filename = os.path.basename(input_path)
            out_path = prefix_join(output_dir, f"outputs/main/{shard_filename}")
            result = write_parquet_file(rows_for(input_path), output_path=out_path, schema=_OUTPUT_SCHEMA)
            yield result

        if flagged_sample_size and reservoir:
            flagged_path = prefix_join(output_dir, f"outputs/flagged_sample/{shard_filename}")
            write_parquet_file(iter(reservoir), output_path=flagged_path, schema=_FLAGGED_SCHEMA)

    return mark_shard


def decon_to_parquet(
    *,
    normalized_data: NormalizedData,
    eval_data_sources: str | list[str] | None = None,
    prebuilt_bloom_dir: str | None = None,
    output_path: str,
    text_field: str = "text",
    ngram: NGramConfig | None = None,
    drop_set_dirs: list[str] | None = None,
    flagged_sample_size: int = 0,
    estimated_doc_count: int = 1_000_000,
    false_positive_rate: float = 1e-9,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    zephyr_context: ZephyrContext | None = None,
) -> DeconAttributes:
    """Mark records in *normalized_data* that overlap with eval text.

    Provide exactly one of:

    * ``eval_data_sources`` — paths to eval data. Builds the bloom inline
      under ``<output_path>/_bloom/`` (single-corpus pattern).
    * ``prebuilt_bloom_dir`` — directory produced by :func:`build_eval_bloom`
      or :func:`merge_eval_blooms`. Skips the build stage; the same bloom can
      be reused by many corpus marks (multi-corpus pattern, used by
      ``experiments/decontamination/all_sources_decon.py``).

    Args:
        normalized_data: Upstream :class:`NormalizedData` artifact. Reads from
            ``normalized_data.main_output_dir`` (the flat, co-partitioned
            Parquet directory produced by datakit normalize). Records must
            have ``id``, ``text``, and ``partition_id`` columns.
        eval_data_sources: Eval source directory or list of directories. Walked
            recursively for files with zephyr-readable extensions; sidecar/metadata
            files (e.g. ``README``, ``_SUCCESS``, ``provenance.json``, hidden dirs
            like ``.metrics/``) are skipped. Read once to build the bloom filter.
            Multiple sources are merged into one filter; per-eval-record
            attribution is preserved in the ``eval_hash_index`` sidecar. The
            attribution ``eval_id`` is ``record["id"]`` when present, else
            ``f"{full_path}::{idx}"`` (full path keeps fallback IDs unique across
            nested or multi-source eval directories that share file basenames).
            Mutually exclusive with ``prebuilt_bloom_dir``.
        prebuilt_bloom_dir: Directory containing ``_bloom/filter.bin`` and
            ``_bloom/eval_hash_index.parquet`` to reuse instead of building.
            Mutually exclusive with ``eval_data_sources``. ``ngram``,
            ``estimated_doc_count``, ``false_positive_rate`` are ignored for
            the bloom but still drive the mark stage (``ngram`` must match
            whatever was used at build time).
        output_path: Directory for co-partitioned Parquet attributes. One
            output file is written per input partition, preserving filenames.
        text_field: Text column name in both input and eval records.
        ngram: Word-ngram matching config. ``None`` = exact whole-paragraph match.
            ``ngram.overlap_threshold`` gates which paragraphs are marked
            contaminated, and ``ngram.min_matched_features`` rejects marks with
            too little distinct evidence. Exact-paragraph mode records any
            non-zero match.
        drop_set_dirs: Optional directories of corpus-common ngram hashes.
            Ngrams from every directory are excluded from each paragraph
            overlap. :func:`decon_step` passes the source-local and global
            outputs from :func:`all_source_drop_sets_step`.
        estimated_doc_count, false_positive_rate: Bloom sizing parameters; size
            for expected total *ngram* count across the eval suite (not record
            count). Defaults handle ~1M unique ngrams cleanly. Ignored when
            ``prebuilt_bloom_dir`` is set.
        worker_resources: Per-shard resource request for the marking pipeline.
            Defaults to 2 CPU / 4GB RAM.
        max_workers: Max Zephyr workers. Defaults to Zephyr's own default.

    Returns:
        :class:`DeconAttributes` describing the output dataset and counters.
    """
    if (eval_data_sources is None) == (prebuilt_bloom_dir is None):
        raise ValueError("provide exactly one of eval_data_sources or prebuilt_bloom_dir")

    input_path = normalized_data.main_output_dir
    files = sorted(str(m) for m in StoragePath(f"{input_path.rstrip('/')}/**/*.parquet").glob())
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {input_path}")
    num_partitions = len(files)
    logger.info("decon: %s → %s, %d input partitions", input_path, output_path, num_partitions)

    if prebuilt_bloom_dir is not None:
        bloom_path, index_path = bloom_paths(prebuilt_bloom_dir)
        logger.info("decon: reusing prebuilt bloom at %s", bloom_path)
    else:
        eval_paths = [eval_data_sources] if isinstance(eval_data_sources, str) else list(eval_data_sources)  # type: ignore[arg-type]
        if not eval_paths:
            raise ValueError("eval_data_sources must be non-empty")
        bloom_path, index_path = bloom_paths(output_path)
        _build_filter(
            eval_paths=eval_paths,
            bloom_path=bloom_path,
            index_path=index_path,
            text_field=text_field,
            ngram=ngram,
            estimated_doc_count=estimated_doc_count,
            false_positive_rate=false_positive_rate,
        )

    drop_hashes = _load_drop_sets(drop_set_dirs) if drop_set_dirs else frozenset()
    if drop_hashes:
        logger.info("decon: filtering %d corpus-common ngrams from %s", len(drop_hashes), drop_set_dirs)
    pipeline = Dataset.from_list(files).map_shard(
        _make_marker(bloom_path, output_path, text_field, ngram, drop_hashes, flagged_sample_size)
    )

    resources = worker_resources or ResourceConfig(cpu=2, ram="4g")
    ctx_kwargs: dict[str, Any] = {"name": "decon-mark", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = zephyr_context or ZephyrContext(**ctx_kwargs)
    outcome = ctx.execute(
        pipeline,
        map_task_resources=resources,
    )

    return DeconAttributes(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        flagged_output_dir=prefix_join(output_path, "outputs/flagged_sample"),
        num_partitions=num_partitions,
        eval_hash_index_path=index_path,
        counters=dict(outcome.counters),
    )


def build_eval_bloom(
    *,
    eval_data_sources: str | list[str],
    output_path: str,
    text_field: str = "text",
    ngram: NGramConfig | None = None,
    estimated_doc_count: int = 1_000_000,
    false_positive_rate: float = 1e-9,
    exclude_eval_dirs: frozenset[str] = frozenset(),
    required_eval_manifest_path: str | None = None,
    required_eval_corpus_version: str | None = None,
    required_eval_names: tuple[str, ...] = (),
    best_effort_eval_manifest_path: str | None = None,
    best_effort_eval_corpus_version: str | None = None,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    zephyr_context: ZephyrContext | None = None,
) -> EvalBloom:
    """Build a reusable bloom + hash-index sidecar from one or more eval sources.

    Writes ``<output_path>/_bloom/filter.bin`` and
    ``<output_path>/_bloom/eval_hash_index.parquet``. The resulting directory
    can be passed to :func:`decon_to_parquet`'s ``prebuilt_bloom_dir`` to scan
    many corpora against the same bloom without re-doing this work.

    For multi-eval suites where you want to cache per-eval results
    independently (so adding one new eval invalidates only one build), call
    this once per eval and then :func:`merge_eval_blooms` to combine.

    Args:
        eval_data_sources: One eval source or a list. Walked recursively for
            zephyr-readable files; sidecar/hidden files skipped. ``eval_id``
            attribution comes from the record's ``id`` field, falling back to
            ``"{full_path}::{record_idx}"``.
        output_path: Directory to write the bloom + sidecar under.
        text_field: Text column name in eval records.
        ngram: Word-ngram matching config. ``None`` = whole-paragraph hashing.
        estimated_doc_count, false_positive_rate: Bloom sizing parameters. Per-eval
            blooms intended for :func:`merge_eval_blooms` MUST share both
            values across all per-eval builds — ``dupekit.Bloom.update``
            requires identical sizing.
        exclude_eval_dirs: Eval task directory names to skip while walking
            ``eval_data_sources`` (see :func:`_discover_eval_files`). Excludes
            those tasks from the bloom without regenerating the eval corpus.
        required_eval_manifest_path: Optional manifest that must report a
            complete required eval suite before Bloom creation starts. When
            set, only the exact artifacts in this manifest enter the Bloom.
        required_eval_corpus_version: Version that the required manifest must
            report. Set this with ``required_eval_manifest_path``.
        required_eval_names: Exact benchmark names that the required manifest
            must contain.
        best_effort_eval_manifest_path: Optional manifest with the exact
            best-effort artifacts to include. Artifacts must be below the
            manifest directory.
        best_effort_eval_corpus_version: Version that the best-effort manifest
            must report.
        worker_resources: Resource request for each eval-file shard.
        max_workers: Maximum Zephyr workers for the Bloom build.
        zephyr_context: Optional shared Zephyr worker context.

    Returns:
        :class:`EvalBloom` artifact pointing at the produced files.
    """
    eval_paths = [eval_data_sources] if isinstance(eval_data_sources, str) else list(eval_data_sources)
    if not eval_paths:
        raise ValueError("eval_data_sources must be non-empty")
    if (required_eval_manifest_path is None) != (required_eval_corpus_version is None):
        raise ValueError("required eval manifest path and corpus version must be set together")
    if required_eval_manifest_path is not None:
        assert required_eval_corpus_version is not None
        eval_paths = _validate_required_eval_manifest(
            required_eval_manifest_path,
            required_eval_corpus_version,
            required_eval_names,
            text_field=text_field,
            ngram=ngram,
        )
    best_effort_parameters = (
        best_effort_eval_manifest_path,
        best_effort_eval_corpus_version,
    )
    if any(value is not None for value in best_effort_parameters) and not all(
        value is not None for value in best_effort_parameters
    ):
        raise ValueError("best-effort eval manifest path and corpus version must be set together")
    if best_effort_eval_manifest_path is not None:
        assert best_effort_eval_corpus_version is not None
        eval_paths.extend(
            _validate_best_effort_eval_manifest(
                best_effort_eval_manifest_path,
                best_effort_eval_corpus_version,
            )
        )

    eval_files = sorted(_discover_eval_files(eval_paths, exclude_eval_dirs))
    bloom_path, index_path = bloom_paths(output_path)
    parts_dir = prefix_join(output_path, "_bloom/_index_parts")
    resources = worker_resources or ResourceConfig(cpu=2, ram="4g")
    ctx_kwargs: dict[str, Any] = {"name": "decon-bloom", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = zephyr_context or ZephyrContext(**ctx_kwargs)
    if eval_files:
        pipeline = (
            Dataset.from_list(eval_files)
            .map_shard(
                lambda paths, shard: _build_eval_index_part(
                    paths,
                    shard,
                    parts_dir=parts_dir,
                    text_field=text_field,
                    ngram=ngram,
                )
            )
            .reshard(1)
            .map_shard(
                lambda parts, shard: _merge_eval_index_shard(
                    parts,
                    shard,
                    bloom_path=bloom_path,
                    index_path=index_path,
                    parts_dir=parts_dir,
                    estimated_doc_count=estimated_doc_count,
                    false_positive_rate=false_positive_rate,
                )
            )
        )
        outcome = ctx.execute(
            pipeline,
            map_task_resources=resources,
        )
        if len(outcome.results) != 1:
            raise RuntimeError(f"Bloom build returned {len(outcome.results)} merge results, expected one")
        n_records = outcome.results[0]
    else:
        n_records = _merge_eval_index_parts(
            [],
            bloom_path=bloom_path,
            index_path=index_path,
            estimated_doc_count=estimated_doc_count,
            false_positive_rate=false_positive_rate,
        )
    return EvalBloom(
        bloom_dir=output_path,
        bloom_path=bloom_path,
        eval_hash_index_path=index_path,
        estimated_doc_count=estimated_doc_count,
        false_positive_rate=false_positive_rate,
        n_eval_records=n_records,
    )


def _validate_required_eval_manifest(
    manifest_path: str,
    expected_corpus_version: str,
    expected_names: tuple[str, ...],
    *,
    text_field: str,
    ngram: NGramConfig | None,
) -> list[str]:
    """Validate required artifacts and confirm that each record has a feature."""
    manifest_storage = StoragePath(manifest_path)
    if not manifest_storage.exists():
        raise ValueError(f"required eval manifest does not exist: {manifest_path}")
    with manifest_storage.open("r") as source:
        manifest = json.load(source)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"required eval manifest must be an object: {manifest_path}")
    if manifest.get("corpus_version") != expected_corpus_version:
        raise ValueError(
            f"required eval manifest version is {manifest.get('corpus_version')!r}, expected {expected_corpus_version!r}"
        )
    if manifest.get("required") is not True:
        raise ValueError("required eval manifest does not mark the suite as required")
    if manifest.get("status") != "complete":
        raise ValueError(f"required eval manifest status is {manifest.get('status')!r}, expected 'complete'")

    benchmarks = manifest.get("benchmarks")
    if not isinstance(benchmarks, list) or not all(isinstance(entry, Mapping) for entry in benchmarks):
        raise ValueError("required eval manifest benchmarks must be a list of objects")
    if not all(isinstance(entry.get("name"), str) for entry in benchmarks):
        raise ValueError("required eval manifest benchmark entries need string names")
    actual_names = tuple(entry["name"] for entry in benchmarks)
    if set(actual_names) != set(expected_names) or len(actual_names) != len(expected_names):
        raise ValueError(
            f"required eval manifest benchmarks are {sorted(str(name) for name in actual_names)}, "
            f"expected {sorted(expected_names)}"
        )

    manifest_parent = os.path.dirname(manifest_path)
    paths: list[str] = []
    for entry in benchmarks:
        name = entry.get("name")
        artifact = entry.get("artifact")
        expected_records = entry.get("expected_records")
        if not isinstance(name, str) or not isinstance(artifact, str) or not isinstance(expected_records, int):
            raise ValueError("required eval manifest benchmark entries need name, artifact, and expected_records")
        artifact_path = prefix_join(manifest_parent, artifact)
        artifact_storage = StoragePath(artifact_path)
        if not artifact_storage.exists():
            raise ValueError(f"{name}: required eval artifact does not exist: {artifact_path}")
        with artifact_storage.open("rb") as source:
            parquet = pq.ParquetFile(source)
            actual_records = parquet.metadata.num_rows
            if text_field not in parquet.schema_arrow.names:
                raise ValueError(f"{name}: required eval artifact does not contain text field {text_field!r}")
            texts = parquet.read(columns=[text_field]).column(text_field).to_pylist()
        if actual_records != expected_records:
            raise ValueError(f"{name}: manifest expects {expected_records} records, artifact contains {actual_records}")
        record_statuses = [_record_feature_status(str(text), ngram) if text else (False, False) for text in texts]
        records_with_features = sum(has_features for has_features, _self_matches in record_statuses)
        if records_with_features != expected_records:
            raise ValueError(
                f"{name}: {expected_records - records_with_features} of {expected_records} required eval records "
                "produce no matchable features"
            )
        self_matching_records = sum(self_matches for _has_features, self_matches in record_statuses)
        if self_matching_records != expected_records:
            raise ValueError(
                f"{name}: {expected_records - self_matching_records} of {expected_records} required eval records "
                "do not match an exact copy under the mark policy"
            )
        paths.append(artifact_path)
    return paths


def _validate_best_effort_eval_manifest(
    manifest_path: str,
    expected_corpus_version: str,
) -> list[str]:
    """Validate a best-effort manifest and return its exact artifact paths."""
    manifest_storage = StoragePath(manifest_path)
    if not manifest_storage.exists():
        raise ValueError(f"best-effort eval manifest does not exist: {manifest_path}")
    with manifest_storage.open("r") as source:
        manifest = json.load(source)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"best-effort eval manifest must be an object: {manifest_path}")
    if manifest.get("corpus_version") != expected_corpus_version:
        raise ValueError(
            f"best-effort eval manifest version is {manifest.get('corpus_version')!r}, "
            f"expected {expected_corpus_version!r}"
        )
    if manifest.get("required") is not False:
        raise ValueError("best-effort eval manifest must mark the suite as not required")
    if manifest.get("status") not in {"complete", "complete_with_failures"}:
        raise ValueError(
            f"best-effort eval manifest status is {manifest.get('status')!r}, "
            "expected 'complete' or 'complete_with_failures'"
        )

    included_tasks = manifest.get("included_leaf_tasks")
    artifacts = manifest.get("artifacts")
    if not isinstance(included_tasks, list) or not all(isinstance(task, str) for task in included_tasks):
        raise ValueError("best-effort eval manifest included_leaf_tasks must be a list of strings")
    if not isinstance(artifacts, list) or not all(isinstance(entry, Mapping) for entry in artifacts):
        raise ValueError("best-effort eval manifest artifacts must be a list of objects")

    manifest_parent = os.path.dirname(manifest_path)
    paths: list[str] = []
    artifact_tasks: list[str] = []
    for entry in artifacts:
        task = entry.get("task")
        artifact = entry.get("artifact")
        expected_records = entry.get("records")
        if not isinstance(task, str) or not isinstance(artifact, str) or not isinstance(expected_records, int):
            raise ValueError("best-effort eval artifacts need task, artifact, and records")
        if artifact.startswith("/") or ".." in artifact.split("/"):
            raise ValueError(f"{task}: best-effort artifact must be relative to its root: {artifact}")
        artifact_path = prefix_join(manifest_parent, artifact)
        artifact_storage = StoragePath(artifact_path)
        if not artifact_storage.exists():
            raise ValueError(f"{task}: best-effort eval artifact does not exist: {artifact_path}")
        with artifact_storage.open("rb") as source:
            actual_records = pq.ParquetFile(source).metadata.num_rows
        if actual_records != expected_records:
            raise ValueError(
                f"{task}: best-effort manifest expects {expected_records} records, artifact contains {actual_records}"
            )
        artifact_tasks.append(task)
        paths.append(artifact_path)

    if sorted(artifact_tasks) != sorted(included_tasks) or len(artifact_tasks) != len(included_tasks):
        raise ValueError("best-effort eval artifact tasks do not match included_leaf_tasks")
    return paths


def merge_eval_blooms(
    *,
    per_eval_bloom_dirs: list[str],
    output_path: str,
) -> EvalBloom:
    """Merge N pre-built per-eval blooms into one combined bloom + index.

    Bit-OR-merges the bloom filters via :meth:`dupekit.Bloom.update` (which
    requires identical sizing across inputs) and concatenates the per-eval
    ``eval_hash_index.parquet`` sidecars. Output layout matches
    :func:`build_eval_bloom`.

    Args:
        per_eval_bloom_dirs: Directories produced by :func:`build_eval_bloom`,
            each containing ``_bloom/filter.bin`` and
            ``_bloom/eval_hash_index.parquet``.
        output_path: Directory to write the combined bloom + sidecar under.

    Returns:
        :class:`EvalBloom` artifact pointing at the merged files.
    """
    if not per_eval_bloom_dirs:
        raise ValueError("per_eval_bloom_dirs must be non-empty")

    out_bloom_path, out_index_path = bloom_paths(output_path)
    out_dir = os.path.dirname(out_bloom_path)
    if out_dir:
        StoragePath(out_dir).mkdirs()

    # Bit-OR merge of input blooms (dupekit raises on size mismatch).
    merged: dupekit.Bloom | None = None
    for d in per_eval_bloom_dirs:
        src_bloom, _ = bloom_paths(d)
        bf = dupekit.Bloom.load_bytes(StoragePath(src_bloom).read_bytes())
        if merged is None:
            merged = bf
        else:
            merged.update(bf)
    assert merged is not None  # non-empty list checked above
    StoragePath(out_bloom_path).write_bytes(merged.save_bytes())

    # Concatenate per-eval hash-index parquets, streaming row-by-row.
    src_indexes = [bloom_paths(d)[1] for d in per_eval_bloom_dirs]

    def emit_rows() -> Iterator[dict[str, Any]]:
        # Use zephyr's load_file so the read goes through rigging's
        # CrossRegionGuardedFS wrapper. Passing it directly to pq.read_table's
        # filesystem= kwarg trips pyarrow's strict type check (it expects a
        # native pyarrow.fs.FileSystem).
        for src in src_indexes:
            yield from load_file(src)

    write_parquet_file(emit_rows(), output_path=out_index_path, schema=_INDEX_SCHEMA)

    # Roll up sizing + record counts from upstream artifacts (best-effort —
    # informational; merge doesn't actually need these values to succeed).
    estimated = 0
    fpr = 0.0
    n_records = 0
    for d in per_eval_bloom_dirs:
        try:
            up: EvalBloom = read_artifact(d, EvalBloom)
        except FileNotFoundError:
            continue
        if estimated == 0:
            estimated = up.estimated_doc_count
            fpr = up.false_positive_rate
        n_records += up.n_eval_records

    logger.info("decon: merged %d per-eval blooms → %s", len(per_eval_bloom_dirs), output_path)
    return EvalBloom(
        bloom_dir=output_path,
        bloom_path=out_bloom_path,
        eval_hash_index_path=out_index_path,
        estimated_doc_count=estimated,
        false_positive_rate=fpr,
        n_eval_records=n_records,
    )


def build_eval_bloom_step(
    *,
    name: str,
    eval_data_sources: list[str | StepSpec],
    text_field: str = "text",
    ngram_length: int | None = 13,
    overlap_threshold: float = 0.5,
    paragraph_delimiter: str = DEFAULT_PARAGRAPH_DELIMITER,
    estimated_doc_count: int = 1_000_000,
    false_positive_rate: float = 1e-9,
    exclude_eval_dirs: frozenset[str] = frozenset(),
    required_eval_manifest_path: str | None = None,
    required_eval_corpus_version: str | None = None,
    required_eval_names: tuple[str, ...] = (),
    best_effort_eval_manifest_path: str | None = None,
    best_effort_eval_corpus_version: str | None = None,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    zephyr_context: ZephyrContext | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """StepSpec factory for :func:`build_eval_bloom`.

    Args:
        name: Step name (e.g. ``"datakit/bloom/mmlu"``).
        eval_data_sources: Mix of raw paths (str) and upstream StepSpecs. Raw
            paths go into ``hash_attrs`` (so changing them invalidates the
            cache); StepSpec entries become DAG deps.
        text_field, ngram_length, overlap_threshold, paragraph_delimiter: ngram
            config (see :class:`NGramConfig`). ``paragraph_delimiter`` MUST match
            the consuming :func:`decon_step` for the bloom to be reusable.
        estimated_doc_count, false_positive_rate: bloom sizing.
        exclude_eval_dirs: Eval task directory names to drop from the bloom
            (see :func:`build_eval_bloom`). Folded into ``hash_attrs`` so
            changing the exclusion set rebuilds the bloom at a fresh path.
        required_eval_manifest_path, required_eval_corpus_version,
            required_eval_names: Required-suite gate passed to
            :func:`build_eval_bloom`. The path is an external input. The
            expected version and names enter the step hash.
        best_effort_eval_manifest_path, best_effort_eval_corpus_version:
            Optional best-effort suite. Only the exact artifacts below its
            immutable, versioned manifest directory enter the Bloom.
        worker_resources: Resource request for each eval-file shard.
        max_workers: Maximum Zephyr workers for the Bloom build.
        zephyr_context: Optional shared Zephyr worker context.
        output_path_prefix, override_output_path: StepSpec routing.
    """
    raw_paths: list[str] = []
    step_deps: list[StepSpec] = []
    for s in eval_data_sources:
        if isinstance(s, StepSpec):
            step_deps.append(s)
            raw_paths.append(s.output_path)
        else:
            raw_paths.append(s)

    ngram: NGramConfig | None = (
        NGramConfig(
            ngram_length=ngram_length, overlap_threshold=overlap_threshold, paragraph_delimiter=paragraph_delimiter
        )
        if ngram_length is not None
        else None
    )

    hash_attrs: dict[str, Any] = {
        "text_field": text_field,
        "ngram_length": ngram_length,
        "overlap_threshold": overlap_threshold,
        "paragraph_delimiter": paragraph_delimiter,
        "feature_filter_version": FEATURE_FILTER_VERSION,
        "bloom_build_version": BLOOM_BUILD_VERSION,
        "estimated_doc_count": estimated_doc_count,
        "false_positive_rate": false_positive_rate,
        # Raw paths aren't deps — fingerprint them so swapping a path
        # invalidates the cache.
        "eval_data_sources": tuple(sorted(s for s in raw_paths if s not in (d.output_path for d in step_deps))),
        "exclude_eval_dirs": tuple(sorted(exclude_eval_dirs)),
        "required_eval_corpus_version": required_eval_corpus_version,
        "required_eval_names": tuple(sorted(required_eval_names)),
        "best_effort_eval_corpus_version": best_effort_eval_corpus_version,
    }

    return StepSpec(
        name=name,
        fn=lambda output_path: build_eval_bloom(
            eval_data_sources=raw_paths,
            output_path=output_path,
            text_field=text_field,
            ngram=ngram,
            estimated_doc_count=estimated_doc_count,
            false_positive_rate=false_positive_rate,
            exclude_eval_dirs=exclude_eval_dirs,
            required_eval_manifest_path=required_eval_manifest_path,
            required_eval_corpus_version=required_eval_corpus_version,
            required_eval_names=required_eval_names,
            best_effort_eval_manifest_path=best_effort_eval_manifest_path,
            best_effort_eval_corpus_version=best_effort_eval_corpus_version,
            worker_resources=worker_resources,
            max_workers=max_workers,
            zephyr_context=zephyr_context,
        ),
        deps=step_deps,
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )


def merge_eval_blooms_step(
    *,
    name: str,
    per_eval_bloom_steps: list[StepSpec],
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """StepSpec factory for :func:`merge_eval_blooms`."""
    return StepSpec(
        name=name,
        fn=lambda output_path: merge_eval_blooms(
            per_eval_bloom_dirs=[s.output_path for s in per_eval_bloom_steps],
            output_path=output_path,
        ),
        deps=list(per_eval_bloom_steps),
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )


# ---------------------------------------------------------------------------
# Corpus-common ngram filters (marin#6852, marin#7126): remove eval ngrams
# ubiquitous within one source or repeated across several sources.
# ---------------------------------------------------------------------------


class SourceDropSet(BaseModel):
    """Outcome of :func:`build_source_drop_set`: a source's common-ngram hashes.

    Consumers read the drop hashes from ``output_dir`` (via :func:`_load_drop_set`);
    the counts are informational.
    """

    output_dir: DatakitArtifactPath
    n_sampled: int
    n_dropped: int


def _iter_normalized_texts(main_output_dir: str, text_field: str) -> Iterator[str]:
    files = sorted(str(m) for m in StoragePath(f"{main_output_dir.rstrip('/')}/**/*.parquet").glob())
    for path in files:
        for record in load_file(path):
            text = record.get(text_field)
            if text:
                yield str(text)


def _load_drop_set(drop_set_dir: str) -> frozenset[int]:
    drop_path = StoragePath(f"{drop_set_dir.rstrip('/')}/drop.parquet")
    if not drop_path.exists():
        return frozenset()
    with drop_path.open("rb") as fh:
        return frozenset(pq.read_table(fh, columns=["hash"]).column("hash").to_pylist())


def _load_drop_sets(drop_set_dirs: list[str]) -> frozenset[int]:
    return frozenset().union(*(_load_drop_set(drop_set_dir) for drop_set_dir in drop_set_dirs))


def _document_frequency_counts(
    df_sample_dir: str,
    bf: dupekit.Bloom,
    text_field: str,
    ngram: NGramConfig | None,
    sample_docs: int,
) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    n = 0
    for text in islice(_iter_normalized_texts(df_sample_dir, text_field), sample_docs):
        n += 1
        counts.update({h for feat in _extract_features(text, ngram) if (h := _bloom_hash(feat)) in bf})
    return counts, n


def _drop_set_for_source(
    df_sample_dir: str,
    bf: dupekit.Bloom,
    text_field: str,
    ngram: NGramConfig | None,
    sample_docs: int,
    common_frac: float,
    common_min_abs: int,
) -> tuple[list[int], int, int]:
    """Core DF count for one source given a *loaded* bloom → (drop_hashes, n_sampled, threshold).

    Reads a prefix of *sample_docs* docs from *df_sample_dir* (shuffled upstream,
    so a prefix is representative), counts how many contain each eval ngram
    (membership via the bloom — the only ngrams a drop-set can hold), and keeps
    those in at least ``max(common_min_abs, common_frac * n_sampled)`` docs."""
    counts, n = _document_frequency_counts(df_sample_dir, bf, text_field, ngram, sample_docs)
    threshold = max(common_min_abs, int(common_frac * n))
    return [h for h, c in counts.items() if c >= threshold], n, threshold


def _write_drop_set(output_dir: str, drop: list[int]) -> str:
    StoragePath(output_dir).mkdirs()
    out_file = f"{output_dir.rstrip('/')}/drop.parquet"
    with StoragePath(out_file).open("wb") as fh:
        pq.write_table(pa.table({"hash": pa.array(drop, pa.uint64())}), fh, compression="zstd")
    return out_file


def build_source_drop_set(
    *,
    df_sample_dir: str,
    prebuilt_bloom_dir: str,
    output_path: str,
    text_field: str = "text",
    ngram: NGramConfig | None,
    sample_docs: int,
    common_frac: float,
    common_min_abs: int,
) -> SourceDropSet:
    """Single-source drop-set (loads the bloom, counts DF, writes ``drop.parquet``).

    The building block; :func:`build_all_source_drop_sets` distributes this over
    many sources. *df_sample_dir* should point at a pool large enough to estimate
    DF (~5k docs); it need not be the sample being deconned (DF is a source
    property, so a 100M mark can reuse a drop-set estimated from a 1T sample).
    """
    bloom_path, _ = bloom_paths(prebuilt_bloom_dir)
    bf = dupekit.Bloom.load_bytes(StoragePath(bloom_path).read_bytes())
    drop, n, threshold = _drop_set_for_source(
        df_sample_dir, bf, text_field, ngram, sample_docs, common_frac, common_min_abs
    )
    out_file = _write_drop_set(output_path, drop)
    logger.info("decon drop-set: sampled %d docs, %d common ngrams (df>=%d) → %s", n, len(drop), threshold, out_file)
    return SourceDropSet(output_dir=output_path, n_sampled=n, n_dropped=len(drop))


class AllSourceDropSets(BaseModel):
    """Outcome of :func:`build_all_source_drop_sets`.

    Per-source hashes live at ``<output_dir>/<source>/drop.parquet``. Globally
    common hashes live at ``<global_output_dir>/drop.parquet`` with document
    and source frequencies retained for threshold audits.
    """

    output_dir: DatakitArtifactPath
    global_output_dir: DatakitArtifactPath
    num_sources: int
    n_global_dropped: int
    counters: dict[str, int | float]


@dataclass(frozen=True)
class DropSetSource:
    """A normalized source sampled by :func:`all_source_drop_sets_step`.

    Set *dependency* when *data_path* is produced by another step. Omit it for
    a pre-materialized path.
    """

    name: str
    data_path: str
    dependency: StepSpec | None = None


def _source_sample_shards(
    source: tuple[str, str],
    *,
    text_field: str,
    sample_docs: int,
    global_sample_docs: int,
) -> list[dict[str, Any]]:
    """Plan balanced row-range samples for one normalized source."""
    source_name, data_path = source
    ranges: list[dict[str, Any]] = []
    rows_planned = 0
    files = sorted(str(path) for path in StoragePath(f"{data_path.rstrip('/')}/**/*.parquet").glob())
    for path in files:
        for row_start, row_end in compute_parquet_splits(path, DROP_SET_SAMPLE_SHARD_BYTES):
            rows_remaining = global_sample_docs - rows_planned
            if rows_remaining <= 0:
                break
            clipped_end = min(row_end, row_start + rows_remaining)
            row_count = clipped_end - row_start
            if row_count <= 0:
                continue
            local_row_count = min(row_count, max(0, sample_docs - rows_planned))
            ranges.append(
                {
                    "path": path,
                    "row_start": row_start,
                    "row_end": clipped_end,
                    "local_row_count": local_row_count,
                    "row_count": row_count,
                }
            )
            rows_planned += row_count
        if rows_planned >= global_sample_docs:
            break

    num_shards = min(DROP_SET_SHARDS_PER_SOURCE, len(ranges))
    if num_shards == 0:
        return [{"sample_shard_id": f"{source_name}:0", "source": source_name, "text_field": text_field, "ranges": []}]

    buckets: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    bucket_rows = [0] * num_shards
    for sample_range in sorted(ranges, key=lambda item: item["row_count"], reverse=True):
        bucket_index = min(range(num_shards), key=bucket_rows.__getitem__)
        buckets[bucket_index].append(sample_range)
        bucket_rows[bucket_index] += sample_range["row_count"]
    return [
        {
            "sample_shard_id": f"{source_name}:{bucket_index}",
            "source": source_name,
            "text_field": text_field,
            "ranges": bucket,
        }
        for bucket_index, bucket in enumerate(buckets)
    ]


def _materialize_sample_shard(sample_shard_id: str, items: Iterator[dict[str, Any]]) -> dict[str, Any]:
    iterator = iter(items)
    sample_shard = next(iterator)
    if next(iterator, None) is not None:
        raise ValueError(f"duplicate decontamination sample shard: {sample_shard_id}")
    return sample_shard


def _sample_drop_set_shard(
    sample_shards: Iterator[dict[str, Any]],
    _shard: ShardInfo,
    *,
    bloom_path: str,
    ngram: NGramConfig | None,
) -> Iterator[dict[str, Any]]:
    """Count matching eval features in one group of source row ranges."""
    bf = dupekit.Bloom.load_bytes(StoragePath(bloom_path).read_bytes())
    is_nonempty = False
    for sample_shard in sample_shards:
        if not is_nonempty:
            counters.pipeline.update_counter("decon_drop/nonempty_sampling_shards", 1)
            is_nonempty = True
        local_counts: Counter[int] = Counter()
        global_counts: Counter[int] = Counter()
        local_documents = 0
        global_documents = 0
        text_field = sample_shard["text_field"]
        for sample_range in sample_shard["ranges"]:
            spec = InputFileSpec(
                path=sample_range["path"],
                columns=[text_field],
                row_start=sample_range["row_start"],
                row_end=sample_range["row_end"],
            )
            for row_index, record in enumerate(load_file(spec)):
                text = record.get(text_field)
                if not text:
                    continue
                hashes = {h for feature in _extract_features(str(text), ngram) if (h := _bloom_hash(feature)) in bf}
                global_counts.update(hashes)
                global_documents += 1
                if row_index < sample_range["local_row_count"]:
                    local_counts.update(hashes)
                    local_documents += 1

        counters.pipeline.update_counter("decon_drop/sample_shards", 1)
        yield {
            "source": sample_shard["source"],
            "hash": None,
            "local_document_frequency": 0,
            "global_document_frequency": 0,
            "local_documents": local_documents,
            "global_documents": global_documents,
        }
        for hash_value in local_counts.keys() | global_counts.keys():
            yield {
                "source": sample_shard["source"],
                "hash": hash_value,
                "local_document_frequency": local_counts[hash_value],
                "global_document_frequency": global_counts[hash_value],
                "local_documents": 0,
                "global_documents": 0,
            }


def _reduce_source_drop_set(
    source_name: str,
    items: Iterator[dict[str, Any]],
    *,
    output_path: str,
    common_frac: float,
    common_min_abs: int,
) -> Iterator[dict[str, int]]:
    """Merge sample shards, write one local drop set, and emit global counts."""
    local_counts: Counter[int] = Counter()
    global_counts: Counter[int] = Counter()
    local_documents = 0
    global_documents = 0
    for item in items:
        local_documents += item["local_documents"]
        global_documents += item["global_documents"]
        hash_value = item["hash"]
        if hash_value is None:
            continue
        local_counts[hash_value] += item["local_document_frequency"]
        global_counts[hash_value] += item["global_document_frequency"]

    threshold = max(common_min_abs, int(common_frac * local_documents))
    drop = [hash_value for hash_value, count in local_counts.items() if count >= threshold]
    _write_drop_set(f"{output_path.rstrip('/')}/{source_name}", drop)
    counters.pipeline.update_counter("decon_drop/sources", 1)
    counters.pipeline.update_counter("decon_drop/ngrams_dropped", len(drop))
    counters.pipeline.update_counter("decon_drop/global_documents_sampled", global_documents)
    counters.pipeline.update_counter("decon_drop/global_candidates", len(global_counts))
    logger.info(
        "decon drop-set %s: local=%d docs/%d ngrams (df>=%d), global=%d docs/%d candidates",
        source_name,
        local_documents,
        len(drop),
        threshold,
        global_documents,
        len(global_counts),
    )
    for hash_value, document_frequency in global_counts.items():
        yield {"hash": hash_value, "document_frequency": document_frequency, "source_frequency": 1}


def _global_drop_row(
    hash_value: int,
    items: Iterator[dict[str, int]],
    *,
    common_min_abs: int,
    common_min_sources: int,
) -> dict[str, int] | None:
    document_frequency = 0
    source_frequency = 0
    for item in items:
        document_frequency += item["document_frequency"]
        source_frequency += item["source_frequency"]
    if document_frequency < common_min_abs or source_frequency < common_min_sources:
        return None
    return {
        "hash": hash_value,
        "document_frequency": document_frequency,
        "source_frequency": source_frequency,
    }


def _write_global_drop_set(output_dir: str, rows: list[dict[str, int]]) -> str:
    StoragePath(output_dir).mkdirs()
    out_file = f"{output_dir.rstrip('/')}/drop.parquet"
    schema = pa.schema(
        [
            pa.field("hash", pa.uint64()),
            pa.field("document_frequency", pa.int64()),
            pa.field("source_frequency", pa.int64()),
        ]
    )
    write_parquet_file(iter(rows), output_path=out_file, schema=schema)
    return out_file


def build_all_source_drop_sets(
    *,
    sources: list[tuple[str, str]],
    prebuilt_bloom_dir: str,
    output_path: str,
    text_field: str = "text",
    ngram: NGramConfig | None,
    sample_docs: int,
    common_frac: float,
    common_min_abs: int,
    global_sample_docs: int,
    global_common_min_abs: int,
    global_common_min_sources: int,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    zephyr_context: ZephyrContext | None = None,
) -> AllSourceDropSets:
    """Build per-source and cross-source common eval-ngram drop sets.

    Each source sample is divided into balanced Parquet row-range shards. A
    source reduce writes the local drop set and emits its document frequencies.
    A second distributed reduce sums those counts across sources. A globally
    common ngram must meet both the corpus document-frequency threshold and the
    distinct-source threshold, so a repeated eval item concentrated in one
    source remains matchable.
    """
    if not sources:
        raise ValueError("sources must be non-empty")
    source_names = {source_name for source_name, _ in sources}
    if len(source_names) != len(sources):
        raise ValueError("source names must be unique")
    if _GLOBAL_DROP_SET_DIRECTORY in source_names:
        raise ValueError(f"{_GLOBAL_DROP_SET_DIRECTORY!r} is reserved for the global drop set")
    if global_sample_docs < sample_docs:
        raise ValueError("global_sample_docs must be at least sample_docs")
    if global_common_min_abs <= 0 or global_common_min_sources <= 0:
        raise ValueError("global common thresholds must be positive")
    if len(sources) < global_common_min_sources:
        logger.warning(
            "decon global drop-set cannot meet sources>=%d with only %d sources",
            global_common_min_sources,
            len(sources),
        )

    bloom_path, _ = bloom_paths(prebuilt_bloom_dir)

    pipeline = (
        Dataset.from_list(sources)
        .flat_map(
            lambda source: _source_sample_shards(
                source,
                text_field=text_field,
                sample_docs=sample_docs,
                global_sample_docs=global_sample_docs,
            )
        )
        .group_by(
            key=lambda row: row["sample_shard_id"],
            reducer=_materialize_sample_shard,
            num_output_shards=DROP_SET_SHARDS_PER_SOURCE * len(sources),
        )
        .map_shard(lambda items, shard: _sample_drop_set_shard(items, shard, bloom_path=bloom_path, ngram=ngram))
        .group_by(
            key=lambda row: row["source"],
            reducer=lambda source_name, items: _reduce_source_drop_set(
                source_name,
                items,
                output_path=output_path,
                common_frac=common_frac,
                common_min_abs=common_min_abs,
            ),
            num_output_shards=len(sources),
        )
        .group_by(
            key=lambda row: row["hash"],
            reducer=lambda hash_value, items: _global_drop_row(
                hash_value,
                items,
                common_min_abs=global_common_min_abs,
                common_min_sources=global_common_min_sources,
            ),
            num_output_shards=len(sources),
        )
        .filter(lambda row: row is not None)
    )
    resources = worker_resources or ResourceConfig(cpu=2, ram="4g")
    ctx_kwargs: dict[str, Any] = {"name": "decon-drop-set", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = zephyr_context or ZephyrContext(**ctx_kwargs)
    outcome = ctx.execute(
        pipeline,
        map_task_resources=resources,
    )
    global_rows = list(outcome.results)
    global_output_dir = f"{output_path.rstrip('/')}/{_GLOBAL_DROP_SET_DIRECTORY}"
    out_file = _write_global_drop_set(global_output_dir, global_rows)
    counters_out = dict(outcome.counters)
    counters_out["decon_drop/global_ngrams_dropped"] = len(global_rows)
    logger.info(
        "decon global drop-set: %d ngrams (df>=%d, sources>=%d) → %s",
        len(global_rows),
        global_common_min_abs,
        global_common_min_sources,
        out_file,
    )
    return AllSourceDropSets(
        output_dir=output_path,
        global_output_dir=global_output_dir,
        num_sources=len(sources),
        n_global_dropped=len(global_rows),
        counters=counters_out,
    )


def all_source_drop_sets_step(
    *,
    name: str,
    sources: list[DropSetSource],
    prebuilt_bloom: StepSpec,
    text_field: str = "text",
    ngram_length: int | None = 13,
    paragraph_delimiter: str = DEFAULT_PARAGRAPH_DELIMITER,
    sample_docs: int,
    common_frac: float,
    common_min_abs: int,
    global_sample_docs: int,
    global_common_min_abs: int,
    global_common_min_sources: int,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    zephyr_context: ZephyrContext | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """StepSpec for corpus-side common-ngram filters.

    Each source names a normalized parquet directory used to estimate DF and
    optionally its producer step. ``sample_docs`` controls the source-local
    estimate; ``global_sample_docs`` controls the larger cross-source estimate.
    ``ngram_length`` / ``paragraph_delimiter`` MUST match the consuming
    :func:`decon_step`.
    """
    ngram: NGramConfig | None = (
        NGramConfig(ngram_length=ngram_length, paragraph_delimiter=paragraph_delimiter)
        if ngram_length is not None
        else None
    )
    raw_sources: list[tuple[str, str]] = []
    dependent_sources: list[tuple[str, str, str]] = []
    source_dependencies: list[StepSpec] = []
    runtime_sources: list[tuple[str, str]] = []
    for source in sources:
        runtime_sources.append((source.name, source.data_path))
        dependency = source.dependency
        if dependency is None:
            raw_sources.append((source.name, source.data_path))
            continue
        source_dependencies.append(dependency)
        dependency_path = dependency.output_path.rstrip("/")
        if source.data_path != dependency_path and not source.data_path.startswith(f"{dependency_path}/"):
            raise ValueError(f"{source.name} path {source.data_path} is outside dependency output {dependency_path}")
        dependent_sources.append(
            (source.name, dependency.name_with_hash, source.data_path.removeprefix(dependency_path))
        )

    hash_attrs: dict[str, Any] = {
        "raw_sources": tuple(sorted(raw_sources)),
        "dependent_sources": tuple(sorted(dependent_sources)),
        "text_field": text_field,
        "ngram_length": ngram_length,
        "paragraph_delimiter": paragraph_delimiter,
        "feature_filter_version": FEATURE_FILTER_VERSION,
        "drop_set_build_version": DROP_SET_BUILD_VERSION,
        "sample_docs": sample_docs,
        "common_frac": common_frac,
        "common_min_abs": common_min_abs,
        "global_sample_docs": global_sample_docs,
        "global_common_min_abs": global_common_min_abs,
        "global_common_min_sources": global_common_min_sources,
    }
    return StepSpec(
        name=name,
        fn=lambda output_path: build_all_source_drop_sets(
            sources=runtime_sources,
            prebuilt_bloom_dir=prebuilt_bloom.output_path,
            output_path=output_path,
            text_field=text_field,
            ngram=ngram,
            sample_docs=sample_docs,
            common_frac=common_frac,
            common_min_abs=common_min_abs,
            global_sample_docs=global_sample_docs,
            global_common_min_abs=global_common_min_abs,
            global_common_min_sources=global_common_min_sources,
            worker_resources=worker_resources,
            max_workers=max_workers,
            zephyr_context=zephyr_context,
        ),
        deps=[prebuilt_bloom, *source_dependencies],
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )


def decon_step(
    *,
    name: str,
    normalized: StepSpec | None = None,
    input_dir: str | None = None,
    eval_data_sources: list[StepSpec] | None = None,
    prebuilt_bloom: StepSpec | None = None,
    drop_sets: StepSpec | None = None,
    drop_set_source: str | None = None,
    text_field: str = "text",
    ngram_length: int | None = 13,
    overlap_threshold: float = 0.5,
    min_matched_features: int = 2,
    paragraph_delimiter: str = DEFAULT_PARAGRAPH_DELIMITER,
    flagged_sample_size: int = 0,
    estimated_doc_count: int = 1_000_000,
    false_positive_rate: float = 1e-9,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    zephyr_context: ZephyrContext | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that decontaminates a normalized dataset.

    Provide exactly one of ``eval_data_sources`` (build bloom inline,
    single-corpus pattern) or ``prebuilt_bloom`` (reuse a shared bloom
    produced by :func:`build_eval_bloom_step` / :func:`merge_eval_blooms_step`,
    multi-corpus pattern).

    Args:
        name: Step name (e.g. ``"fineweb/decon"``).
        normalized: Upstream datakit normalize step whose output is the input.
            Provide exactly one of *normalized* or *input_dir*.
        input_dir: Directory of pre-materialized normalized parquet (``id``,
            ``text``) to mark directly — for deconning an already-built sample
            that isn't a step in this DAG (e.g. the fixed 1T testbed root). Folded
            into ``hash_attrs`` (not a dep). Mutually exclusive with *normalized*.
        eval_data_sources: List of eval source steps (any zephyr-readable
            format) to build the bloom filter from. All eval sources are
            merged into one bloom; per-eval attribution is preserved in the
            ``eval_hash_index`` sidecar. Mutually exclusive with ``prebuilt_bloom``.
        prebuilt_bloom: Pre-built bloom StepSpec (output of
            :func:`build_eval_bloom_step` or :func:`merge_eval_blooms_step`).
            Mutually exclusive with ``eval_data_sources``.
        drop_sets: Optional :func:`all_source_drop_sets_step` output. Combined
            with *drop_set_source*, this excludes both hashes common within the
            source and hashes common across several sources. Feature policy
            (ngram/delimiter) must match this step's.
        drop_set_source: This source's subdir name under *drop_sets* (e.g.
            ``"cp/usgpo"``). Required when *drop_sets* is set.
        text_field: Text column name in both input and eval records.
        ngram_length: Word ngram length. ``None`` = exact whole-paragraph match.
        overlap_threshold: Per-paragraph overlap fraction needed to mark a record
            contaminated. Ignored in exact-paragraph mode.
        min_matched_features: Minimum distinct matches in one paragraph. A
            complete document with one feature can still match. Ignored in
            exact-paragraph mode.
        paragraph_delimiter: Paragraph split string (see :class:`NGramConfig`).
            When reusing a ``prebuilt_bloom``, MUST match the delimiter the bloom
            was built with, or the two feature sets won't line up.
        estimated_doc_count, false_positive_rate: Bloom sizing parameters.
            Ignored when ``prebuilt_bloom`` is set.
        worker_resources, max_workers: Zephyr execution knobs.
        output_path_prefix, override_output_path: StepSpec routing.
    """
    if (eval_data_sources is None) == (prebuilt_bloom is None):
        raise ValueError("provide exactly one of eval_data_sources or prebuilt_bloom")
    if (normalized is None) == (input_dir is None):
        raise ValueError("provide exactly one of normalized or input_dir")

    ngram: NGramConfig | None = (
        NGramConfig(
            ngram_length=ngram_length,
            overlap_threshold=overlap_threshold,
            min_matched_features=min_matched_features,
            paragraph_delimiter=paragraph_delimiter,
        )
        if ngram_length is not None
        else None
    )

    # Mark stage hash_attrs: only what actually affects per-record marking
    # output. Bloom sizing is hashed into the bloom-producing step's
    # output_path (a dep below), so it propagates without polluting the
    # mark cache.
    hash_attrs: dict[str, Any] = {
        "text_field": text_field,
        "ngram_length": ngram_length,
        "overlap_threshold": overlap_threshold,
        "min_matched_features": min_matched_features,
        "paragraph_delimiter": paragraph_delimiter,
        "feature_filter_version": FEATURE_FILTER_VERSION,
        "attribute_schema_version": DECON_ATTRIBUTES_VERSION,
        "input_dir": input_dir,
    }
    # Only fold in when enabled: the flagged sidecar is *additional* output, so a
    # run with sampling off keeps the same address as before this feature landed.
    if flagged_sample_size:
        hash_attrs["flagged_sample_size"] = flagged_sample_size

    if drop_sets is not None and drop_set_source is None:
        raise ValueError("drop_set_source is required when drop_sets is set")
    drop_deps = [drop_sets] if drop_sets is not None else []
    drop_dirs = (
        [
            f"{drop_sets.output_path.rstrip('/')}/{drop_set_source}",
            f"{drop_sets.output_path.rstrip('/')}/{_GLOBAL_DROP_SET_DIRECTORY}",
        ]
        if drop_sets is not None
        else None
    )
    norm_deps = [normalized] if normalized is not None else []

    def _read_norm() -> NormalizedData:
        if input_dir is not None:
            return NormalizedData(main_output_dir=input_dir, dup_output_dir="", counters={})
        return read_artifact(normalized.output_path, NormalizedData)

    if prebuilt_bloom is not None:
        bloom_step = prebuilt_bloom
        return StepSpec(
            name=name,
            fn=lambda output_path: decon_to_parquet(
                normalized_data=_read_norm(),
                prebuilt_bloom_dir=bloom_step.output_path,
                output_path=output_path,
                text_field=text_field,
                ngram=ngram,
                drop_set_dirs=drop_dirs,
                flagged_sample_size=flagged_sample_size,
                worker_resources=worker_resources,
                max_workers=max_workers,
                zephyr_context=zephyr_context,
            ),
            deps=[*norm_deps, bloom_step, *drop_deps],
            hash_attrs=hash_attrs,
            output_path_prefix=output_path_prefix,
            override_output_path=override_output_path,
        )

    assert eval_data_sources is not None  # mutex check above
    eval_steps = list(eval_data_sources)
    # Inline-build path adds bloom sizing to hash_attrs since this step
    # owns both the build and the mark.
    inline_hash_attrs = {
        **hash_attrs,
        "estimated_doc_count": estimated_doc_count,
        "false_positive_rate": false_positive_rate,
    }
    return StepSpec(
        name=name,
        fn=lambda output_path: decon_to_parquet(
            normalized_data=_read_norm(),
            eval_data_sources=[s.output_path for s in eval_steps],
            output_path=output_path,
            text_field=text_field,
            ngram=ngram,
            drop_set_dirs=drop_dirs,
            flagged_sample_size=flagged_sample_size,
            estimated_doc_count=estimated_doc_count,
            false_positive_rate=false_positive_rate,
            worker_resources=worker_resources,
            max_workers=max_workers,
            zephyr_context=zephyr_context,
        ),
        deps=[*norm_deps, *eval_steps, *drop_deps],
        hash_attrs=inline_hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
