# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontaminate normalized data against eval sources.

Reads datakit-normalized Parquet (``id``, ``text``), builds an in-memory
bloom filter from the eval text, and emits a co-partitioned Parquet
attributes dataset marking which records overlap with eval text.

Schema of the emitted Parquet attributes (datakit ``{id, attributes}`` convention,
consumable by :func:`marin.processing.classification.consolidate.consolidate`):

    id                       : string         — matches source document id
    partition_id             : int            — source partition index (from sorted file order)
    attributes               : struct
        contaminated         : bool           — max paragraph overlap meets the threshold
        max_overlap          : float          — highest paragraph overlap fraction in [0, 1]
        matched_hashes       : list[uint64]   — bloom-hit ngram hashes from this record

Build also emits ``<output>/_bloom/eval_hash_index.parquet`` with columns
``hash: uint64, eval_id: string`` (flattened, one row per (hash, eval_id) pair).
Join ``attributes.matched_hashes`` against this sidecar to attribute
contamination back to specific eval records.

Output is co-partitioned with the source: one ``part-NNNNN-of-MMMMM.parquet``
per input partition, preserving the source filenames so consolidate can
sorted-merge-join without a shuffle.

The bloom can also be built once and shared across many corpus marks via
:func:`build_eval_bloom` (single-source) and :func:`merge_eval_blooms`
(combine pre-built per-eval blooms). Pass the resulting directory to
:func:`decon_to_parquet` as ``prebuilt_bloom_dir`` to skip the inline build.
"""

import hashlib
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import dupekit
import pyarrow as pa
from fray import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ShardInfo, ZephyrContext, counters, write_parquet_file
from zephyr.readers import SUPPORTED_EXTENSIONS, load_file

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NGramConfig:
    """Word-ngram matching parameters.

    Attributes:
        ngram_length: Size of each ngram in whitespace-split word tokens.
        stride: Step between successive ngrams. 0 = contiguous (every position).
        overlap_threshold: Minimum fraction of paragraph ngrams that must hit
            the filter for the paragraph to count as contaminated.
    """

    ngram_length: int = 13
    stride: int = 0
    overlap_threshold: float = 0.5


class DeconAttributes(BaseModel):
    """Outcome of :func:`decon_to_parquet`: a co-partitioned attributes dataset.

    Persisted as the step's ``.artifact`` so downstream consumers can locate
    the output without re-running the pipeline.

    Attributes:
        output_dir: Directory containing ``part-NNNNN-of-MMMMM.parquet`` files.
        num_partitions: Number of output partitions; matches the source.
        eval_hash_index_path: Path to the ``hash → eval_id`` sidecar Parquet.
            Join the per-record ``attributes.matched_hashes`` column against
            this to attribute contamination to specific eval records.
        counters: Aggregated zephyr counters from the marking pipeline.
    """

    version: str = "v2"
    output_dir: str
    num_partitions: int
    eval_hash_index_path: str
    counters: dict[str, int | float]


_BLOOM_FILENAME = "filter.bin"
_INDEX_FILENAME = "eval_hash_index.parquet"


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
    bloom_dir: str
    bloom_path: str
    eval_hash_index_path: str
    estimated_doc_count: int
    false_positive_rate: float
    n_eval_records: int = 0


def _bloom_hash(x: str) -> int:
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "big")


def _extract_ngrams(text: str, n: int, stride: int) -> Iterator[str]:
    tokens = text.split()
    for i in range(0, len(tokens) - n + 1, stride + 1):
        yield " ".join(tokens[i : i + n])


def _extract_features(text: str, ngram: NGramConfig | None) -> Iterator[str]:
    """Yield matchable features: ngrams within each paragraph, or whole paragraphs.

    In n-gram mode, paragraphs with fewer than ``ngram_length`` tokens
    contribute nothing — no fallback. A whole-paragraph fallback would be
    symmetric with the consumer side but creates trivial collisions on very
    short paragraphs (e.g. ``"..."``, ``"A."``, ``"##"`` that show up
    routinely in eval and pretraining text alike). See PR #5656 for the
    smoke-test finding (~18% phantom contamination on MMLU vs nemotron-math
    came from the literal ``"..."`` short-paragraph artifact).
    """
    for para in text.split("\n"):
        if not para:
            continue
        if ngram is None:
            yield para
            continue
        yield from _extract_ngrams(para, ngram.ngram_length, ngram.stride)


def _paragraph_overlap_and_matches(
    paragraph: str, bf: dupekit.Bloom, ngram: NGramConfig | None
) -> tuple[float, list[int]]:
    """Return ``(overlap_score, matched_hashes)`` for a single paragraph.

    Score is 0.0 or 1.0 in exact-paragraph mode and the fraction of bloom-hit
    ngrams otherwise. *matched_hashes* is the list of ngram hashes that hit
    the bloom (in iteration order, with duplicates if the same ngram repeats).

    Paragraphs with fewer than ``ngram_length`` tokens in n-gram mode return
    ``(0.0, [])`` — see :func:`_extract_features` for why we don't fall back
    to whole-paragraph hashing.
    """
    if ngram is None:
        h = _bloom_hash(paragraph)
        return (1.0, [h]) if h in bf else (0.0, [])
    ngrams = list(_extract_ngrams(paragraph, ngram.ngram_length, ngram.stride))
    if not ngrams:
        return 0.0, []
    hashes = [_bloom_hash(ng) for ng in ngrams]
    matched = [h for h in hashes if h in bf]
    return len(matched) / len(hashes), matched


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

    Single-process by design: the build is never the bottleneck (~seconds for a
    full lm-eval-style suite, vs. hours for marking over a pretraining corpus).
    If a very large eval suite ever forces this to dominate, parallelize via
    Zephyr: per-shard ``dupekit.Bloom`` + shared shards of the sidecar Parquet,
    merging blooms with ``bf.update(other)``. The mark side stays as-is.
    """
    bf = dupekit.Bloom(estimated_doc_count, false_positive_rate)
    stats = {"n_records": 0, "n_index_rows": 0}

    def emit_index_rows() -> Iterator[dict[str, Any]]:
        for path in _discover_eval_files(eval_paths, exclude_dir_names):
            for idx, record in enumerate(load_file(path)):
                text = record.get(text_field)
                if not text:
                    continue
                # Use the full path (not basename) so fallback IDs stay unique across
                # nested or multi-source eval directories that share file basenames.
                eval_id = str(record.get("id") or f"{path}::{idx}")
                seen_in_record: set[int] = set()
                for feat in _extract_features(str(text), ngram):
                    h = _bloom_hash(feat)
                    bf.add(h)
                    if h in seen_in_record:
                        continue
                    seen_in_record.add(h)
                    stats["n_index_rows"] += 1
                    yield {"hash": h, "eval_id": eval_id}
                stats["n_records"] += 1

    # Stream the index parquet; this iteration also fills the bloom.
    fs_idx, ip = url_to_fs(index_path)
    idx_dir = os.path.dirname(ip)
    if idx_dir:
        fs_idx.makedirs(idx_dir, exist_ok=True)
    write_parquet_file(emit_index_rows(), output_path=index_path, schema=_INDEX_SCHEMA)

    # Persist the populated bloom.
    fs_bf, bp = url_to_fs(bloom_path)
    bloom_dir = os.path.dirname(bp)
    if bloom_dir:
        fs_bf.makedirs(bloom_dir, exist_ok=True)
    with fs_bf.open(bp, "wb") as f:
        f.write(bf.save_bytes())

    logger.info(
        "decon: built bloom + index from %d eval records (%d index rows) → bloom=%s, index=%s",
        stats["n_records"],
        stats["n_index_rows"],
        bloom_path,
        index_path,
    )
    return stats["n_records"]


_ATTRIBUTES_STRUCT = pa.struct(
    [
        pa.field("contaminated", pa.bool_()),
        pa.field("max_overlap", pa.float64()),
        pa.field("matched_hashes", pa.list_(pa.uint64())),
    ]
)

# Wrapped-attributes schema -- matches the datakit convention consumed by
# ``marin.processing.classification.consolidate``: top-level ``id`` (join key)
# and ``partition_id`` (co-partitioning invariant), with the per-record decon
# facts grouped under ``attributes`` so a ``FilterConfig(name="contaminated")``
# resolves correctly.
_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("partition_id", pa.int64()),
        pa.field("attributes", _ATTRIBUTES_STRUCT),
    ]
)


def _make_marker(
    bloom_path: str,
    output_dir: str,
    text_field: str,
    ngram: NGramConfig | None,
) -> Callable[[Iterator[str], ShardInfo], Iterator[dict[str, Any]]]:
    """Return a ``map_shard`` function that processes one input parquet → one output parquet."""

    # Threshold is only meaningful for n-gram mode; in exact-paragraph mode score is 0 or 1
    # so any non-zero match is always recorded.
    threshold = ngram.overlap_threshold if ngram is not None else 0.0

    def mark_shard(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict[str, Any]]:
        # Load bloom once per shard.
        fs, bp = url_to_fs(bloom_path)
        with fs.open(bp, "rb") as f:
            bf = dupekit.Bloom.load_bytes(f.read())

        for input_path in paths:

            def rows_for(p: str) -> Iterator[dict[str, Any]]:
                for record in load_file(p):
                    text = str(record.get(text_field, "") or "")
                    max_score = 0.0
                    matched: set[int] = set()
                    for para in text.split("\n"):
                        if not para:
                            continue
                        score, hits = _paragraph_overlap_and_matches(para, bf, ngram)
                        if score > max_score:
                            max_score = score
                        matched.update(hits)
                    contaminated = max_score > 0 and max_score >= threshold
                    counters.pipeline.update_counter("decon/contaminated" if contaminated else "decon/clean", 1)
                    # Dataset.from_list yields one shard per item in input order, so shard.shard_idx
                    # matches the input's "part-NNNNN-of-NNNNN" partition number on a sorted file list.
                    yield {
                        "id": record["id"],
                        "partition_id": shard.shard_idx,
                        "attributes": {
                            "contaminated": contaminated,
                            "max_overlap": max_score,
                            "matched_hashes": list(matched),
                        },
                    }

            out_filename = os.path.basename(input_path)
            out_path = f"{output_dir.rstrip('/')}/{out_filename}"
            result = write_parquet_file(rows_for(input_path), output_path=out_path, schema=_OUTPUT_SCHEMA)
            yield result

    return mark_shard


def decon_to_parquet(
    *,
    normalized_data: NormalizedData,
    eval_data_sources: str | list[str] | None = None,
    prebuilt_bloom_dir: str | None = None,
    output_path: str,
    text_field: str = "text",
    ngram: NGramConfig | None = None,
    estimated_doc_count: int = 1_000_000,
    false_positive_rate: float = 1e-9,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
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
            contaminated; exact-paragraph mode records any non-zero match.
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
    files = sorted(fsspec_glob(f"{input_path.rstrip('/')}/**/*.parquet"))
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

    pipeline = Dataset.from_list(files).map_shard(_make_marker(bloom_path, output_path, text_field, ngram))

    resources = worker_resources or ResourceConfig(cpu=2, ram="4g")
    ctx_kwargs: dict[str, Any] = {"name": "decon-mark", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = ZephyrContext(**ctx_kwargs)
    outcome = ctx.execute(pipeline)

    return DeconAttributes(
        output_dir=output_path,
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

    Returns:
        :class:`EvalBloom` artifact pointing at the produced files.
    """
    eval_paths = [eval_data_sources] if isinstance(eval_data_sources, str) else list(eval_data_sources)
    if not eval_paths:
        raise ValueError("eval_data_sources must be non-empty")

    bloom_path, index_path = bloom_paths(output_path)
    n_records = _build_filter(
        eval_paths=eval_paths,
        bloom_path=bloom_path,
        index_path=index_path,
        text_field=text_field,
        ngram=ngram,
        estimated_doc_count=estimated_doc_count,
        false_positive_rate=false_positive_rate,
        exclude_dir_names=exclude_eval_dirs,
    )
    return EvalBloom(
        bloom_dir=output_path,
        bloom_path=bloom_path,
        eval_hash_index_path=index_path,
        estimated_doc_count=estimated_doc_count,
        false_positive_rate=false_positive_rate,
        n_eval_records=n_records,
    )


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
    fs_bf, bp = url_to_fs(out_bloom_path)
    out_dir = os.path.dirname(bp)
    if out_dir:
        fs_bf.makedirs(out_dir, exist_ok=True)

    # Bit-OR merge of input blooms (dupekit raises on size mismatch).
    merged: dupekit.Bloom | None = None
    for d in per_eval_bloom_dirs:
        src_bloom, _ = bloom_paths(d)
        fs, p = url_to_fs(src_bloom)
        with fs.open(p, "rb") as f:
            bf = dupekit.Bloom.load_bytes(f.read())
        if merged is None:
            merged = bf
        else:
            merged.update(bf)
    assert merged is not None  # non-empty list checked above
    with fs_bf.open(bp, "wb") as f:
        f.write(merged.save_bytes())

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
    estimated_doc_count: int = 1_000_000,
    false_positive_rate: float = 1e-9,
    exclude_eval_dirs: frozenset[str] = frozenset(),
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """StepSpec factory for :func:`build_eval_bloom`.

    Args:
        name: Step name (e.g. ``"datakit/bloom/mmlu"``).
        eval_data_sources: Mix of raw paths (str) and upstream StepSpecs. Raw
            paths go into ``hash_attrs`` (so changing them invalidates the
            cache); StepSpec entries become DAG deps.
        text_field, ngram_length, overlap_threshold: ngram config.
        estimated_doc_count, false_positive_rate: bloom sizing.
        exclude_eval_dirs: Eval task directory names to drop from the bloom
            (see :func:`build_eval_bloom`). Folded into ``hash_attrs`` so
            changing the exclusion set rebuilds the bloom at a fresh path.
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
        NGramConfig(ngram_length=ngram_length, overlap_threshold=overlap_threshold) if ngram_length is not None else None
    )

    hash_attrs: dict[str, Any] = {
        "text_field": text_field,
        "ngram_length": ngram_length,
        "overlap_threshold": overlap_threshold,
        "estimated_doc_count": estimated_doc_count,
        "false_positive_rate": false_positive_rate,
        # Raw paths aren't deps — fingerprint them so swapping a path
        # invalidates the cache.
        "eval_data_sources": tuple(sorted(s for s in raw_paths if s not in (d.output_path for d in step_deps))),
        "exclude_eval_dirs": tuple(sorted(exclude_eval_dirs)),
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


def decon_step(
    *,
    name: str,
    normalized: StepSpec,
    eval_data_sources: list[StepSpec] | None = None,
    prebuilt_bloom: StepSpec | None = None,
    text_field: str = "text",
    ngram_length: int | None = 13,
    overlap_threshold: float = 0.5,
    estimated_doc_count: int = 1_000_000,
    false_positive_rate: float = 1e-9,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
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
        eval_data_sources: List of eval source steps (any zephyr-readable
            format) to build the bloom filter from. All eval sources are
            merged into one bloom; per-eval attribution is preserved in the
            ``eval_hash_index`` sidecar. Mutually exclusive with ``prebuilt_bloom``.
        prebuilt_bloom: Pre-built bloom StepSpec (output of
            :func:`build_eval_bloom_step` or :func:`merge_eval_blooms_step`).
            Mutually exclusive with ``eval_data_sources``.
        text_field: Text column name in both input and eval records.
        ngram_length: Word ngram length. ``None`` = exact whole-paragraph match.
        overlap_threshold: Per-paragraph overlap fraction needed to mark a record
            contaminated. Ignored in exact-paragraph mode.
        estimated_doc_count, false_positive_rate: Bloom sizing parameters.
            Ignored when ``prebuilt_bloom`` is set.
        worker_resources, max_workers: Zephyr execution knobs.
        output_path_prefix, override_output_path: StepSpec routing.
    """
    if (eval_data_sources is None) == (prebuilt_bloom is None):
        raise ValueError("provide exactly one of eval_data_sources or prebuilt_bloom")

    ngram: NGramConfig | None = (
        NGramConfig(ngram_length=ngram_length, overlap_threshold=overlap_threshold) if ngram_length is not None else None
    )

    # Mark stage hash_attrs: only what actually affects per-record marking
    # output. Bloom sizing is hashed into the bloom-producing step's
    # output_path (a dep below), so it propagates without polluting the
    # mark cache.
    hash_attrs: dict[str, Any] = {
        "text_field": text_field,
        "ngram_length": ngram_length,
        "overlap_threshold": overlap_threshold,
    }

    if prebuilt_bloom is not None:
        bloom_step = prebuilt_bloom
        return StepSpec(
            name=name,
            fn=lambda output_path: decon_to_parquet(
                normalized_data=read_artifact(normalized.output_path, NormalizedData),
                prebuilt_bloom_dir=bloom_step.output_path,
                output_path=output_path,
                text_field=text_field,
                ngram=ngram,
                worker_resources=worker_resources,
                max_workers=max_workers,
            ),
            deps=[normalized, bloom_step],
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
            normalized_data=read_artifact(normalized.output_path, NormalizedData),
            eval_data_sources=[s.output_path for s in eval_steps],
            output_path=output_path,
            text_field=text_field,
            ngram=ngram,
            estimated_doc_count=estimated_doc_count,
            false_positive_rate=false_positive_rate,
            worker_resources=worker_resources,
            max_workers=max_workers,
        ),
        deps=[normalized, *eval_steps],
        hash_attrs=inline_hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
