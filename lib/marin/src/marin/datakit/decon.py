# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontaminate normalized data against eval sources.

Reads datakit-normalized Parquet (``id``, ``text``, ``partition_id``), builds an
in-memory bloom filter from the eval text, and emits a co-partitioned Parquet
attributes dataset marking which records overlap with eval text.

Schema of the emitted Parquet attributes:
    id           : string  — matches source document id
    partition_id : int     — matches source partition
    contaminated : bool    — max paragraph overlap meets the threshold
    max_overlap  : float   — highest paragraph overlap fraction in [0, 1]

Output is co-partitioned with the source: one ``part-NNNNN-of-MMMMM.parquet``
per input partition, preserving the source filenames so consolidate can
sorted-merge-join without a shuffle.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import dupekit
from fray import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ShardInfo, ZephyrContext, counters, write_parquet_file
from zephyr.readers import load_file

from marin.execution.step_spec import StepSpec

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
        counters: Aggregated zephyr counters from the marking pipeline.
    """

    version: str = "v1"
    output_dir: str
    num_partitions: int
    counters: dict[str, int]


def _bloom_hash(x: str) -> int:
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "big")


def _extract_ngrams(text: str, n: int, stride: int) -> Iterator[str]:
    tokens = text.split()
    for i in range(0, len(tokens) - n + 1, stride + 1):
        yield " ".join(tokens[i : i + n])


def _extract_features(text: str, ngram: NGramConfig | None) -> Iterator[str]:
    """Yield matchable features: ngrams within each paragraph, or whole paragraphs."""
    for para in text.split("\n"):
        if not para:
            continue
        if ngram is not None:
            yield from _extract_ngrams(para, ngram.ngram_length, ngram.stride)
        else:
            yield para


def _paragraph_overlap(paragraph: str, bf: dupekit.Bloom, ngram: NGramConfig | None) -> float:
    """Fraction of ngrams in *paragraph* that hit the bloom; 0 or 1 for exact-paragraph mode."""
    if ngram is None:
        return 1.0 if _bloom_hash(paragraph) in bf else 0.0
    ngrams = list(_extract_ngrams(paragraph, ngram.ngram_length, ngram.stride))
    if not ngrams:
        # Paragraph too short for n-grams — fall back to exact paragraph match.
        return 1.0 if _bloom_hash(paragraph) in bf else 0.0
    matches = sum(1 for ng in ngrams if _bloom_hash(ng) in bf)
    return matches / len(ngrams)


def _discover_parquet_partitions(input_path: str) -> list[str]:
    """Walk *input_path* recursively, return sorted list of .parquet files."""
    fs, resolved = url_to_fs(input_path)
    protocol = input_path.split("://")[0] if "://" in input_path else ""

    discovered: list[str] = []
    for root, _dirs, files in fs.walk(resolved):
        for fname in files:
            if fname.startswith(".") or not fname.endswith(".parquet"):
                continue
            full = os.path.join(root, fname)
            discovered.append(f"{protocol}://{full}" if protocol else full)
    discovered.sort()
    return discovered


def _discover_eval_files(eval_paths: list[str]) -> Iterator[str]:
    """Walk all *eval_paths* recursively and yield any zephyr-readable file."""
    for source in eval_paths:
        fs, resolved = url_to_fs(source)
        protocol = source.split("://")[0] if "://" in source else ""
        for root, _dirs, files in fs.walk(resolved):
            for fname in files:
                if fname.startswith("."):
                    continue
                full = os.path.join(root, fname)
                yield f"{protocol}://{full}" if protocol else full


def _build_filter(
    eval_paths: list[str],
    bloom_path: str,
    text_field: str,
    ngram: NGramConfig | None,
    estimated_doc_count: int,
    false_positive_rate: float,
) -> None:
    """Build a bloom filter from eval source(s) and persist it to *bloom_path*.

    Eval sets are small enough that we read them serially in-process. If this
    becomes a bottleneck (very large eval suite), wrap in a Zephyr pipeline.
    """
    bf = dupekit.Bloom(estimated_doc_count, false_positive_rate)
    n_records = 0
    for path in _discover_eval_files(eval_paths):
        for record in load_file(path):
            text = record.get(text_field)
            if not text:
                continue
            for feat in _extract_features(str(text), ngram):
                bf.add(_bloom_hash(feat))
            n_records += 1
    logger.info("decon: built bloom from %d eval records → %s", n_records, bloom_path)
    fs, path = url_to_fs(bloom_path)
    bloom_dir = os.path.dirname(path)
    if bloom_dir:
        fs.makedirs(bloom_dir, exist_ok=True)
    with fs.open(path, "wb") as f:
        f.write(bf.save_bytes())


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
                    text = record.get(text_field, "") or ""
                    paragraph_scores = (_paragraph_overlap(para, bf, ngram) for para in str(text).split("\n") if para)
                    max_score = max(paragraph_scores, default=0.0)
                    contaminated = max_score > 0 and max_score >= threshold
                    counters.increment("decon/contaminated" if contaminated else "decon/clean")
                    yield {
                        "id": record["id"],
                        "partition_id": record["partition_id"],
                        "contaminated": contaminated,
                        "max_overlap": max_score,
                    }

            out_filename = os.path.basename(input_path)
            out_path = f"{output_dir.rstrip('/')}/{out_filename}"
            result = write_parquet_file(rows_for(input_path), output_path=out_path)
            yield result

    return mark_shard


def decon_to_parquet(
    *,
    input_path: str,
    decontaminate_source: str | list[str],
    output_path: str,
    text_field: str = "text",
    ngram: NGramConfig | None = None,
    estimated_doc_count: int = 1_000_000,
    false_positive_rate: float = 1e-9,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
) -> DeconAttributes:
    """Mark records in *input_path* that overlap with text in *decontaminate_source*.

    Args:
        input_path: Datakit-normalized Parquet directory. Records must have
            ``id``, ``text``, and ``partition_id`` columns.
        decontaminate_source: Eval source directory (or list of dirs). Any
            zephyr-readable file format. Read once to build the bloom filter.
        output_path: Directory for co-partitioned Parquet attributes. One
            output file is written per input partition, preserving filenames.
        text_field: Text column name in both input and eval records.
        ngram: Word-ngram matching config. ``None`` = exact whole-paragraph match.
            ``ngram.overlap_threshold`` gates which paragraphs are marked
            contaminated; exact-paragraph mode records any non-zero match.
        estimated_doc_count, false_positive_rate: Bloom sizing parameters; size
            for expected total *ngram* count across the eval suite (not record
            count). Defaults handle ~1M unique ngrams cleanly.
        worker_resources: Per-shard resource request for the marking pipeline.
            Defaults to 2 CPU / 4GB RAM.
        max_workers: Max Zephyr workers. Defaults to Zephyr's own default.

    Returns:
        :class:`DeconAttributes` describing the output dataset and counters.
    """
    eval_paths = [decontaminate_source] if isinstance(decontaminate_source, str) else list(decontaminate_source)
    if not eval_paths:
        raise ValueError("decontaminate_source must be non-empty")

    files = _discover_parquet_partitions(input_path)
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {input_path}")
    num_partitions = len(files)
    logger.info("decon: %s → %s, %d input partitions", input_path, output_path, num_partitions)

    bloom_path = os.path.join(output_path, "_bloom", "filter.bin")
    _build_filter(
        eval_paths=eval_paths,
        bloom_path=bloom_path,
        text_field=text_field,
        ngram=ngram,
        estimated_doc_count=estimated_doc_count,
        false_positive_rate=false_positive_rate,
    )

    pipeline = (
        Dataset.from_list(files)
        .reshard(num_shards=num_partitions)
        .map_shard(_make_marker(bloom_path, output_path, text_field, ngram))
    )

    resources = worker_resources or ResourceConfig(cpu=2, ram="4g")
    ctx_kwargs: dict[str, Any] = {"name": "decon-mark", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = ZephyrContext(**ctx_kwargs)
    outcome = ctx.execute(pipeline)

    return DeconAttributes(
        output_dir=output_path,
        num_partitions=num_partitions,
        counters=dict(outcome.counters),
    )


def decon_step(
    *,
    name: str,
    normalized: StepSpec,
    decontaminate_source: list[StepSpec],
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
    """Create a StepSpec that decontaminates a normalized dataset against eval sources.

    Args:
        name: Step name (e.g. ``"fineweb/decon"``).
        normalized: Upstream datakit normalize step whose output is the input.
        decontaminate_source: List of eval source steps (any zephyr-readable
            format) to build the bloom filter from.
        text_field: Text column name in both input and eval records.
        ngram_length: Word ngram length. ``None`` = exact whole-paragraph match.
        overlap_threshold: Per-paragraph overlap fraction needed to mark a record
            contaminated. Ignored in exact-paragraph mode.
        estimated_doc_count, false_positive_rate: Bloom sizing parameters.
        worker_resources, max_workers: Zephyr execution knobs.
        output_path_prefix, override_output_path: StepSpec routing.
    """
    ngram: NGramConfig | None = (
        NGramConfig(ngram_length=ngram_length, overlap_threshold=overlap_threshold) if ngram_length is not None else None
    )

    hash_attrs: dict[str, Any] = {
        "text_field": text_field,
        "ngram_length": ngram_length,
        "overlap_threshold": overlap_threshold,
        "estimated_doc_count": estimated_doc_count,
        "false_positive_rate": false_positive_rate,
    }

    return StepSpec(
        name=name,
        fn=lambda output_path: decon_to_parquet(
            input_path=normalized.output_path,
            decontaminate_source=[s.output_path for s in decontaminate_source],
            output_path=output_path,
            text_field=text_field,
            ngram=ngram,
            estimated_doc_count=estimated_doc_count,
            false_positive_rate=false_positive_rate,
            worker_resources=worker_resources,
            max_workers=max_workers,
        ),
        deps=[normalized, *decontaminate_source],
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )
