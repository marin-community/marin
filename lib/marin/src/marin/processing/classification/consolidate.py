# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Consolidate takes a set of documents with corresponding attributes and writes
out a subset of the documents based on various filters defined with respect to
the attributes.  Handles three cases:
- Quality filtering produces attributes (e.g., fasttext-quality) with labels
  (e.g., __label__hq), filter on threshold.
- Span removal produces attributes (e.g., duplicate_text spans). Remove text spans.
- Document removal via attribute produced by deduplication.

Joins documents with their attribute files via a streaming map-side merge:
the datakit convention guarantees that attribute files share the input file
partitioning (1:1 file pairing, sorted by id), so each shard can be processed
independently with constant memory per filter.
"""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from fray import ResourceConfig
from marin.utils import (
    fsspec_exists,
    fsspec_glob,
    rebase_file_path,
)
from zephyr import Dataset, ZephyrContext
from zephyr.readers import InputFileSpec, load_file


class FilterType(StrEnum):
    CLASSIFY = "classify"
    REMOVE_SPANS = "remove_spans"
    REMOVE_DOC = "remove_docs"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterConfig:
    """Config for filtering operation on Marin data"""

    type: FilterType
    """The type of filter to apply."""

    attribute_path: str
    """Base path where the files with the attributes are stored."""

    name: str
    """Name of attribute to use for filtering."""

    label: str | None = None
    """The label under the attribute name."""

    lower_threshold: float | None = None
    """Keep documents where the value is above this."""

    keep_fraction: float | None = None
    """Keep documents where the score is in the top percentile. Calculates the threshold from the entire dataset."""

    upper_threshold: float | None = None
    """Keep documents where the value is below this."""

    reverse: bool = False
    """Reverse the filter."""

    attribute_filetype: str | None = None
    """File extension for attribute files (e.g. 'jsonl.gz', 'vortex'). If None, uses the input filetype."""

    keep_if_missing: bool = False
    """If True, keep docs that have no attribute entry. If False (default), reject them."""


# Dictionary-based navigation guide for extracting IDs from different corpus types
CORPUS_TYPE_TO_ID_GUIDE = {
    "default": {"key": "id"},  # Direct key access
    "dclm": {"key": "metadata", "nested": {"key": "WARC-Record-ID"}},  # Nested dictionary access
}


def _is_valid(doc: dict, filt: FilterConfig, attributes: dict) -> bool:
    assert filt.type == FilterType.CLASSIFY
    attribute_value = attributes[filt.name]

    # Handle nested attributes structure if a label is specified
    if filt.label is not None:
        if isinstance(attribute_value, dict) and filt.label in attribute_value:
            value = attribute_value[filt.label]
        else:
            raise ValueError(f"Label {filt.label} not found in attribute {filt.name} for document {doc}")
    else:
        value = attribute_value

    # Check both lower and upper bounds if specified
    accepted = True
    if filt.lower_threshold is not None and value < filt.lower_threshold:
        accepted = False
    if filt.upper_threshold is not None and value > filt.upper_threshold:
        accepted = False

    if filt.reverse:
        accepted = not accepted

    return accepted


def _remove_spans_from_doc(doc: dict, filt: FilterConfig, attributes: dict) -> dict:
    def _remove_spans(text: str, spans: list[list[int]]) -> str:
        """Return ``text`` with ``spans`` removed.

        Example: text = "hello", spans = [[1, 4]], returns "ho"
        """
        # Sort spans in reverse order to avoid index shifting
        sorted_spans = sorted(spans, key=lambda x: x[1], reverse=True)
        for span in sorted_spans:
            start, end = span[0], span[1]
            text = text[:start] + text[end:]

        return text

    spans = attributes[filt.name]
    new_text = _remove_spans(doc["text"], spans)
    return {**doc, "text": new_text}


def extract_id(row: dict, corpus_type: str) -> str:
    """Extract ID from row based on corpus type.

    Recursively navigates nested structures as defined in CORPUS_TYPE_TO_ID_GUIDE."""
    guide = CORPUS_TYPE_TO_ID_GUIDE[corpus_type]

    # grab the key, then navigate nested if needed
    val = row[guide["key"]]

    while "nested" in guide:
        nested = guide["nested"]
        assert isinstance(nested, dict)
        val = val[nested["key"]]
        guide = nested

    return val


def _compute_percentile_threshold(
    attr_paths: list[str], attr_name: str, attr_label: str | None, keep_fraction: float
) -> float:
    """Compute percentile threshold for a single filter using DDSketch reduction.

    Args:
        attr_paths: Paths to attribute files
        attr_name: Name of attribute to extract
        attr_label: Optional label within attribute (for nested dicts)
        keep_fraction: Fraction of documents to keep (0-1)

    Returns:
        Threshold value at the (1 - keep_fraction) quantile
    """
    from ddsketch import DDSketch

    def local_reducer(rows: Iterator[dict], attr_name: str = attr_name, attr_label: str | None = attr_label) -> DDSketch:
        """Build DDSketch from rows in a single shard."""
        sketch = DDSketch()
        for row in rows:
            attributes = row["attributes"]
            value = attributes[attr_name][attr_label] if attr_label else attributes[attr_name]
            sketch.add(value)
        return sketch

    def global_reducer(sketches: Iterator[DDSketch]) -> DDSketch:
        """Merge all shard sketches into one."""
        combined = DDSketch()
        for sketch in sketches:
            combined.merge(sketch)
        return combined

    ctx = ZephyrContext(name="consolidate-stats")
    result = ctx.execute(
        Dataset.from_list(attr_paths)
        .load_file()
        .select("attributes")
        .reduce(local_reducer=local_reducer, global_reducer=global_reducer)
    ).results

    combined_sketch = next(iter(result))
    threshold = combined_sketch.get_quantile_value(1 - keep_fraction)
    return threshold


def calculate_percentile_thresholds(
    *,
    input_path: str,
    filters: list[FilterConfig],
    filetype: str = "jsonl.gz",
) -> list[FilterConfig]:
    """Resolve ``keep_fraction`` filters to ``lower_threshold`` via percentile calculation.

    Returns a new list of filters with percentile-based thresholds resolved.
    """
    updated_filters = []
    input_paths = fsspec_glob(os.path.join(input_path, f"**/*.{filetype}"))

    for filt in filters:
        # Validate threshold configuration
        if filt.keep_fraction is not None and filt.lower_threshold is not None:
            raise ValueError("Cannot specify both keep_fraction and lower_threshold. Please specify only one.")

        # Skip if no percentile calculation needed
        if filt.keep_fraction is None:
            updated_filters.append(filt)
            continue

        if not (0 < filt.keep_fraction < 1):
            raise ValueError("keep_fraction must be between 0 and 1")

        # Only applies to CLASSIFY filters
        if filt.type != FilterType.CLASSIFY:
            logger.warning(f"keep_fraction only applies to CLASSIFY filters, ignoring for {filt.name}")
            updated_filters.append(filt)
            continue

        attr_paths = [_resolve_attribute_path(input_path, inp, filt, filetype) for inp in input_paths]
        attr_paths = [p for p in attr_paths if p is not None]

        threshold = _compute_percentile_threshold(attr_paths, filt.name, filt.label, filt.keep_fraction)
        logger.info(f"Calculated threshold {threshold} for {filt.name} to keep {filt.keep_fraction} of documents")
        updated_filters.append(replace(filt, lower_threshold=threshold, keep_fraction=None))

    return updated_filters


class _PeekableIter:
    """Iterator with one-element lookahead used by the merge walk.

    ``peek()`` returns the next item without consuming it (or ``None`` if exhausted);
    ``advance()`` discards the current peeked item.
    """

    __slots__ = ("_has_peek", "_it", "_peeked")

    def __init__(self, iterator):
        self._it = iter(iterator)
        self._peeked = None
        self._has_peek = False

    def peek(self):
        if not self._has_peek:
            try:
                self._peeked = next(self._it)
                self._has_peek = True
            except StopIteration:
                return None
        return self._peeked

    def advance(self) -> None:
        if self._has_peek:
            self._has_peek = False
            self._peeked = None
        else:
            try:
                next(self._it)
            except StopIteration:
                pass


def _resolve_attribute_path(input_base: str, input_path: str, filt: FilterConfig, filetype: str) -> str | None:
    """Map an input file path to its attribute file path, with glob fallback for compression suffixes."""
    new_extension = f".{filt.attribute_filetype}" if filt.attribute_filetype else f".{filetype}"
    attr_path = rebase_file_path(
        input_base,
        input_path,
        filt.attribute_path,
        new_extension=new_extension,
        old_extension=f".{filetype}",
    )
    if fsspec_exists(attr_path):
        return attr_path
    candidates = fsspec_glob(f"{attr_path}.*")
    if candidates:
        return candidates[0]
    logger.warning(f"Attribute file not found: {attr_path}")
    return None


def process_file_shard(*, shard, filters: list[FilterConfig], input_base: str, filetype: str) -> Iterator[dict]:
    """Filter documents in a file shard using a streaming merge join with each attribute file.

    Both the input file and each attribute file are sorted by id (datakit convention),
    so we can walk all streams in lockstep with constant memory per filter.
    """
    input_path = next(iter(shard))
    corpus_type = "dclm" if "dclm" in input_path else "default"
    id_key = CORPUS_TYPE_TO_ID_GUIDE[corpus_type]["key"]

    # Open one peekable attribute stream per filter (1:1 file pairing).
    attr_iters: list[_PeekableIter] = []
    for filt in filters:
        attr_path = _resolve_attribute_path(input_base, input_path, filt, filetype)
        if attr_path is None:
            attr_iters.append(_PeekableIter(iter([])))
            continue
        attr_iters.append(_PeekableIter(load_file(InputFileSpec(path=attr_path, columns=[id_key, "attributes"]))))

    logger.info(f"Processing {input_path}")
    for doc in load_file(input_path):
        doc_id = extract_id(doc, corpus_type)
        keep = True

        for filt, attr_iter in zip(filters, attr_iters, strict=True):
            if not keep:
                # NOTE: if at any point the doc is rejected, stop processing further filters.
                break

            # Advance the attribute stream past any orphan ids (id < doc_id).
            while True:
                peek = attr_iter.peek()
                if peek is None or extract_id(peek, corpus_type) >= doc_id:
                    break
                attr_iter.advance()

            peek = attr_iter.peek()
            if peek is None or extract_id(peek, corpus_type) != doc_id:
                # No matching attribute for this doc.
                if filt.keep_if_missing:
                    continue
                keep = False
                break

            filt_attrs: dict[str, Any] = peek["attributes"]
            attr_iter.advance()

            if filt.type == FilterType.CLASSIFY:
                keep = _is_valid(doc, filt, filt_attrs)
            elif filt.type == FilterType.REMOVE_DOC:
                keep = not filt_attrs.get(filt.name, False)
            else:
                assert filt.type == FilterType.REMOVE_SPANS
                doc = _remove_spans_from_doc(doc, filt, filt_attrs)

        if keep and doc.get("text"):
            yield doc


def consolidate(
    *,
    input_path: str,
    output_path: str,
    filters: list[FilterConfig],
    filetype: str = "jsonl.gz",
    worker_resources: ResourceConfig | None = None,
) -> None:
    """Consolidate documents by applying filters based on attributes.

    Joins each input file with its (co-partitioned, sorted) attribute files via a
    streaming merge — no in-memory hash table is materialized. Output is written
    to Parquet in ``output_path``.

    Args:
        input_path: Directory (recursively) containing input documents.
        output_path: Destination directory for filtered Parquet output.
        filters: List of filters to apply (see :class:`FilterConfig`).
        filetype: Extension of the input documents (default: ``"jsonl.gz"``).
        worker_resources: Optional Zephyr worker resource config (defaults to Zephyr defaults).
    """
    filters = calculate_percentile_thresholds(input_path=input_path, filters=filters, filetype=filetype)
    input_paths = fsspec_glob(os.path.join(input_path, f"**/*.{filetype}"))
    logger.info(f"Consolidating {len(input_paths)} document files")

    output_pattern = f"{output_path}/part-{{shard:04d}}.parquet"

    ctx = ZephyrContext(name="consolidate-filter", **({"resources": worker_resources} if worker_resources else {}))
    results = ctx.execute(
        Dataset.from_list(input_paths)
        .map_shard(
            lambda shard, _: process_file_shard(shard=shard, filters=filters, input_base=input_path, filetype=filetype)
        )
        .write_parquet(output_pattern)
    ).results

    logger.info(f"Consolidation complete. Wrote {len(results)} output files")
