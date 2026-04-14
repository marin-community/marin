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

Joins documents with their attribute files via Zephyr's ``sorted_merge_join``:
the datakit convention guarantees that attribute files share the input file
partitioning (1:1 file pairing, sorted by id), so each shard pairs with its
corresponding attribute shard without a shuffle. Multiple filters are chained
as successive left joins.
"""

import logging
import os
from collections.abc import Callable, Iterator
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


def _make_id_extractor(corpus_type: str) -> Callable[[dict], Any]:
    return lambda r: extract_id(r, corpus_type)


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

        attr_paths = _attribute_paths_for_filter(input_path, input_paths, filt, filetype)
        attr_paths = [p for p in attr_paths if p is not None]

        threshold = _compute_percentile_threshold(attr_paths, filt.name, filt.label, filt.keep_fraction)
        logger.info(f"Calculated threshold {threshold} for {filt.name} to keep {filt.keep_fraction} of documents")
        updated_filters.append(replace(filt, lower_threshold=threshold, keep_fraction=None))

    return updated_filters


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
    return None


def _attribute_paths_for_filter(input_base: str, input_paths: list[str], filt: FilterConfig, filetype: str) -> list[str]:
    """Resolve the 1:1 input→attribute paths for a filter.

    Raises if any shard's attribute file is missing — the datakit invariant is
    that all attribute files exist. ``keep_if_missing`` governs missing *rows*
    within a file, not missing files.
    """
    resolved = []
    for inp in input_paths:
        path = _resolve_attribute_path(input_base, inp, filt, filetype)
        if path is None:
            raise FileNotFoundError(
                f"No attribute file for filter '{filt.name}' corresponding to input {inp} "
                f"under {filt.attribute_path}"
            )
        resolved.append(path)
    return resolved


def _make_filter_combiner(filt: FilterConfig) -> Callable[[dict, dict | None], dict | None]:
    """Build a combiner for one filter.

    Called by ``sorted_merge_join`` with the current doc (``left``) and the
    matching attribute row or ``None`` (``right``). Returns the doc (possibly
    with mutated text for ``REMOVE_SPANS``) or ``None`` to drop it.
    """

    def combine(left: dict, right: dict | None) -> dict | None:
        if right is None:
            return left if filt.keep_if_missing else None

        attrs = right["attributes"]
        if filt.type == FilterType.CLASSIFY:
            return left if _is_valid(left, filt, attrs) else None
        if filt.type == FilterType.REMOVE_DOC:
            return left if not attrs.get(filt.name, False) else None
        assert filt.type == FilterType.REMOVE_SPANS
        mutated = _remove_spans_from_doc(left, filt, attrs)
        return mutated if mutated.get("text") else None

    return combine


def consolidate(
    *,
    input_path: str,
    output_path: str,
    filters: list[FilterConfig],
    filetype: str = "jsonl.gz",
    worker_resources: ResourceConfig | None = None,
) -> None:
    """Consolidate documents by applying filters based on attributes.

    Joins each input file with its (co-partitioned, sorted) attribute files via
    chained ``sorted_merge_join`` ops — one left join per filter, with the
    filter's keep/mutate/drop logic encoded in its combiner. No in-memory hash
    table is materialized.

    Args:
        input_path: Directory (recursively) containing input documents.
        output_path: Destination directory for filtered Parquet output.
        filters: List of filters to apply (see :class:`FilterConfig`).
        filetype: Extension of the input documents (default: ``"jsonl.gz"``).
        worker_resources: Optional Zephyr worker resource config (defaults to Zephyr defaults).
    """
    filters = calculate_percentile_thresholds(input_path=input_path, filters=filters, filetype=filetype)
    input_paths = sorted(fsspec_glob(os.path.join(input_path, f"**/*.{filetype}")))
    if not input_paths:
        raise ValueError(f"No input files matched {input_path}/**/*.{filetype}")
    logger.info(f"Consolidating {len(input_paths)} document files via {len(filters)} sorted_merge_join(s)")

    # Determine id key; assume a uniform corpus across shards (matches prior per-shard behavior
    # since datakit inputs are all "default" — "dclm" was the only alternative).
    corpus_type = "dclm" if any("dclm" in p for p in input_paths) else "default"
    id_key = CORPUS_TYPE_TO_ID_GUIDE[corpus_type]["key"]
    key_fn = _make_id_extractor(corpus_type)

    # Resolve attribute paths up front so the plan can be built before execution.
    filter_attr_paths = [
        (filt, _attribute_paths_for_filter(input_path, input_paths, filt, filetype)) for filt in filters
    ]

    ds = Dataset.from_list(input_paths).load_parquet()
    for filt, attr_paths in filter_attr_paths:
        attrs = Dataset.from_list(attr_paths).load_parquet(columns=[id_key, "attributes"])
        ds = ds.sorted_merge_join(
            attrs,
            left_key=key_fn,
            right_key=key_fn,
            combiner=_make_filter_combiner(filt),
            how="left",
        )
        # Drop rejected docs before the next join so its key extractor never sees None.
        ds = ds.filter(lambda r: r is not None)

    output_pattern = f"{output_path}/part-{{shard:04d}}.parquet"
    ctx = ZephyrContext(
        name="consolidate-filter",
        **({"resources": worker_resources} if worker_resources else {}),
    )
    results = ctx.execute(ds.write_parquet(output_pattern)).results
    logger.info(f"Consolidation complete. Wrote {len(results)} output files")
