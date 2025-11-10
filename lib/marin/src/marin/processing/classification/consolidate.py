# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Consolidate takes a set of documents with corresponding attributes and writes
out a subset of the documents based on various filters defined with respect to
the attributes.  Handles two cases:
- Quality filtering produces attributes (e.g., fasttext-quality) with labels
  (e.g., __label__hq), filter on threshold.
- Deduplication produces attributes (e.g., duplicate_text).  Remove duplicates.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --memory=512MB --cluster=us-central2 \\
    --entry-point consolidate \\
    lib/marin/src/marin/processing/classification/consolidate.py \\
    --input_path gs://marin-us-central2/processed/documents \\
    --output_path gs://marin-us-central2/processed/filtered \\
    --filters '[{"type": "classify", "attribute_path": "gs://...attributes/quality",
                 "name": "fasttext", "label": "__label__hq", "lower_threshold": 0.5}]'
"""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum

import draccus
from marin.utils import (
    fsspec_exists,
    fsspec_glob,
    rebase_file_path,
)
from zephyr import Dataset, flow_backend, load_file


class FilterType(StrEnum):
    CLASSIFY = "classify"
    REMOVE_SPANS = "remove_spans"


logger = logging.getLogger("ray")


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


@dataclass(frozen=True)
class ConsolidateConfig:
    """Config for Consolidation operation on Marin data."""

    input_path: str
    """The input path to a directory (recursively) containing documents."""

    output_path: str
    """The output path to save the filtered (consolidated) data."""

    filters: list[FilterConfig]
    """List of filters to apply to the documents."""

    filetype: str = "jsonl.gz"
    """The filetype of the input data."""


# Dictionary-based navigation guide for extracting IDs from different corpus types
CORPUS_TYPE_TO_ID_GUIDE = {
    "dolma": {"key": "id"},  # Direct key access
    "dclm": {"key": "metadata", "nested": {"key": "WARC-Record-ID"}},  # Nested dictionary access
}


def remove_spans(text: str, spans: list[list[int]]) -> str:
    """
    Return `text` with `spans` removed.
    Example: text = "hello", spans = [[1, 4]], returns "ho"
    """
    # Sort spans in reverse order to avoid index shifting
    sorted_spans = sorted(spans, key=lambda x: x[1], reverse=True)
    for start, end, _ in sorted_spans:
        text = text[:start] + text[end:]

    return text


def passes_filter(doc: dict, filt: FilterConfig, attributes: dict | None) -> bool:
    """Check if document passes filter using its attributes.

    Args:
        doc: Document to check
        filt: Filter configuration
        attributes: Attributes for this document (or None if not found)

    Returns:
        True if document passes filter, False otherwise
    """
    if attributes is None:
        return False

    if filt.type == FilterType.CLASSIFY:
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
    elif filt.type == FilterType.REMOVE_SPANS:
        # not really a filter, but...
        spans = attributes[filt.name]
        new_text = remove_spans(doc["text"], spans)

        # If the deduped text doesn't have actual content, filter it out
        if new_text.strip() == "":
            return False

        # Mutate document in place with new text
        doc["text"] = new_text
        return True

    return True


def get_corpus_type(filename: str) -> str:
    if "dclm" in filename:
        return "dclm"
    else:  # Assume it's in dolma format
        return "dolma"


def get_id_column_name(corpus_type: str) -> str | dict[str, str]:
    """Get the top-level key that contains the ID (or the ID itself for flat structures)."""
    return CORPUS_TYPE_TO_ID_GUIDE[corpus_type]["key"]


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

    backend = flow_backend()
    result = backend.execute(
        Dataset.from_list(attr_paths)
        .flat_map(lambda p: load_file(p, columns=["attributes"]))
        .reduce(local_reducer=local_reducer, global_reducer=global_reducer)
    )

    combined_sketch = next(iter(result))
    threshold = combined_sketch.get_quantile_value(1 - keep_fraction)
    return threshold


def calculate_percentile_thresholds(config: ConsolidateConfig) -> list[FilterConfig]:
    """Resolve keep_fraction filters to lower_threshold using percentile calculation.

    Args:
        config: Consolidation configuration

    Returns:
        Updated filters with percentile thresholds resolved
    """
    updated_filters = []
    input_paths = fsspec_glob(os.path.join(config.input_path, f"**/*.{config.filetype}"))

    for filt in config.filters:
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

        # Build list of attribute file paths
        attr_paths = [rebase_file_path(config.input_path, inp, filt.attribute_path) for inp in input_paths]

        # Compute threshold using reduction
        threshold = _compute_percentile_threshold(attr_paths, filt.name, filt.label, filt.keep_fraction)
        logger.info(f"Calculated threshold {threshold} for {filt.name} to keep {filt.keep_fraction} of documents")
        updated_filters.append(replace(filt, lower_threshold=threshold, keep_fraction=None))

    return updated_filters


def process_file_shard(shard, filters: list[FilterConfig], input_base: str) -> Iterator[dict]:
    """Filter documents in a file shard based on provided filters."""
    # Shard has __iter__, iterate to get the single file path
    input_path = next(iter(shard))
    corpus_type = get_corpus_type(input_path)

    # Load all attribute files for this input file and build mapping
    attr_file_paths = {filt.name: rebase_file_path(input_base, input_path, filt.attribute_path) for filt in filters}

    attrs = {}
    for filt_name, attr_path in attr_file_paths.items():
        # Try exact path first, then look for compressed versions (.gz, .zst, etc.)
        if fsspec_exists(attr_path):
            final_attr_path = attr_path
        else:
            candidates = fsspec_glob(f"{attr_path}.*")
            if candidates:
                final_attr_path = candidates[0]
            else:
                logger.warning(f"Attribute file not found: {attr_path}")
                continue

        # Build dict mapping doc_id -> attributes
        attr_dict = {}
        for row in load_file(final_attr_path, columns=[get_id_column_name(corpus_type), "attributes"]):
            doc_id = extract_id(row, corpus_type)
            attr_dict[doc_id] = row["attributes"]

        attrs[filt_name] = attr_dict

    logger.info(f"Processing {input_path}")
    for doc in load_file(input_path):
        doc_id = extract_id(doc, corpus_type)

        # Apply all filters - if any filter fails, skip document
        keep_doc = True
        for filt in filters:
            filt_attrs = attrs.get(filt.name, {}).get(doc_id)
            if not passes_filter(doc, filt, filt_attrs):
                keep_doc = False
                break

        if keep_doc:
            yield doc


@draccus.wrap()
def consolidate(config: ConsolidateConfig):
    """Consolidate documents by applying filters based on attributes."""
    backend = flow_backend()

    filters = calculate_percentile_thresholds(config)
    input_paths = fsspec_glob(os.path.join(config.input_path, f"**/*.{config.filetype}"))
    logger.info(f"Consolidating {len(input_paths)} document files")

    output_pattern = f"{config.output_path}/part-{{shard:04d}}.{config.filetype}"

    results = list(
        backend.execute(
            Dataset.from_list(input_paths)
            .map_shard(lambda shard: process_file_shard(shard, filters, config.input_path))
            .write_jsonl(output_pattern)
        )
    )

    logger.info(f"Consolidation complete. Wrote {len(results)} output files")
