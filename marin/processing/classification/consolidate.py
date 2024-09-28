"""
Consolidate takes a set of documents with corresponding attributes and writes
out a subset of the documents based on various filters defined with respect to
the attributes.  Handles two cases:
- Quality filtering produces attributes (e.g., fasttext-quality) with labels
  (e.g., __label__hq), filter on threshold.
- Deduplication produces attributes (e.g., duplicate_text).  Remove duplicates.
"""

import json
import logging
import os
from dataclasses import dataclass, replace
from typing import Any

import draccus
import fsspec
import ray

from marin.utils import (
    fsspec_exists,
    fsspec_glob,
    rebase_file_path,
)

FILTER_TYPE_CLASSIFY = "classify"
FILTER_TYPE_SPLICE = "splice"

logger = logging.getLogger("ray")


@dataclass(frozen=True)
class FilterConfig:
    """Config for filtering operation on Marin data"""

    type: str
    """The type of filter to apply."""

    attribute_path: str
    """Base path where the files with the attributes are stored."""

    name: str
    """Name of attribute to use for filtering."""

    label: str | None = None
    """The label under the attribute name."""

    threshold: float | None = 0.5
    """Keep documents where the value is above this."""


@dataclass(frozen=True)
class ConsolidateConfig:
    """Config for Consolidation operation on Marin data"""

    input_path: str
    """The input path to a directory (recursively) containing documents."""

    output_path: str  # The output path to save the consolidated data
    """The output path to save the filtered (consolidated) data."""

    filters: list[FilterConfig]
    """List of filters to apply to the documents."""

    max_tasks_in_flight: int = 1000  # The maximum number of flights in a task


def remove_spans(text: str, spans: list[list[int]]) -> str:
    """
    Return `text` with `spans` removed.
    Example: text = "hello", spans = [[1, 4]], returns "ho"
    """
    # Sort spans in reverse order to avoid index shifting
    sorted_spans = sorted(spans, key=lambda x: x[1], reverse=True)

    # Remove spans
    for start, end, _ in sorted_spans:
        text = text[:start] + text[end:]

    return text


def apply_filter(input_data: dict, doc_filter: FilterConfig, attributes: dict[str, Any]) -> dict | None:
    if doc_filter.type == FILTER_TYPE_CLASSIFY:
        # Check attribute >= threshold?
        scores = attributes[doc_filter.name]
        score = scores[doc_filter.label]
        return input_data if score >= doc_filter.threshold else None

    elif doc_filter.type == FILTER_TYPE_SPLICE:
        # Remove spans (e.g., because they are duplicates)
        new_text = remove_spans(input_data["text"], attributes[doc_filter.name])
        return dict(input_data, text=new_text)

    else:
        raise ValueError(f"Unknown filter type: {doc_filter.type}")


@ray.remote
def process_file(input_path: str, filters: list[FilterConfig], output_path: str):
    """
    Read documents from `input_path`, apply the `filters` (involves reading the
    attributes paths) and writes the subset of documents to `output_path`.
    """

    logger.info(f"Processing {input_path} and {[doc_filter.attribute_path for doc_filter in filters]}")

    # Open all files simultaneously, and read in parallel.
    input_file = fsspec.open(input_path, "rt", compression="gzip").open()
    attribute_files = []
    for doc_filter in filters:
        if fsspec_exists(doc_filter.attribute_path):
            attribute_files.append(fsspec.open(doc_filter.attribute_path, compression="gzip").open())
        else:
            logger.warning(f"Attribute file not found: {doc_filter.attribute_path}")
            attribute_files.append(None)
    output_file = fsspec.open(output_path, "wt", compression="gzip").open()

    num_kept = 0
    num_total = 0
    for input_line in input_file:
        num_total += 1

        # Read document and attributes for that document
        input_data = json.loads(input_line)
        all_attributes = [json.loads(attr_file.readline()) if attr_file else None for attr_file in attribute_files]

        # Apply filters
        for doc_filter, attributes in zip(filters, all_attributes, strict=False):
            if attributes is None:
                continue
            try:
                input_data = apply_filter(input_data, doc_filter, attributes["attributes"])
            except Exception as e:
                logger.error(f"Error applying filter {doc_filter} to line {num_total}: {e}")
                input_data = None  # Skip this example

            if input_data is None:
                break

        # Write output
        if input_data is not None:
            num_kept += 1
            print(json.dumps(input_data), file=output_file)

    logger.info(f"Kept {num_kept}/{num_total} from {input_path}")

    # Close all files
    input_file.close()
    for attr_file in attribute_files:
        if attr_file:
            attr_file.close()
    output_file.close()


@ray.remote
def consolidate(config: ConsolidateConfig):
    input_paths = fsspec_glob(os.path.join(config.input_path, "**/*.jsonl.gz"))
    logger.info(f"Consolidating {len(input_paths)} documents")

    tasks = []
    ready_refs = []
    for input_path in input_paths:
        if len(tasks) > config.max_tasks_in_flight:
            ready_refs, tasks = ray.wait(tasks, num_returns=1)
            ray.get(ready_refs)

        filters = [
            replace(
                doc_filter, attribute_path=rebase_file_path(config.input_path, input_path, doc_filter.attribute_path)
            )
            for doc_filter in config.filters
        ]
        output_path = rebase_file_path(config.input_path, input_path, config.output_path)

        task = process_file.remote(input_path, filters, output_path)
        tasks.append(task)

    try:
        ray.get(tasks)
    except Exception as e:
        print(f"Error processing: {e}")


@draccus.wrap()
def main(cfg: ConsolidateConfig):
    ray.get(consolidate.remote(cfg))


if __name__ == "__main__":
    main()
