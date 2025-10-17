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

NOTE(chris): Remove the usage of read_dataset and write_dataset.
Right now, this code utilizes read_dataset and write_dataset but this is
probably not needed because we could theoretically stream the data in as well.
The reason why we need it currently is for two reasons:
1. We have attributes in one file and documents in another file. However, we are not
guaranteed that they are read in the same order. So, our solution is to just
read in all of the attributes and then map the id of the document to the id of the attribute.
We can then filter based on the value. However, this means we cannot simply stream the data in
since we have to store the mapping of id -> attributes.
2. We use some of the builtin Huggingface Dataset .map and .filter functions which may not work with
the streaming data paradigm (it might but not sure).
"""

import logging
import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import draccus
import numpy as np
import ray
import pandas as pd
import datasets

from marin.core.runtime import cached_or_construct_output
from marin.utils import (
    fsspec_exists,
    fsspec_glob,
    rebase_file_path,
)

FILTER_TYPE_CLASSIFY = "classify"
FILTER_TYPE_REMOVE_SPANS = "remove_spans"

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

    threshold: float | None = None
    """Keep documents where the value is above this."""

    keep_fraction: float | None = None
    """Keep documents where the score is in the top percentile. Calculates the threshold from the entire dataset."""

    upper_threshold: float | None = None
    """Keep documents where the value is below this."""

    reverse: bool = False
    """Reverse the filter."""


@dataclass(frozen=True)
class ConsolidateConfig:
    """Config for Consolidation operation on Marin data"""

    input_path: str
    """The input path to a directory (recursively) containing documents."""

    output_path: str  # The output path to save the consolidated data
    """The output path to save the filtered (consolidated) data."""

    filters: list[FilterConfig]
    """List of filters to apply to the documents."""

    filetype: str = "jsonl.gz"
    """The filetype of the input data."""

    max_tasks_in_flight: int = 1000  # The maximum number of flights in a task

    ray_memory_limit_gb: float = 0.5
    """The memory limit for the task in GB."""


# Dictionary-based navigation guide for extracting IDs from different corpus types
CORPUS_TYPE_TO_ID_GUIDE = {
    "dolma": {"key": "id"},  # Direct key access
    "dclm": {"key": "metadata", "nested": {"key": "WARC-Record-ID"}},  # Nested dictionary access
}


def read_dataset(input_filename: str, columns: list[str] | None = None):
    """Read in a data source and return as a Huggingface Dataset

    Args:
        input_filename: str
            The path to the input file. Currently supports .jsonl.gz and .parquet

    Returns:
        datasets.Dataset: A Huggingface Dataset in-memory without using the disk
    """
    datasets.disable_caching()  # Disabling caching or else it spills to disk to cache
    datasets.logging.set_verbosity_warning()
    # We use pandas to read in the file so that we don't have to materialize
    # the entire dataset in disk since we have limited disk space.
    # Huggingface datasets loads the dataset into disk first and mmaps.
    if input_filename.endswith(".jsonl.gz"):
        df = pd.read_json(input_filename, compression="gzip", lines=True)
    elif input_filename.endswith(".jsonl.zst"):
        df = pd.read_json(input_filename, compression="zstd", lines=True)
    elif input_filename.endswith(".parquet"):
        df = pd.read_parquet(input_filename, columns=columns)
    else:
        raise ValueError(f"Unsupported filetype: {input_filename}")

    return datasets.Dataset.from_pandas(df)


def write_dataset(dataset, output_filename: str):
    """Writes a Huggingface Dataset to a file (remote or local)"""
    if output_filename.endswith(".jsonl.gz"):
        dataset.to_json(output_filename, compression="gzip")
    elif output_filename.endswith(".jsonl.zst"):
        df_pandas = dataset.to_pandas()
        df_pandas.to_json(output_filename, orient="records", compression="zstd", lines=True)
        # dataset.to_json(output_filename, to_json_kwargs={"compression": "zstd", "lines": True})
    elif output_filename.endswith(".parquet"):
        dataset.to_parquet(output_filename)
    else:
        raise ValueError(f"Unsupported filetype: {output_filename}")


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


def apply_filter_remove_spans(
    input_data: dict, doc_filter: FilterConfig, id_to_attributes: dict[str, Any]
) -> dict | None:
    attributes = id_to_attributes[input_data["id"]]
    # Remove spans (e.g., because they are duplicates)
    new_text = remove_spans(input_data["text"], attributes[doc_filter.name])

    # if the deduped text doesn't have actual content, we can skip this document
    # this is to avoid cases where there are just newlines or spaces
    if new_text.strip() == "":
        return dict(input_data, keep=False)

    return dict(input_data, text=new_text, keep=True)


def apply_filter_classify(input_data: dict, doc_filter: FilterConfig, id_to_attributes: dict[str, Any]) -> bool:
    attributes = id_to_attributes[input_data["id"]]

    # Get value from attributes
    attribute_value = attributes[doc_filter.name]

    # Handle nested attributes structure if a label is specified
    accepted = True
    if doc_filter.label is not None:
        if isinstance(attribute_value, dict) and doc_filter.label in attribute_value:
            value = attribute_value[doc_filter.label]
        else:
            logger.warning(
                f"Label {doc_filter.label} not found in attribute {doc_filter.name} for document {input_data['id']}"
            )
            accepted = False
    else:
        # If no label specified, use the attribute value directly
        # This handles cases like compression_ratio where the value is a scalar
        value = attribute_value

    # Check both lower and upper bounds if specified
    if doc_filter.threshold is not None and value < doc_filter.threshold:
        accepted = False
    if doc_filter.upper_threshold is not None and value > doc_filter.upper_threshold:
        accepted = False

    if doc_filter.reverse:
        accepted = not accepted

    return accepted


def get_corpus_type(filename: str) -> str:
    if "dclm" in filename:
        return "dclm"
    else:  # Assume it's in dolma format
        return "dolma"


def get_id_column_name(corpus_type: str) -> str:
    """Get the top-level key that contains the ID (or the ID itself for flat structures)."""
    return CORPUS_TYPE_TO_ID_GUIDE[corpus_type]["key"]


def get_nested_id_object(row: dict, corpus_type: str) -> dict:
    """
    Extract the ID from a row using a dictionary-based navigation guide.

    The guide specifies how to traverse the nested dictionary structure to find the ID.
    For example, for DCLM dataset, the ID is nested within the "metadata" column under "WARC-Record-ID":
    {"metadata": {"WARC-Record-ID": "1234567890"}}

    Our guide for this would be:
    {"key": "metadata", "nested": {"key": "WARC-Record-ID"}}

    Args:
        row: The row containing the document data
        corpus_type: The type of corpus (e.g., "dclm", "dolma")

    Returns:
        The row with the "id" field set to the extracted ID
    """
    guide = CORPUS_TYPE_TO_ID_GUIDE[corpus_type]

    # Start with the top-level key
    current_value = row[guide["key"]]

    # If there's no nested structure, we're done
    if "nested" not in guide:
        row["id"] = current_value
        return row

    # Navigate through nested keys
    current_guide = guide["nested"]
    while True:
        current_value = current_value[current_guide["key"]]

        # If there are no more nested levels, we've reached the ID
        if "nested" not in current_guide:
            break

        # Move to the next level in the guide
        current_guide = current_guide["nested"]

    row["id"] = current_value
    return row


def read_attributes_as_dict(attribute_filename: str) -> dict[str, Any]:
    """Given some attribute filename, return a dictionary mapping from id to attributes

    Inputs:
        attribute_filename: str
            The path to the attribute file
        filetype: str
            The filetype of the attribute file
    """

    id_column_name = get_id_column_name(get_corpus_type(attribute_filename))
    corpus_type = get_corpus_type(attribute_filename)
    table = read_dataset(attribute_filename, columns=[id_column_name, "attributes"])
    table = table.map(lambda row: get_nested_id_object(row, corpus_type))

    data = {}
    for row_id, attr in zip(table["id"], table["attributes"], strict=True):
        data[row_id] = attr
    return data


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(
    input_path: str,
    output_path: str,
    filters: list[FilterConfig],
):
    """
    Read documents from `input_path`, apply the `filters` (involves reading the
    attributes paths) and writes the subset of documents to `output_path`.
    """

    logger.info(f"Processing {input_path} and {[doc_filter.attribute_path for doc_filter in filters]}")

    # Open all files simultaneously, and read in parallel.
    attribute_files = []
    for doc_filter in filters:
        if fsspec_exists(doc_filter.attribute_path):
            attribute_files.append(read_attributes_as_dict(doc_filter.attribute_path))
        else:
            logger.warning(f"Attribute file not found: {doc_filter.attribute_path}")
            attribute_files.append(None)

    dataset = read_dataset(input_path)
    dataset = dataset.map(lambda row: get_nested_id_object(row, get_corpus_type(input_path)))

    total_examples = len(dataset)

    for doc_filter, id_to_attributes in zip(filters, attribute_files, strict=True):
        print(f"Applying filter {doc_filter.name} with label {doc_filter.label} with dataset length {len(dataset)}")
        if id_to_attributes is None:
            continue

        if doc_filter.type == FILTER_TYPE_CLASSIFY:
            dataset = dataset.filter(
                partial(apply_filter_classify, doc_filter=doc_filter, id_to_attributes=id_to_attributes)
            )
        elif doc_filter.type == FILTER_TYPE_REMOVE_SPANS:
            dataset = dataset.map(
                partial(apply_filter_remove_spans, doc_filter=doc_filter, id_to_attributes=id_to_attributes)
            )
            dataset = dataset.filter(lambda x: x["keep"])
        else:
            raise ValueError(f"Unknown filter type: {doc_filter.type}")

    write_dataset(dataset, output_path)

    total_kept = len(dataset)

    logger.info(f"Kept {total_kept}/{total_examples} from {input_path}")


@ray.remote
def get_scores(attribute_filename: str, attribute_name: str, label: str) -> np.ndarray:
    scores = np.array([])
    attributes = read_attributes_as_dict(attribute_filename)
    for _, attr in attributes.items():
        scores = np.append(scores, attr[attribute_name][label])
    return scores


def calculate_percentile_threshold(
    base_input_path: str,
    input_paths: list[str],
    attribute_path: str,
    attribute_name: str,
    label: str,
    keep_fraction: float,
) -> float:
    from ddsketch import DDSketch

    attribute_paths = [rebase_file_path(base_input_path, input_path, attribute_path) for input_path in input_paths]
    scores_refs = [get_scores.remote(attribute_path, attribute_name, label) for attribute_path in attribute_paths]

    sketch = DDSketch()
    for scores_ref in scores_refs:
        scores = ray.get(scores_ref)
        for score in scores:
            sketch.add(score)

    threshold = sketch.get_quantile_value(1 - keep_fraction)
    logger.info(f"Calculated the threshold {threshold} to keep {keep_fraction} of the documents")

    return threshold


@ray.remote
def consolidate(config: ConsolidateConfig):
    input_paths = fsspec_glob(os.path.join(config.input_path, f"**/*.{config.filetype}"))
    logger.info(f"Consolidating {len(input_paths)} documents")

    updated_filters = []
    for doc_filter in config.filters:
        assert (
            doc_filter.keep_fraction is None or doc_filter.threshold is None
        ), "Cannot specify both percentile threshold and threshold. Please specify only one."

        if doc_filter.keep_fraction is not None:
            assert doc_filter.keep_fraction > 0 and doc_filter.keep_fraction < 1, "Keep fraction must be between 0 and 1"

        # Calculate the minimum threshold required to keep `keep_fraction` of the documents
        if doc_filter.keep_fraction is not None and doc_filter.type == FILTER_TYPE_CLASSIFY:
            threshold = calculate_percentile_threshold(
                config.input_path,
                input_paths,
                doc_filter.attribute_path,
                doc_filter.name,
                doc_filter.label,
                doc_filter.keep_fraction,
            )
            updated_filters.append(replace(doc_filter, threshold=threshold, keep_fraction=None))
        else:
            updated_filters.append(doc_filter)

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
            for doc_filter in updated_filters
        ]
        output_path = rebase_file_path(config.input_path, input_path, config.output_path)

        task = process_file.options(memory=config.ray_memory_limit_gb * 1024 * 1024 * 1024, num_cpus=2).remote(
            input_path, output_path, filters
        )
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
