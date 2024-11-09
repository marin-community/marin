"""
Consolidate takes a set of documents with corresponding attributes and writes
out a subset of the documents based on various filters defined with respect to
the attributes.  Handles two cases:
- Quality filtering produces attributes (e.g., fasttext-quality) with labels
  (e.g., __label__hq), filter on threshold.
- Deduplication produces attributes (e.g., duplicate_text).  Remove duplicates.
"""

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import ray

from marin.core.runtime import cached_or_construct_output
from marin.processing.classification.inference import read_dataset, write_dataset
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

    threshold: float | None = 0.5
    """Keep documents where the value is above this."""

    keep_fraction: float | None = None
    """Keep documents where the score is in the top percentile. Calculates the threshold from the entire dataset."""


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

    memory_limit_gb: float = 0.5
    """The memory limit for the task in GB."""


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


def apply_filter(input_data: dict, doc_filter: FilterConfig, id_to_attributes: dict[str, Any]) -> bool | dict | None:
    attributes = id_to_attributes[input_data["id"]]
    if doc_filter.type == FILTER_TYPE_CLASSIFY:
        # Check attribute >= threshold?
        scores = attributes[doc_filter.name]
        score = scores[doc_filter.label]

        if score >= doc_filter.threshold:
            return True

        return False

    elif doc_filter.type == FILTER_TYPE_REMOVE_SPANS:

        # Remove spans (e.g., because they are duplicates)
        new_text = remove_spans(input_data["text"], attributes[doc_filter.name])

        # if the deduped text doesn't have actual content, we can skip this document
        # this is to avoid cases where there are just newlines or spaces
        if new_text.strip() == "":
            return dict(input_data, keep=False)

        return dict(input_data, text=new_text, keep=True)

    else:
        raise ValueError(f"Unknown filter type: {doc_filter.type}")


def apply_filter_batch_func(doc_filter, id_to_attributes):
    # The huggingface dataset map API passes a batch of data to the function and returns a batch of data.
    # We have additional data to pass in such as doc_filter and id_to_attributes, so we define a wrapper
    # function here.
    def apply_filter_func(batch):
        return apply_filter(batch, doc_filter, id_to_attributes)

    return apply_filter_func


def get_filetype(input_path: str):
    filename = input_path.split("/")[-1]
    filetype = "".join(Path(filename).suffixes)
    filetype = filetype.lstrip(".")
    return filetype


def read_attributes_as_dict(attribute_filename: str) -> dict[str, Any]:
    """Given some attribute filename, return a dictionary mapping from id to attributes

    Inputs:
        attribute_filename: str
            The path to the attribute file
        filetype: str
            The filetype of the attribute file
    """
    table = read_dataset(attribute_filename)
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

    total_examples = len(dataset)

    for doc_filter, id_to_attributes in zip(filters, attribute_files, strict=True):
        print(f"Applying filter {doc_filter.name} with label {doc_filter.label} with dataset length {len(dataset)}")
        if id_to_attributes is None:
            continue

        apply_filter_func = apply_filter_batch_func(doc_filter, id_to_attributes)

        if doc_filter.type == FILTER_TYPE_CLASSIFY:
            dataset = dataset.filter(apply_filter_func)
        elif doc_filter.type == FILTER_TYPE_REMOVE_SPANS:
            dataset = dataset.map(apply_filter_func)
        else:
            raise ValueError(f"Unknown filter type: {doc_filter.type}")

    # In the previous map step, we set the "keep" key for documents that should be removed
    dataset = dataset.filter(lambda x: x["keep"])

    write_dataset(dataset, output_path)

    total_kept = len(dataset)

    logger.info(f"Kept {total_kept}/{total_examples} from {input_path}")


@ray.remote
def get_scores(attribute_filename: str, attribute_name: str, label: str) -> list[float]:
    scores = []
    attributes = read_attributes_as_dict(attribute_filename)
    for _, attr in attributes.items():
        scores.append(attr[attribute_name][label])
    return scores


def calculate_percentile_threshold(scores: list[list[float]], keep_fraction: float) -> float:
    percentile = 100 - keep_fraction * 100
    scores = np.concatenate(scores)
    return np.percentile(scores, percentile)


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
            attribute_paths = [
                rebase_file_path(config.input_path, input_path, doc_filter.attribute_path) for input_path in input_paths
            ]
            scores = ray.get(
                [
                    get_scores.remote(attribute_path, doc_filter.name, doc_filter.label)
                    for attribute_path in attribute_paths
                ]
            )
            threshold = calculate_percentile_threshold(scores, doc_filter.keep_fraction)
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

        task = process_file.options(memory=config.memory_limit_gb * 1024 * 1024 * 1024, num_cpus=2).remote(
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
