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
import numpy as np
import ray
from transformers import AutoTokenizer

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

    threshold: float | None = 0.5
    """Keep documents where the value is above this."""

    percentile_threshold: float | None = None
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

    max_total_tokens: int | None = None
    """The maximum total number of tokens to keep. This is measured by llama-3 tokenizer."""

    max_tasks_in_flight: int = 1000  # The maximum number of flights in a task


@ray.remote
class TokenCounter:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.num_tokens = 0

    def get_num_tokens(self) -> int:
        return self.num_tokens

    def add_text(self, text: str, max_total_tokens: int):
        """
        Add `text` to the token counter, trimming if necessary to not exceed `max_total_tokens`.

        Returns:
            The trimmed `text` if necessary, otherwise the original `text`.
            If the number of tokens exceeds `max_total_tokens`, we return an empty string to signal
            that the text should not be added.
        """

        if self.num_tokens >= max_total_tokens:  # Already at max tokens
            return ""
        elif self.num_tokens + self.count_tokens(text) > max_total_tokens:  # Exceeding max tokens at this current point
            text = self.get_trimmed_text(text, max_total_tokens)  # Trim text to max tokens
            self.num_tokens = max_total_tokens
        else:  # Not exceeding max tokens
            self.num_tokens += self.count_tokens(text)

        return text

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_trimmed_text(self, text: str, max_total_tokens: int) -> str:
        """
        Get the text that has been trimmed to `max_total_tokens`.
        """
        return self.tokenizer.decode(self.tokenizer.encode(text)[:max_total_tokens])


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

    elif doc_filter.type == FILTER_TYPE_REMOVE_SPANS:
        # Remove spans (e.g., because they are duplicates)
        new_text = remove_spans(input_data["text"], attributes[doc_filter.name])
        return dict(input_data, text=new_text)

    else:
        raise ValueError(f"Unknown filter type: {doc_filter.type}")


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(
    input_path: str,
    output_path: str,
    filters: list[FilterConfig],
    token_counter: TokenCounter,
    max_total_tokens: int | None = None,
):
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
        if max_total_tokens is not None:
            num_tokens = ray.get(token_counter.get_num_tokens.remote())
            if num_tokens >= max_total_tokens:
                break

        # Read document and attributes for that document
        input_data = json.loads(input_line)
        all_attributes = [json.loads(attr_file.readline()) if attr_file else None for attr_file in attribute_files]

        # Apply filters
        for doc_filter, attributes in zip(filters, all_attributes, strict=True):
            if attributes is None:
                continue
            try:
                assert attributes["id"] == input_data["id"]
                input_data = apply_filter(input_data, doc_filter, attributes["attributes"])
            except Exception as e:
                logger.error(f"Error applying filter {doc_filter} to line {num_total}: {e}")
                input_data = None  # Skip this example

            if input_data is None:
                break

        # Write output
        if input_data is not None:
            if max_total_tokens is not None:
                output_text = ray.get(token_counter.add_text.remote(input_data["text"], max_total_tokens))
                if output_text == "":  # Finished processing
                    break
                input_data["text"] = output_text

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
def get_scores(attribute_filename: str, attribute_name: str, label: str) -> list[float]:
    scores = []
    with fsspec.open(attribute_filename, "rt", compression="gzip") as f:
        for line in f:
            attributes = json.loads(line)
            scores.append(attributes["attributes"][attribute_name][label])
    return scores


def calculate_percentile_threshold(scores: list[list[float]], percentile: float) -> float:
    scores = np.concatenate(scores)
    return np.percentile(scores, percentile)


@ray.remote
def consolidate(config: ConsolidateConfig):
    input_paths = fsspec_glob(os.path.join(config.input_path, "**/*.jsonl.gz"))
    logger.info(f"Consolidating {len(input_paths)} documents")

    token_counter = TokenCounter.remote()

    updated_filters = []
    for doc_filter in config.filters:
        assert (
            doc_filter.percentile_threshold is None or doc_filter.threshold is None
        ), "Cannot specify both percentile threshold and threshold. Please specify only one."

        # Calculate percentile threshold if specified, otherwise use the provided threshold
        if doc_filter.percentile_threshold is not None:
            attribute_paths = [
                rebase_file_path(config.input_path, input_path, doc_filter.attribute_path) for input_path in input_paths
            ]
            scores = ray.get(
                [
                    get_scores.remote(attribute_path, doc_filter.name, doc_filter.label)
                    for attribute_path in attribute_paths
                ]
            )
            threshold = calculate_percentile_threshold(scores, doc_filter.percentile_threshold)
            updated_filters.append(replace(doc_filter, threshold=threshold, percentile_threshold=None))
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

        task = process_file.remote(input_path, output_path, filters, token_counter, config.max_total_tokens)
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
