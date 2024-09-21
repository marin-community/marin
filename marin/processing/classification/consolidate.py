import copy
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import draccus
import fsspec
import ray

from marin.core.runtime import cached_or_construct_output
from marin.utils import (
    fsspec_exists,
    fsspec_get_atomic_directories,
    fsspec_glob,
    fsspec_isdir,
    fsspec_mkdirs,
    rebase_file_path,
    validate_marin_gcp_path,
)


@dataclass
class FilterConfig:
    """Config for filtering operation on Marin data"""

    type: str
    attribute_path: str
    name: str
    label: str | None = None
    threshold: float | None = 0.5
    min_score: float = 0.0
    max_score: float = 1e6

    def __post_init__(self):
        if not (self.min_score < self.threshold < self.max_score):
            raise ValueError(
                f"""
                Scores must satisfy: \
                    min_score ({self.min_score}) < threshold ({self.threshold}) \
                    < max_score ({self.max_score})
                """
            )

        if "dedupe" in self.type:
            self.filter_func = dedupe_filter_func
        elif "classify" in self.type:
            self.filter_func = quality_filter_func
        else:
            raise ValueError(f"Unknown attribute type: {self.type}")


@dataclass
class ConsolidateConfig:
    """Config for Consolidation operation on Marin data"""

    input_path: str  # The input path to the Marin data
    output_path: str  # The output path to save the consolidated data
    filters: list[FilterConfig]  # The list of filters to apply

    max_tasks_in_flight: int = 1000  # The maximum number of flights in a task


def is_high_quality(
    attributes: dict[str, Any], attribute_name: str, threshold: float, label: str, min_score: float, max_score: float
) -> bool:
    if attribute_name in attributes:
        quality_scores = attributes[attribute_name]
        if label not in quality_scores:
            raise ValueError(f"Label '{label}' not found in quality scores for attribute '{attribute_name}'!")
        score = quality_scores.get(label, 0)
        return min_score <= score <= max_score and score >= threshold
    else:
        raise ValueError("No valid attribute found!")


def remove_duplicates(input_data: dict[str, Any], duplicate_spans: list[list[int]]) -> dict[str, Any]:
    text = input_data["text"]
    # Sort spans in reverse order to avoid index shifting
    sorted_spans = sorted(duplicate_spans, key=lambda x: x[1], reverse=True)
    # Remove duplicate spans
    for start, end, _ in sorted_spans:
        text = text[:start] + text[end:]

    # return the deduped data
    input_data["text"] = text
    return input_data


def quality_filter_func(
    input_data: dict[str, Any],
    attributes_data: dict[str, Any],
    attribute_name: str,
    threshold: float,
    label: str,
    min_score: float,
    max_score: float,
) -> dict[str, Any]:
    if is_high_quality(attributes_data, attribute_name, threshold, label, min_score, max_score):
        return input_data
    return None


def dedupe_filter_func(
    input_data: dict[str, Any],
    attributes_data: dict[str, Any],
    attribute_name: str,
    threshold: float,
    label: str,
    min_score: float,
    max_score: float,
) -> dict[str, Any]:
    # Dolma dedupe has a fixed attribute name and a binary decision
    # So there is no need to check the threshold
    duplicate_spans = attributes_data.get("attributes", {}).get(attribute_name, [])
    if duplicate_spans:
        return remove_duplicates(input_data, duplicate_spans)
    return input_data


def get_filter_func(filter_type: str) -> Callable:
    if "dedupe" in filter_type:
        return dedupe_filter_func
    elif "classify" in filter_type:
        return quality_filter_func
    else:
        raise ValueError(f"Unknown attribute name: {filter_type}")


def load_all_attributes(attribute_filenames: list[str]) -> dict[str, dict[str, Any]]:

    all_attributes = {}
    for filename in attribute_filenames:
        print(f"Loading attributes from {filename}")
        if not fsspec_exists(filename):
            print(f"Warning: Attribute file or directory {filename} does not exist. Skipping.")
            continue

        # If filename is a directory, glob all jsonl.gz files in it and its subdirectories
        if fsspec_isdir(filename):
            matching_files = fsspec_glob(os.path.join(filename, "**/*.jsonl.gz"))
        else:
            matching_files = [filename]

        for file in matching_files:
            print(f"Processing file: {file}")
            with fsspec.open(file, "rt", compression="gzip") as attr_file:
                for line in attr_file:
                    attr_data = json.loads(line)
                    doc_id = attr_data["id"]
                    if doc_id not in all_attributes:
                        all_attributes[doc_id] = {}
                    all_attributes[doc_id].update(attr_data.get("attributes", {}))

    print(f"Loaded attributes for {len(all_attributes)} documents")
    return all_attributes


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(
    input_filename: str,
    output_filename: str,
    all_attributes: dict[str, dict[str, Any]],
    filters: list[tuple[str, float, Callable]],
):
    with (
        fsspec.open(input_filename, "rt", compression="gzip") as input_file,
        fsspec.open(output_filename, "wt", compression="gzip") as output_file,
    ):
        for input_line in input_file:

            input_data = json.loads(input_line)
            doc_id = input_data["id"]

            if doc_id not in all_attributes:
                print(f"No attributes found for input ID: {doc_id}")
                continue

            attributes = all_attributes[doc_id]
            filtered_data = input_data

            for doc_filter in filters:
                filtered_data = doc_filter.filter_func(
                    filtered_data,
                    attributes,
                    filter.name,
                    filter.threshold,
                    filter.label,
                    filter.min_score,
                    filter.max_score,
                )
                if filtered_data is None:
                    break

            if filtered_data:
                output_line = json.dumps(filtered_data) + "\n"
                output_file.write(output_line)


def rebase_filter_filepath(input_subdir: str, input_path: str, doc_filter: FilterConfig) -> FilterConfig:
    """Changes the attribute path of a filter to point to a more specific file/subdirectory

    Similar to how we rebase_file_path to get the output path from the input path and the subdirectory
    we are processing, we need to rebase the attribute path to get the attribute subdirectory from the
    input path and the subdirectory we are processing.
    """
    attribute_path = rebase_file_path(input_subdir, input_path, doc_filter.attribute_path)
    assert fsspec_exists(attribute_path), f"Warning: Attribute path {attribute_path} does not exist."

    sub_filter = copy.deepcopy(doc_filter)
    sub_filter.attribute_path = attribute_path
    return sub_filter


@ray.remote
def process_directory(input_subdir: str, output_subdir: str, filters: list[FilterConfig]):
    files = fsspec_glob(os.path.join(input_subdir, "**/*.jsonl.gz"))
    for input_filename in files:
        output_filename = rebase_file_path(input_subdir, input_filename, output_subdir)
        file_filters = [rebase_filter_filepath(input_subdir, input_filename, doc_filter) for doc_filter in filters]
        process_file(input_filename, output_filename, file_filters)


def apply_filters(input_path: str, output_path: str, filters: list[FilterConfig], max_tasks_in_flight: int):
    subdirectories = fsspec_get_atomic_directories(input_path)
    print(f"subdirectories: {subdirectories}")

    tasks = []
    ready_refs = []
    for input_subdir in subdirectories:
        if len(tasks) > max_tasks_in_flight:
            ready_refs, tasks = ray.wait(tasks, num_returns=1)
            ray.get(ready_refs)

        print(f"Processing {input_subdir}")
        output_subdir = rebase_file_path(input_path, input_subdir, output_path)
        subdir_filters = [rebase_filter_filepath(input_path, input_subdir, doc_filter) for doc_filter in filters]
        fsspec_mkdirs(output_subdir)

        task = process_directory.remote(input_subdir, output_subdir, subdir_filters)
        tasks.append(task)

    try:
        ray.get(tasks)
    except Exception as e:
        print(f"Error processing: {e}")

    return output_path


@ray.remote
def main_ray(cfg: ConsolidateConfig):
    input_path = validate_marin_gcp_path(cfg.input_path)
    output_path = validate_marin_gcp_path(cfg.output_path)

    for doc_filter in cfg.filters:
        print(f"Filter enabled: {doc_filter.name} with threshold {doc_filter.threshold})")

    output_path = apply_filters(input_path, output_path, cfg.filters, cfg.max_tasks_in_flight)
    print(f"Processing complete. Final output path: {output_path}")


@draccus.wrap()
def main(cfg: ConsolidateConfig):
    ray.init()
    ray.get(main_ray.remote(cfg))


if __name__ == "__main__":
    main()
