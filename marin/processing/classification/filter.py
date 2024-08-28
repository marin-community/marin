"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.filter --input_dir gs://marin-data/processed/fineweb/fw-v1.0/md/CC-MAIN-2020-24/ --output_dir gs://marin-data/filtered/dclm-fasttext-quality/fineweb/fw-v1.0/md/CC-MAIN-2020-24/

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.filter --input_dir gs://marin-data/processed/fineweb/fw-v1.0/md/CC-MAIN-2020-10/000_00000/ --attribute_name fineweb-edu-quality --threshold 3
"""

import json
import fsspec
import os
from typing import Dict, Any, List, Callable, Optional
import ray
from dataclasses import dataclass, field
import draccus

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_get_atomic_directories, fsspec_glob, rebase_file_path, fsspec_mkdirs, fsspec_exists, validate_marin_gcp_path



    
@dataclass
class ConsolidateConfig:
    """Config for Consolidation operation on Marin data"""
    input_path: str  # The input path to the Marin data
    output_path: str  # The output path to save the consolidated data
    max_tasks_in_flight: int = field(default=1000)  # The maximum number of flights in a task

    dedupe: bool = False  # Whether to dedupe the data or not
    dedupe_path: Optional[str] = None  # The path to save the deduped data

    fasttext: bool = False # Whether to filter the data based on fasttext scores or not
    fasttext_threshold: float = field(default=0.5)  # The threshold for the fasttext scores
    fasttext_path: Optional[str] = None  # The path to save the filtered data based on fasttext scores
    fasttext_name: Optional[str] = None # The name of the fasttext attribute

def is_high_quality(attributes: Dict[str, Any], attribute_name: str, threshold: float) -> bool:
    _ATTRIBUTE_NAME_TO_LABEL_DICT = {
        "dclm-fasttext-quality": "__label__hq",
        "dolma-fasttext-quality": "__label__hq",
        "fineweb-edu-quality": "score",
    }
    if attribute_name in attributes:
        quality_scores = attributes[attribute_name]
        label = _ATTRIBUTE_NAME_TO_LABEL_DICT[attribute_name]
        return quality_scores.get(label, 0) >= threshold

    return False


def remove_duplicates(input_data: Dict[str, Any], duplicate_spans: List[List[int]]) -> Dict[str, Any]:
    text = input_data['text']
    # Sort spans in reverse order to avoid index shifting
    sorted_spans = sorted(duplicate_spans, key=lambda x: x[1], reverse=True)
    # Remove duplicate spans
    for start, end, _ in sorted_spans:
        text = text[:start] + text[end:]
    
    # return the deduped data
    input_data['text'] = text
    return input_data

def quality_filter_func(input_data: Dict[str, Any], attributes_data: Dict[str, Any], attribute_name: str, threshold: float) -> Dict[str, Any]:
    if is_high_quality(attributes_data.get("attributes", {}), attribute_name, threshold):
        return input_data
    return None

def dedupe_filter_func(input_data: Dict[str, Any], attributes_data: Dict[str, Any], attribute_name: str, threshold: float) -> Dict[str, Any]:
    # Dolma dedupe has a fixed attribute name and a binary decision
    duplicate_spans = attributes_data.get("attributes", {}).get("duplicate_text", [])
    if duplicate_spans:
        return remove_duplicates(input_data, duplicate_spans)
    return input_data

def get_filter_func(attribute_name: str) -> Callable:
    if "dedupe" in attribute_name:
        return dedupe_filter_func
    elif "quality" in attribute_name:
        return quality_filter_func
    else:
        raise ValueError(f"Unknown attribute name: {attribute_name}")


def process_file(input_filename: str, output_filename: str, attributes_filename: str, attribute_name: str, threshold: float):
    if not fsspec_exists(attributes_filename):
        raise ValueError(f"Attributes file does not exist: {attributes_filename}")

    filter_func = get_filter_func(attribute_name)

    # First, read all attributes into a dictionary
    attributes_dict = {}
    with fsspec.open(attributes_filename, "rt", compression="gzip") as attributes_file:
        for attributes_line in attributes_file:
            attributes_data = json.loads(attributes_line)
            attributes_dict[attributes_data["id"]] = attributes_data

    with (
        fsspec.open(input_filename, "rt", compression="gzip") as input_file,
        fsspec.open(output_filename, "wt", compression="gzip") as output_file,
    ):
        for input_line in input_file:
            input_data = json.loads(input_line)
            
            # Look up attributes by ID
            attributes_data = attributes_dict.get(input_data["id"])
            
            if attributes_data is None:
                print(f"No attributes found for input ID: {input_data['id']}")
                continue

            filtered_data = filter_func(input_data, attributes_data, attribute_name, threshold)
            if filtered_data:
                output_line = json.dumps(filtered_data) + "\n"
                output_file.write(output_line)


@ray.remote
def process_dir(input_subdir: str, output_subdir: str, attribute_dir: str, attribute_name: str, threshold: float):
    files = fsspec_glob(os.path.join(input_subdir, "**/*.jsonl.gz"))
    for input_filename in files:
        output_filename = rebase_file_path(input_subdir, input_filename, output_subdir)
        attributes_filename = rebase_file_path(input_subdir, input_filename, attribute_dir)
        process_file(input_filename, output_filename, attributes_filename, attribute_name, threshold)


def filter_attribute(input_dir: str, output_dir: str, attribute_dir: str, attribute_name: str, threshold: float, max_tasks_in_flight: int):
    print(f"input_dir: {input_dir}, output_dir: {output_dir}, attribute_dir: {attribute_dir}, attribute_name: {attribute_name}, threshold: {threshold}")
    subdirectories = fsspec_get_atomic_directories(input_dir)
    print(f"subdirectories: {subdirectories}")
    responses = []
    for input_subdir in subdirectories:
        print(f"Processing {input_subdir}")
        if len(responses) > max_tasks_in_flight:
            ready_refs, responses = ray.wait(responses, num_returns=1)
            ray.get(ready_refs)

        output_subdir = rebase_file_path(input_dir, input_subdir, output_dir)
        fsspec_mkdirs(output_subdir)
        attribute_subdir = rebase_file_path(input_dir, input_subdir, attribute_dir)
        result_ref = process_dir.options(
            memory=500 * 1024 * 1024,
        ).remote(input_subdir, output_subdir, attribute_subdir, attribute_name, threshold)

        responses.append(result_ref)
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error: {e}")
    return output_dir
    
@draccus.wrap()
def main(cfg: ConsolidateConfig):
    if not cfg.dedupe and not cfg.fasttext:
        raise ValueError("At least one operation should be enabled")
    
    input_path = validate_marin_gcp_path(cfg.input_path)
    output_path = validate_marin_gcp_path(cfg.output_path)
    
    print(f"MAX_TASKS_IN_FLIGHT: {cfg.max_tasks_in_flight}")

    if cfg.dedupe:
        if cfg.dedupe_path is None:
            raise ValueError("dedupe_path is required for dedupe operation")
        dedupe_path = validate_marin_gcp_path(cfg.dedupe_path)
        print("Running dedupe filter")
        output_path = filter_attribute(input_path, output_path, dedupe_path, "dedupe", 0, cfg.max_tasks_in_flight)
        input_path = output_path  # Update input_path for potential next step

    if cfg.fasttext:
        if cfg.fasttext_path is None or cfg.fasttext_name is None:
            raise ValueError("Both fasttext_path and fasttext_name are required for fasttext operation")
        fasttext_path = validate_marin_gcp_path(cfg.fasttext_path)
        print(f"Running fasttext filter {cfg.fasttext_name} with threshold {cfg.fasttext_threshold}")
        output_path = filter_attribute(input_path, output_path, fasttext_path, cfg.fasttext_name, cfg.fasttext_threshold, cfg.max_tasks_in_flight)

    print(f"Processing complete. Final output path: {output_path}")
    
if __name__ == "__main__":
    main()