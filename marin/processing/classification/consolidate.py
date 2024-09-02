import json
import fsspec
import os
from typing import Dict, Any, List, Tuple, Callable, Optional
import ray
from dataclasses import dataclass, field
import draccus

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_get_atomic_directories, fsspec_glob, rebase_file_path, fsspec_mkdirs, fsspec_exists, validate_marin_gcp_path, fsspec_isdir


    
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
    else:
        raise ValueError("No valid attriubte found!")


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
    if is_high_quality(attributes_data, attribute_name, threshold):
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

def load_all_attributes(attribute_filenames: List[str]) -> Dict[str, Dict[str, Any]]:
    
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
def process_file(input_filename: str, output_filename: str, all_attributes: Dict[str, Dict[str, Any]], filters: List[Tuple[str, float, Callable]]):
    with fsspec.open(input_filename, "rt", compression="gzip") as input_file, \
         fsspec.open(output_filename, "wt", compression="gzip") as output_file:
        for input_line in input_file:

            input_data = json.loads(input_line)
            doc_id = input_data["id"]
            
            if doc_id not in all_attributes:
                print(f"No attributes found for input ID: {doc_id}")
                continue
            
            attributes = all_attributes[doc_id]
            filtered_data = input_data
            
            for attr_name, threshold, filter_func in filters:
                filtered_data = filter_func(filtered_data, attributes, attr_name, threshold)
                if filtered_data is None:
                    break
            
            if filtered_data:
                output_line = json.dumps(filtered_data) + "\n"
                output_file.write(output_line)

@ray.remote
def process_directory(input_subdir: str, output_subdir: str, all_attributes: Dict[str, Dict[str, Any]], filters: List[Tuple[str, float, Callable]]):
    files = fsspec_glob(os.path.join(input_subdir, "**/*.jsonl.gz"))
    for input_filename in files:
        output_filename = rebase_file_path(input_subdir, input_filename, output_subdir)
        process_file(input_filename, output_filename, all_attributes, filters)

def apply_filters(input_dir: str, output_dir: str, attribute_files: List[str], filters: List[Tuple[str, float, Callable]], max_tasks_in_flight: int):
    
    all_attributes = load_all_attributes(attribute_files)

    
    subdirectories = fsspec_get_atomic_directories(input_dir)
    print(f"subdirectories: {subdirectories}")

    tasks = []
    for input_subdir in subdirectories:
        print(f"Processing {input_subdir}")
        output_subdir = rebase_file_path(input_dir, input_subdir, output_dir)
        fsspec_mkdirs(output_subdir)

        task = process_directory.remote(input_subdir, output_subdir, all_attributes, filters)
        tasks.append(task)

        if len(tasks) >= max_tasks_in_flight:
            ray.get(tasks.pop(0))

    ray.get(tasks)
    return output_dir

@draccus.wrap()
def main(cfg: ConsolidateConfig):
    if not cfg.dedupe and not cfg.fasttext:
        raise ValueError("At least one operation should be enabled")

    input_path = validate_marin_gcp_path(cfg.input_path)
    output_path = validate_marin_gcp_path(cfg.output_path)

    attribute_files = []
    filters = []

    if cfg.dedupe:
        if cfg.dedupe_path is None:
            raise ValueError("dedupe_path is required for dedupe operation")
        dedupe_path = validate_marin_gcp_path(cfg.dedupe_path)
        attribute_files.append(dedupe_path)
        filters.append(("dedupe", 0, dedupe_filter_func))
        print("Dedupe filter enabled")

    if cfg.fasttext:
        if cfg.fasttext_path is None or cfg.fasttext_name is None:
            raise ValueError("Both fasttext_path and fasttext_name are required for fasttext operation")
        fasttext_path = validate_marin_gcp_path(cfg.fasttext_path)
        attribute_files.append(fasttext_path)
        filters.append((cfg.fasttext_name, cfg.fasttext_threshold, quality_filter_func))
        print(f"Fasttext filter enabled: {cfg.fasttext_name} with threshold {cfg.fasttext_threshold}")

    output_path = apply_filters(input_path, output_path, attribute_files, filters, cfg.max_tasks_in_flight)
    print(f"Processing complete. Final output path: {output_path}")

if __name__ == "__main__":
    main()