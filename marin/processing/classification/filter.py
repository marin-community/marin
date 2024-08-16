"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.filter --input_dir gs://marin-data/processed/fineweb/fw-v1.0/md/CC-MAIN-2020-24/ --output_dir gs://marin-data/filtered/dclm-fasttext-quality/fineweb/fw-v1.0/md/CC-MAIN-2020-24/

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.filter --input_dir gs://marin-data/processed/fineweb/fw-v1.0/md/CC-MAIN-2020-10/000_00000/ --attribute_name fineweb-edu-quality --threshold 3
"""

import json
import gzip
import fsspec
import os
from typing import Dict, Any, List
import ray

from marin.core.runtime import map_files_in_directory, cached_or_construct_output
from marin.utils import fsspec_get_atomic_directories, fsspec_glob, rebase_file_path, fsspec_mkdirs, fsspec_exists, validate_gcp_path


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
    deduped_data = input_data.copy()
    text = deduped_data['text']
    
    # Sort spans in reverse order to avoid index shifting
    sorted_spans = sorted(duplicate_spans, key=lambda x: x[1], reverse=True)
    
    # Remove duplicate spans
    for start, end, _ in sorted_spans:
        text = text[:start] + text[end:]
    
    deduped_data['text'] = text
    return deduped_data

@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(input_filename: str, output_filename: str, attributes_filename: str, attribute_name: str, threshold: float):
    if not fsspec_exists(attributes_filename):
        raise ValueError(f"Attributes file does not exist: {attributes_filename}")

    with (
        fsspec.open(input_filename, "rt", compression="gzip") as input_file,
        fsspec.open(attributes_filename, "rt", compression="gzip") as attributes_file,
        fsspec.open(output_filename, "wt", compression="gzip") as output_file,
    ):
        for input_line, attributes_line in zip(input_file, attributes_file):
            input_data = json.loads(input_line)
            attributes_data = json.loads(attributes_line)
            
            if attributes_data["id"] != input_data["id"]:
                print(f"ID of attribute row and input row do not match: {attributes_data['id']} != {input_data['id']}")
                continue

            if attribute_name == "dedupe":
                duplicate_spans = attributes_data.get("attributes", {}).get("duplicate_text", [])
                if duplicate_spans:
                    deduped_data = remove_duplicates(input_data, duplicate_spans)
                    output_line = json.dumps(deduped_data) + "\n"
                    output_file.write(output_line)
                else:
                    output_file.write(input_line)
            elif is_high_quality(attributes_data.get("attributes", {}), attribute_name, threshold):
                output_file.write(input_line)


@ray.remote
def process_dir(input_subdir: str, output_subdir: str, attribute_dir: str, attribute_name: str, threshold: float):
    files = fsspec_glob(os.path.join(input_subdir, "**/*.jsonl.gz"))
    for input_filename in files:
        output_filename = rebase_file_path(input_subdir, input_filename, output_subdir)
        attributes_filename = rebase_file_path(input_subdir, input_filename, attribute_dir)
        process_file(input_filename, output_filename, attributes_filename, attribute_name, threshold)


def main(input_dir: str, output_dir: str, attribute_dir: str, attribute_name: str, threshold: float):
    ray.init()

    input_dir = validate_gcp_path(input_dir)
    output_dir = validate_gcp_path(output_dir)
    attribute_dir = validate_gcp_path(attribute_dir)

    MAX_FLIGHTS_IN_TASK = 1000
    subdirectories = fsspec_get_atomic_directories(input_dir)
    responses = []
    for input_subdir in subdirectories:
        if len(responses) > MAX_FLIGHTS_IN_TASK:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter high-quality data based on DCLM FastText scores.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing original data files")
    parser.add_argument("--attributes_dir", type=str, required=True, help="Directory containing attribute files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save filtered data")
    parser.add_argument(
        "--attribute_name",
        type=str,
        default="dclm-fasttext-quality",
        required=True,
        choices=["dclm-fasttext-quality", "dolma-fasttext-quality", "fineweb-edu-quality", "dedupe"],
        help="Attribute name of the quality score",
    )
    parser.add_argument("--threshold", type=float, default=0.5, required=False, help="Threshold for the quality score, ignored for dedupe")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.attributes_dir, args.attribute_name, args.threshold)
