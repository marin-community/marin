"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.filter --input_dir gs://marin-data/processed/fineweb/fw-v1.0/md/CC-MAIN-2020-24/ --output_dir gs://marin-data/filtered/dclm-fasttext-quality/fineweb/fw-v1.0/md/CC-MAIN-2020-24/

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.filter --input_dir gs://marin-data/processed/fineweb/fw-v1.0/md/CC-MAIN-2020-10/000_00000/ --file_format md --attribute_name fineweb-edu-quality --threshold 3
"""

import json
import gzip
import fsspec
import os
from typing import Dict, Any
import ray

from marin.core.runtime import map_files_in_directory, cached_or_construct_output
from marin.utils import fsspec_get_atomic_directories, fsspec_glob, rebase_file_path, fsspec_mkdirs, fsspec_exists


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


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(input_filename: str, output_filename: str, file_format: str, attribute_name: str, threshold: float):
    print(f"Processing file: {input_filename}")

    attributes_filename = input_filename.replace(f"{file_format}/", f"attributes_{file_format}/{attribute_name}/")

    if not fsspec_exists(attributes_filename):
        print(f"Attributes file does not exist: {attributes_filename}")
        return

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

            if is_high_quality(attributes_data.get("attributes", {}), attribute_name, threshold):
                output_file.write(input_line)


@ray.remote
def process_dir(input_subdir: str, output_subdir: str, file_format: str, attribute_name: str, threshold: float):
    files = fsspec_glob(os.path.join(input_subdir, "**/*.jsonl.gz"))
    for input_filename in files:
        output_filename = rebase_file_path(input_subdir, input_filename, output_subdir)
        process_file(input_filename, output_filename, file_format, attribute_name, threshold)


def main(input_dir: str, output_dir: str, file_format: str, attribute_name: str, threshold: float):
    ray.init()

    if not output_dir:
        output_dir = input_dir.replace("documents", f"filtered/{attribute_name}-{threshold}")
    
    print(f"Output directory is: {output_dir}")

    MAX_FLIGHTS_IN_TASK = 1000
    subdirectories = fsspec_get_atomic_directories(input_dir)
    responses = []
    for input_subdir in subdirectories:
        if len(responses) > MAX_FLIGHTS_IN_TASK:
            ready_refs, responses = ray.wait(responses, num_returns=1)
            ray.get(ready_refs)

        output_subdir = rebase_file_path(input_dir, input_subdir, output_dir)
        fsspec_mkdirs(output_subdir)

        result_ref = process_dir.options(
            memory=500 * 1024 * 1024,
        ).remote(input_subdir, output_subdir, file_format, attribute_name, threshold)

        responses.append(result_ref)
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter high-quality data based on DCLM FastText scores.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing original data files")
    parser.add_argument("--file_format", type=str, default="md", required=True, help="File format of the data")
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory to save filtered data. If not set we assume Dolma directory structure")
    parser.add_argument(
        "--attribute_name",
        type=str,
        default="dclm-fasttext-quality",
        required=True,
        help="Attribute name of the quality score",
    )
    parser.add_argument("--threshold", type=float, default=0.5, required=True, help="Threshold for the quality score")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.file_format, args.attribute_name, args.threshold)
