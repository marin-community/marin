"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.quality.filter --input_dir gs://marin-data/processed/fineweb/fw-v1.0/md/CC-MAIN-2020-24/ --output_dir gs://marin-data/filtered/dclm-fasttext-quality/fineweb/fw-v1.0/md/CC-MAIN-2020-24/
"""

import json
import gzip
import fsspec
from typing import Dict, Any
import ray

from marin.core.runtime import map_files_in_directory, cached_or_construct_output


def is_high_quality(attributes: Dict[str, Any], attribute_name: str) -> bool:
    _ATTRIBUTE_NAME_TO_THRESHOLD_DICT = {
        "dclm-fasttext-quality": 0.5,
        "dolma-fasttext-quality": 0.5,
        "fineweb-edu-quality": 3,
    }

    _ATTRIBUTE_NAME_TO_LABEL_DICT = {
        "dclm-fasttext-quality": "__label__hq",
        "dolma-fasttext-quality": "__label__hq",
        "fineweb-edu-quality": "score",
    }

    if attribute_name in attributes:
        quality_scores = attributes[attribute_name]
        label = _ATTRIBUTE_NAME_TO_LABEL_DICT[attribute_name]
        threshold = _ATTRIBUTE_NAME_TO_THRESHOLD_DICT[attribute_name]
        return quality_scores.get(label, 0) > threshold

    return False


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(input_filename: str, output_filename: str, file_format: str, attribute_name: str):
    print(f"Processing file: {input_filename}")

    attributes_filename = input_filename.replace(f"{file_format}/", f"attributes_{file_format}/{attribute_name}/")

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

            if is_high_quality(attributes_data.get("attributes", {}), attribute_name):
                output_file.write(input_line)


def main(input_dir: str, output_dir: str):
    ray.init()

    responses = map_files_in_directory(
        process_file.remote, input_dir, "**/*.jsonl.gz", output_dir, None, "md", "dclm-fasttext-quality"
    )

    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter high-quality data based on DCLM FastText scores.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing original data files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for filtered data")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
