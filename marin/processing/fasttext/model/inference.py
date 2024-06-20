"""
Usage:
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.fasttext.model.inference --input_dir gs://marin-data/processed/fineweb/fw-v1.0/CC-MAIN-2023-50/000_00000/  --output_dir gs://marin-data/scratch/chrisc/quality/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/
"""
import argparse
import datetime
import json
import os

import fsspec
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from marin.processing.fasttext.model.classifier import FasttextQualityClassifier

def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def make_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(element) for element in obj)
    elif not is_json_serializable(obj):
        return str(obj)
    return obj

@ray.remote(memory= 5 * 1024 * 1024 * 1024, runtime_env={"pip": ["fasttext"]})
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(input_filename, output_filename):
    ds = ray.data.read_json(input_filename)

    print(f"[*] Read in dataset {input_filename}")
    
    # NOTE(chris): Could we parallelize using map_batches?
    # There are sometimes some issues with initialization when concurrency increases
    results = ds.map(
        FasttextQualityClassifier,
        # concurrency=(1,16),
        concurrency=1,
        fn_constructor_args=("",)
    )

    print("[*] Finished quality classification")
    
    with fsspec.open(input_filename, "rt", compression="gzip") as f_in:
        with fsspec.open(output_filename, "wt", compression="gzip") as f_out:
            for row in results.iter_rows():                
                json_row = json.dumps(make_serializable(row))
                f_out.write(json_row + "\n")

def main(input_dir, output_dir):
    ray.init()

    # TODO(chris): At some point we should be able to place the fasttext object in the global object store. Currently seems not serializable
    # Load fasttext classifier in the global object store
    # model = fasttext.load_model(MODEL_PATH)
    # model_ref = ray.put(model)

    # output_filename = "gs://marin-data/scratch/chrisc/0_processed_md.jsonl.gz"
    # for filename in FILENAMES:
    #     ray.get(process_file.remote(filename, output_filename))

    responses = map_files_in_directory(process_file.remote, input_dir, "**/*_md.jsonl.gz", output_dir)

    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to convert fineweb html to markdown.")
    parser.add_argument('--input_dir', type=str, help='Path to the unprocessed data directory', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to store data updated with quality scores directory', required=True)

    args = parser.parse_args()

    main(input_dir=args.input_dir, output_dir=args.output_dir)