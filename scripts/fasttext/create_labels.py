import argparse
import json
import os

import fsspec
import ray

from marin.core.runtime import RayConfig, TaskConfig, cached_or_construct_output, map_files_in_directory

from typing import Callable, Iterator, List, Optional
from ray.remote_function import RemoteFunction
from marin.utils import fsspec_exists, fsspec_glob, fsspec_mkdirs, rebase_file_path

@cached_or_construct_output(success_suffix="SUCCESS") # We use this decorator to make this function idempotent
def json_to_labels(input_file_path, output_file_path, labels):
    # Read the input file
    if fsspec_exists(output_file_path):
        tmp_file_path = output_file_path.split(".jsonl.gz")[0] + "-tmp" + ".jsonl.gz"
        with fsspec.open(output_file_path, "rt", compression="gzip") as f_in, \
                fsspec.open(tmp_file_path, "wt", compression="gzip") as f_out:
            for line in f_in:
                json_obj = json.loads(line)
                attributes["labels"] += labels
                f_out.write(json.dumps({"id": json_obj["id"],
                                        "source": json_obj["source"],
                                        "attributes": attributes
                                        }) + "\n")
        with fsspec.open(tmp_file_path, "rt", compression="gzip") as f_in, \
                fsspec.open(output_file_path, "wt", compression="gzip") as f_out:
            for line in f_in:
                f_out.write(line)
        os.remove(tmp_file_path)
    else:
        with fsspec.open(input_file_path, "rt", compression="gzip") as f_in, \
                fsspec.open(output_file_path, "wt", compression="gzip") as f_out:
            for line in f_in:
                json_obj = json.loads(line)
                attributes = {}
                attributes["labels"] = labels
                f_out.write(json.dumps({"id": json_obj["id"],
                                        "source": json_obj["source"],
                                        "attributes": attributes
                                        }) + "\n")

    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Example script to convert fineweb html to markdown.")
    
    parser.add_argument('--good_dir', type=str, help='Path to positive examples', required=True)
    parser.add_argument('--bad_dir', type=str, help='Path to negative examples', required=True)

    args = parser.parse_args()

    @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
    def json_to_positive(input_file_path,output_file_path):
        return json_to_labels(input_file_path,output_file_path,["good"])
    
    @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
    def json_to_negative(input_file_path,output_file_path):
        return json_to_labels(input_file_path,output_file_path,[])

    ray.init()

    output_dir = rebase_file_path("gs://marin-data/processed", args.good_dir, "gs://marin-data/attributes/fasttext-labels")
    responses = map_files_in_directory(json_to_positive.remote, args.good_dir, "**/*.jsonl.gz", output_dir)
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")

    output_dir = rebase_file_path("gs://marin-data/processed", args.bad_dir, "gs://marin-data/attributes/fasttext-labels")
    responses = map_files_in_directory(json_to_negative.remote, args.bad_dir, "**/*.jsonl.gz", output_dir)
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")