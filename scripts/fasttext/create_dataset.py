import argparse
import json
import os

import fsspec
import ray

from marin.core.runtime import RayConfig, TaskConfig, cached_or_construct_output, simple_backpressure

from typing import Callable, Iterator, List, Optional
from ray.remote_function import RemoteFunction
from marin.utils import fsspec_exists, fsspec_glob, fsspec_mkdirs, rebase_file_path

def map_files_in_directory(
    func: Callable | RemoteFunction,
    input_dir,
    pattern,
    output_dir,
    attr_dir,
    task_config: TaskConfig = TaskConfig(),  # noqa
    *args,
    **kwargs,
):
    """
    Map a function to all files in a directory.
    If the function is a ray.remote function, then it will be executed in parallel.

    Args:
        func: The function to map
        input_dir: The input directory
        output_dir: The output directory
        task_config: TaskConfig object

    Returns:
        List: A list of outputs from the function.
    """
    # Get a list of all files in the input directory
    files = fsspec_glob(os.path.join(input_dir, pattern))

    def func_to_call(input_file):
        # Construct the output file path
        output_file = rebase_file_path(input_dir, input_file, output_dir)
        attr_file = rebase_file_path("gs://marin-data/processed", input_file, attr_dir)

        dir_name = os.path.dirname(output_file)
        fsspec_mkdirs(dir_name)
        return func(input_file, output_file, attr_file)

    if isinstance(func, ray.remote_function.RemoteFunction):
        # If the function is a ray.remote function, then execute it in parallel
        responses = simple_backpressure(func_to_call, iter(files), task_config.max_in_flight, fetch_local=True)
        return responses
    else:
        # Map the function to all files
        outputs = []
        for file in files:
            outputs.append(func_to_call(file))

    return outputs

@ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
@cached_or_construct_output(success_suffix="SUCCESS") # We use this decorator to make this function idempotent
def json_to_fasttext(input_file_path, output_file_path, attr_file_path):
    # Read the input file
    with fsspec.open(input_file_path, "rt", compression="gzip") as f_in, \
            fsspec.open(output_file_path, "wt", compression="gzip") as f_out, \
                fsspec.open(attr_file_path, "wt", compression="gzip") as f_attr:
        for input_line,attr_line in zip(f_in,f_attr):
            json_obj = json.loads(input_line)
            attr_obj = json.loads(attr_line)

            text = json_obj["text"].replace("\n"," ")
            labels = ''.join([f" __label__{label}" for label in attr_obj["attributes"]["fasttext-labels"]])
            
            line = labels + " " + text + "\n"
            f_out.write(line)

    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Example script to convert fineweb html to markdown.")
    
    parser.add_argument('--input_dir', type=str, help='Path to processed data', required=True)
    parser.add_argument('--attr_dir', type=str, help='Path to data labels', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to store training dataset', required=True)

    args = parser.parse_args()

    ray.init()

    responses = map_files_in_directory(json_to_fasttext.remote, args.input_dir, "**/*.jsonl.gz", args.output_dir, args.attr_dir)

    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")