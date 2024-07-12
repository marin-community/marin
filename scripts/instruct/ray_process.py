# Example script on how to run jobs on ray and process large amount of data.
# The scripts contains comments with critical information to follow along.
# Anything which is not inside ray.remote function will be executed on the head node, so keep it to minimum.
# path: scripts/fineweb/process_parquet_fw.py
# Inputs: jsonl.gz files in dolma format having html content, Output: jsonl.gz files in dolma format having markdown
# Example Usage:
# ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
# python scripts/hello_world_fw/process.py \
# --input_dir gs://marin-data/hello_world_fw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/
# --output_dir gs://marin-data/scratch/user/hello_world_fw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/
import argparse
import json
from dataclasses import dataclass

import draccus
import fsspec
import os
import ray
import pandas as pd

from marin.core.runtime import RayConfig, TaskConfig, cached_or_construct_output, map_files_in_directory
from tqdm import tqdm
from marin.processing.instruct.utils import process_instruction



# This function will be executed on the worker nodes. It is important to keep the function idempotent and resumable.
# default memory is unbound, default runtime_env is empty, default num_cpus is 1
# IMPORTANT:Ray resources are logical and not physical: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
# Ray will not impose any physical limits on the resources used by the function, these numbers are used for scheduling.
@ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
@cached_or_construct_output(success_suffix="SUCCESS")
def process_raw_chunk(input_file_path, output_dir):
    input_type = "parquet" if input_file_path.endswith(".parquet") else "jsonl"
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    print(f"\n input type: {input_type}, input_file_path: {input_file_path}, output_dir: {output_dir}")
    md_json_files = []
    text_json_files = []

    if input_type == 'jsonl':
        with fsspec.open(input_file_path, "rt", compression="gzip") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                id = data[0].get("dataset", f"{idx}")
                try:
                    md_content, text_content = process_instruction(source, id, data)
                except Exception as e:
                    print(f"Error {e} in processing {idx = }, file: {input_file_path}")
                    continue

                metadata = data[0].get("metadata", {})
                md_json_files.append({"id": id, "text": md_content, "source": source, "metadata": metadata})
                text_json_files.append({"id": id, "text": text_content, "source": source, "metadata": metadata})

    elif input_type == 'parquet':
        print(f"\n\n  Processing parquet file {input_file_path} \n\n")
        df = pd.read_parquet(input_file_path)
        for idx, row in tqdm(enumerate(df.iterrows()), total=len(df)):
            if len(row) != 2:
                raise ValueError("Row should have 2 elements, index and data")
            data_idx, data_series = row
            data_dict = data_series.to_dict()
            id = data_dict.get("id", f"{data_idx}")
            source = data_dict.get("dataset", "")
            try:
                md_content, text_content = process_instruction(source, id, data_dict)
            except Exception as e:
                print(f"Error {e} in processing {idx = }, file: {input_file_path}")
                continue

            
            metadata = data_dict.get("metadata", {})
            md_json_files.append({"id": id, "text": md_content, "source": source, "metadata": metadata})
            text_json_files.append({"id": id, "text": text_content, "source": source, "metadata": metadata})

    # Save the Markdown JSONL file
    version = "v1_olmo_mix"
    save_path = "/".join(output_dir.split("/")[:5])

    
    md_output_path = os.path.join(save_path,  version, "md", f"{base_name}.jsonl.gz")
    print(f"MD path Saving to {md_output_path} with input file path {input_file_path}")
    print(f" Number of md files {len(md_json_files)}")
    with fsspec.open(md_output_path, "at", compression="gzip") as md_file:
        for md_json in md_json_files:
            md_file.write(json.dumps(md_json) + "\n")

    # Save the Text JSONL file
    text_output_path = os.path.join(save_path, version, "text", f"{base_name}.jsonl.gz")
    print(f"HTML path Saving to {text_output_path} with input file path {input_file_path}")
    print(f" Number of html files {len(text_json_files)}")
    with fsspec.open(text_output_path, "at", compression="gzip") as text_file:
        for text_json in text_json_files:
            text_file.write(json.dumps(text_json) + "\n")

    return True


@dataclass
class RayConfig:
    input_dir: str   # Path to the fineweb html directory
    output_dir: str  # Path to store fineweb markdown files
    task: TaskConfig = TaskConfig()
    ray: RayConfig = RayConfig()


@draccus.wrap()
def main(config: RayConfig):
    config.ray.initialize()
    # Example of input_dir = gs://marin-data/hello_world_fw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/
    # We will process all html_jsonl.gz files in this directory.
    # As a reminder all the processing in this function will be done on the head node. We should try
    # to keep number of jobs (started using ray job submit command) to minimum while trying to use tasks(ray.remote
    # function) as much as possible. For reference, fw uses only 1 job to process a complete dump which is about
    # 400GB of data and spawns about 75000 tasks (ray.remote functions).


    responses = map_files_in_directory(
        process_raw_chunk,
        config.input_dir,
        "**/*.jsonl.gz",
        config.output_dir,
        task_config=config.task
    )
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")
        # Put your retry logic here, incase you want to get the file for which the processing failed, please see:
        # https://docs.ray.io/en/latest/ray-core/fault-tolerance.html
        # In practice, since we make process_raw_chunk resumable and idempotent, you can just look at the logs in Ray dashboard
        # And retry the same command after fixing the bug.


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Example script to convert HF instruction data to markdown.")
    # Example of input_dir = gs://marin-data/hello_world_fw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/
    # We will process all html_jsonl.gz files in this directory.
    # As a reminder all the processing in this function will be done on the head node. We should try
    # to keep number of jobs (started using ray job submit command) to minimum while trying to use tasks(ray.remote
    # function) as much as possible. For reference, fw uses only 1 job to process a complete dump which is about
    # 400GB of data and spawns about 75000 tasks (ray.remote functions).
    parser.add_argument('--input_dir', type=str, help='Path to  raw parquet/jsonl directory', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to processed html/markdown files', required=True)
    parser.add_argument('--input_type', type=str, choices=['jsonl', 'parquet'], default='jsonl', help='Type of input file')

    args = parser.parse_args()

    ray.init()
    if args.input_type == 'parquet':
        responses = map_files_in_directory(process_raw_chunk.remote, args.input_dir, "**/*.parquet", args.output_dir, args.input_type)
    else:
        responses = map_files_in_directory(process_raw_chunk.remote, args.input_dir, "**/*.jsonl.gz", args.output_dir, args.input_type)

    # Wait for all the tasks to finish.
    # The try and catch is important here as incase process_raw_chunk throws any exception, that exception is passed here,
    # And if we don't catch it here, the script will exit, which will kill all the other tasks.
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")
        # Put your retry logic here, incase you want to get the file for which the processing failed, please see:
        # https://docs.ray.io/en/latest/ray-core/fault-tolerance.html
        # In practice, since we make process_raw_chunk resumable and idempotent, you can just look at the logs in Ray dashboard
        # And retry the same command after fixing the bug.
