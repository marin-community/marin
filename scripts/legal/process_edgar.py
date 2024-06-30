"""
Anything which is not inside ray.remote function will be executed on the h
path: scripts/legal/process_edgar.py
Inputs: raw parquet files, Output: jsonl.gz files in dolma format

Example Usage:
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
python scripts/legal/process_edgar.py \
--input_dir gs://marin-data/raw/huggingface.co/datasets/eloukas/edgar-corpus/resolve/f7d3ba73d65ff10194a95b84c75eb484d60b0ede/full/partial-train \
--output_dir gs://marin-data/processed/law/edgar-v1.0
"""

import argparse
import json
from dataclasses import dataclass

import draccus
import fsspec
import ray

import pandas as pd

from marin.core.runtime import RayConfig, TaskConfig, cached_or_construct_output, map_files_in_directory


# This function will be executed on the worker nodes. It is important to keep the function idempotent and resumable.
# default memory is unbound, default runtime_env is empty, default num_cpus is 1
# IMPORTANT:Ray resources are logical and not physical: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
# Ray will not impose any physical limits on the resources used by the function, these numbers are used for scheduling.
@ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
@cached_or_construct_output(success_suffix="SUCCESS")  # We use this decorator to make this function idempotent
def convert_to_dolma(input_file_path, output_file_path):
    # The runtime for this function should be low (less than 5-10 min), as the machines are preemptible
    # Example of input_path = gs://marin-data/hello_world_fw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/0_processed_html.jsonl.gz

    # Read the input file
    df = pd.read_parquet('0007.parquet', engine='fastparquet')
    source = "edgar"

    with fsspec.open(output_file_path, "wt", compression="gzip") as output:
        for row in df.iterrows():
            sections = ['section_1', 'section_1A', 'section_1B',
                        'section_2', 'section_3', 'section_4', 'section_5', 'section_6',
                        'section_7', 'section_7A', 'section_8', 'section_9', 'section_9A',
                        'section_9B', 'section_10', 'section_11', 'section_12', 'section_13',
                        'section_14', 'section_15']
            text = "\n\n".join([row[section] for section in sections])

            output.write(json.dumps({
                "id": row['cik'],
                "text": text,
                "source": source,
                "metadata": {
                    f"year": row['year'],
                    f"filename": row['filename'],
                }
            }) + "\n")
    return True


@dataclass
class HelloWorldConfig:
    input_dir: str  # Path to the edgar raw directory
    output_dir: str  # Path to store edgar dolma files
    task: TaskConfig = TaskConfig()
    ray: RayConfig = RayConfig()


@draccus.wrap()
def main(config: HelloWorldConfig):
    config.ray.initialize()
    # Example of input_dir = gs://marin-data/hello_world_fw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/
    # We will process all .parquet files in this directory.
    # As a reminder all the processing in this function will be done on the head node. We should try
    # to keep number of jobs (started using ray job submit command) to minimum while trying to use tasks(ray.remote
    # function) as much as possible. For reference, fw uses only 1 job to process a complete dump which is about
    # 400GB of data and spawns about 75000 tasks (ray.remote functions).

    responses = map_files_in_directory(
        convert_to_dolma,
        config.input_dir,
        "**/*.parquet",
        config.output_dir,
        task_config=config.task
    )
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")
        # Put your retry logic here, incase you want to get the file for which the processing failed, please see:
        # https://docs.ray.io/en/latest/ray-core/fault-tolerance.html
        # In practice, since we make html_to_md resumable and idempotent, you can just look at the logs in Ray dashboard
        # And retry the same command after fixing the bug.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script to convert fineweb html to markdown.")
    # Example of input_dir = gs://marin-data/hello_world_fw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/
    # We will process all html_jsonl.gz files in this directory.
    # As a reminder all the processing in this function will be done on the head node. We should try
    # to keep number of jobs (started using ray job submit command) to minimum while trying to use tasks(ray.remote
    # function) as much as possible. For reference, fw uses only 1 job to process a complete dump which is about
    # 400GB of data and spawns about 75000 tasks (ray.remote functions).
    parser.add_argument('--input_dir', type=str, help='Path to the edgar raw directory', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to store edgar dolma files', required=True)

    args = parser.parse_args()

    ray.init()

    responses = map_files_in_directory(convert_to_dolma.remote, args.input_dir, "**/*.parquet", args.output_dir)

    # Wait for all the tasks to finish.
    # The try and catch is important here as incase html_to_md throws any exception, that exception is passed here,
    # And if we don't catch it here, the script will exit, which will kill all the other tasks.
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")
        # Put your retry logic here, incase you want to get the file for which the processing failed, please see:
        # https://docs.ray.io/en/latest/ray-core/fault-tolerance.html
        # In practice, since we make html_to_md resumable and idempotent, you can just look at the logs in Ray dashboard
        # And retry the same command after fixing the bug.
