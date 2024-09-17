"""
Anything which is not inside ray.remote function will be executed on the head node, so keep it to minimum.
path: scripts/legal/transform_edgar.py
Inputs: raw parquet files, Output: jsonl.gz files in dolma format

Example Usage:
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python operations/transform/legal/transform_edgar.py \
        --input_path gs://marin-data/raw/law/edgar \
        --output_path gs://marin-data/processed/law/edgar-v1.0/txt/documents
"""

import argparse
import json
import os

import fsspec
import pandas as pd
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory


# This function will be executed on the worker nodes. It is important to keep the function idempotent and resumable.
# default memory is unbound, default runtime_env is empty, default num_cpus is 1
# IMPORTANT:Ray resources are logical and not physical: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
# Ray will not impose any physical limits on the resources used by the function, these numbers are used for scheduling.
@ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["fastparquet"]}, num_cpus=1)  # 1 GB
@cached_or_construct_output(success_suffix="SUCCESS")  # We use this decorator to make this function idempotent
def convert_to_dolma(input_file_path, output_file_path):
    # The runtime for this function should be low (less than 5-10 min), as the machines are preemptible

    # Read the input file
    df = pd.read_parquet(input_file_path, engine="fastparquet")
    source = "edgar"

    with fsspec.open(os.path.splitext(output_file_path)[0] + ".jsonl.gz", "wt", compression="gzip") as output:
        for _, row in df.iterrows():
            sections = [
                "section_1",
                "section_1A",
                "section_1B",
                "section_2",
                "section_3",
                "section_4",
                "section_5",
                "section_6",
                "section_7",
                "section_7A",
                "section_8",
                "section_9",
                "section_9A",
                "section_9B",
                "section_10",
                "section_11",
                "section_12",
                "section_13",
                "section_14",
                "section_15",
            ]
            text = "\n\n".join([row[section] for section in sections]).strip()

            output.write(
                json.dumps(
                    {
                        "id": row["cik"],
                        "text": text,
                        "source": source,
                        "metadata": {
                            "year": row["year"],
                            "filename": row["filename"],
                        },
                    }
                )
                + "\n"
            )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to convert edgar data to dolma format.")
    # As a reminder all the processing in this function will be done on the head node. We should try
    # to keep number of jobs (started using ray job submit command) to minimum while trying to use tasks(ray.remote
    # function) as much as possible. For reference, fw uses only 1 job to process a complete dump which is about
    # 400GB of data and spawns about 75000 tasks (ray.remote functions).
    parser.add_argument("--input_path", type=str, help="Path to the edgar raw directory", required=True)
    parser.add_argument("--output_path", type=str, help="Path to store edgar dolma files", required=True)

    args = parser.parse_args()

    ray.init()

    responses = map_files_in_directory(convert_to_dolma.remote, args.input_path, "**/*.parquet", args.output_path)

    # Wait for all the tasks to finish.
    # The try and catch is important here as incase convert_to_dolma throws any exception, that exception is passed here,
    # And if we don't catch it here, the script will exit, which will kill all the other tasks.
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")
        # Put your retry logic here, incase you want to get the file for which the processing failed, please see:
        # https://docs.ray.io/en/latest/ray-core/fault-tolerance.html
        # In practice, since we make html_to_md resumable and idempotent, you can just look at the logs in Ray dashboard
        # And retry the same command after fixing the bug.
