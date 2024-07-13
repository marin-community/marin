"""
Anything which is not inside ray.remote function will be executed on the head node, so keep it to minimum.
path: scripts/legal/process_australianlegalcorpus.py
Inputs: one raw jsonl file, Output: one jsonl.gz file in dolma format

Example Usage:
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
python scripts/legal/process_australianlegalcorpus.py \
--input_dir gs://marin-data/raw/huggingface.co/datasets/umarbutler/open-australian-legal-corpus/resolve/66e7085ff50b8d71d3089efbf60e02ef5b53cf46 \
--output_dir gs://marin-data/processed/law/txt/australianlegalcorpus-v1.0/documents
"""

import argparse
import json
import os

import fsspec
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory


# This function will be executed on the worker nodes. It is important to keep the function idempotent and resumable.
# default memory is unbound, default runtime_env is empty, default num_cpus is 1
# IMPORTANT:Ray resources are logical and not physical: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
# Ray will not impose any physical limits on the resources used by the function, these numbers are used for scheduling.
# 10 GB because the corpus.jsonl is just one large (8.4GB) file
@ray.remote(memory=10 * 1024 * 1024 * 1024, runtime_env={"pip": ["fastparquet"]}, num_cpus=1)
@cached_or_construct_output(success_suffix="SUCCESS")  # We use this decorator to make this function idempotent
def convert_to_dolma(input_file_path, output_file_path):
    # The runtime for this function should be low (less than 5-10 min), as the machines are preemptible

    source = "australianlegalcorpus"
    change_extension = lambda filepath: os.path.splitext(filepath)[0] + '.jsonl.gz'

    # Read the input file
    with fsspec.open(input_file_path, "rt") as f, \
            fsspec.open(change_extension(output_file_path), "wt", compression="gzip") as output:
        for idx, line in enumerate(f):
            row = json.loads(line)

            output.write(json.dumps({
                "id": row['version_id'],
                "text": row['text'],
                "source": source,
                "created": row['date'],
                "added": row['when_scraped'],
                "metadata": {
                    f"type": row['type'],
                    f"jurisdiction": row['jurisdiction'],
                    f"source": row['source'],
                    f"citation": row['citation'],
                    f"mime": row['mime'],
                    f"url": row['url'],
                }
            }) + "\n")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to convert australianlegalcorpus data to dolma format.")
    # As a reminder all the processing in this function will be done on the head node. We should try
    # to keep number of jobs (started using ray job submit command) to minimum while trying to use tasks(ray.remote
    # function) as much as possible. For reference, fw uses only 1 job to process a complete dump which is about
    # 400GB of data and spawns about 75000 tasks (ray.remote functions).
    parser.add_argument('--input_dir', type=str, help='Path to the australianlegalcorpus raw directory', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to store australianlegalcorpus dolma files', required=True)

    args = parser.parse_args()

    ray.init()

    responses = map_files_in_directory(convert_to_dolma.remote, args.input_dir, "**/*.jsonl", args.output_dir)

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
