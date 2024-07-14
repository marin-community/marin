import argparse
import json

import fsspec
import ray

from marin.core.runtime import RayConfig, TaskConfig, cached_or_construct_output, map_files_in_directory
from scripts.reddit.utils import convert_thread

@ray.remote(memory=10 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
@cached_or_construct_output(success_suffix="SUCCESS") # We use this decorator to make this function idempotent
def json_to_md(input_file_path, output_file_path):
    # Read the input file
    with fsspec.open(input_file_path, "rt", compression="gzip") as f, \
            fsspec.open(output_file_path, "wt", compression="gzip") as output:
        for line in f:
            data = json.loads(line)

            # data is in dolma format hence
            id = data["id"]
            source = data["source"]
            metadata = data["metadata"]

            url = f"https://www.reddit.com/r/{metadata['subreddit']}/comments/{metadata['thread_id']}/"

            # Convert page can throw exception based on the html content (e.g. invalid html, Empty page)
            try:
                md = convert_thread(url)
            except Exception as e:
                print(f"Error {e} in processing {id = }, {url = }, file: {input_file_path}")
                # You can choose to raise it or ignore it depending upon the use case
                # raise e
                continue

            output.write(json.dumps({"id": id,
                                     "text": md,
                                     "source": source,
                                     "metadata": {
                                         key: value for key, value in metadata.items()
                                     }
                                     }) + "\n")
    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Example script to convert fineweb html to markdown.")
    
    parser.add_argument('--input_dir', type=str, default='gs://marin-data/raw/dolma/dolma-v1.7/', help='Path to the dolma 1.7 directory', required=True)
    parser.add_argument('--output_dir', type=str, default='gs://marin-data/scratch/rohithk/', help='Path to store reddit markdown files', required=True)

    args = parser.parse_args()

    ray.init()

    responses = map_files_in_directory(json_to_md.remote, args.input_dir, "**/reddit*.json.gz", args.output_dir)

    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")
