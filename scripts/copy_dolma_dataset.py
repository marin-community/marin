"""
file: copy_dolma_dataset.py
---
This program copies a specified subset of the Dolma data and formats it to match Marin schema.
It is assumed that all Dolma data is stored as *.json.gz file in the bucket specified by the dolma_path arg.
Different subsets of the data are identified by prefixing the filename. For instance, "algebraic-stack-train-0000.json.gz" belongs to the "algebraic-stack" domain.

This program:
(1) Groups data into example_per_file chunks and saves them as *.jsonl.gz files (note that json->jsonl)
(2) Checks that the json matches the Dolma schema and makes corrections if necessary

This program was used to process the OpenWebMath, AlgebraicStack, and StarCoder datasets, among possible others.

Example usage:
    python scripts/copy_dolma_dataset.py --dolma_path=gs://marin-data/raw/dolma/dolma-v1.7/ --domain=open-web-math --output_path=gs://marin-data/processed/openwebmath/2023-09-06/md
"""

import argparse
import json
import os
import gzip

import fsspec
import ray


@ray.remote(memory=1 * 1024 * 1024 * 1024)  # 1 GB
def process_one_dolma_file(input_file_path, output_dir_path, domain, examples_per_file):
    """
    Takes in raw Dolma file, splits it into chunks, and writes to a new directory.
    Performs sanity checking to ensure that data is formatted correctly. However,
    the Dolma data should already be in the correct format.
    """
    gfs = fsspec.filesystem("gcs")

    # read the input file
    with gfs.open(input_file_path, mode="rb") as f_in:
        with gzip.open(f_in, mode="rt", encoding="utf-8") as gz_in:
            # sanitize_entry checks that entry matches Dolma format
            data = [sanitize_entry(json.loads(line), domain) for line in gz_in]

    # split data into chunks
    chunks = [data[i : i + examples_per_file] for i in range(0, len(data), examples_per_file)]

    # write each chunk to a new jsonl.gz file
    input_file_basename = os.path.basename(input_file_path)
    for i, chunk in enumerate(chunks):
        output_file_path = os.path.join(
            output_dir_path, f"{input_file_basename}_chunk_{i:04d}.jsonl.gz"
        )
        with fsspec.open(output_file_path, mode="wb") as f_out:
            with gzip.open(f_out, mode="wt", encoding="utf-8") as gz_out:
                for item in chunk:
                    gz_out.write(json.dumps(item) + "\n")

    print(f"Processed {input_file_path} into {len(chunks)} files.")
    return True  # success


def sanitize_entry(entry, domain):
    """
    This function takes a Dolma example in json format and checks that it matches the required schema.

    Dolma schema is detailed at https://github.com/allenai/dolma/blob/main/docs/data-format.md
    """
    required_keys = {"id", "text"}  # also "source" but we can add that ourselves
    optional_keys = {"added", "created", "metadata"}
    # ensure all mandatory fields exist
    for key in required_keys:
        if key not in entry:
            raise ValueError(f"Mandatory field {key} is missing in entry {entry}")

    if "source" not in entry:  # this is missing from some data...
        entry["source"] = domain

    # move any additional fields to metadata
    for key in list(entry.keys()):
        if key not in required_keys and key not in optional_keys:
            if "metadata" not in entry:
                entry["metadata"] = {}
            entry["metadata"][key] = entry.pop(key)

    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dolma_path",
        type=str,
        required=True,
        help="Path to the bucket containing Dolma datasets",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="The prefix of the Dolma data to copy and format",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the formatted Dolma dataset. This should probably match the form 'gs://marin-data/processed/{domain}/{version}/{format}'.",
    )
    parser.add_argument(
        "--examples_per_file",
        type=int,
        default=10000,
        help="Number of examples to include in each output file. Larger input files are split up into chunks of this size.",
    )
    args = parser.parse_args()

    gfs = fsspec.filesystem("gcs")
    files = gfs.glob(os.path.join(args.dolma_path, f"{args.domain}*.json.gz"))

    ray.init()
    result_refs = []
    for file in files:
        result_refs.append(
            process_one_dolma_file.remote(
                file, args.output_path, args.domain, args.examples_per_file
            )
        )

    # Wait for all the tasks to finish
    try:
        ray.get(result_refs)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
