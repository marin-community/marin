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
def process_one_dolma_file(input_file_path, output_path_path, domain, examples_per_file):
    """
    Takes in raw Dolma file, splits it into chunks, and writes to a new directory.
    Performs sanity checking to ensure that data is formatted correctly. However,
    the Dolma data should already be in the correct format.
    """
    gfs = fsspec.filesystem("gcs")
    input_file_basename = os.path.basename(input_file_path)

    out_file_handler = None

    # stream read input file
    with gfs.open(input_file_path, "rt", compression="gzip") as in_file:
        for idx, line in enumerate(in_file):
            if idx % examples_per_file == 0:
                if out_file_handler:
                    out_file_handler.close()
                out_file_path = os.path.join(
                    output_path_path,
                    f"{input_file_basename.split('.')[0]}-chunk{idx // examples_per_file:04d}.jsonl.gz",
                )
                out_file_handler = gfs.open(out_file_path, "wt", compression="gzip")

            entry = json.loads(line)
            entry = sanitize_entry(entry, domain)
            out_file_handler.write(json.dumps(entry) + "\n")

    if out_file_handler:
        out_file_handler.close()

    print(f"Finished processing {input_file_path}")
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
        default=100000,
        help="Number of examples to include in each output file. Larger input files are split up into chunks of this size.",
    )
    args = parser.parse_args()

    gfs = fsspec.filesystem("gcs")
    files = gfs.glob(os.path.join(args.dolma_path, f"{args.domain}*.json.gz"))

    result_refs = []
    for file in files:
        result_refs.append(process_one_dolma_file.remote(file, args.output_path, args.domain, args.examples_per_file))

    # Wait for all the tasks to finish
    try:
        ray.get(result_refs)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
