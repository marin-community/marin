"""
file: process_raw_owm.py
---
Processes the OpenWebMath data.

OpenWebMath randomly converts doc to either plaintext or markdown with the following probabilities:
- headings: 0.9 markdown, 0.1 plaintext
- code: 0.95 markdown, 0.05 plaintext
See https://github.com/keirp/OpenWebMath/blob/main/extract_from_cc/configs/randomized_all.yaml
"""

import argparse
import json
import os

import fsspec
import pandas as pd
import ray

from marin.utils import fsspec_exists, get_gcs_path


def get_owm_paths(input_file_path, output_dir):
    """
    Given a parquet file path, return the path to the output jsonl file and success file
    Args:
        input_file_path (str): The input file to process
    """
    filename = os.path.basename(input_file_path).replace(".parquet", "")
    # filename from HF is of the form train-00011-of-00114-da8ee2fcf07be148.parquet
    # pick out just the "train-00011" part
    filename = "-".join(filename.split("-")[:2])
    output_file = get_gcs_path(
        os.path.join(
            output_dir,
            filename + "_processed_md.jsonl.gz",
        )
    )
    success_file = output_file + ".SUCCESS"
    return output_file, success_file


@ray.remote(memory=1 * 1024 * 1024 * 1024)  # 1 GB
def process_one_parquet_file(input_file_path, output_dir):
    """
    Takes in raw OpenWebMath parquet file and writes to a new directory in Dolma format.
    Args:
        input_file_path (str): The input file to process
    """
    output_file, success_file = get_owm_paths(input_file_path, output_dir)
    if fsspec_exists(success_file):
        print(f"File {output_file} already processed. Skipping...")
        return True

    try:
        df = pd.read_parquet(input_file_path)
    except FileNotFoundError as e:
        print(f"Error reading the parquet file: {e}")
        return False

    if isinstance(df["metadata"].iloc[0], str):
        df["metadata"] = df["metadata"].apply(json.loads)

    with fsspec.open(output_file, "wb", compression="gzip") as f:
        for _, row in df.iterrows():
            out_fw = row.to_dict()
            out_dolma = {
                "id": out_fw["url"],
                "text": out_fw["text"],
                "source": "openwebmath",
                "added": "",
                "created": out_fw["date"],
                "metadata": out_fw["metadata"],
            }
            f.write(json.dumps(out_dolma).encode("utf-8") + b"\n")

    with fsspec.open(success_file, "w") as f:
        f.write("SUCCESS")

    print(f"Successfully processed and uploaded {output_file}")

    return True  # Success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw OpenWebMath data")
    # Example of input_dir = gs://marin-data/raw/openwebmath/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8de2300f5e778f56261843dab89f230815/data/
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The directory containing the unmodified OpenWebMath data in parquet format",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to store processed OpenWebMath data, which will be formatted as jsonl files",
    )

    args = parser.parse_args()
    gfs = fsspec.filesystem("gcs")
    files = gfs.glob(os.path.join(args.input_dir, "*.parquet"))
    ray.init()
    result_refs = []
    for file in files:
        gs_file = get_gcs_path(file)
        print(f"Starting Processing for the fw parquet file: {gs_file}")
        result_refs.append(process_one_parquet_file.remote(gs_file, args.output_dir))

    # Wait for all the tasks to finish
    try:
        ray.get(result_refs)
    except Exception as e:
        print(f"Error processing the group: {e}")
