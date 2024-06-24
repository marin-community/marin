"""
Usage:
python3 -m marin.processing.download_gcs_data --output-dir ~/data
"""

from google.cloud import storage
import argparse
import os
import gzip
import shutil
import fsspec
import json

filenames = [
                # "gs://marin-data/raw/dolma/dolma-v1.7/wiki-0000.json.gz",
                # "gs://marin-data/raw/dolma/dolma-v1.7/wiki-0001.json.gz",
                # "gs://marin-data/raw/dolma/dolma-v1.7/c4-0001.json.gz",
                # "gs://marin-data/scratch/chrisc/test.json.gz",
                # "gs://marin-data/scratch/chrisc/dataset.txt.gz",
                # "gs://marin-data/scratch/chrisc/fasttext_train.txt.gz",
                # "gs://marin-data/scratch/chrisc/fasttext_test.txt.gz",
                "gs://marin-data/scratch/chrisc/processed/fineweb/fw-v1.0/CC-MAIN-2024-10/attributes/fasttext-quality/000_00000/0_processed_md.jsonl.gz"
            ]

def download_file(filename, output_dir):
    output_filename = os.path.basename(filename).replace(".gz", "")
    output_file_path = os.path.join(output_dir, output_filename)
    file_format = os.path.basename(output_filename).split(".")[1]

    with fsspec.open(filename, 'r', compression="gzip") as f:
        with open(output_file_path, "w", encoding="utf-8") as f_out:
            for line in f:
                if file_format == "json" or file_format == "jsonl":
                    json_line = json.loads(line)
                    f_out.write(json.dumps(json_line) + "\n")
                elif file_format == "txt":
                    f_out.write(line)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, help="The output directory")

    args = parser.parse_args()

    for filename in filenames:
        download_file(filename, args.output_dir)