"""
Usage:
python3 -m marin.processing.download_gcs_data --output-dir <output-dir>
nlprun -q john -c 16 -r 40G 'python marin/processing/download_gcs_data.py --output-dir /nlp/scr/cychou'
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
                "gs://marin-data/raw/dolma/dolma-v1.7/wiki-0001.json.gz",
                # "gs://marin-data/raw/dolma/dolma-v1.7/c4-0001.json.gz",
                # "gs://marin-data/scratch/chrisc/test.json.gz"
            ]

# Function to decompress .gz file to .json
def decompress_gz_to_json(gz_file_path, json_file_path):
    with gzip.open(gz_file_path, 'rb') as f_in:
        with open(json_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def download_file(filename, output_dir):
    with fsspec.open(filename, 'r', compression="gzip") as f:
        json_filename = os.path.basename(filename).replace(".gz", "")
        json_file_path = os.path.join(output_dir, json_filename)
        with open(json_file_path, "w", encoding="utf-8") as f_out:
            for line in f:
                json_line = json.loads(line)
                f_out.write(json.dumps(json_line) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, help="The output directory")

    args = parser.parse_args()

    for filename in filenames:
        download_file(filename, args.output_dir)