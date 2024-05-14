"""
Usage:
python3 -m marin.processing.dolma_wiki --output-dir <output-dir> --convert-to-json
"""

from google.cloud import storage
import argparse
import os
import gzip
import shutil

filenames = [
                "markweb/dolma-v1.7/wiki-0000.json.gz",
                "markweb/dolma-v1.7/wiki-0001.json.gz",
            ]

# Function to decompress .gz file to .json
def decompress_gz_to_json(gz_file_path, json_file_path):
    with gzip.open(gz_file_path, 'rb') as f_in:
        with open(json_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def download_file(filename, output_dir, convert_to_json):
    blob = bucket.blob(f"{filename}")
    gz_file_path = os.path.join(output_dir, filename.split("/")[-1])
    blob.download_to_filename(gz_file_path)
    
    if convert_to_json:
        json_file_path = os.path.join(output_dir, filename.split("/")[-1].replace(".gz", ""))
        decompress_gz_to_json(gz_file_path, json_file_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, help="The output directory")
    parser.add_argument("--convert-to-json", action="store_true", default=False, help="Whether to convert the downloaded file to json")

    args = parser.parse_args()

    storage_client = storage.Client("hai-gcp-models")
    bucket = storage_client.get_bucket("levanter-data")

    for filename in filenames:
        download_file(filename, args.output_dir, args.convert_to_json)