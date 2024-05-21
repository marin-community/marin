'''Convert fineweb to markdown'''
import argparse
import json
import logging
import os
import time
import traceback

import fsspec
import ray

import pandas as pd
import requests
from warcio import ArchiveIterator
from marin.web.convert import convert_page
from marin.utils import gcs_file_exists
from scripts.fineweb.utils import get_warc_parquet_success_path

# Initialize S3 file system
logging.basicConfig(level=logging.INFO)

@ray.remote(memory=1*1024*1024*1024) # 1 GB
def process_one_warc_file(input_file_path):
    """
    Takes in the input file and processes it to get the html and md content.
    It scans the s3 bucket in input_file and returns the content of the urls in the input_file
    Args:
    input_file_path (str): The input file to process
    """
    # Example of input_file_path = gs://marin-data/processed/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/0.parquet
    output_file, success_file = get_warc_parquet_success_path(input_file_path)

    if gcs_file_exists(success_file):
        print(f"Output file already processed. Skipping {input_file_path}")
        if gcs_file_exists(input_file_path):
            fs = fsspec.filesystem("gcs")
            fs.rm(input_file_path)
        return True

    df = pd.read_parquet(input_file_path)
    df["md"] = None
    df["html"] = None

    urls = df["url"].tolist()
    s3_url = df["file_path"].iloc[0]
    index = df.index.tolist()
    # url_dict is url to index in df so that we can update that record
    url_dict = {url: idx for idx, url in zip(index, urls)}
    num_urls_processed = 0  # Used to early terminate
    length_url_inp_list = len(urls)

    # Logging variables
    print(f"Processing {s3_url} in {input_file_path}")
    length_warc = 0
    stat_time = time.time()

    s3_url = s3_url.replace("s3://commoncrawl/", "https://data.commoncrawl.org/")
    response = requests.get(s3_url, stream=True)
    response.raise_for_status()

    if response.status_code == 200:
        for record in ArchiveIterator(response.raw):
            if num_urls_processed == length_url_inp_list:
                break
            # Check if it's a response record
            if record.rec_type == 'response':
                # Process the record
                url = record.rec_headers.get_header('WARC-Target-URI')
                length_warc += 1

                if length_warc % 1000 == 0:
                    print(f"Processed {length_warc} records in {time.time() - stat_time} seconds")

                if url in url_dict:
                    num_urls_processed += 1
                    url_idx_in_df = url_dict[url]
                    try:
                        # Read the response body
                        content = record.content_stream().read()
                        html_decoded = content.decode(errors='ignore')
                        # Writing this above ensures that we write html even if there's an error in markdown conversion
                        df.loc[url_idx_in_df, "html"] = html_decoded
                        markdown = convert_page(html_decoded, url)
                        df.loc[url_idx_in_df, "md"] = markdown["content"].encode('utf-8', 'ignore').decode('utf-8')
                    except Exception as e:
                        print(f"Error processing {url} in {s3_url} for {input_file_path}: {e}")
                        traceback.print_exc()

    print(f"Processed {s3_url}, found {length_warc} records, {length_url_inp_list} urls, "
          f"{length_warc/length_url_inp_list} ratio")

    # Write the output to a file with md information. MD will have html in the metadata
    output_file = input_file_path.replace(".parquet", "_processed_md.jsonl.gz")
    with fsspec.open(output_file, 'wb', compression='gzip') as f: #md output
        for index, row in df.iterrows():
            out_fw  = row.to_dict()
            out_dolma = {"id": out_fw["id"],
                         "text": out_fw["md"],
                         "source": "fineweb"
            }
            all_except_md = {key: value for key, value in out_fw.items() if key != 'md'}
            out_dolma["metadata"] = {"fineweb_metadata": all_except_md}
            f.write(json.dumps(out_dolma).encode('utf-8') + b"\n")

    # Write the output to a file with html format.
    output_file_html = output_file.replace("_processed_md.jsonl.gz", "_processed_html.jsonl.gz")
    with fsspec.open(output_file_html, 'wb', compression='gzip') as f: #html output
        for index, row in df.iterrows():
            out_fw  = row.to_dict()
            out_dolma = {"id": out_fw["id"],
                         "text": out_fw["html"],
                         "source": "fineweb"
            }
            all_except_md_html = {key: value for key, value in out_fw.items() if key not in ('md', 'html')}
            out_dolma["metadata"] = {"fineweb_metadata": all_except_md_html}
            f.write(json.dumps(out_dolma).encode('utf-8') + b"\n")

    # remove the input file
    fs = fsspec.filesystem("gcs")
    fs.rm(input_file_path)

    with fsspec.open(success_file, 'w') as f:
        f.write("SUCCESS")

    if num_urls_processed != length_url_inp_list:
        print(f"All urls should be processed, only process {num_urls_processed} out of {length_url_inp_list} urls,")

    return True

@ray.remote(memory=10*1024*1024*1024) # 10 GB
def process_fw_parquet(input_file_path):
    """
       Converts fineweb files to html and markdown. This will essentially take in fineweb and split different groups based
       on file_path and write all those file paths to a new folder and then run ray for each group

       Parameters:
       input_path (str): Path to the fineweb parquet file
    """

    # Example of input_path = gs://marin-data/raw/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000.parquet
    # Example of output_path = gs://marin-data/processed/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/
    print(f"Processing {input_file_path}")
    output_dir_path = input_file_path.replace("raw", "processed").replace(".parquet", "")
    success_file = output_dir_path + "/_SUCCESS"
    if gcs_file_exists(success_file):
        print(f"Output file already processed. Skipping {input_file_path}")
        return True

    df = pd.read_parquet(input_file_path)
    success_refs = {"ray_waitable": [], "file_path": []}
    # file_path is s3 url
    grouped = df.groupby("file_path")

    for index, (file_url, group_df) in enumerate(grouped):
        filename = os.path.join(output_dir_path, f"{index}.parquet")

        _, success_file_group = get_warc_parquet_success_path(filename)
        # Save the group to a parquet file
        if gcs_file_exists(success_file_group):
            print(f"Output file already processed. Skipping {filename}")
            continue
        group_df.to_parquet(filename)
        success_refs["ray_waitable"].append(process_one_warc_file.remote(filename))
        success_refs["file_path"].append(filename)

    TASK_TIMEOUT = 600.0 #10 minutes
    done, in_progress = ray.wait(success_refs["ray_waitable"], num_returns=len(success_refs["ray_waitable"]),
                                 timeout=TASK_TIMEOUT, fetch_local=False)
    in_progress_indices = [success_refs["ray_waitable"].index(task) for task in in_progress]

    for i in in_progress_indices:
        print(f"Error: Timeout processing {success_refs['file_path'][i]}")

    if len(in_progress_indices) == 0:
        print(f"Processed {input_file_path}")
        with fsspec.open(success_file, 'w') as f:
            f.write("SUCCESS")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert fineweb to markdown.")
    # Example of input_folder =
    parser.add_argument('--input_file_path', type=str, help='Path to the fineweb parquet file', required=True)

    args = parser.parse_args()
    ray.init()
    ray.get(process_fw_parquet.remote(args.input_file_path))


