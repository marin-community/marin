'''Convert fineweb to markdown'''
import argparse
import json
import os
import traceback
from datetime import datetime

import fsspec
import pandas as pd
import ray
from warcio import ArchiveIterator

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_exists, fsspec_glob, fsspec_rm
from marin.web.convert import convert_page


@ray.remote(memory=1.5 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]})  # 1 GB
@cached_or_construct_output(success_suffix="SUCCESS")  # We use this decorator to make this function idempotent
def process_one_warc_file(input_file_path, output_file):
    """
    Takes in the input file and processes it to get the html and md content.
    It scans the s3 bucket in input_file and returns the content of the urls in the input_file
    Args:
    input_file_path (str): The input file to process
    """
    # Example of input_file_path = gs://marin-data/processed/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/0.parquet
    # Example of output_file = gs://marin-data/processed/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/0_processed_md.jsonl.gz
    # output_file, success_file = get_warc_parquet_success_path(input_file_path)

    try:
        df = pd.read_parquet(input_file_path)
    except FileNotFoundError as e:
        print(f"Error reading the parquet file: {e}")
        raise e

    df["md"] = None
    df["html"] = None

    urls = df["url"].tolist()
    s3_url = df["file_path"].iloc[0]
    index = df.index.tolist()

    # url_dict is url to index in df so that we can update that record
    url_dict = {url: idx for idx, url in zip(index, urls)}
    num_urls_found = 0  # Used to early terminate
    num_urls_processed = 0
    length_url_inp_list = len(urls)

    # Logging variables
    print(f"Processing {s3_url} in {input_file_path}")
    length_warc = 0
    s3_fs = fsspec.filesystem("s3", anon=False, key="AKIAXVYC7AAF6JHGUA5K",
                              secret="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")
    with s3_fs.open(s3_url, mode='rb') as file_stream:
        for record in ArchiveIterator(file_stream):
            if num_urls_found == length_url_inp_list:
                break

            # Check if it's a response record
            if record.rec_type == 'response':

                # Process the record
                url = record.rec_headers.get_header('WARC-Target-URI')
                length_warc += 1

                if url in url_dict:
                    num_urls_found += 1
                    url_idx_in_df = url_dict[url]
                    if num_urls_found % 100 == 0:
                        print(
                            f"Found Url {num_urls_found = }, Processed Url {num_urls_processed = }, length of warc {length_warc = }")
                    try:
                        content = record.content_stream().read()

                        html_decoded = content.decode(errors='ignore')
                        df.loc[url_idx_in_df, "html"] = html_decoded

                        markdown = convert_page(html_decoded, url)
                        df.loc[url_idx_in_df, "md"] = markdown["content"]

                        num_urls_processed += 1
                    except Exception as e:
                        print(f"Error processing {url} in {s3_url} for {input_file_path}: {e}")
                        traceback.print_exc()

    print(f"Processed {input_file_path}, found {length_warc} records, {length_url_inp_list} urls, "
          f"{length_warc / length_url_inp_list} ratio")

    # Write the output to a file with md information. MD will have html in the metadata
    # output_file = input_file_path.replace(".parquet", "_processed_md.jsonl.gz")
    print(f"Writing to {output_file}")
    with fsspec.open(output_file, 'wt', compression='gzip') as f:  # md output
        for index, row in df.iterrows():
            out_fw = row.to_dict()
            out_dolma = {"id": out_fw["id"],
                         "text": out_fw["md"],
                         "source": "fineweb",
                         "metadata": {
                             f"fw_{key}": value for key, value in out_fw.items() if key not in ('md', 'html', 'text')
                         }
                         }

            f.write(json.dumps(out_dolma) + "\n")

    # Write the output to a file with html format.
    output_file_html = output_file.replace("_processed_md.jsonl.gz", "_processed_html.jsonl.gz")
    print(f"Writing to {output_file_html}")
    with fsspec.open(output_file_html, 'wt', compression='gzip') as f:  # html output
        for index, row in df.iterrows():
            out_fw = row.to_dict()
            out_dolma = {"id": out_fw["id"],
                         "text": out_fw["html"],
                         "source": "fineweb",
                         "metadata": {
                             f"fw_{key}": value for key, value in out_fw.items() if key not in ('md', 'html')
                         }
                         }
            f.write(json.dumps(out_dolma) + "\n")

    # remove the input file
    fsspec_rm(input_file_path)

    if num_urls_found != length_url_inp_list:
        print(f"All urls should be processed, "
              f"Found: {num_urls_found}, Processed: {num_urls_processed}, out of {length_url_inp_list} urls, "
              f"in {input_file_path}"
              f"AWS URL: {s3_url}"
              f"Found {length_warc} records in the WARC file")

    return True


@ray.remote(memory=10 * 1024 * 1024 * 1024)  # 10 GB
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
    datetime_start = datetime.utcnow()

    if fsspec_exists(success_file):
        print(f"Output file already processed. Skipping {input_file_path}")
        return True

    try:
        df = pd.read_parquet(input_file_path)
    except FileNotFoundError as e:
        print(f"Error reading the parquet file: {e}")
        raise e

    success_refs = {"ray_waitable": [], "file_path": []}
    # file_path is s3 url
    grouped = df.groupby("file_path")

    for index, (file_url, group_df) in enumerate(grouped):
        filename = os.path.join(output_dir_path, f"{index}.parquet")

        # output_file_name, success_file_group = get_warc_parquet_success_path(filename)
        output_file_name = filename.replace(".parquet", "_processed_md.jsonl.gz")

        #
        # if fsspec_exists(success_file_group):
        #     print(f"Output file already processed. Skipping {filename}")
        #     continue

        # Save the group to a parquet file
        group_df.to_parquet(filename)
        print(f"Processing the group: {filename}, into {output_file_name}")
        success_refs["ray_waitable"].append(process_one_warc_file.remote(filename, output_file_name))
        success_refs["file_path"].append(filename)
    was_successful = True

    for waitable, filename in zip(success_refs["ray_waitable"], success_refs["file_path"]):
        try:
            ray.get(waitable)
        except Exception as e:
            print(f"Error processing {filename = }, Error: {e}")
            was_successful = False
        finally:
            fsspec_rm(filename)

    datetime_end = datetime.utcnow()

    if not was_successful:
        return False

    with fsspec.open(success_file, 'w') as f:
        metadata = {
            "input_file_path": input_file_path,
            "output_file_path": output_dir_path,
            "datetime_start": str(datetime_start),
            "datetime_end": str(datetime_end),
        }
        f.write(json.dumps(metadata))

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert fineweb to markdown.")
    # Example of input_dir = gs://marin-data/raw/fineweb/fw-v1.0/CC-MAIN-2024-10/
    parser.add_argument('--input_dir', type=str, help='Path to the fineweb parquet diretory', required=True)

    args = parser.parse_args()
    # gfs = fsspec.filesystem("gcs")
    # gs://marin-data/processed/hello_world_fw/fw-v1.0/CC-MAIN-2024-10/000_00000/
    files = fsspec_glob(os.path.join(args.input_dir, "*.parquet"))
    MAX_NUM_PENDING_TASKS = 15  # Max number of parquet files we want to process in pending state
    NUM_TASKS = len(files)
    ray.init()
    result_refs = []
    for file in files:
        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            # update result_refs to only
            # track the remaining tasks.
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                print(f"Error processing the group: {e}")
                continue

        print(f"Starting Processing for the fw parquet file: {file}")
        result_refs.append(process_fw_parquet.remote(file))

    # Wait for all the tasks to finish
    try:
        ray.get(result_refs)
    except Exception as e:
        print(f"Error processing the group: {e}")
