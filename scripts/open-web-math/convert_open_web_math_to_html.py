"""
Convert open-web-math to HTML:


```
python scripts/open-web-math/convert_open_web_math_to_html.py \
    --input_path gs://marin-us-central2/raw/open-web-math-fde8ef8/fde8ef8/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8/data/ \
    --html_output_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/
```

```
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python scripts/open-web-math/convert_open_web_math_to_html.py \
    --input_path gs://marin-us-central2/raw/open-web-math-fde8ef8/fde8ef8/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8/data/ \
    --html_output_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/
```
"""

import os
import ray
import json
import fsspec
import draccus
import logging
import traceback
import random
import pandas as pd

import hashlib

from datetime import datetime
from dataclasses import dataclass
from warcio import ArchiveIterator

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_glob, fsspec_exists


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_warc_path_from_open_web_math_metadata(metadata_str):
    metadata_dict = json.loads(metadata_str)
    return metadata_dict["warc_path"]


@ray.remote(memory=2 * 1024 * 1024 * 1024)  # 2 GB
@cached_or_construct_output(success_suffix="SUCCESS")  # We use this decorator to make this function idempotent
def process_one_shard(
    input_path: str,
    output_path: str,
):
    """
    Takes in the input file and processes it to get the html content.
    Download the WARC path in input_path and returns the content of the urls in the input_path

    Args:
    input_path (str): The input parquet shard to process
    output_path (str): Path to write gzipped JSONL with HTML for URLs in the input parquet shard
    """
    logger.info(f"Processing {input_path}, writing to {output_path}")

    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError as e:
        logger.exception(f"Error reading the parquet file: {e}")
        raise e

    urls = df["url"].tolist()
    # All frames will have same file_path, by design
    s3_url = df["file_path"].iloc[0]
    index = df.index.tolist()

    # url_dict is url to index in df so that we can update that record
    url_dict = {url: idx for idx, url in zip(index, urls)}
    num_urls_found = 0  # Used to early terminate
    num_urls_processed = 0
    length_url_inp_list = len(urls)

    logger.info(f"Processing {s3_url} in {input_path}")
    length_warc = 0
    # NOTE: make sure s3 keys are setup, either on the cluster
    # or by manually initializing with:
    # fsspec.filesystem("s3", anon=False, key="...", secret="...")
    s3_fs = fsspec.filesystem(
        "s3",
        anon=False,
    )

    with s3_fs.open(s3_url, mode="rb") as file_stream:
        for record in ArchiveIterator(file_stream):
            if num_urls_found == length_url_inp_list:
                break

            # Check if it's a response record
            if record.rec_type == "response":
                # Process the record
                url = record.rec_headers.get_header("WARC-Target-URI")
                length_warc += 1

                if url in url_dict:
                    num_urls_found += 1
                    url_idx_in_df = url_dict[url]
                    if num_urls_found % 100 == 0:
                        logger.info(
                            f"Found Url {num_urls_found = }, Processed Url {num_urls_processed = }, "
                            f"length of warc {length_warc = }"
                        )

                    try:
                        content = record.content_stream().read()
                        html_decoded = content.decode(errors="ignore")
                        df.loc[url_idx_in_df, "html"] = html_decoded
                        num_urls_processed += 1
                    except Exception as e:
                        # We are just ignoring the error and moving forward as these errors are generally not a lot
                        logger.exception(f"Error processing {url} in {s3_url} for {input_path}: {e}")
                        traceback.print_exc()

    logger.info(
        f"Processed {input_path}, found {length_warc} records, {length_url_inp_list} urls, "
        f"{length_warc / length_url_inp_list} ratio"
    )

    with fsspec.open(output_path, "wt", compression="gzip") as f:  # html output
        for index, row in df.iterrows():
            out_open_web_math = row.to_dict()
            out_dolma = {
                # NOTE: open-web-math doesn't have an ID field, so we
                # take the md5hash of its url and the date
                "id": hashlib.md5((str(out_open_web_math["url"]) + str(out_open_web_math["date"])).encode()).hexdigest(),
                "source": "open-web-math",
                "format": "html",
                "html": out_open_web_math["html"],
            }
            print(json.dumps(out_dolma), file=f)

    # num_urls_found should be equal to length_url_inp_list
    logger.info(
        f"Found: {num_urls_found}, Processed: {num_urls_processed}, out of {length_url_inp_list} urls, "
        f"in {input_path}"
        f"AWS URL: {s3_url}"
        f"Found {length_warc} records in the WARC file"
    )

    return True


@dataclass
class ParquetOpenWebMathConfig:
    input_path: str
    html_output_path: str


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def write_group_to_parquet(output_file, group_df, group_success_file):
    # Check if the success file already exists
    if fsspec_exists(group_success_file):
        logger.info(f"Shard {output_file} already exists, skipping...")
        return output_file

    # Save the group to a parquet file
    group_df.to_parquet(output_file)
    # Create the group success file
    with fsspec.open(group_success_file, "w") as f:
        metadata = {
            "datetime": str(datetime.utcnow()),
        }
        print(json.dumps(metadata), file=f)

    return output_file


@ray.remote(memory=256 * 1024 * 1024 * 1024)
def group_open_web_math_by_warc(input_paths: list[str], output_path: str):
    """
    Given open-web-math files, group the examples by their source WARC.

    Parameters:
    input_paths (str): Path to the open-web-math parquet files
    output_path (str): Path to the output folder where we will write the open-web-math examples,
                       grouped by their source WARC.
    """
    success_file_path = os.path.join(output_path, f"_examples_groupby_warc_success")
    if fsspec_exists(success_file_path):
        logger.info(f"Already grouped open-web-math by WARC, skipping...")
        return

    logger.info(f"Grouping examples at {input_paths} by their source WARC")
    datetime_start = datetime.utcnow()

    try:
        df = pd.concat([pd.read_parquet(file) for file in input_paths], ignore_index=True)
    except FileNotFoundError as e:
        logger.exception(f"Error reading the parquet file: {e}")
        raise e

    df["file_path"] = df["metadata"].apply(lambda x: json.loads(x)["warc_path"])
    grouped = df.groupby("file_path")

    remote_refs = []

    # Using Ray to parallelize the writing of groups
    for index, (_, group_df) in enumerate(grouped):
        output_file = os.path.join(output_path, f"{index}_warc_examples.parquet")
        group_success_file = os.path.join(output_path, f"{index}_warc_examples.success")

        # Queue up remote task for writing this group
        remote_refs.append(write_group_to_parquet.remote(output_file, group_df, group_success_file))

    # Wait for all groups to be written
    ray.get(remote_refs)

    datetime_end = datetime.utcnow()

    # Create the overall success file
    with fsspec.open(success_file_path, "w") as f:
        metadata = {
            "input_paths": input_paths,
            "output_path": output_path,
            "datetime_start": str(datetime_start),
            "datetime_end": str(datetime_end),
        }
        print(json.dumps(metadata), file=f)


@draccus.wrap()
def process_open_web_math(cfg: ParquetOpenWebMathConfig):
    files = fsspec_glob(os.path.join(cfg.input_path, "*.parquet"))
    # Group open-web-math examples by their source WARC
    groupby_ref = group_open_web_math_by_warc.remote(files, cfg.html_output_path)
    _ = ray.get(groupby_ref)
    shard_paths_to_process = fsspec_glob(os.path.join(cfg.html_output_path, "*_warc_examples.parquet"))
    logger.info("Found {len(shard_paths_to_process)} shards to fetch HTML for")

    # Process each of the example shards by downloading the original WARC
    # and picking out the HTML for the URLs of interest
    # Set a limit on the number of concurrent tasks so we don't overwhelm CC
    MAX_CONCURRENT_TASKS = 500

    # Shuffle to encourage different workers to hit different AWS prefixes,
    # so we don't run into per-prefix rate limits.
    random.shuffle(shard_paths_to_process)

    # Launch the initial MAX_CONCURRENT_TASKS batch of tasks
    unfinished = []
    for _ in range(min(MAX_CONCURRENT_TASKS, len(shard_paths_to_process))):
        # shard_to_process is of form gs://<html_output_path>/0_warc_examples.parquet
        shard_to_process = shard_paths_to_process.pop()
        # shard_output_path is of form gs://<html_output_path>/0.parquet
        shard_output_path = shard_to_process.replace("_warc_examples.parquet", ".jsonl.gz")
        unfinished.append(process_one_shard.remote(shard_to_process, shard_output_path))

    while unfinished:
        # Returns the first ObjectRef that is ready.
        finished, unfinished = ray.wait(unfinished, num_returns=1)
        try:
            _ = ray.get(finished[0])
        except Exception as e:
            logger.exception(f"Error processing shard: {e}")

        # If we have more shard paths left to process, add them to the unfinished queue.
        if shard_paths_to_process:
            # shard_to_process is of form gs://<html_output_path>/0_warc_examples.parquet
            shard_to_process = shard_paths_to_process.pop()
            # shard_output_path is of form gs://<html_output_path>/0.jsonl.gz
            shard_output_path = shard_to_process.replace("_warc_examples.parquet", ".jsonl.gz")
            unfinished.append(process_one_shard.remote(shard_to_process, shard_output_path))


if __name__ == "__main__":
    process_open_web_math()
