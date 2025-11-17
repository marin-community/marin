# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Download DCLM HQ HTML data by fetching HTML content from Common Crawl.

Processes DCLM HQ JSONL files and enriches them with HTML content fetched from Common Crawl
via a custom index server. Uses zephyr for parallel processing with flattened parallelism.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=800 --memory=2GB \
    lib/marin/src/marin/download/dclm_hq/download_dclm_hq_html.py \
    --input_path gs://marin-us-central2/raw/dclm-baseline-1.0-parquet/global/ \
    --output_path gs://marin-data/processed/dclm-hq-html/
"""

import io
import json
import logging
import os
import re
from dataclasses import dataclass

import draccus
import fsspec
import requests
import warcio
from marin.utils import fsspec_glob
from tqdm import tqdm
from zephyr import Dataset, flow_backend
from zephyr.writers import ensure_parent_dir

CC_IDX_HOST_URL = "http://34.72.201.218:8080"
logger = logging.getLogger("ray")


@dataclass
class DCLMHQDownloadConfig:
    input_path: str
    output_path: str


@dataclass
class FileTask:
    """Represents a single file processing task."""

    input_file_path: str
    output_file_path: str


def fetch_warc_from_cc(s3_warc_path: str, length: int, offset: int) -> str:
    """
    Fetch a WARC record from Common Crawl S3 bucket using byte range requests we get
    from the CC index via `find_html_in_cc`.
    Args:
        s3_warc_path: Path to WARC file in S3 bucket
        length: Length of the record in bytes
        offset: Byte offset of the record in the WARC file
    Returns:
        The WARC record content as a string
    """
    # Convert string values to integers
    offset = int(offset)
    length = int(length)

    # Make range request to CommonCrawl
    response = requests.get(
        f"https://data.commoncrawl.org/{s3_warc_path}", headers={"Range": f"bytes={offset}-{offset + length - 1}"}
    )
    response.raise_for_status()

    # Parse WARC record and extract HTML content
    with io.BytesIO(response.content) as stream:
        for record in warcio.ArchiveIterator(stream):
            content = record.content_stream().read()
            return content.decode(errors="ignore")

    raise ValueError(f"No WARC records found in response from {s3_warc_path}")


def find_html_in_cc(split_id: str, target_uri: str) -> str | None:
    """
    We host our own index of the Common Crawl over GCP which we use in this function.
    For each call we receive a list of chunks that contain the HTML content for the given target URI.
    We then fetch each chunk and concatenate them together to form the complete HTML content.
    Args:
        split_id: The split ID of the Common Crawl
        target_uri: The target URI to find the HTML content for
    Returns:
        The HTML content as a string
    """
    resp = requests.get(f"{CC_IDX_HOST_URL}/{split_id}-index?url={target_uri}&output=json")

    resp.raise_for_status()

    chunks = [json.loads(chunk) for chunk in resp.text.split("\n") if chunk]
    sorted_chunks = sorted(chunks, key=lambda x: x["offset"])

    html_content = ""

    for chunk in sorted_chunks:
        warc_path = chunk["filename"]
        length = chunk["length"]
        offset = chunk["offset"]

        warc_record = fetch_warc_from_cc(warc_path, length, offset)

        html_content += warc_record

    return html_content


def process_file(task: FileTask) -> None:
    """Process a single DCLM file, fetching HTML from Common Crawl.

    Args:
        task: FileTask containing input and output file paths
    """
    logger.info(f"Starting processing of file {task.input_file_path}")
    logger.info(f"Source: {task.input_file_path}")
    logger.info(f"Destination: {task.output_file_path}")
    try:
        ensure_parent_dir(task.output_file_path)
        with (
            fsspec.open(task.input_file_path, compression="zstd") as source,
            fsspec.open(task.output_file_path, "wt", compression="gzip") as output,
        ):
            text_wrapper = io.TextIOWrapper(source, encoding="utf-8")

            for line in tqdm(text_wrapper, desc="Processing lines"):
                row = json.loads(line.strip())

                # We need to extract the split from where the record was for querying the index
                # The only place we have this information is in the warcinfo key in DCLM HQ
                # The format is:
                # warc-type: WARC/1.1
                # ...
                # isPartOf: CC-MAIN-2024-01
                # This however is a string and not a key-value pair, so we need to extract
                # the split from it via regex pattern `isPartOf:\s*(CC-MAIN-\d{4}-\d{2})`.
                # This pattern groups the value of the key `isPartOf` that is of the form
                # `CC-MAIN-xxxx-xx` where `xxxx` is a year and `xx` is a month.
                match = re.search(r"isPartOf:\s*(CC-MAIN-\d{4}-\d{2})", row["metadata"]["warcinfo"])
                if match is None:
                    logger.error(f"No split found for record ID: {row['metadata']['WARC-Record-ID']}")
                    continue

                is_part_of = match.group(1)

                try:
                    html_string = find_html_in_cc(is_part_of, row["metadata"]["WARC-Target-URI"])

                    if html_string is None:
                        logger.error(f"No HTML found for record ID: {row['metadata']['WARC-Record-ID']}")
                        continue

                    if "text" in row:
                        row.pop("text")

                    row["html"] = html_string

                    print(json.dumps(row), file=output)
                except Exception as e:
                    logger.exception(f"Error processing line: {e}")
                    continue

        logger.info("\nProcessing completed successfully!")
        logger.info(f"File available at: {task.output_file_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


# TODO: add executor step for this in the experiments
@draccus.wrap()
def extract_dclm_hq_dump(cfg: DCLMHQDownloadConfig) -> None:
    """Process the DCLM HQ dump in the input path and save the results to the output path.

    Flattens the nested directory structure (shards → files) into a single list of files
    and processes them in parallel using zephyr.
    """
    logger.info(f"Starting processing of DCLM HQ dump in {cfg.input_path}")

    # Flatten nested structure: discover all files upfront
    all_files = []
    paths = [i.split("/")[-1] for i in fsspec_glob(os.path.join(cfg.input_path, "*"))]

    logger.info(f"Found {len(paths)} shards to process")

    for path in paths:
        input_path = os.path.join(cfg.input_path, path)
        shard_paths = fsspec_glob(os.path.join(input_path, "*.json.zst"))

        for shard_path in shard_paths:
            input_file_path = shard_path
            output_file_path = os.path.join(cfg.output_path, path, os.path.basename(shard_path)).replace(
                ".json.zst", ".jsonl.gz"
            )

            all_files.append(FileTask(input_file_path=input_file_path, output_file_path=output_file_path))

    logger.info(f"Found {len(all_files)} files to process")

    # Single-level parallelism over all files
    backend = flow_backend()
    pipeline = Dataset.from_list(all_files).map(process_file)

    list(backend.execute(pipeline))

    logger.info("Processing completed successfully!")
