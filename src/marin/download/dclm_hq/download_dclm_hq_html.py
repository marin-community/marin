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

import io
import json
import logging
import os
import re
from dataclasses import dataclass

import draccus
import fsspec
import ray
import requests
import warcio
from tqdm import tqdm

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_glob

CC_IDX_HOST_URL = "http://34.72.201.218:8080"
logger = logging.getLogger("ray")


@dataclass
class DCLMHQDownloadConfig:
    input_path: str
    output_path: str


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


@ray.remote(memory=2 * 1024 * 1024 * 1024, max_retries=5)
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(
    input_file_path: str,
    output_file_path: str,
) -> None:
    logger.info(f"Starting processing of file {input_file_path}")
    logger.info(f"Source: {input_file_path}")
    logger.info(f"Destination: {output_file_path}")
    try:
        with (
            fsspec.open(input_file_path, compression="zstd") as source,
            fsspec.open(output_file_path, "wt", compression="gzip") as output,
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
        logger.info(f"File available at: {output_file_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


@ray.remote(memory=2 * 1024 * 1024 * 1024, max_retries=5)
def process_dclm_shard(
    input_path: str,
    output_path: str,
) -> None:
    logger.info(f"Processing DCLM shard {input_path}")
    logger.info(f"Output path: {output_path}")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 16

    shard_paths = [i.split("/")[-1] for i in fsspec_glob(os.path.join(input_path, "*.json.zst"))]

    for shard_path in shard_paths:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        input_file_path = os.path.join(input_path, shard_path)
        output_file_path = os.path.join(output_path, shard_path).replace(".json.zst", ".jsonl.gz")
        result_refs.append(process_file.remote(input_file_path, output_file_path))

    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")


# TODO: add executor step for this in the experiments
@draccus.wrap()
def extract_dclm_hq_dump(cfg: DCLMHQDownloadConfig) -> None:
    """
    Process the DCLM HQ dump in the input path and save the results to the output path.
    """
    logger.info(f"Starting processing of DCLM HQ dump in {cfg.input_path}")

    result_refs = []
    MAX_CONCURRENT_WORKERS = 50

    paths = [i.split("/")[-1] for i in fsspec_glob(os.path.join(cfg.input_path, "*"))]

    for path in paths:
        if len(result_refs) > MAX_CONCURRENT_WORKERS:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                logger.exception(f"Error processing the group: {e}")
                continue

        # input_path = "gs://marin-us-central2/raw/dolmino-mix-1124-157960/bb54cab/data/dclm/0000"
        input_path = os.path.join(cfg.input_path, path)
        output_path = os.path.join(cfg.output_path, path)
        result_refs.append(process_dclm_shard.remote(input_path, output_path))
    try:
        ray.get(result_refs)
    except Exception as e:
        logger.exception(f"Error processing the group: {e}")
