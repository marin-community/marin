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

import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
import requests
from zephyr import Dataset, flow_backend

from marin.core.runtime import cached_or_construct_output
from marin.download.nemotron_cc.utils import decompress_zstd_stream
from marin.utils import fsspec_exists

logger = logging.getLogger("ray")

myagent = "marin-nemotron-ingress/1.0"
NCC_PATH_FILE_URL = "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"


@cached_or_construct_output(success_suffix="SUCCESS")
def download_single_nemotron_path(input_file_path: str, output_file_path: str, chunk_size: int):
    """
    Fetches content from a Common Crawl path.
    Args:
        input_file_path (str): Path to the Common Crawl file
        output_file_path (str): Path to the output file
        chunk_size (int): Size of chunks to read
    Returns:
        list: List of content from the JSONL records
    """
    contents = []

    cc_url = f"https://data.commoncrawl.org/{input_file_path}"
    logger.info(f"Downloading Nemotron CC file {cc_url} to {output_file_path}")

    try:
        response = requests.get(cc_url, headers={"user-agent": myagent}, stream=True)

        if response.status_code == 200:
            contents = decompress_zstd_stream(response.raw, response.headers.get("content-length", 0), chunk_size)
        else:
            logger.error(f"Failed to fetch data: {response.status_code}")
            return None

    except Exception as e:
        logger.exception(f"Error fetching content from {cc_url}: {e}")
        return None

    if not contents:
        logger.warning("No valid JSONL records found")
        return None

    with fsspec.open(output_file_path, "w", compression="gzip") as f:
        for content in contents:
            dolma_format = {
                "id": content["warc_record_id"],
                "text": content["text"],
                "source": "nemotron",
                "format": "text",
                "metadata": {
                    f"nemotron_{key}": value for key, value in content.items() if key not in ("warc_record_id", "text")
                },
            }
            print(json.dumps(dolma_format), file=f)


@dataclass
class NemotronIngressConfig:
    output_path: str
    chunk_size: int = 65536


@draccus.wrap()
def download_nemotron_cc(cfg: NemotronIngressConfig):
    files = []

    paths_file_path = os.path.join(cfg.output_path, "data-jsonl.paths")
    logger.info(f"Downloading Nemotron CC path file {paths_file_path}")

    if fsspec_exists(paths_file_path):
        logger.warning(f"Paths file {paths_file_path} already exists. Skipping download.")
    else:
        with fsspec.open(NCC_PATH_FILE_URL, "rb") as f, fsspec.open(paths_file_path, "wb") as f_out:
            f_out.write(f.read())

    logger.info(f"Reading paths from {paths_file_path}")
    with fsspec.open(paths_file_path, "r", compression="gzip") as f:
        for line in f:
            file = line.strip()
            output_file_path = os.path.join(cfg.output_path, file).replace("jsonl.zstd", "jsonl.gz")
            if not fsspec_exists(output_file_path):
                files.append((file, output_file_path))
            else:
                logger.warning(f"Output file {output_file_path} already exists. Skipping download.")

    logger.info(f"Processing {len(files)} Nemotron CC files")

    backend = flow_backend()
    pipeline = Dataset.from_list(files).map(
        lambda file_info: download_single_nemotron_path(file_info[0], file_info[1], cfg.chunk_size)
    )

    list(backend.execute(pipeline))

    logger.info(f"Downloaded Nemotron CC files to {cfg.output_path}")
