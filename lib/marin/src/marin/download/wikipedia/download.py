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
wikipedia/download.py

Download script for the Wikipedia raw HTML data, provided by Wikimedia.

Home Page: https://dumps.wikimedia.org/other/enterprise_html/runs/

Example Usage:
ENWIKI=https://dumps.wikimedia.org/other/enterprise_html/runs/20250320/enwiki-NS0-20250320-ENTERPRISE-HTML.json.tar.gz
uv run zephyr --backend=ray --max-parallelism=10 \
    lib/marin/src/marin/download/wikipedia/download.py \
    --input_urls $ENWIKI \
    --revision 20250320 --output_path gs://path/to/output

Note: The enwiki-NS0 file (English Wikipedia, namespace 0 = articles) is approximately 130 GB compressed.
"""

import logging
import os
import tarfile
from dataclasses import dataclass

import draccus
import fsspec
import requests
from marin.utils import fsspec_exists, fsspec_size
from tqdm_loggable.auto import tqdm
from zephyr import Dataset, flow_backend

logger = logging.getLogger("ray")


@dataclass
class DownloadConfig:
    input_urls: list[str]
    revision: str
    output_path: str


def make_download_function(urls: list[str], output_path: str):
    """Create a download function with URLs and output path in closure."""

    def download_tar(shard_filename: str) -> str:
        # Extract shard index from filename like "task-00001.txt"
        import re

        match = re.search(r"-([0-9]{5})", shard_filename)
        if match is None:
            raise ValueError(f"Could not extract shard number from: {shard_filename}")
        shard_idx = int(match.group(1))

        url = urls[shard_idx]
        output_filename = os.path.join(output_path, f"downloaded-{shard_idx:05d}.tar.gz")

        logger.info(f"Downloading URL: {url} to {output_filename}")

        try:
            total_size = fsspec_size(url)
            pbar = tqdm(total=total_size, desc="Downloading File", unit="B", unit_scale=True)

            with fsspec.open(output_filename, "wb") as f:
                r = requests.get(url, stream=True)

                for chunk in r.raw.stream(20 * 1024 * 1024, decode_content=False):
                    if chunk:
                        f.write(chunk)
                        f.flush()

                        pbar.update(len(chunk))

            return output_filename
        except Exception as e:
            logger.error(f"Error downloading URL: {url}")
            raise e

    return download_tar


def process_file(input_file: str, output_path: str) -> None:
    logger.info(f"Processing file: {input_file}")
    logger.info(f"Output path: {output_path}")

    try:
        with fsspec.open(input_file) as f:
            with tarfile.open(fileobj=f, mode="r:gz") as tr:
                for info in tr:
                    extracted_file = tr.extractfile(info)
                    if extracted_file is None:
                        continue
                    with extracted_file as file:
                        file_content = file.read()
                        file_path = os.path.join(output_path, info.name + ".gz")
                        # Each file is a .ndjson file, which contains about 18k-21k articles
                        # per file with size ranging from 200MB to 300MB
                        with fsspec.open(file_path, "wb", compression="gzip") as output_f:
                            output_f.write(file_content)

    except Exception as e:
        logger.error(f"Error processing file: {input_file}")
        raise e


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    """Download and process Wikipedia data."""
    backend = flow_backend()
    logger.info("Starting transfer of Wikipedia dump...")

    output_base = os.path.join(cfg.output_path, cfg.revision)

    # Create task filenames with shard numbers that can be inferred
    task_filenames = [f"task-{i:05d}.txt" for i in range(len(cfg.input_urls))]

    # Single pipeline: download each URL, then extract the downloaded file
    download_fn = make_download_function(cfg.input_urls, cfg.output_path)

    def _output_exists(task_filename: str) -> bool:
        import re

        match = re.search(r"-([0-9]{5})", task_filename)
        if match is None:
            return False
        shard_idx = int(match.group(1))
        output_file = os.path.join(cfg.output_path, f"downloaded-{shard_idx:05d}.tar.gz")
        return fsspec_exists(output_file)

    pipeline = (
        Dataset.from_list(task_filenames)
        .filter(lambda task: not _output_exists(task))
        .map(download_fn)
        .map(lambda file: process_file(file, output_base))
    )

    list(backend.execute(pipeline))
