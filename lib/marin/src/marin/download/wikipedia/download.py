# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
wikipedia/download.py

Download script for the Wikipedia raw HTML data, provided by Wikimedia.

Home Page: https://dumps.wikimedia.org/other/enterprise_html/runs/

Example Usage (production, large dataset):
ENWIKI=https://dumps.wikimedia.org/other/enterprise_html/runs/20250320/enwiki-NS0-20250320-ENTERPRISE-HTML.json.tar.gz
uv run zephyr --backend=ray --max-parallelism=10 \
    lib/marin/src/marin/download/wikipedia/download.py \
    --input_urls $ENWIKI \
    --revision 20250320 --output_path gs://path/to/output

Example Usage (local testing, small dataset):
SIMPLEWIKI=https://dumps.wikimedia.org/other/enterprise_html/runs/20250320/simplewiki-NS0-20250320-ENTERPRISE-HTML.json.tar.gz
uv run zephyr --backend=threadpool --max-parallelism=4 --entry-point=download \
    lib/marin/src/marin/download/wikipedia/download.py \
    --input_urls "[$SIMPLEWIKI]" \
    --revision 20250320 --output_path /tmp/wikipedia_test

Note: The enwiki-NS0 file (English Wikipedia, namespace 0 = articles) is approximately 130 GB compressed.
      The simplewiki-NS0 file (Simple English Wikipedia) is much smaller at ~2 GB compressed.
"""

import logging
import os
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass

import draccus
import fsspec
import requests
from marin.utils import fsspec_size
from tqdm_loggable.auto import tqdm
from zephyr import Dataset, ZephyrContext, atomic_rename, load_jsonl

logger = logging.getLogger("ray")


@dataclass
class DownloadConfig:
    input_urls: list[str]
    revision: str
    output_path: str


def download_tar(url: str, output_prefix) -> str:
    shard_filename = url.split("/")[-1]
    output_filename = os.path.join(output_prefix, shard_filename)
    logger.info(f"Downloading URL: {url} to {output_filename}")

    try:
        total_size = fsspec_size(url)
        pbar = tqdm(total=total_size, desc="Downloading File", unit="B", unit_scale=True)

        with atomic_rename(output_filename) as tmp_filename, fsspec.open(tmp_filename, "wb") as f:
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


def process_file(input_file: str, output_path: str) -> Iterable[str]:
    logger.info(f"Processing file: {input_file}")
    logger.info(f"Output path: {output_path}")

    try:
        with fsspec.open(input_file) as f:
            with tarfile.open(fileobj=f, mode="r:gz") as tr:
                for info in tr:
                    with tr.extractfile(info) as file:
                        file_content = file.read()
                        file_path = os.path.join(output_path, info.name + ".gz")

                    # Each file is a .ndjson file, which contains about 18k-21k articles
                    # per file with size ranging from 200MB to 300MB
                    with (
                        atomic_rename(file_path) as tmpfile_path,
                        fsspec.open(tmpfile_path, "wb", compression="gzip") as output_f,
                    ):
                        output_f.write(file_content)
                        yield file_path

    except Exception as e:
        logger.error(f"Error processing file: {input_file}")
        raise e


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    """Download and process Wikipedia data."""
    logger.info("Starting transfer of Wikipedia dump...")
    output_base = os.path.join(cfg.output_path, cfg.revision)

    with ZephyrContext(name="download-wikipedia") as ctx:
        download_metrics = ctx.execute(
            Dataset.from_list(cfg.input_urls)
            .map(lambda url: download_tar(url, output_base))
            .write_jsonl(f"{output_base}/.metrics/download-{{shard:05d}}.jsonl", skip_existing=True),
        )

        # load all of the output filenames to process
        downloads = ctx.execute(Dataset.from_list(download_metrics).flat_map(load_jsonl))

        extracted = ctx.execute(
            Dataset.from_list(downloads)
            .flat_map(lambda file: process_file(file, output_base))
            .write_jsonl(f"{output_base}/.metrics/process-{{shard:05d}}.jsonl", skip_existing=True),
        )

    logger.info("Wikipedia dump transfer complete, wrote: %s", list(extracted))
