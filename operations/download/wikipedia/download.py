"""
wikipedia/download.py

Download script for the Wikipedia raw HTML data, provided by Wikimedia.

Home Page: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
"""

import logging
import os
import tarfile
from dataclasses import dataclass

import draccus
import fsspec
import ray
import requests
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_exists, fsspec_size

logger = logging.getLogger("ray")


@dataclass
class DownloadConfig:
    input_urls: list[str]
    revision: str
    output_path: str


@ray.remote(memory=10 * 1024 * 1024 * 1024)
def download_tar(url: str, output_path: str) -> None:
    output_path = os.path.join(output_path, url.split("/")[-1])

    logger.info(f"Downloading URL: {url} to {output_path}")

    if fsspec_exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path
    try:
        total_size = fsspec_size(url)
        pbar = tqdm(total=total_size, desc="Downloading File", unit="B", unit_scale=True)

        with fsspec.open(output_path, "wb") as f:
            r = requests.get(url, stream=True)

            for chunk in r.raw.stream(20 * 1024 * 1024, decode_content=False):
                if chunk:
                    f.write(chunk)
                    f.flush()

                    pbar.update(len(chunk))

        return output_path
    except Exception as e:
        logger.error(f"Error downloading URL: {url}")
        raise e


@ray.remote(memory=250 * 1024 * 1024 * 1024)
def process_file(input_file: str, output_path: str) -> None:
    logger.info(f"Processing file: {input_file}")
    logger.info(f"Output path: {output_path}")

    try:
        with fsspec.open(input_file) as f:
            with tarfile.open(fileobj=f, mode="r:gz") as tr:
                file_names = tr.getnames()
                logger.info(f"Extracting files: {file_names}")

                for file_name in tqdm(file_names):
                    file = tr.extractfile(file_name)
                    file_content = file.read()
                    file_path = os.path.join(output_path, file_name + ".gz")

                    # Each file is a .ndjson file, which contains about 18k-21k articles
                    # per file with size ranging from 200MB to 300MB
                    with fsspec.open(file_path, "wb", compression="gzip") as f:
                        f.write(file_content)

    except Exception as e:
        logger.error(f"Error processing file: {input_file}")
        raise e


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    logger.info("Starting transfer of Wikipedia dump...")

    MAX_CONCURRENT_DOWNLOADS = 10
    download_refs = []
    downloaded_files = []
    for url in cfg.input_urls:
        download_refs.append(download_tar.remote(url, cfg.output_path))

        # Wait for downloads to complete, processing MAX_CONCURRENT_DOWNLOADS at a time
        if len(download_refs) >= MAX_CONCURRENT_DOWNLOADS:
            # Wait for at least one download to complete
            ready_refs, download_refs = ray.wait(download_refs, num_returns=1)
            try:
                downloaded_file = ray.get(download_refs)
                downloaded_files.extend(downloaded_file)
            except Exception as e:
                logger.exception(f"Error downloading: {e}")
                raise e

    try:
        downloaded_file = ray.get(download_refs)
        downloaded_files.extend(downloaded_file)
    except Exception as e:
        raise e

    logger.info(f"Downloaded files: {downloaded_files}")

    for file in downloaded_files:
        output_path = os.path.join(cfg.output_path, cfg.revision)
        ray.get(process_file.remote(file, output_path))
