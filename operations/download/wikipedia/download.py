"""
wikipedia/download.py

Download script for the Wikipedia raw HTML data, provided by Wikimedia.

Home Page: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
"""

import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
import ray
from tqdm import tqdm

logger = logging.getLogger("ray")


@dataclass
class DownloadConfig:
    input_urls: list[str]
    revision: str
    output_path: str


@ray.remote
def process_url(url: str, output_path: str) -> None:
    logger.info(f"Processing URL: {url}")
    logger.info(f"Output path: {output_path}")

    try:
        with fsspec.open(url, "rb") as tar_file:
            tar_fs = fsspec.filesystem("tar", fo=tar_file)

            # List all files within the tar archive
            files = tar_fs.ls("/")
            logger.info(f"Files in the tar archive: {files}")

            # Read each file in the tar archive
            for file_info in tqdm(files, desc="Extracting files"):
                file_path = file_info["name"]
                output_file_path = os.path.join(output_path, file_path.replace(".ndjson", ".jsonl.gz"))
                with (
                    tar_fs.open(file_path, "r") as file,
                    fsspec.open(output_file_path, "wb", compression="gzip") as output_file,
                ):
                    content = file.read()
                    logger.info(f"Content of {file_path}:")

                    for line in tqdm(content.split("\n"), desc="Writing content"):
                        out_dict = {
                            "id": line["identifier"],
                            "url": ["url"],
                            "date_published": line["event"]["date_published"],
                            "html": line["article_body"]["html"],
                            "wikitext": line["article_body"]["wikitext"],
                        }
                        logger.info(out_dict, file=output_file)

    except Exception as e:
        logger.error(f"Error processing URL: {url}")
        raise e


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    print("Starting transfer of Wikipedia dump...")

    for url in cfg.input_urls:
        output_path = os.path.join(cfg.output_path, cfg.revision)
        ray.get(process_url.remote(url, output_path))
