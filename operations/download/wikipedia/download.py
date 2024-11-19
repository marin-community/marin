"""
wikipedia/download.py

Download script for the Wikipedia raw HTML data, provided by Wikimedia.

Home Page: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
"""

import bz2
from dataclasses import dataclass
from pathlib import Path

import draccus
import fsspec
import requests
from tqdm import tqdm

fs_http = fsspec.filesystem("http")
fs_gcs = fsspec.filesystem("gcs")


@dataclass
class DownloadConfig:
    input_path: Path
    output_path: str
    chunk_size: int = 1024 * 1024  # 1MB


def get_file_size(url):
    """Get content length from headers without downloading"""
    response = requests.head(url)
    return int(response.headers.get("content-length", 0))


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    total_size = get_file_size(cfg.input_path)

    try:
        print("Starting transfer of Wikipedia dump...")
        print(f"Source: {cfg.input_path}")
        print(f"Destination: {cfg.output_path}")

        decompressor = bz2.BZ2Decompressor()

        with (
            fsspec.open(cfg.input_path, "rb") as source,
            fsspec.open(cfg.output_path, "wb") as destination,
            tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading and decompressing") as pbar,
        ):

            while True:
                chunk = source.read(cfg.chunk_size)
                if not chunk:
                    break

                # Decompress chunk
                try:
                    decompressed_chunk = decompressor.decompress(chunk)
                    if decompressed_chunk:
                        destination.write(decompressed_chunk)
                except EOFError:
                    # Handle end of bz2 stream
                    break

                pbar.update(len(chunk))

        print("\nTransfer completed successfully!")
        print(f"File available at: {cfg.output_path}")

    except Exception as e:
        print(f"Error during transfer: {e}")
        raise
