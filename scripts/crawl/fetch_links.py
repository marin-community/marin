#!/usr/bin/env python3
"""
Given a parquet file with outlinks, fetch the link targets and write the
scraped pages as a WARC file.

Running on FineWeb-Edu:

```
python marin/run/ray_run.py \
    --pip_deps 'warcio' \
    --no_wait -- \
    python scripts/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-1M/ \
    --output_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-1M/
```
"""
import io
import json
import logging
import os
import pathlib
import random
from dataclasses import dataclass
from urllib.parse import urlparse

import draccus
import fsspec
import pandas as pd
import ray
import requests
from tqdm_loggable.auto import tqdm
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_url(session: requests.Session, url: str) -> tuple[requests.Response, None] | tuple[None, str]:
    """Fetch the content of a URL."""
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response, None
    except Exception as e:
        logger.info(f"Failed to fetch {url}: {e}")
        return None, str(e)


@dataclass
class FetchLinksConfig:
    urls_input_directory: str
    output_directory: str


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def fetch_links(urls_path: str, warc_output_path: str, robots_output_path: str, errors_output_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    success_path = warc_output_path + ".SUCCESS"
    if fsspec_exists(success_path):
        logger.info(f"Already processed and wrote WARC to {warc_output_path}, skipping...")
        return

    with fsspec.open(urls_path) as f:
        df = pd.read_parquet(f)
    logger.info(f"Found {len(df)} examples in input file")

    # Extract the URLs from the "link_target" column
    urls = df["link_target"].tolist()

    # Randomly shuffle the URLs to load balance so we aren't repeatedly hitting a particular host
    random.shuffle(urls)

    fetch_to_warc(urls, warc_output_path, robots_output_path, errors_output_path)

    # Create success file
    with fsspec.open(success_path, "w") as fout:
        json.dump(
            {
                "urls_path": urls_path,
                "warc_output_path": warc_output_path,
                "robots_output_path": robots_output_path,
                "errors_output_path": errors_output_path,
            },
            fout,
        )

    logger.info(
        f"WARC file created at: {warc_output_path}\n"
        f"robots.txt data written to {robots_output_path}\n"
        f"errors written to {errors_output_path}"
    )


def fetch_to_warc(urls: list[str], warc_output_path: str, robots_output_path: str, errors_output_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    domains_to_robots: dict[str, str] = {}
    fetch_errors: dict[str, str] = {}

    warc_buffer = io.BytesIO()

    with requests.Session() as session:
        session.headers.update({"User-Agent": "CCBot"})
        writer = WARCWriter(warc_buffer, gzip=True)

        for url in tqdm(urls, desc="Fetching URLs"):
            parsed_url = urlparse(url)
            url_domain = parsed_url.netloc

            if url_domain not in domains_to_robots:
                # Construct the robots.txt URL
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                robots_url = f"{base_url}/robots.txt"

                logger.info(f"Getting robots.txt: {robots_url}")
                robots_response, _ = fetch_url(session, robots_url)
                if robots_response is not None:
                    domains_to_robots[url_domain] = robots_response.text

            logger.info(f"Processing: {url}")
            response, err = fetch_url(session, url)
            if response is None:
                if err:
                    fetch_errors[url] = err
                continue

            # Prepare HTTP headers for WARC
            http_headers = []
            status_line = f"{response.status_code} {response.reason}"
            http_headers.append(("Status", status_line))
            for header, value in response.headers.items():
                http_headers.append((header, value))

            status_headers = StatusAndHeaders(status_line, http_headers, protocol="HTTP/1.1")
            payload_io = io.BytesIO(response.content)

            # Create WARC record
            try:
                warc_record = writer.create_warc_record(url, "response", payload=payload_io, http_headers=status_headers)
                writer.write_record(warc_record)
            except Exception as e:
                logger.info(f"Failed to write WARC record for {url}: {e}")
                fetch_errors[url] = f"WARC write error: {e}"

    # Get the WARC data from our in-memory buffer and write to disk
    warc_data = warc_buffer.getvalue()
    with fsspec.open(warc_output_path, "wb") as fout:
        fout.write(warc_data)

    # Write domains_to_robots data
    with fsspec.open(robots_output_path, "w", compression="gzip") as fout:
        json.dump(domains_to_robots, fout)

    with fsspec.open(errors_output_path, "w", compression="gzip") as fout:
        json.dump(fetch_errors, fout)


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def get_shard_indices_to_process(urls_input_directory: str) -> list[int]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".parquet").removeprefix(f"links."))
        for path in fsspec_glob(os.path.join(urls_input_directory, "links.*.parquet"))
    ]
    shard_indices = sorted(shard_indices)
    logger.info(f"Found {len(shard_indices)} shards to process")
    return shard_indices


@draccus.wrap()
def main(cfg: FetchLinksConfig):
    shard_indices_to_process = ray.get(get_shard_indices_to_process.remote(cfg.urls_input_directory))
    num_shards_to_process = len(shard_indices_to_process)

    # Set a limit on the number of concurrent tasks so we don't make too many network requests
    MAX_CONCURRENT_TASKS = 10
    num_shards_submitted = 0
    unfinished = []

    def submit_shard_task(shard_index):
        nonlocal num_shards_submitted
        input_path = os.path.join(cfg.urls_input_directory, f"links.{shard_index}.parquet")
        warc_output_path = os.path.join(cfg.output_directory, f"links.{shard_index}.warc.gz")
        robots_output_path = os.path.join(cfg.output_directory, f"links.{shard_index}_robots.json.gz")
        errors_output_path = os.path.join(cfg.output_directory, f"links.{shard_index}_errors.json.gz")
        unfinished.append(fetch_links.remote(input_path, warc_output_path, robots_output_path, errors_output_path))

        num_shards_submitted += 1
        if num_shards_submitted % 10 == 0:
            logger.info(
                f"Submitted {num_shards_submitted} / {num_shards_to_process} shards "
                f"({num_shards_submitted / num_shards_to_process})"
            )

    # Launch the initial MAX_CONCURRENT_TASKS batch of tasks
    for _ in range(min(MAX_CONCURRENT_TASKS, len(shard_indices_to_process))):
        submit_shard_task(shard_indices_to_process.pop())

    while unfinished:
        finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
        try:
            _ = ray.get(finished)
        except Exception as e:
            logger.exception(f"Error processing shard: {e}")

        # If we have more shard paths left to process and we haven't hit the max
        # number of concurrent tasks, add tasks to the unfinished queue.
        while shard_indices_to_process and len(unfinished) < MAX_CONCURRENT_TASKS:
            submit_shard_task(shard_indices_to_process.pop())


if __name__ == "__main__":
    main()
