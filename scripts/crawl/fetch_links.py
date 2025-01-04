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
    --urls_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-1M/links.99.parquet \
    --warc_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-1M/links.99.warc.gz \
    --robots_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-1M/links.99_robots.json.gz \
    --errors_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-1M/links.99_errors.json.gz
```
"""
import io
import json
import logging
from dataclasses import dataclass
from urllib.parse import urlparse
import random

import draccus
import fsspec
import ray
import pandas as pd
from tqdm_loggable.auto import tqdm

import requests
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

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
    urls_path: str
    warc_output_path: str
    robots_output_path: str
    errors_output_path: str


@ray.remote(memory=64 * 1024 * 1024 * 1024)
def fetch_links(urls_path: str, warc_output_path: str, robots_output_path: str, errors_output_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    success_path = warc_output_path + ".SUCCESS"
    if fsspec.exists(success_path):
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


@draccus.wrap()
def main(cfg: FetchLinksConfig):
    # Do all the processing in a remote function
    _ = ray.get(fetch_links.remote(cfg.urls_path, cfg.warc_output_path, cfg.robots_output_path, cfg.errors_output_path))


if __name__ == "__main__":
    main()
