#!/usr/bin/env python3
"""
Given a parquet file with outlinks, fetch the link targets and write the
scraped pages as a WARC file.

Running on FineWeb-Edu:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/crawl/sample_outlinks.py \
    --urls_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-1M/links.99.parquet \
    --warc_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-1M/links.99.warc.gz \
    --robots_output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-1M/links.99.json
```
"""
import io
import json
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

import draccus
import fsspec
import ray
from tqdm_loggable.auto import tqdm

import requests
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_url(session: requests.Session, url: str) -> requests.Response | None:
    """Fetch the content of a URL."""
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.info(f"Failed to fetch {url}: {e}")
        return None


@dataclass
class FetchLinksConfig:
    urls_path: str
    warc_output_path: str
    robots_output_path: str


def fetch_links(urls_path: str, warc_output_path: str, robots_output_path: str):
    with fsspec.open(urls_path, "rt", compression="gzip") as fin:
        examples = [json.loads(line.strip()) for line in fin if line.strip()]
    logger.info(f"Found {len(examples)} examples in input file")

    urls = [ex["link_target"] for ex in examples]
    fetch_to_warc(urls, warc_output_path, robots_output_path)
    logger.info(f"WARC file created at: {warc_output_path}, robots.txt data written to {robots_output_path}")


def fetch_to_warc(urls: list[str], warc_output_path: str, robots_output_path: str):
    domains_to_robots: dict[str, str] = {}

    warc_buffer = io.BytesIO()

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "CCBot",
            }
        )
        writer = WARCWriter(warc_buffer, gzip=True)

        for url in tqdm(urls, desc="Fetching URLs"):
            parsed_url = urlparse(url)
            url_domain = parsed_url.netloc

            if url_domain not in domains_to_robots:
                # Construct the robots.txt URL
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                robots_url = f"{base_url}/robots.txt"
                # Fetch the robots.txt for this domain
                logger.info(f"Getting robots.txt: {url}")
                response = fetch_url(session, robots_url)
                if response is not None:
                    domains_to_robots[url_domain] = response.text

            logger.info(f"Processing: {url}")
            response = fetch_url(session, url)
            if response is None:
                continue

            # Prepare HTTP headers for WARC
            http_headers = []
            # Add status line
            status_line = f"{response.status_code} {response.reason}"
            http_headers.append(("Status", status_line))
            # Add other headers
            for header, value in response.headers.items():
                http_headers.append((header, value))

            # Create StatusAndHeaders object
            status_headers = StatusAndHeaders(status_line, http_headers, protocol="HTTP/1.1")

            # Wrap the response content in a BytesIO object
            payload_io = io.BytesIO(response.content)

            # Create WARC record
            try:
                warc_record = writer.create_warc_record(url, "response", payload=payload_io, http_headers=status_headers)
                writer.write_record(warc_record)
            except Exception as e:
                logger.info(f"Failed to write WARC record for {url}: {e}")

    # Get the WARC data from our in-memory buffer and write to disk
    warc_data = warc_buffer.getvalue()
    with fsspec.open(warc_output_path, "wb") as fout:
        fout.write(warc_data)

    with fsspec.open(robots_output_path, "w", compression="gzip") as fout:
        json.dump(domains_to_robots, fout)


@draccus.wrap()
def main(cfg: FetchLinksConfig):
    # Do all the processing in a remote function
    _ = ray.get(fetch_links.remote(cfg.urls_path, cfg.warc_output_path, cfg.robots_output_path))


if __name__ == "__main__":
    main()
