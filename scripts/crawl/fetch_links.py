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
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M/ \
    --output_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/
```

Running on OpenWebMath:

```
python marin/run/ray_run.py \
    --pip_deps 'warcio' \
    --no_wait -- \
    python scripts/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/ \
    --output_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/
```
"""
import io
import json
import logging
import os
import pathlib
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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


@dataclass
class FetchLinksConfig:
    urls_input_directory: str
    output_directory: str
    threads_per_shard: int = 40
    max_concurrent_shards: int = 20


def fetch_url(
    session: requests.Session,
    url: str,
    timeout: float = 30,
    max_size_bytes: int = 10 * 1024 * 1024,
    chunk_size: int = 1 * 1024 * 1024,
) -> tuple[requests.Response, int | None, None, bool] | tuple[None, int | None, str, bool]:
    """Fetch the content of a URL, truncating at 10 MB if necessary.

    Returns a tuple with four items:
    - Response, if the request was successful, else None
    - The response status code, if we received a response, else None
    - The error string if the request was not successful, else None
    - True if the domain was unreachable, else False
      (e.g., successful request or we get a 4xx or 5xx response)
    """
    try:
        # Use stream=True to avoid downloading the entire body into memory at once.
        with session.get(url, timeout=timeout, stream=True) as r:
            r.raise_for_status()

            # Create a mutable Response object that we'll patch with truncated content.
            truncated_response = requests.models.Response()
            truncated_response.status_code = r.status_code
            truncated_response.reason = r.reason
            truncated_response.headers = r.headers
            truncated_response.url = r.url
            truncated_response.request = r.request

            # Read response in chunks, truncate at 10 MB
            content_buffer = io.BytesIO()
            downloaded_bytes = 0

            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    new_size = downloaded_bytes + len(chunk)

                    # If adding this chunk would exceed the 10MB cap, only write the part that fits
                    if new_size > max_size_bytes:
                        allowed_bytes = max_size_bytes - downloaded_bytes
                        content_buffer.write(chunk[:allowed_bytes])
                        # Log that we've truncated the response:
                        logger.info(f"Truncated {url} after {max_size_bytes} bytes.")
                        break
                    else:
                        content_buffer.write(chunk)
                        downloaded_bytes = new_size

            truncated_response._content = content_buffer.getvalue()
            return truncated_response, r.status_code, None, False
    except requests.exceptions.HTTPError as e:
        # We got a response from the server, so the domain is reachable
        logger.exception(f"Error fetching {url}: {e}")
        return None, e.response.status_code, str(e), False
    except Exception as e:
        # Anything else is a connection-level error (DNS, SSL, etc.)
        logger.exception(f"Error fetching {url}: {e}")
        return None, None, str(e), True


def _fetch_one_url(
    url: str,
    session: requests.Session,
    domain_failure_counts: dict[str, int],
    domains_to_robots: dict[str, str],
    robots_fetch_errors: dict[str, str],
    fetch_errors: dict[str, str],
    domain_locks: dict[str, threading.Lock],
    domain_next_allowed: dict[str, float],
    domain_backoff_seconds: dict[str, float],
    logger: logging.Logger,
) -> tuple[str, requests.Response | None]:
    """
    Worker function that:
      1. Ensures we don't hit the same domain in parallel (domain_locks).
      2. Respects "next allowed request time" per domain.
      3. Fetches robots.txt if not already done.
      4. Tries 3 times to fetch the main URL if we get 429.
      5. Returns (url, response or None).
    """
    parsed_url = urlparse(url)
    url_domain = parsed_url.netloc

    # First, possibly fetch robots.txt (done once per domain).
    # We'll grab the domain lock so that no parallel request for robots.txt is done.
    with domain_locks[url_domain]:
        now = time.time()
        if now < domain_next_allowed[url_domain]:
            wait_time = domain_next_allowed[url_domain] - now
            logger.info(f"[robots] Rate limiting {url_domain}; sleeping for {wait_time:.2f}s.")
            time.sleep(wait_time)

        if url_domain not in domains_to_robots and url_domain not in robots_fetch_errors:
            logger.info(f"Getting robots.txt for domain: {url_domain}")
            # Decreasing timeout based on failures
            current_timeout = decreasing_timeout(
                base_timeout=30.0,
                failures=domain_failure_counts[url_domain],
                factor=2.0,
                min_timeout=1.0,
            )
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = f"{base_url}/robots.txt"

            robots_response, robots_response_code, robots_err, robots_conn_err = fetch_url(
                session, robots_url, timeout=current_timeout
            )
            if robots_response is None:
                if robots_err:
                    robots_fetch_errors[url_domain] = robots_err
                if robots_conn_err:
                    domain_failure_counts[url_domain] += 1
                if robots_response_code == 429:
                    domain_backoff_seconds[url_domain] *= 2
                    domain_next_allowed[url_domain] = time.time() + domain_backoff_seconds[url_domain]
            else:
                logger.info(f"Got robots.txt for {url_domain}")
                domains_to_robots[url_domain] = robots_response.text
                # reset domain failures if we succeed
                domain_failure_counts[url_domain] = 0

    # Now fetch the main URL, possibly retrying on 429
    attempts = 0
    max_retries = 3
    response = None
    final_err = None

    while attempts < max_retries:
        attempts += 1
        with domain_locks[url_domain]:
            # Enforce domain-level rate limiting
            now = time.time()
            if now < domain_next_allowed[url_domain]:
                wait_time = domain_next_allowed[url_domain] - now
                logger.info(f"[main URL] Rate limiting {url_domain}; sleeping for {wait_time:.2f}s.")
                time.sleep(wait_time)

            # Decide how small the timeout should be now
            num_domain_failures = domain_failure_counts[url_domain]
            current_timeout = decreasing_timeout(
                base_timeout=30.0, failures=num_domain_failures, factor=2.0, min_timeout=1.0
            )

            logger.info(f"Fetching {url} (attempt {attempts}/{max_retries})")
            fetched_response, fetched_response_code, err, is_conn_err = fetch_url(session, url, timeout=current_timeout)

            if fetched_response is not None:
                # A non-429 response means success or some HTTP status != 429
                # either way we consider that "done"
                response = fetched_response
                domain_failure_counts[url_domain] = 0
                # Slight "cooldown" after success
                domain_backoff_seconds[url_domain] = max(1.0, domain_backoff_seconds[url_domain])
                domain_next_allowed[url_domain] = time.time() + domain_backoff_seconds[url_domain] / 5.0
                break

            # If we reach here, we have an error
            final_err = err

            if is_conn_err:
                # connection-level error
                domain_failure_counts[url_domain] += 1
                # We won't automatically retry connection errors beyond the single attempt
                # but if you wanted to, you could do so here.
                break

            if fetched_response_code == 429:
                logger.warning(f"Hit 429 for {url_domain} on attempt {attempts}")
                # exponential backoff for 429
                domain_backoff_seconds[url_domain] *= 2
                domain_next_allowed[url_domain] = time.time() + domain_backoff_seconds[url_domain]
                # If we haven't exhausted max_retries, loop again
                # We'll attempt again in the next iteration
                if attempts < max_retries:
                    continue
                else:
                    # No more attempts left
                    break

            # If it's some other HTTP error (404, 500, etc.), we won't retry
            break

    # If we exhausted attempts and still have no success, record an error
    if response is None and final_err:
        fetch_errors[url] = final_err

    return url, response


@ray.remote(memory=128 * 1024 * 1024 * 1024, num_cpus=8)
def fetch_links(
    urls_path: str, warc_output_path: str, robots_output_path: str, errors_output_path: str, threads_per_shard: int
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    success_path = warc_output_path + ".SUCCESS"
    if fsspec_exists(success_path):
        logger.info(f"Already processed and wrote WARC to {warc_output_path}, skipping...")
        return

    with fsspec.open(urls_path) as f:
        df = pd.read_parquet(f)
    logger.info(f"Found {len(df)} examples in input file {urls_path}")

    # Extract the URLs from the "link_target" column
    urls = df["link_target"].tolist()
    # Deduplicate the URLs
    urls = list(set(urls))

    # Randomly shuffle the URLs to load balance so we aren't repeatedly hitting a particular host
    random.shuffle(urls)

    logger.info(f"Fetching {len(urls)} deduplicated URLs from input file {urls_path}")
    fetch_to_warc(urls, warc_output_path, robots_output_path, errors_output_path, threads_per_shard)

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


def decreasing_timeout(
    base_timeout: float = 30.0, failures: int = 0, factor: float = 2.0, min_timeout: float = 1.0
) -> float:
    """
    The more failures a domain has, the smaller the timeout we allow.
    We clamp the final value so it doesn't go below min_timeout.
    """
    timeout = base_timeout / (factor ** min(failures / 5, 5))
    return max(timeout, min_timeout)


def fetch_to_warc(
    urls: list[str], warc_output_path: str, robots_output_path: str, errors_output_path: str, threads_per_shard: int
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Data structures to track domain info
    domain_failure_counts = Counter()  # for the decreasing_timeout logic
    domain_next_allowed = defaultdict(float)  # domain -> earliest next request time
    # Start with a minimal backoff of 1 second for each domain:
    domain_backoff_seconds = defaultdict(lambda: 1.0)
    domain_locks = defaultdict(threading.Lock)

    domains_to_robots: dict[str, str] = {}
    fetch_errors: dict[str, str] = {}
    robots_fetch_errors: dict[str, str] = {}

    warc_buffer = io.BytesIO()

    # We'll open a single requests.Session in the main thread
    with requests.Session() as session:
        session.headers.update({"User-Agent": "CCBot"})
        writer = WARCWriter(warc_buffer, gzip=True)

        # We'll process the URLS in parallel:
        with ThreadPoolExecutor(max_workers=threads_per_shard) as executor:
            future_to_url = {
                executor.submit(
                    _fetch_one_url,
                    url,
                    session,
                    domain_failure_counts,
                    domains_to_robots,
                    robots_fetch_errors,
                    fetch_errors,
                    domain_locks,
                    domain_next_allowed,
                    domain_backoff_seconds,
                    logger,
                ): url
                for url in urls
            }

            # We can show a progress bar with as_completed:
            for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Fetching URLs"):
                url = future_to_url[future]
                url, response = future.result()
                # If `response` is not None, write it to WARC
                if response is not None:
                    status_line = f"{response.status_code} {response.reason}"
                    logger.info(f"Got response {status_line} for {url}")

                    http_headers = [("Status", status_line)]
                    for header, value in response.headers.items():
                        http_headers.append((header, value))

                    status_headers = StatusAndHeaders(status_line, http_headers, protocol="HTTP/1.1")
                    payload_io = io.BytesIO(response.content)

                    warc_record = writer.create_warc_record(
                        url, "response", payload=payload_io, http_headers=status_headers
                    )
                    writer.write_record(warc_record)

    # Write final WARC data
    warc_data = warc_buffer.getvalue()
    with fsspec.open(warc_output_path, "wb") as fout:
        fout.write(warc_data)

    # Write domains_to_robots data
    with fsspec.open(robots_output_path, "w", compression="infer") as fout:
        json.dump(domains_to_robots, fout)

    # Write errors
    with fsspec.open(errors_output_path, "w", compression="infer") as fout:
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
    random.shuffle(shard_indices_to_process)

    # Set a limit on the number of concurrent tasks so we don't make too many network requests
    num_shards_submitted = 0
    unfinished = []

    def submit_shard_task(shard_index):
        nonlocal num_shards_submitted
        input_path = os.path.join(cfg.urls_input_directory, f"links.{shard_index}.parquet")
        warc_output_path = os.path.join(cfg.output_directory, f"links.{shard_index}.warc.gz")
        robots_output_path = os.path.join(cfg.output_directory, f"links.{shard_index}_robots.json.gz")
        errors_output_path = os.path.join(cfg.output_directory, f"links.{shard_index}_errors.json.gz")
        unfinished.append(
            fetch_links.remote(
                input_path, warc_output_path, robots_output_path, errors_output_path, cfg.threads_per_shard
            )
        )

        num_shards_submitted += 1
        if num_shards_submitted % 10 == 0:
            logger.info(
                f"Submitted {num_shards_submitted} / {num_shards_to_process} shards "
                f"({num_shards_submitted / num_shards_to_process})"
            )

    # Launch the initial cfg.max_concurrent_shards batch of tasks
    for _ in range(min(cfg.max_concurrent_shards, len(shard_indices_to_process))):
        submit_shard_task(shard_indices_to_process.pop())

    while unfinished:
        finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
        try:
            ray.get(finished)
        except Exception as e:
            logger.exception(f"Error processing shard: {e}")
            raise

        # If we have more shard paths left to process and we haven't hit the max
        # number of concurrent tasks, add tasks to the unfinished queue.
        while shard_indices_to_process and len(unfinished) < cfg.max_concurrent_shards:
            submit_shard_task(shard_indices_to_process.pop())


if __name__ == "__main__":
    main()
