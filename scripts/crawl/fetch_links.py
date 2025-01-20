#!/usr/bin/env python3
"""
Given a parquet file with outlinks, fetch the link targets and write the
scraped pages as a WARC file.

Running on FineWeb-Edu:

```
python marin/run/ray_run.py \
    --pip_deps 'warcio,tldextract' \
    --no_wait -- \
    python scripts/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/ \
    --threads_per_shard 80 \
    --max_concurrent_shards 1
```

Running on OpenWebMath:

```
python marin/run/ray_run.py \
    --pip_deps 'warcio,tldextract' \
    --no_wait -- \
    python scripts/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/ \
    --threads_per_shard 80 \
    --max_concurrent_shards 1
```
"""
import io
import json
import logging
import os
import pathlib
import random
import threading
import queue
import time
from collections import Counter, defaultdict
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
    output_path: str
    threads_per_shard: int = 80
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
        return None, e.response.status_code, str(e), False
    except Exception as e:
        # Anything else is a connection-level error (DNS, SSL, etc.)
        return None, None, str(e), True


def fetch_to_warc(
    urls: list[str],
    warc_output_path: str,
    robots_output_path: str,
    errors_output_path: str,
    threads_per_shard: int,
):
    """
    Main pipeline that:
      - Deduplicates and shuffles URLs
      - Spawns multiple worker threads pulling from a shared queue
      - For each domain, ensures concurrency=1 using domain locks
      - Collects results in a WARC file
      - Writes out robots and error logs
    """
    urls = list(set(urls))
    random.shuffle(urls)
    logger.info(f"fetch_to_warc: {len(urls)} unique URLs to fetch.")

    # Each netloc has a lock to ensure that we only have one concurrent
    # thread making a request to each netloc. For example, if a thread is making
    # a request to abc.wordpress.com/page1.html , another thread cannot touch
    # abc.wordpress.com/page2.html until it acquires the lock.
    netloc_locks = defaultdict(threading.Lock)

    # For each netloc , we keep track of how many consecutive connection errors
    # we've encountered (e.g., if the name cannot be resolved). The more consecutive connection
    # errors we've encountered, the shorter a timeout we use when making requests to URLs
    # from this netloc (since it's pretty likely that the site is just dead).
    netloc_connection_error_counts = Counter()
    # If we've hit 5 or more consecutive connection errors for a netloc, skip all
    # further URLs from this netloc.
    MAX_NUM_CONNECTION_ERROR = 5

    # For each netloc, keep track of the next time (Unix epoch time)
    # that we are allowed to make a request to a URL from this netloc.
    # This is important for handling 429s, since we want to wait a bit until the next time
    # we make a request to this netloc.
    netloc_next_allowed = defaultdict(float)

    # When we hit a 429 error for a URL, we back off (up to 30 seconds)
    # until the next fetch to a URL from that netloc
    netloc_backoff_seconds = defaultdict(lambda: 5.0)
    max_delay_429_5xx = 30
    # For each netloc, count the number of consecutive 429s that we encounter.
    netloc_429_5xx_counts = Counter()
    # If we encounter MAX_NUM_429 429 responses for a netloc (despite the backoff
    # between requests), skip all further URLs from this netloc.
    MAX_NUM_429_5xx = 3

    # Robot storage
    netloc_to_robots = {}
    netloc_to_robots_fetch_error = {}

    # Fetch errors
    url_to_fetch_errors = {}
    # Store successful responses in memory, then write them at the end.
    # TODO(nfliu): consider streaming directly to the WARC
    # list of (url, response)
    successful_responses = []

    # A queue of tasks. Each is (url, retries_left, is_robots)
    work_queue = queue.Queue()

    retries_per_url = 3
    # Enqueue the main tasks
    for u in urls:
        work_queue.put((u, retries_per_url, False))

    pbar = tqdm(total=len(urls), desc="Fetching")
    pbar_lock = threading.Lock()

    # Worker function
    def worker():
        session = requests.Session()
        session.headers.update({"User-Agent": "CCBot"})

        while True:
            try:
                url, retries_left, is_robots = work_queue.get_nowait()
            except queue.Empty:
                return  # no more tasks

            netloc = urlparse(url).netloc
            now = time.time()

            # 1) Acquire lock for this netloc
            lock_acquired = netloc_locks[netloc].acquire(blocking=False)
            if not lock_acquired:
                # Another thread is fetching this domain. Re-queue and move on.
                work_queue.put((url, retries_left, is_robots))
                work_queue.task_done()
                continue

            try:
                if netloc_429_5xx_counts[netloc] >= MAX_NUM_429_5xx:
                    # This netloc has seen more than `MAX_NUM_429` consecutive
                    # 429 responses (despite backing off between requests), so
                    # we give up on all URLs from this netloc.
                    if is_robots and (netloc not in netloc_to_robots and netloc not in netloc_to_robots_fetch_error):
                        # Don't overwrite robots.txt if we already successfully fetched it, or an existing error
                        # if there already is one.
                        netloc_to_robots_fetch_error[netloc] = "netloc passed max number of 429/5xx"
                    else:
                        url_to_fetch_errors[url] = "netloc passed max number of 429/5xx"
                    work_queue.task_done()
                    continue
                if netloc_connection_error_counts[netloc] >= MAX_NUM_CONNECTION_ERROR:
                    if is_robots and (netloc not in netloc_to_robots and netloc not in netloc_to_robots_fetch_error):
                        # Don't overwrite robots.txt if we already successfully fetched it, or an existing error
                        # if there already is one.
                        netloc_to_robots_fetch_error[netloc] = "netloc passed max number of connection errors"
                    else:
                        url_to_fetch_errors[url] = f"netloc passed max number of connection errors"
                    work_queue.task_done()
                    continue

                # 2) Check if netloc is currently rate-limited
                if now < netloc_next_allowed[netloc]:
                    # Not ready yet, re-queue and work on another URL
                    work_queue.put((url, retries_left, is_robots))
                    work_queue.task_done()
                    continue

                # 3) If this is a robots fetch
                if is_robots:
                    # If already fetched or permanently errored, skip
                    # We have to check netloc here, since different subdomains might
                    # have different robots.txt
                    if netloc in netloc_to_robots or netloc in netloc_to_robots_fetch_error:
                        work_queue.task_done()
                        continue

                    current_timeout = decreasing_timeout(
                        base_timeout=30.0,
                        failures=netloc_connection_error_counts[netloc],
                        factor=2.0,
                        min_timeout=1.0,
                    )
                    resp, code, err, conn_err = fetch_url(session, url, timeout=current_timeout)

                    if resp is not None:
                        # Succeeded
                        netloc_to_robots[netloc] = resp.text
                        netloc_connection_error_counts[netloc] = 0
                        netloc_429_5xx_counts[netloc] = 0
                        # reset the backoff value
                        netloc_backoff_seconds[netloc] = 5.0
                        netloc_next_allowed[netloc] = time.time()

                    else:
                        # Some error
                        if conn_err:
                            netloc_connection_error_counts[netloc] += 1
                        if code and (code == 429 or code > 499):
                            netloc_backoff_seconds[netloc] = min(netloc_backoff_seconds[netloc] * 2, max_delay_429_5xx)

                            netloc_429_5xx_counts[netloc] += 1
                            # Back off to delay the the next time we hit this netloc
                            netloc_next_allowed[netloc] = time.time() + netloc_backoff_seconds[netloc]

                            # Re-queue if we have retries left and this netloc hasn't reached the max
                            # number of 429s or 5xxs
                            if retries_left > 0 and netloc_429_5xx_counts[netloc] < MAX_NUM_429_5xx:
                                work_queue.put((url, retries_left - 1, True))
                            else:
                                # out of retries => final error
                                netloc_to_robots_fetch_error[netloc] = f"{code}, no more retries"
                        else:
                            # e.g., 4xx or permanent robots error
                            netloc_to_robots_fetch_error[netloc] = err or "robots fetch error"

                    work_queue.task_done()
                    continue  # end is_robots block

                # 4) If we need robots for this domain but haven't fetched or errored
                if netloc not in netloc_to_robots and netloc not in netloc_to_robots_fetch_error:
                    # Insert a robots task for it
                    robots_url = f"{urlparse(url).scheme}://{netloc}/robots.txt"
                    work_queue.put((robots_url, 2, True))
                    # Re-queue the original request
                    work_queue.put((url, retries_left, False))
                    work_queue.task_done()
                    continue

                # 5) Actually fetch the main URL
                current_timeout = decreasing_timeout(
                    base_timeout=30.0,
                    failures=netloc_connection_error_counts[netloc],
                    factor=2.0,
                    min_timeout=1.0,
                )
                resp, code, err, conn_err = fetch_url(session, url, timeout=current_timeout)

                if resp is not None:
                    # Successfully fetched the URL
                    successful_responses.append((url, resp))
                    if len(successful_responses) % 1000 == 0:
                        logger.info(f"Fetched {len(successful_responses)} successful responses")
                    netloc_connection_error_counts[netloc] = 0
                    netloc_429_5xx_counts[netloc] = 0

                    # reset the backoff value
                    netloc_backoff_seconds[netloc] = 5.0
                    netloc_next_allowed[netloc] = time.time()

                    # Final outcome => increment progress
                    with pbar_lock:
                        pbar.update(1)

                else:
                    # error
                    if conn_err:
                        # e.g. DNS error
                        netloc_connection_error_counts[netloc] += 1
                        url_to_fetch_errors[url] = err
                        # Final outcome => increment progress
                        with pbar_lock:
                            pbar.update(1)
                    elif code and (code == 429 or code > 499):
                        netloc_429_5xx_counts[netloc] += 1
                        # back off until the next time we can fetch a URL from
                        # this netloc
                        netloc_backoff_seconds[netloc] = min(netloc_backoff_seconds[netloc] * 2, max_delay_429_5xx)
                        netloc_next_allowed[netloc] = time.time() + netloc_backoff_seconds[netloc]
                        # Re-queue if we have retries left and this netloc hasn't reached the max
                        # number of 429s or 5xxs
                        if retries_left > 0 and netloc_429_5xx_counts[netloc] < MAX_NUM_429_5xx:
                            work_queue.put((url, retries_left - 1, False))
                        else:
                            # out of retries => final error
                            url_to_fetch_errors[url] = f"{code}, no more retries"
                            with pbar_lock:
                                pbar.update(1)
                    else:
                        # any other HTTP error
                        url_to_fetch_errors[url] = err
                        with pbar_lock:
                            pbar.update(1)
                work_queue.task_done()

            finally:
                # Always release the domain lock if we acquired it
                netloc_locks[netloc].release()

    # 6) Launch worker threads
    threads = []
    for _ in range(threads_per_shard):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    # 7) Wait for queue to be empty
    work_queue.join()
    for t in threads:
        t.join()

    # 8) Now we have successful responses in memory; write them to WARC
    warc_buffer = io.BytesIO()
    writer = WARCWriter(warc_buffer, gzip=True)
    for url, response in successful_responses:
        status_line = f"{response.status_code} {response.reason}"
        http_headers = [("Status", status_line)]
        for h, v in response.headers.items():
            http_headers.append((h, v))
        status_headers = StatusAndHeaders(status_line, http_headers, protocol="HTTP/1.1")
        payload_io = io.BytesIO(response.content)
        record = writer.create_warc_record(url, "response", payload=payload_io, http_headers=status_headers)
        writer.write_record(record)

    # Write the WARC file
    with fsspec.open(warc_output_path, "wb") as fout:
        fout.write(warc_buffer.getvalue())

    # Write robots data
    with fsspec.open(robots_output_path, "w", compression="infer") as fout:
        json.dump(netloc_to_robots, fout)

    # Write errors
    all_errors = {
        "url_to_fetch_errors": url_to_fetch_errors,
        "netloc_to_robots_fetch_error": netloc_to_robots_fetch_error,
    }
    with fsspec.open(errors_output_path, "w", compression="infer") as fout:
        json.dump(all_errors, fout)

    logger.info(
        f"Wrote {len(successful_responses)} successful WARC responses to {warc_output_path}.\n"
        f"Fetched robots for {len(netloc_to_robots)} domains, encountered {len(netloc_to_robots_fetch_error)} robots errors.\n"
        f"Recorded {len(url_to_fetch_errors)} fetch errors."
    )


@ray.remote(memory=128 * 1024 * 1024 * 1024, num_cpus=16)
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
        warc_output_path = os.path.join(cfg.output_path, f"links.{shard_index}.warc.gz")
        robots_output_path = os.path.join(cfg.output_path, f"links.{shard_index}_robots.json.gz")
        errors_output_path = os.path.join(cfg.output_path, f"links.{shard_index}_errors.json.gz")
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
