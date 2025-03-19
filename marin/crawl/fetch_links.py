#!/usr/bin/env python3
"""
Given a parquet file with `Outlink`s, download the link targets and write the
scraped pages as a parquet file.

The high-level design of this fetcher is heavily inspired by the Apache Nutch crawler:

- Input: N shards of URLs (each shard has ~100K URLs)
- Each shard is processed by a different ray remote function. `--max_concurrent_shards` controls
  the number of shards that can be processed concurrently. The processes don't talk to each other,
  and each process writes out a parquet with the fetched contents of the URLs in the shard.
  - Writing is handled by a separate thread, and results are flushed out to the parquet every 5000
    successful responses. This enables us to restart from partial results if the shard is pre-empted.
- Each process runs multiple threads (`--threads_per_shard`) to fetch in parallel.
  The threads share a queue of URLs to fetch.
  - In addition, each host (netloc) we're fetching from has a separate lock to ensure that within a single process,
    we aren't making multiple simultaneous requests to a particular host.
  - When we see a 429 or 5xx response, the thread adaptively backs-off by requeuing the URL
    for retrying at a later point in time and goes to work on another URL.
  - When we see too many (10) consecutive 429s or 5xx from a particular host, we skip all
    further URLs from that host.
  - Finally, when we see too many (10) consecutive connection errors from a particular host,
    we skip all further URLs from that host.

After fetching to parquet, use `convert_responses_parquet_to_warc.py` to convert the parquet
responses to WARC.

Running on FineWeb-Edu-10M:

```
python marin/run/ray_run.py \
    --pip_deps 'fastparquet' \
    --no_wait -- \
    python marin/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/ \
    --threads_per_shard 160 \
    --max_concurrent_shards 40
```

Running on FineWeb-Edu-10M-cc-deduplicated:

```
python marin/run/ray_run.py \
    --pip_deps 'fastparquet' \
    --no_wait -- \
    python marin/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M-cc-deduplicated/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M-cc-deduplicated/ \
    --threads_per_shard 160 \
    --max_concurrent_shards 40
```


Running on OpenWebMath-10M:

```
python marin/run/ray_run.py \
    --pip_deps 'fastparquet' \
    --no_wait -- \
    python marin/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/ \
    --threads_per_shard 160 \
    --max_concurrent_shards 40
```

Running on OpenWebMath-10M-cc-deduplicated:

```
python marin/run/ray_run.py \
    --pip_deps 'fastparquet' \
    --no_wait -- \
    python marin/crawl/fetch_links.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --threads_per_shard 160 \
    --max_concurrent_shards 40
```
"""
import io
import json
import logging
import os
import pathlib
import queue
import random
import shutil
import tempfile
import threading
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from urllib.parse import urlparse

import draccus
import fastparquet
import fsspec
import pandas as pd
import ray
import requests
import urllib3
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_cp, fsspec_exists, fsspec_glob

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
    """Fetch the content of a URL, truncating the response at `max_size_bytes` if necessary.

    Args:
    session (requests.Session): Session object to use for fetching the URL
    url (str): URL to fetch
    timeout (float, default=30): timeout to use when issuing request
    max_size_bytes (int, default=10*1024*1024 (10MB)): truncate responses after this many bytes.
    chunk_size (int, default=1*1024*1024 (1MB)): chunk size to use when streaming responses.

    Returns a tuple with four items:
    - The response, if the request was successful, else None
    - The response status code, if we received a response, else None
    - The error string if the request was not successful, else None
    - True if the domain was unreachable, else False (e.g., successful request
      or we get a 4xx or 5xx response)
    """
    try:
        with session.get(url, timeout=timeout, stream=True, verify=False) as r:
            r.raise_for_status()

            # Create a mutable Response object that we'll patch with truncated content.
            truncated_response = requests.models.Response()
            truncated_response.status_code = r.status_code
            truncated_response.reason = r.reason
            truncated_response.headers = r.headers
            truncated_response.url = r.url
            truncated_response.request = r.request

            # Read response in chunks, truncate at `max_size_bytes`
            content_buffer = io.BytesIO()
            downloaded_bytes = 0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    new_size = downloaded_bytes + len(chunk)

                    # If adding this chunk would exceed the `max_size_bytes` cap,
                    # only write the part that fits
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
        # We got a response from the server, so the domain is reachable, but
        # the request was not successful (e.g., a 4xx or 5xx error).
        return None, e.response.status_code, str(e), False
    except Exception as e:
        # Anything else is a connection-level error (DNS, SSL, etc.),
        # where we didn't even get a response back.
        return None, None, str(e), True


def write_parquet_chunk(
    chunk: list[dict],
    local_parquet_path: str,
):
    """Write or append a list of dict records to a local Parquet file."""
    if not chunk:
        return
    df = pd.DataFrame.from_records(chunk)

    if not os.path.exists(local_parquet_path):
        fastparquet.write(local_parquet_path, df, compression="snappy")
    else:
        fastparquet.write(local_parquet_path, df, append=True, compression="snappy")


def fetch_to_parquet(
    urls: list[str],
    shard_id: str,
    parquet_output_path: str,
    robots_output_path: str,
    errors_output_path: str,
    threads_per_shard: int,
    parquet_chunk_size: int = 5_000,
):
    """
    Main pipeline that:
      - Deduplicates and shuffles URLs
      - Spawns multiple worker threads pulling from a shared queue
      - Uses per-netloc locks to ensure that only one thread concurrently
        making a request to each netloc.
      - Collects results in a parquet file
      - Writes out robots and error logs

    Args:
    urls (list[str]): List of URLs to fetch
    shard_id (str): ID of this shard, used for logging
    parquet_output_path (str): Path to write the output parquet file with the fetched responses
    robots_output_path (str): Path to write JSON object with mapping of netloc to robots.txt
    errors_output_path (str): Path to write JSON object with fetch errors for robots.txts and
      and URLs. The JSON object has two keys: (1) "url_to_fetch_errors" and
      (2) "netloc_to_robots_fetch_error". "url_to_fetch_errors" maps to a mapping of URLs to
      the error encountered when fetching (if we saw one), and "netloc_to_robots_fetch_error"
      maps to a mapping of robots.txt URLs (one per netloc) to the error encountered when
      fetching (if we saw one),
    threads_per_shard (int): Number of threads to use for concurrent fetching.
    parquet_chunk_size (int): The number of results to write out at once to the parquet file.
    """
    urls = list(set(urls))
    random.shuffle(urls)
    logger.info(f"Found {len(urls)} unique URLs to fetch.")

    already_fetched_urls = load_already_fetched_urls(parquet_output_path)

    # Store the number of records we've already written
    num_written_records_lock = threading.Lock()
    logger.info(f"Shard {shard_id}: Found {len(already_fetched_urls)} already fetched URLs in {parquet_output_path}")

    # Load or init the mapping from netloc to robots
    existing_robots = load_json_if_exists(robots_output_path)
    netloc_to_robots = deepcopy(existing_robots)

    # Load or init the mappings from:
    # 1. netloc to robots fetch errors
    # 2. URL to fetch errors
    existing_errors = load_json_if_exists(errors_output_path)
    netloc_to_robots_fetch_error = deepcopy(existing_errors.get("netloc_to_robots_fetch_error", {}))
    url_to_fetch_errors = deepcopy(existing_errors.get("url_to_fetch_errors", {}))

    # Locks to protect concurrent access to the mappings
    netloc_to_robots_lock = threading.Lock()
    netloc_to_robots_fetch_error_lock = threading.Lock()
    url_to_fetch_errors_lock = threading.Lock()

    # If a URL is already in the existing output or has a final error, skip it.
    urls_to_fetch = []
    for url in urls:
        if url in already_fetched_urls:
            # We've already fetched this URL, so skip it.
            continue
        if url in url_to_fetch_errors:
            # We've already recorded a terminal error for this URL, so skip it.
            continue
        urls_to_fetch.append(url)
    random.shuffle(urls_to_fetch)
    logger.info(f"Found {len(urls_to_fetch)} URLs to fetch after skipping already-fetched or errored URLs.")

    # Each netloc has a lock to ensure that we only have one concurrent
    # thread making a request to each netloc. For example, if a thread is making
    # a request to abc.wordpress.com/page1.html , another thread cannot work on
    # abc.wordpress.com/page2.html until it acquires the lock.
    netloc_locks = defaultdict(threading.Lock)

    # For each netloc, we keep track of how many consecutive connection errors
    # we've encountered (e.g., if the name cannot be resolved). The more consecutive connection
    # errors we've encountered, the shorter a timeout we use when making requests to URLs
    # from this netloc (since it's pretty likely that the site is just dead).
    netloc_connection_error_counts = Counter()
    # If we've hit 5 or more consecutive connection errors for a netloc, skip all
    # further URLs from this netloc.
    MAX_NUM_CONNECTION_ERROR = 10

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
    # If we encounter `MAX_NUM_429_5xx` consecutive 429 or 5xx responses for a
    # netloc (despite the backoff between requests), skip all further URLs from this netloc.
    MAX_NUM_429_5xx = 10

    # Queue for the writer thread
    results_queue = queue.Queue()
    # We use a sentinel (None) to signal the writer thread when we're done
    SENTINEL = None

    def writer_thread():
        """
        Continuously dequeues results and writes them to Parquet in chunks locally,
        then uploads the Parquet file to a remote gs:// path. This prevents
        accumulating too many results in memory and allows recovery from pre-emption.
        """
        buffer = []
        num_written_records = len(already_fetched_urls)

        # Create a temporary directory for Parquet files
        with tempfile.TemporaryDirectory() as temp_dir:
            local_parquet_path = os.path.join(temp_dir, f"{shard_id}_temp_output.parquet")
            # Check if a partially-written parquet already exists on gcloud. If so, download it.
            if fsspec_exists(parquet_output_path):
                logger.info(f"Parquet output path {parquet_output_path} already exists, downloading it locally")
                # Open parquet_output_path with fsspec and write its contents to local_parquet_path
                with (
                    fsspec.open(parquet_output_path, "rb", block_size=1 * 1024 * 1024 * 1024) as remote_file,
                    open(local_parquet_path, "wb") as local_file,
                ):
                    shutil.copyfileobj(remote_file, local_file)
                logger.info(f"Downloaded file at parquet output path {parquet_output_path} to {local_parquet_path}")
            while True:
                item = results_queue.get()
                if item is SENTINEL:
                    # Flush any leftover items
                    if buffer:
                        # Write robots data to output path
                        with netloc_to_robots_lock:
                            netloc_to_robots_output = deepcopy(netloc_to_robots)
                        with fsspec.open(
                            robots_output_path, "w", compression="infer", block_size=1 * 1024 * 1024 * 1024
                        ) as fout:
                            json.dump(netloc_to_robots_output, fout)

                        # Write errors to output path
                        with url_to_fetch_errors_lock:
                            url_to_fetch_errors_output = deepcopy(url_to_fetch_errors)
                        with netloc_to_robots_fetch_error_lock:
                            netloc_to_robots_fetch_error_output = deepcopy(netloc_to_robots_fetch_error)
                        with fsspec.open(
                            errors_output_path, "w", compression="infer", block_size=1 * 1024 * 1024 * 1024
                        ) as fout:
                            json.dump(
                                {
                                    "url_to_fetch_errors": url_to_fetch_errors_output,
                                    "netloc_to_robots_fetch_error": netloc_to_robots_fetch_error_output,
                                },
                                fout,
                            )

                        # Write leftover items to local parquet
                        logger.info(f"[shard {shard_id}] Writing {len(buffer)} examples to {local_parquet_path}")
                        write_parquet_chunk(buffer, local_parquet_path)
                        logger.info(f"[shard {shard_id}] Wrote {len(buffer)} examples to {local_parquet_path}")

                        # Upload the local parquet to remote gs:// path
                        try:
                            logger.info(f"[shard {shard_id}] uploading final parquet chunk to {parquet_output_path}")
                            fsspec_cp(local_parquet_path, parquet_output_path)
                            logger.info(f"[shard {shard_id}] uploaded final parquet chunk to {parquet_output_path}")
                        except Exception as e:
                            logger.error(f"Failed to upload parquet file to {parquet_output_path}: {e}")

                        with num_written_records_lock:
                            num_written_records += len(buffer)
                            logger.info(f"[shard {shard_id}] wrote {num_written_records} records so far")

                    results_queue.task_done()
                    break

                buffer.append(item)
                results_queue.task_done()

                # If buffer is large enough, flush to local Parquet and upload
                if len(buffer) >= parquet_chunk_size:
                    # Write robots data to output path
                    with netloc_to_robots_lock:
                        netloc_to_robots_output = deepcopy(netloc_to_robots)
                    with fsspec.open(
                        robots_output_path, "w", compression="infer", block_size=1 * 1024 * 1024 * 1024
                    ) as fout:
                        json.dump(netloc_to_robots_output, fout)

                    # Write errors to output path
                    with url_to_fetch_errors_lock:
                        url_to_fetch_errors_output = deepcopy(url_to_fetch_errors)
                    with netloc_to_robots_fetch_error_lock:
                        netloc_to_robots_fetch_error_output = deepcopy(netloc_to_robots_fetch_error)
                    with fsspec.open(
                        errors_output_path, "w", compression="infer", block_size=1 * 1024 * 1024 * 1024
                    ) as fout:
                        json.dump(
                            {
                                "url_to_fetch_errors": url_to_fetch_errors_output,
                                "netloc_to_robots_fetch_error": netloc_to_robots_fetch_error_output,
                            },
                            fout,
                        )

                    # Write buffer to local parquet
                    logger.info(f"[shard {shard_id}] Writing {len(buffer)} examples to {local_parquet_path}")
                    write_parquet_chunk(buffer, local_parquet_path)
                    logger.info(f"[shard {shard_id}] Wrote {len(buffer)} examples to {local_parquet_path}")

                    # Upload the local parquet to remote gs:// path
                    try:
                        logger.info(f"[shard {shard_id}] Uploading parquet chunk to {parquet_output_path}")
                        fsspec_cp(local_parquet_path, parquet_output_path)
                        logger.info(f"[shard {shard_id}] Uploaded parquet chunk to {parquet_output_path}")
                    except Exception as e:
                        logger.error(f"Failed to upload parquet file to {parquet_output_path}: {e}")

                    with num_written_records_lock:
                        num_written_records += len(buffer)
                        logger.info(f"[shard {shard_id}] wrote {num_written_records} records so far")

                    buffer.clear()

            logger.info(f"Writer thread for shard {shard_id} finished.")

    writer = threading.Thread(target=writer_thread, daemon=True)
    writer.start()

    # A queue of tasks. Each is (url, retries_left, is_robots)
    work_queue = queue.Queue()
    retries_per_url = 3
    # Enqueue the URL fetching tasks
    for url in urls_to_fetch:
        work_queue.put((url, retries_per_url, False))

    pbar = tqdm(total=len(urls_to_fetch), desc=f"Fetching shard {shard_id}")
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
                        # Don't overwrite robots.txt if we already successfully fetched it, or
                        # if there is an existing error
                        with netloc_to_robots_fetch_error_lock:
                            netloc_to_robots_fetch_error[netloc] = "netloc passed max number of 429/5xx"
                    elif not is_robots:
                        with url_to_fetch_errors_lock:
                            url_to_fetch_errors[url] = "netloc passed max number of 429/5xx"
                        # Final outcome => increment progress
                        with pbar_lock:
                            pbar.update(1)
                    work_queue.task_done()
                    continue
                elif netloc_connection_error_counts[netloc] >= MAX_NUM_CONNECTION_ERROR:
                    if is_robots and (netloc not in netloc_to_robots and netloc not in netloc_to_robots_fetch_error):
                        # Don't overwrite robots.txt if we already successfully fetched it, or
                        # if there is an existing error
                        with netloc_to_robots_fetch_error_lock:
                            netloc_to_robots_fetch_error[netloc] = "netloc passed max number of connection errors"
                    elif not is_robots:
                        with url_to_fetch_errors_lock:
                            url_to_fetch_errors[url] = "netloc passed max number of connection errors"
                        # Final outcome => increment progress
                        with pbar_lock:
                            pbar.update(1)
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
                        with netloc_to_robots_lock:
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
                                with netloc_to_robots_fetch_error_lock:
                                    netloc_to_robots_fetch_error[netloc] = f"{code}, no more retries"
                        else:
                            # e.g., 4xx or permanent robots error
                            with netloc_to_robots_fetch_error_lock:
                                netloc_to_robots_fetch_error[netloc] = err or "robots fetch error"

                    work_queue.task_done()
                    continue  # end is_robots block

                # 4) If we need robots for this domain but haven't fetched or errored
                if netloc not in netloc_to_robots and netloc not in netloc_to_robots_fetch_error:
                    # Insert a robots task for it
                    robots_url = f"{urlparse(url).scheme}://{netloc}/robots.txt"
                    work_queue.put((robots_url, retries_per_url, True))
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

                    # Enqueue a dictionary that we will write to Parquet
                    result_record = {
                        "url": url,
                        "status_code": resp.status_code,
                        "reason": resp.reason,
                        "headers": dict(resp.headers),
                        "content": resp.content,  # could store as bytes or text
                        "fetched_at": time.time(),
                    }
                    results_queue.put(result_record)

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
                        with url_to_fetch_errors_lock:
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
                            with url_to_fetch_errors_lock:
                                url_to_fetch_errors[url] = f"{code}, no more retries"
                            with pbar_lock:
                                pbar.update(1)
                    else:
                        # any other HTTP error
                        with url_to_fetch_errors_lock:
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
    # All fetching is done. Tell the writer thread to exit.
    results_queue.put(SENTINEL)

    for t in threads:
        t.join()

    # Wait until the writer has finished flushing everything
    results_queue.join()
    writer.join()

    logger.info(
        f"Fetched robots for {len(netloc_to_robots)} domains, encountered {len(netloc_to_robots_fetch_error)} "
        f"robots errors.\nRecorded {len(url_to_fetch_errors)} fetch errors."
    )


def load_already_fetched_urls(parquet_path: str):
    """
    If a parquet already exists, read in the URLs that we've successfully fetched.
    Return a set() of URLs so we can skip re-fetching them.
    """
    if fsspec_exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if "url" in df.columns:
            return set(df["url"].dropna().unique().tolist())
    else:
        logger.info(f"Fetched URLs parquet path {parquet_path} does not already exist")
    return set()


def load_json_if_exists(json_path: str):
    """Load a JSON file if it exists, else return an empty dict."""
    if fsspec_exists(json_path):
        with fsspec.open(json_path, compression="infer", block_size=1 * 1024 * 1024 * 1024) as fin:
            return json.load(fin)
    return {}


@ray.remote(memory=128 * 1024 * 1024 * 1024, num_cpus=16)
def fetch_links(
    urls_path: str, parquet_output_path: str, robots_output_path: str, errors_output_path: str, threads_per_shard: int
):
    """
    Given a parquet with links to fetch, fetch the responses and write the output as parquet
    to `parquet_output_path`. In addition, the robots.txt for each fetched netloc and any errors
    encountered while fetching are written to `robots_output_path` and `errors_output_path`,
    respectively.

    Args:
    urls_path (str): Path to parquet with links to fetch.
    parquet_output_path (str): Path to write the output parquet file with the fetched responses
    robots_output_path (str): Path to write JSON object with mapping of netloc to robots.txt
    errors_output_path (str): Path to write JSON object with fetch errors for robots.txts and
      and URLs. The JSON object has two keys: (1) "url_to_fetch_errors" and
      (2) "netloc_to_robots_fetch_error". "url_to_fetch_errors" maps to a mapping of URLs to
      the error encountered when fetching (if we saw one), and "netloc_to_robots_fetch_error"
      maps to a mapping of robots.txt URLs (one per netloc) to the error encountered when
      fetching (if we saw one),
    threads_per_shard (int): Number of threads to use for concurrent fetching.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Disable SSL verification warnings.
    urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

    success_path = parquet_output_path + ".SUCCESS"
    if fsspec_exists(success_path):
        logger.info(f"Already processed and wrote parquet to {parquet_output_path}, skipping...")
        return

    with fsspec.open(urls_path, block_size=1 * 1024 * 1024 * 1024) as f:
        df = pd.read_parquet(f)
    logger.info(f"Found {len(df)} examples in input file {urls_path}")

    # Extract the URLs from the "link_target" column
    urls = df["link_target"].tolist()
    # Deduplicate the URLs
    urls = list(set(urls))

    # Randomly shuffle the URLs to load balance so we aren't repeatedly hitting a particular host
    random.shuffle(urls)
    shard_id = os.path.basename(urls_path)
    logger.info(f"Fetching {len(urls)} deduplicated URLs from input file {urls_path}")
    fetch_to_parquet(urls, shard_id, parquet_output_path, robots_output_path, errors_output_path, threads_per_shard)

    # Create success file
    with fsspec.open(success_path, "w", block_size=1 * 1024 * 1024 * 1024) as fout:
        json.dump(
            {
                "urls_path": urls_path,
                "parquet_output_path": parquet_output_path,
                "robots_output_path": robots_output_path,
                "errors_output_path": errors_output_path,
            },
            fout,
        )

    logger.info(
        f"Parquet file created at: {parquet_output_path}\n"
        f"robots.txt data written to {robots_output_path}\n"
        f"errors written to {errors_output_path}"
    )


def decreasing_timeout(
    base_timeout: float = 30.0, failures: int = 0, factor: float = 2.0, min_timeout: float = 1.0
) -> float:
    """
    The more failures a domain has, the (exponentially) smaller the timeout we allow.
    We clamp the final value so it doesn't go below min_timeout.
    """
    timeout = base_timeout / (factor ** min(failures / 5, 5))
    return max(timeout, min_timeout)


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def get_shard_indices_to_process(urls_input_directory: str) -> list[int]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".parquet").removeprefix("links."))
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
        parquet_output_path = os.path.join(cfg.output_path, f"fetched_links.{shard_index}.parquet")
        robots_output_path = os.path.join(cfg.output_path, f"links.{shard_index}_robots.json.gz")
        errors_output_path = os.path.join(cfg.output_path, f"links.{shard_index}_errors.json.gz")
        unfinished.append(
            fetch_links.remote(
                input_path, parquet_output_path, robots_output_path, errors_output_path, cfg.threads_per_shard
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
