# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter the focus-crawl fetch artifact down to a sampled key set, on-cluster.

Zephyr fan-out: the 1,773 fetch shards are split into chunks of 100, and each map task scans
its chunk next to the storage, keeping rows whose ``(warc_filename, warc_record_offset)``
appears in the uploaded key parquet. One compact output shard per input shard makes the scan
resumable. Ad-hoc companion to the 10k OCR-quality sample; the driver downloads the output
afterwards.

    uv run iris --cluster=marin job run --job-name pdf-sample-bytes-zephyr \\
        -- python -m experiments.build_pdf_source._pull_sample_pdf_bytes
"""

import concurrent.futures
import logging

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

logger = logging.getLogger(__name__)

KEYS_PATH = "marin-us-east-02a/marin/tmp/pdf_ocr_sample10k/keys.parquet"
OUT_PREFIX = "marin-us-east-02a/marin/tmp/pdf_ocr_sample10k/pdfs"
FETCH_GLOB = "marin-us-east-02a/marin/data/datakit/raw/common_crawl_focus_2026_22_pdf_e70aa547/" "outputs/main/*.parquet"
OUTPUT_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]
SHARDS_PER_CHUNK = 100
THREADS_PER_TASK = 16
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="32g", disk="8g")


def filter_shard(fs, shard: str, index: int, wanted: set[tuple[str, int]]) -> int:
    out_path = f"{OUT_PREFIX}/part-{index:05d}.parquet"
    if fs.exists(out_path):
        return 0
    with fs.open(shard, "rb") as f:
        table = pq.read_table(f, columns=OUTPUT_COLUMNS)
    mask = [
        (name, offset) in wanted
        for name, offset in zip(table["warc_filename"].to_pylist(), table["warc_record_offset"].to_pylist(), strict=True)
    ]
    matched = table.filter(mask) if any(mask) else table.slice(0, 0)
    with fs.open(out_path, "wb") as f:
        pq.write_table(matched, f)
    return matched.num_rows


def process_chunk(chunk: list[tuple[int, str]]) -> int:
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    with fs.open(KEYS_PATH, "rb") as f:
        keys = pq.read_table(f)
    wanted = set(zip(keys["warc_filename"].to_pylist(), keys["warc_record_offset"].to_pylist(), strict=True))

    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS_PER_TASK) as ex:
        futures = [ex.submit(filter_shard, fs, shard, index, wanted) for index, shard in chunk]
        for future in concurrent.futures.as_completed(futures):
            total += future.result()
    logger.info("chunk starting at shard %d: %d rows matched", chunk[0][0], total)
    return total


def main() -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    shards = sorted(fs.glob(FETCH_GLOB))
    indexed = list(enumerate(shards))
    chunks = [indexed[start : start + SHARDS_PER_CHUNK] for start in range(0, len(indexed), SHARDS_PER_CHUNK)]
    logger.info("%d chunks over %d shards", len(chunks), len(indexed))

    outcome = ZephyrContext(
        name="pdf-sample-bytes",
        resources=_WORKER_RESOURCES,
        max_workers=len(chunks),
        stage_runner_factory=SubprocessRunner,
        # One chunk per worker: costing a map task at the full worker keeps zephyr from
        # packing every chunk onto a single worker.
        map_task_resources=_WORKER_RESOURCES,
    ).execute(Dataset.from_list(chunks).map(process_chunk))
    logger.info("done: %s", dict(outcome.counters))


if __name__ == "__main__":
    main()
