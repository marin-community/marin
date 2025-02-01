#!/usr/bin/env python3
"""
Convert fineweb-edu to HTML. Given the parquet shards from a dump of the raw
fineweb-edu dataset, we take each example and fetch its corresponding HTML
source from common crawl. The output is written as sharded JSONL files, where
each record is a Dolma-format fineweb-edu example in HTML.

To run on a single dump:

```
python scripts/fineweb-edu/convert_fineweb_edu_to_html.py \
    --input_path gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2021-39/ \
    --html_output_path gs://marin-us-central2/documents/fineweb-edu/html/CC-MAIN-2021-39/
```

```
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python scripts/fineweb-edu/convert_fineweb_edu_to_html.py \
    --input_path gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2021-39/ \
    --html_output_path gs://marin-us-central2/documents/fineweb-edu/html/CC-MAIN-2021-39/
```

Launch jobs for all dumps:

```
for fineweb_edu_dump_path in $(gcloud storage ls gs://marin-us-central2/raw/fineweb-edu); do
    dump_name=$(basename -- ${fineweb_edu_dump_path})

    ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python scripts/fineweb-edu/convert_fineweb_edu_to_html.py \
    --input_path ${fineweb_edu_dump_path} \
    --html_output_path gs://marin-us-central2/documents/fineweb-edu/html/${dump_name}/
done
```

"""
import json
import logging
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import draccus
import fsspec
import pandas as pd
import ray
from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from warcio import ArchiveIterator

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def decode_html(html: bytes) -> str | None:
    """
    Given HTML (bytes), decode it into a string if possible. First try with
    utf-8. If that doesn't work, try to detect the encoding.
    """
    try:
        html = bytes_to_str(html, "utf-8")
    except Exception:
        encoding = detect_encoding(html)
        if encoding is None or encoding == "utf-8":
            return
        try:
            html = bytes_to_str(html, encoding)
        except Exception:
            return
    return html


@ray.remote(memory=4 * 1024 * 1024 * 1024)  # 4 GB
@cached_or_construct_output(success_suffix="SUCCESS", verbose=False)
def process_one_shard(
    input_path: str,
    output_path: str,
):
    """
    Takes an parquet with FineWeb records as input and processes it to get the html content.
    Download the WARC path in input_path and returns the content of the urls in the input_path.

    Args:
    input_path (str): The input parquet shard to process.
    output_path (str): Path to write gzipped JSONL with HTML for URLs in the input parquet shard.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df = pd.read_parquet(input_path, columns=["url", "file_path", "id"])

    urls = df["url"].tolist()
    # All frames will have same file_path, by design
    s3_url = df["file_path"].iloc[0]
    index = df.index.tolist()

    # url_dict is url to index in df so that we can update that record
    url_dict = {url: idx for idx, url in zip(index, urls)}
    num_urls_found = 0  # Used to early terminate
    num_urls_processed = 0
    num_urls_failed_decoding = 0
    num_urls_to_find = len(urls)

    length_warc = 0
    # NOTE: make sure s3 keys are set up by passing them as
    # environment variables when submitting the job
    s3_fs = fsspec.filesystem(
        "s3",
        anon=False,
        default_block_size=100 * 2**20,
    )
    s3_fs.retries = 100

    # HACK: some of the fineweb-edu filepaths are invalid (seem to point to files on
    # HF local disk), but they be easily re-written to the corresponding s3 WARCs.
    pattern = r"^/fsx/guilherme/cc2023-50/r\d+/input/"
    if re.match(pattern, s3_url):
        logger.info(f"Found invalid s3 URL {s3_url}")
        s3_url = re.sub(pattern, "s3://commoncrawl/crawl-data/CC-MAIN-2023-50/segments/", s3_url)
        logger.info(f"Re-wrote s3 URL as {s3_url}")

    with s3_fs.open(s3_url, mode="rb") as file_stream:
        for record in ArchiveIterator(file_stream):
            if num_urls_found == num_urls_to_find:
                break

            # Check if it's a response record
            if record.rec_type == "response":
                # Process the record
                url = record.rec_headers.get_header("WARC-Target-URI")
                length_warc += 1

                if url in url_dict:
                    num_urls_found += 1
                    url_idx_in_df = url_dict[url]

                    content = record.content_stream().read()
                    html_decoded: str | None = decode_html(content)

                    if html_decoded:
                        df.loc[url_idx_in_df, "html"] = html_decoded
                    else:
                        df.loc[url_idx_in_df, "html"] = ""
                        num_urls_failed_decoding += 1
                    num_urls_processed += 1

    with fsspec.open(output_path, "wt", compression="gzip") as f:  # html output
        for index, row in df.iterrows():
            out_fineweb_edu = row.to_dict()
            # If this example failed decoding, don't write it out
            if not out_fineweb_edu["html"]:
                continue
            out_dolma = DolmaFormattedFineWebEduRecord(
                id=out_fineweb_edu["id"],
                source="fineweb-edu",
                format="html",
                html=out_fineweb_edu["html"],
                metadata={
                    "url": out_fineweb_edu["url"],
                    "file_path": out_fineweb_edu["file_path"],
                },
            )

            print(json.dumps(asdict(out_dolma)), file=f)

    # num_urls_found should be equal to num_urls_to_find
    logger.info(
        f"Found: {num_urls_found}, Processed: {num_urls_processed}, out of {num_urls_to_find} urls, "
        f"in {input_path} . {num_urls_failed_decoding} failed HTML decoding "
        f"AWS URL: {s3_url}"
        f"Found {length_warc} records in the WARC file"
    )


@dataclass(frozen=True)
class DolmaFormattedFineWebEduRecord:
    id: str
    source: str
    format: str
    html: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ParquetFineWebEduConfig:
    input_path: str
    html_output_path: str


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def write_group_to_parquet(output_file: str, group_df: pd.DataFrame, group_success_file: str):
    # Check if the success file already exists
    if fsspec_exists(group_success_file):
        logger.info(f"Shard {output_file} already exists, skipping...")
        return output_file

    # Save the group to a parquet file
    group_df.to_parquet(output_file)
    # Create the group success file
    with fsspec.open(group_success_file, "w") as f:
        metadata = {
            "datetime": str(datetime.utcnow()),
        }
        print(json.dumps(metadata), file=f)

    return output_file


@ray.remote(memory=32 * 1024 * 1024 * 1024)
def group_fineweb_edu_by_warc(input_paths: list[str], output_path: str):
    """
    Given fineweb-edu files, group the examples by their source WARC.

    Parameters:
    input_paths (str): Path to the fineweb-edu parquet files.
    output_path (str): Path to the output folder where we will write the fineweb-edu examples,
                       grouped by their source WARC.
    """
    success_file_path = os.path.join(output_path, f"_examples_groupby_warc_success")
    if fsspec_exists(success_file_path):
        logger.info(f"Already grouped fineweb-edu input paths by WARC, skipping...")
        return

    logger.info(f"Grouping examples at {input_paths} by their source WARC")
    datetime_start = datetime.utcnow()

    df = pd.concat(
        [pd.read_parquet(file, columns=["url", "file_path", "id"]) for file in input_paths], ignore_index=True
    )

    grouped = df.groupby("file_path")

    remote_refs = []

    # Using Ray to parallelize the writing of groups
    for index, (_, group_df) in enumerate(grouped):
        output_file = os.path.join(output_path, f"{index}_warc_examples.parquet")
        group_success_file = os.path.join(output_path, f"{index}_warc_examples.success")

        # Queue up remote task for writing this group
        remote_refs.append(write_group_to_parquet.remote(output_file, group_df, group_success_file))

    # Wait for all groups to be written
    ray.get(remote_refs)

    datetime_end = datetime.utcnow()

    # Create the overall success file
    with fsspec.open(success_file_path, "w") as f:
        metadata = {
            "input_paths": input_paths,
            "output_path": output_path,
            "datetime_start": str(datetime_start),
            "datetime_end": str(datetime_end),
        }
        print(json.dumps(metadata), file=f)


@ray.remote(memory=32 * 1024 * 1024 * 1024)
def get_shards_to_process(shard_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    all_shard_paths = fsspec_glob(os.path.join(shard_path, "*_warc_examples.parquet"))
    num_total_shards = len(all_shard_paths)
    already_processed_shard_paths = set(fsspec_glob(os.path.join(shard_path, "*.jsonl.gz.SUCCESS")))
    num_already_processed_shards = len(already_processed_shard_paths)

    shard_indices_to_process = [
        int(os.path.basename(path).replace("_warc_examples.parquet", ""))
        for path in all_shard_paths
        if path.replace("_warc_examples.parquet", ".jsonl.gz.SUCCESS") not in already_processed_shard_paths
    ]
    logger.info(
        f"Found {len(shard_indices_to_process)} shards to fetch HTML for ("
        f"{num_total_shards} total shards, {num_already_processed_shards}) already finished."
    )
    return shard_indices_to_process


@draccus.wrap()
def process_fineweb_edu(cfg: ParquetFineWebEduConfig):
    files = fsspec_glob(os.path.join(cfg.input_path, "*.parquet"))
    # Group fine-web-edu examples by their source WARC
    groupby_ref = group_fineweb_edu_by_warc.remote(files, cfg.html_output_path)
    ray.get(groupby_ref)

    shard_indices_to_process = ray.get(get_shards_to_process.remote(cfg.html_output_path))
    num_shards_to_process = len(shard_indices_to_process)

    # Set a limit on the number of concurrent tasks so we don't overwhelm CC
    MAX_CONCURRENT_TASKS = 500

    # Shuffle to encourage different workers to hit different AWS prefixes,
    # so we don't run into per-prefix rate limits.
    random.shuffle(shard_indices_to_process)
    num_shards_submitted = 0
    unfinished = []

    def submit_shard_task(shard_index):
        """Submit a shard processing task and log progress every 1000 shards."""
        nonlocal num_shards_submitted
        # shard_path_to_process is of form gs://<html_output_path>/0_warc_examples.parquet
        shard_path = os.path.join(cfg.html_output_path, f"{shard_index}_warc_examples.parquet")
        # shard_output_path is of form gs://<html_output_path>/0.jsonl.gz
        shard_output_path = shard_path.replace("_warc_examples.parquet", ".jsonl.gz")
        unfinished.append(process_one_shard.remote(shard_path, shard_output_path))
        num_shards_submitted += 1
        if num_shards_submitted % 1000 == 0:
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
            ray.get(finished)
        except Exception as e:
            logger.exception(f"Error processing shard: {e}")

        # If we have more shard paths left to process and we haven't hit the max
        # number of concurrent tasks, add tasks to the unfinished queue.
        while shard_indices_to_process and len(unfinished) < MAX_CONCURRENT_TASKS:
            submit_shard_task(shard_indices_to_process.pop())


if __name__ == "__main__":
    process_fineweb_edu()
