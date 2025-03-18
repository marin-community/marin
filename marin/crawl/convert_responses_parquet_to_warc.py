#!/usr/bin/env python3
"""
Given a parquet file with fetched responses, convert to WARC.

Running on FineWeb-Edu-10M:

```
python marin/run/ray_run.py \
    --pip_deps 'warcio[all]' \
    --no_wait -- \
    python marin/crawl/convert_responses_parquet_to_warc.py \
    --input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M/
```

Running on open-web-math-10M (cc deduplicated):

```
python marin/run/ray_run.py \
    --pip_deps 'warcio[all]' \
    --no_wait -- \
    python marin/crawl/convert_responses_parquet_to_warc.py \
    --input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/
```

Running on fineweb-edu-10M (cc deduplicated):

```
python marin/run/ray_run.py \
    --pip_deps 'warcio[all]' \
    --no_wait -- \
    python marin/crawl/convert_responses_parquet_to_warc.py \
    --input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M-cc-deduplicated/ \
    --output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-10M-cc-deduplicated/
```

After converting to WARC, you may want to delete the parquets with fetched responses
to conserve GCS space.
"""
import io
import json
import logging
import os
import pathlib
import random
from dataclasses import dataclass
from http import HTTPStatus

import draccus
import fsspec
import pandas as pd
import ray
from tqdm_loggable.auto import tqdm
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConvertResponsesToWARCConfig:
    input_directory: str
    output_path: str


def get_reason_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        logger.info(f"Found unknown status code {status_code}")
        return "Unknown Status Code"


@ray.remote(memory=128 * 1024 * 1024 * 1024, num_cpus=16)
@cached_or_construct_output(success_suffix="SUCCESS")
def convert_parquet_to_warc(input_path: str, output_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    with fsspec.open(input_path) as f:
        records = pd.read_parquet(f).to_dict("records")
    logger.info(f"Found {len(records)} responses in input file {input_path}")

    warc_buffer = io.BytesIO()
    writer = WARCWriter(warc_buffer, gzip=True)

    for record in tqdm(records, desc="Converting responses to WARC"):
        status_reason = record.get("reason", get_reason_phrase(record["status_code"]))
        status_line = f"{record['status_code']} {status_reason}"
        http_headers = [("Status", status_line)]
        for h, v in json.loads(record["headers"]).items():
            # Only keep headers with ascii names and values, since warcio errors out
            # headers with values including Unicode characters. For example, the URL
            # https://lib.utsa.edu/research/ causes issues because one of the returned headers is:
            # ('Strict-Transport-Security', 'max-age=31536000;Â\xa0includeSubDomains; preload')
            if h and not h.isascii():
                logger.info(f"Skipping non-ASCII header in {h}: {v}")
                continue
            if v and not v.isascii():
                logger.info(f"Skipping non-ASCII value in {h}: {v}")
                continue
            http_headers.append((h, v))
        status_headers = StatusAndHeaders(status_line, http_headers, protocol="HTTP/1.1")
        payload_io = io.BytesIO(record["content"])
        record = writer.create_warc_record(record["url"], "response", payload=payload_io, http_headers=status_headers)
        try:
            writer.write_record(record)
        except Exception:
            logger.exception(f"Got exception when writing WARC record.\nurl: {record['url']}\nheaders: {status_headers}")
            raise

    # Write the WARC file to output path
    logger.info(f"Writing WARC to {output_path}")
    with fsspec.open(output_path, "wb") as fout:
        fout.write(warc_buffer.getvalue())
    logger.info(f"Wrote WARC to {output_path}")


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def get_shard_indices_to_process(urls_input_directory: str) -> list[int]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".parquet").removeprefix("fetched_links."))
        for path in fsspec_glob(os.path.join(urls_input_directory, "fetched_links.*.parquet"))
    ]
    shard_indices = sorted(shard_indices)
    logger.info(f"Found {len(shard_indices)} shards to process")
    return shard_indices


@draccus.wrap()
def main(cfg: ConvertResponsesToWARCConfig):
    shard_indices_to_process = ray.get(get_shard_indices_to_process.remote(cfg.input_directory))
    num_shards_to_process = len(shard_indices_to_process)
    logger.info(f"Found {num_shards_to_process} shards to process")
    random.shuffle(shard_indices_to_process)

    refs = []
    for shard_index in shard_indices_to_process:
        input_path = os.path.join(cfg.input_directory, f"fetched_links.{shard_index}.parquet")
        output_path = os.path.join(cfg.input_directory, f"links.{shard_index}.warc.gz")
        refs.append(convert_parquet_to_warc.remote(input_path, output_path))
    ray.get(refs)


if __name__ == "__main__":
    main()
