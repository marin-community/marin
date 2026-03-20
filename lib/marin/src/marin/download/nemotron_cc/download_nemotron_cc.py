# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Download and process Nemotron-CC dataset from Common Crawl.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=100 --memory=4GB \
    lib/marin/src/marin/download/nemotron_cc/download_nemotron_cc.py \
    --output_path gs://bucket/nemotron-output
"""

import json
import logging
import os
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass

import draccus
import requests
import zstandard
from fray.cluster import ResourceConfig
from iris.marin_fs import open_url, url_to_fs
from marin.execution import THIS_OUTPUT_PATH
from marin.utils import fsspec_exists
from requests.adapters import HTTPAdapter
from urllib3.exceptions import ProtocolError
from urllib3.util import Retry
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)

myagent = "marin-nemotron-ingress/1.0"
NCC_PATH_FILE_URL = "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"


def _iter_jsonl_from_zstd_stream(raw_stream) -> Iterator[dict]:
    """Yield parsed JSON objects from a zstd-compressed JSONL stream."""
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(raw_stream) as reader:
        buf = bytearray()
        while True:
            chunk = reader.read(1048576)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                newline_pos = buf.find(b"\n")
                if newline_pos < 0:
                    break
                line_bytes = bytes(buf[:newline_pos])
                del buf[: newline_pos + 1]
                if not line_bytes.strip():
                    continue
                yield json.loads(line_bytes)


_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_BACKOFF = 30.0


def download_single_nemotron_path(input_file_path: str, output_file_path: str) -> dict:
    """Fetches content from a Common Crawl path, streaming records to zstd output."""
    cc_url = f"https://data.commoncrawl.org/{input_file_path}"

    for attempt in range(_DOWNLOAD_MAX_RETRIES):
        try:
            return _download_and_upload(input_file_path, output_file_path, cc_url)
        except (requests.exceptions.ConnectionError, ProtocolError) as e:
            if attempt == _DOWNLOAD_MAX_RETRIES - 1:
                raise
            wait = _DOWNLOAD_BACKOFF * (attempt + 1)
            logger.warning(f"Download failed (attempt {attempt + 1}): {e}, retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _download_and_upload(input_file_path: str, output_file_path: str, cc_url: str) -> dict:
    logger.info(f"Downloading Nemotron CC file {cc_url} to {output_file_path}")

    session = requests.Session()
    retries = Retry(total=10, backoff_factor=2.0, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    response = session.get(cc_url, headers={"user-agent": myagent}, stream=True)
    response.raise_for_status()

    num_records = 0
    # Write locally then fs.put() — R2 rejects streaming multipart with unequal part sizes.
    with tempfile.NamedTemporaryFile(suffix=".jsonl.zst", delete=True) as local_tmp:
        cctx = zstandard.ZstdCompressor(level=2, threads=1)
        with cctx.stream_writer(local_tmp, closefd=False) as zst:
            for record in _iter_jsonl_from_zstd_stream(response.raw):
                dolma_record = {
                    "id": record["warc_record_id"],
                    "text": record["text"],
                    "source": "nemotron",
                    "format": "text",
                    "metadata": {f"nemotron_{k}": v for k, v in record.items() if k not in ("warc_record_id", "text")},
                }
                zst.write(json.dumps(dolma_record).encode() + b"\n")
                num_records += 1
        local_tmp.flush()

        fs, resolved_path = url_to_fs(output_file_path)
        fs.mkdirs(os.path.dirname(resolved_path), exist_ok=True)
        fs.put(local_tmp.name, resolved_path)

    return {"input_file": input_file_path, "output_file": output_file_path, "num_records": num_records}


@dataclass
class NemotronIngressConfig:
    output_path: str = THIS_OUTPUT_PATH


def download_nemotron_cc(cfg: NemotronIngressConfig):
    paths_file_path = os.path.join(cfg.output_path, "data-jsonl.paths")
    logger.info(f"Downloading Nemotron CC path file {paths_file_path}")

    with open_url(NCC_PATH_FILE_URL, "rb") as f, open_url(paths_file_path, "wb") as f_out:
        f_out.write(f.read())

    logger.info(f"Reading paths from {paths_file_path}")
    all_files = []
    with open_url(paths_file_path, "r", compression="gzip") as f:
        for line in f:
            file = line.strip()
            output_file_path = os.path.join(cfg.output_path, file).replace("jsonl.zstd", "jsonl.zst")
            all_files.append((file, output_file_path))

    logger.info(f"Processing {len(all_files)} Nemotron CC files")

    pipeline = (
        Dataset.from_list(all_files)
        .filter(lambda file_info: not fsspec_exists(file_info[1]))
        .map(lambda file_info: download_single_nemotron_path(*file_info))
        .write_jsonl(os.path.join(cfg.output_path, ".metrics/download-{shard:05d}.jsonl"), skip_existing=True)
    )

    # Each worker downloads a ~350MB zstd file and decompresses to ~1.5-2GB in memory.
    # Default ZephyrContext resources (1GB) causes OOMKill; 4GB gives sufficient headroom.
    ctx = ZephyrContext(name="download-nemotron-cc", resources=ResourceConfig(cpu=1, ram="4g"))
    ctx.execute(pipeline)

    logger.info(f"Downloaded Nemotron CC files to {cfg.output_path}")


@draccus.wrap()
def main(cfg: NemotronIngressConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    download_nemotron_cc(cfg)
