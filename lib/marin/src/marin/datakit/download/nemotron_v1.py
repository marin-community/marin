# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and process Nemotron-CC dataset from Common Crawl"""

import json
import logging
import os
from collections.abc import Iterator

import requests
import zstandard
from fray.cluster import ResourceConfig
from requests.adapters import HTTPAdapter
from rigging.filesystem import open_url
from urllib3.util import Retry
from zephyr import Dataset, ZephyrContext
from zephyr.writers import atomic_rename

from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_exists

logger = logging.getLogger(__name__)

myagent = "marin-nemotron-ingress/1.0"
NCC_BASE_URL = "https://data.commoncrawl.org"
NCC_PATHS_SUFFIX = "contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"


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


def download_single_nemotron_path(input_file_path: str, output_file_path: str, base_url: str = NCC_BASE_URL) -> dict:
    """Fetches content from a Common Crawl path, streaming records to zstd output."""
    cc_url = f"{base_url}/{input_file_path}"
    logger.info(f"Downloading Nemotron CC file {cc_url} to {output_file_path}")

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    response = session.get(cc_url, headers={"user-agent": myagent}, stream=True)
    response.raise_for_status()

    num_records = 0
    with atomic_rename(output_file_path) as temp_path:
        with open_url(temp_path, "w", compression="zstd") as out:
            for record in _iter_jsonl_from_zstd_stream(response.raw):
                dolma_record = {
                    "id": record["warc_record_id"],
                    "text": record["text"],
                    "source": "nemotron",
                    "format": "text",
                    "metadata": {f"nemotron_{k}": v for k, v in record.items() if k not in ("warc_record_id", "text")},
                }
                print(json.dumps(dolma_record), file=out)
                num_records += 1

    return {"input_file": input_file_path, "output_file": output_file_path, "num_records": num_records}


def download_nemotron_cc(output_path: str, base_url: str = NCC_BASE_URL) -> None:
    """Download and process Nemotron-CC dataset from Common Crawl."""

    paths_file_path = os.path.join(output_path, "data-jsonl.paths")
    paths_file_url = f"{base_url}/{NCC_PATHS_SUFFIX}"
    logger.info(f"Downloading Nemotron CC path file {paths_file_path}")

    with open_url(paths_file_url, "rb") as f, open_url(paths_file_path, "wb") as f_out:
        f_out.write(f.read())

    logger.info(f"Reading paths from {paths_file_path}")
    all_files = []
    with open_url(paths_file_path, "r", compression="gzip") as f:
        for line in f:
            file = line.strip()
            output_file_path = os.path.join(output_path, file).replace("jsonl.zstd", "jsonl.zst")
            all_files.append((file, output_file_path))

    logger.info(f"Processing {len(all_files)} Nemotron CC files")

    pipeline = (
        Dataset.from_list(all_files)
        .filter(lambda file_info: not fsspec_exists(file_info[1]))
        .map(lambda file_info: download_single_nemotron_path(file_info[0], file_info[1], base_url=base_url))
        .write_jsonl(os.path.join(output_path, ".metrics/download-{shard:05d}.jsonl"), skip_existing=True)
    )

    # Each worker downloads a ~350MB zstd file and decompresses to ~1.5-2GB in memory.
    # Default ZephyrContext resources (1GB) causes OOMKill; 4GB gives sufficient headroom.
    ctx = ZephyrContext(name="download-nemotron-cc", resources=ResourceConfig(cpu=1, ram="4g"))
    ctx.execute(pipeline)

    logger.info(f"Downloaded Nemotron CC files to {output_path}")


def download_nemotron_v1_step() -> StepSpec:
    """Create a StepSpec that downloads the Nemotron-CC dataset from Common Crawl."""

    return StepSpec(
        name="raw/nemotron_v1",
        fn=lambda output_path: download_nemotron_cc(output_path=output_path),
        # NOTE: use the existing output to avoid re-downloading. Yes this is missing the `n`.
        override_output_path="raw/nemotro-cc-eeb783",
    )


_NEMOTRON_V1_DATA_ROOT = "contrib/Nemotron/Nemotron-CC/data-jsonl"

# Maps split name → relative path under data-jsonl/ that the normalize
# step should point at. Each split gets its own normalize StepSpec because
# normalize now processes a single directory (no subdirectory grouping).
NEMOTRON_V1_SPLITS: dict[str, str] = {
    "hq_actual": "quality=high/kind=actual",
    "hq_synth": "quality=high/kind=synthetic",
    "medium_high": "quality=medium-high",
    "medium": "quality=medium",
    "medium_low": "quality=medium-low",
    "low_actual": "quality=low/kind=actual",
    "low_synth": "quality=low/kind=synthetic",
}


def normalize_nemotron_v1_step(download: StepSpec, *, split: str) -> StepSpec:
    """Normalize one Nemotron-CC v1 split.

    The download writes dolma-format records ``{id, text, source, format,
    metadata}`` as ``.jsonl.zst`` under nested ``quality=<x>/kind=<y>/``
    directories. Each split gets its own normalize step pointing at the
    corresponding subdirectory.
    """
    if split not in NEMOTRON_V1_SPLITS:
        raise ValueError(f"Unknown split {split!r}. Choose from: {sorted(NEMOTRON_V1_SPLITS)}")
    rel_path = NEMOTRON_V1_SPLITS[split]
    return normalize_step(
        name=f"normalized/nemotron_v1/{split}",
        download=download,
        text_field="text",
        id_field="id",
        file_extensions=(".jsonl.zst",),
        relative_input_path=f"{_NEMOTRON_V1_DATA_ROOT}/{rel_path}",
    )
