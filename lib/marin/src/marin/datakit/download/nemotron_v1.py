# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and process Nemotron-CC dataset from Common Crawl"""

import json
import logging

from fray.cluster import ResourceConfig
from rigging.filesystem import StoragePath, atomic_rename, open_url, prefix_join
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit.download.http_session import build_retrying_session
from marin.datakit.download.zstd_jsonl import iter_jsonl_from_zstd_stream
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

myagent = "marin-nemotron-ingress/1.0"
NCC_BASE_URL = "https://data.commoncrawl.org"
NCC_PATHS_SUFFIX = "contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"


def download_single_nemotron_path(input_file_path: str, output_file_path: str, base_url: str = NCC_BASE_URL) -> dict:
    """Fetches content from a Common Crawl path, streaming records to zstd output."""
    cc_url = f"{base_url}/{input_file_path}"
    logger.info(f"Downloading Nemotron CC file {cc_url} to {output_file_path}")

    session = build_retrying_session(status_forcelist=(500, 502, 503, 504))

    response = session.get(cc_url, headers={"user-agent": myagent}, stream=True)
    response.raise_for_status()

    num_records = 0
    with atomic_rename(output_file_path) as temp_path:
        with open_url(temp_path, "w", compression="zstd") as out:
            for record in iter_jsonl_from_zstd_stream(response.raw):
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

    paths_file_path = prefix_join(output_path, "data-jsonl.paths")
    paths_file_url = f"{base_url}/{NCC_PATHS_SUFFIX}"
    logger.info(f"Downloading Nemotron CC path file {paths_file_path}")

    with open_url(paths_file_url, "rb") as f, open_url(paths_file_path, "wb") as f_out:
        f_out.write(f.read())

    logger.info(f"Reading paths from {paths_file_path}")
    all_files = []
    with open_url(paths_file_path, "r", compression="gzip") as f:
        for line in f:
            file = line.strip()
            output_file_path = prefix_join(output_path, file).replace("jsonl.zstd", "jsonl.zst")
            all_files.append((file, output_file_path))

    logger.info(f"Processing {len(all_files)} Nemotron CC files")

    pipeline = (
        Dataset.from_list(all_files)
        .filter(lambda file_info: not StoragePath(file_info[1]).exists())
        .map(lambda file_info: download_single_nemotron_path(file_info[0], file_info[1], base_url=base_url))
        .write_jsonl(prefix_join(output_path, ".metrics/download-{shard:05d}.jsonl"), skip_existing=True)
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
